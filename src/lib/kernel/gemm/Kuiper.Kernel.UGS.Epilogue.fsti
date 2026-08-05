module Kuiper.Kernel.UGS.Epilogue
#lang-pulse

(* Foundation-level witness for wmma_reference/wmma_ugs.cu's pass-2
   [swiglu_epilogue_kernel]: one (conceptual) thread per output element
   (m, n), reading the FP16 gate/up projections out of [P : half[M,2N]]
   (deinterleaved via the [kPairGroupN = 64] group/pos split), and
   reproducing the CUTLASS epilogue arithmetic exactly:

     g      = __half2float(gate_h)
     sig    = 1.0f / (1.0f + expf(-g))
     silu_f = g * sig
     silu_h = __float2half_rn(silu_f)
     out    = __hmul(silu_h, up_h)

   Here this is done with a single-threaded loop (foundation-level, see
   [Klas.UGS.SwiGLU] for the composed, extractable entry point), matching
   this codebase's approach in [Kuiper.Kernel.UGS.Round]. *)

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

module SZ = Kuiper.SizeT

(* The pair-group width from wmma_ugs.cu's [kPairGroupN]. *)
unfold
let pair_group_n : nat = 64

(* Pure per-cell spec of the SwiGLU epilogue, using exactly the scalar
   operations ([cast_f16_to_f32], [sub], [add], [div], [fexp],
   [cast_f32_to_f16], native FP16 [mul]) that wmma_ugs.cu's C expression
   compiles to (modulo -use_fast_math's substitution of [expf]/[/], which
   is a hardware/compiler detail outside this abstract model; see
   [Kuiper.Floating.Base.fexp]'s own opaque spec). *)
[@@CPrologue "__device__"; "KrmlPrivate"]
let epilogue_cell (gate_h up_h : half) : half =
  let g : float = cast_f16_to_f32 gate_h in
  let sgm : float = div one (add one (fexp (zero `sub` g))) in
  let silu_f : float = mul g sgm in
  let silu_h : half = cast_f32_to_f16 silu_f in
  mul silu_h up_h

(* The exact real-number ("mathematical") SwiGLU epilogue that
   [epilogue_cell] is meant to approximate: [g * sigmoid(g) * u] with
   [sigmoid(g) = 1 / (1 + exp(-g))], written directly in terms of
   [Kuiper.Real]'s [real] arithmetic ([+.], [-.], [*.], [/.]) and
   [FStar.Math.Exp.exp], mirroring [epilogue_cell]'s own
   [div one (add one (fexp (zero \`sub\` g)))] / [mul g sgm] / [mul silu_h up_h]
   structure exactly (just at the real, rather than floating-point,
   level -- no separate FP32/FP16 rounding steps, since real numbers are
   exact). *)
unfold
let real_epilogue_cell (g u : real) : real =
  (g /. (1.0R +. exp (0.0R -. g))) *. u

(* [epilogue_cell] is a numerically faithful (admit/assume/magic-free)
   approximation of [real_epilogue_cell]: if the FP16 [gate_h]/[up_h]
   approximate ([Kuiper.Approximates]'s [%~]) real numbers [gR]/[uR],
   then the exact device computation [epilogue_cell gate_h up_h]
   approximates [real_epilogue_cell gR uR]. This is proved purely by
   composing this codebase's existing floating-point/real approximation
   algebra ([Kuiper.Approximates.Base]'s [a_add]/[a_mul]/[sub_approx]/
   [exp_approx]/[div_approx], plus [Kuiper.Float.Casts.Base]'s
   [cast_f16_to_f32_ok]/[cast_f32_to_f16_ok]), using
   [FStar.Math.Exp.exp_positive] to justify that the sigmoid's
   denominator is never zero. *)
val epilogue_cell_approx (gate_h up_h : half) (gR uR : real)
  : Lemma (requires gate_h %~ gR /\ up_h %~ uR)
          (ensures epilogue_cell gate_h up_h %~ real_epilogue_cell gR uR)

(* [gate_col_of c] / [up_col_of c] deinterleave a logical output column [c]
   (in [0, n)) into the two physical columns (in [0, 2n)) of the [P]
   buffer holding the gate and up projections, per wmma_ugs.cu's
   [group = n / 64; pos = n % 64; gate_col = group*128+pos; up_col =
   gate_col + 64]. *)
unfold
let gate_col_of (c : nat) : nat = 128 * (c / pair_group_n) + (c % pair_group_n)
unfold
let up_col_of (c : nat) : nat = gate_col_of c + pair_group_n

(* If [n] is a multiple of the pair-group width, every logical column
   [c < n] deinterleaves into two physical columns strictly below [2n]. *)
val lemma_gate_up_bound (n c : nat)
  : Lemma (requires n % pair_group_n == 0 /\ c < n)
          (ensures gate_col_of c < 2 * n /\ up_col_of c < 2 * n)

(* [gate_col_of]/[up_col_of], packaged with their in-bounds proof so call
   sites never need to re-derive [lemma_gate_up_bound] themselves. *)
unfold
let gate_col_idx (n : nat { n % pair_group_n == 0 }) (c : natlt n) : natlt (2 * n) =
  lemma_gate_up_bound n c;
  gate_col_of c

unfold
let up_col_idx (n : nat { n % pair_group_n == 0 }) (c : natlt n) : natlt (2 * n) =
  lemma_gate_up_bound n c;
  up_col_of c

(* The matrix-level real-number SwiGLU epilogue: applies
   [real_epilogue_cell] to the deinterleaved gate/up columns of an
   [m]-by-[2n] real "pre-activation" matrix [rP] (e.g. the exact
   real-number matmul of a projection's real inputs), producing an
   [m]-by-[n] real matrix. This is the mathematical function that a
   correct SwiGLU implementation's FP16 output is meant to approximate. *)
unfold
let real_epilogue
  (#m : nat) (n : nat { n % pair_group_n == 0 })
  (rP : chest2 real m (2 * n))
  : chest2 real m n
  = mk2 (fun (r : natlt m) (c : natlt n) ->
      real_epilogue_cell (acc2 rP r (gate_col_idx n c)) (acc2 rP r (up_col_idx n c)))

(* Matrix-level lifting of [epilogue_cell_approx]: if [vP] (the FP16
   gate/up projection buffer) approximates a real matrix [rP], and
   [eOut] is *exactly* the (device) epilogue applied pointwise to [vP]'s
   deinterleaved gate/up columns (i.e. [eOut] is any chest satisfying the
   same pointwise equation [epilogue]'s own postcondition establishes for
   its result), then [eOut] approximates [real_epilogue n rP]. *)
val epilogue_matrix_approx
  (#m : nat) (n : nat { n % pair_group_n == 0 })
  (vP : chest2 half m (2 * n))
  (rP : chest2 real m (2 * n))
  (eOut : chest2 half m n)
  : Lemma (requires
             vP %~ rP /\
             (forall (r : natlt m) (c : natlt n).
                acc2 eOut r c ==
                  epilogue_cell (acc2 vP r (gate_col_idx n c)) (acc2 vP r (up_col_idx n c))))
          (ensures eOut %~ real_epilogue n rP)

(* Overwrites every cell of [gOut] (an [m]-by-[n] chest) with the SwiGLU
   epilogue applied to the corresponding gate/up cells of [gP] (an
   [m]-by-[n2] chest, [n2 = 2n]), leaving [gP] untouched. [n] must be a
   multiple of [pair_group_n] (64), matching wmma_ugs.cu's implicit
   assumption that the gate/up interleave groups tile [N] exactly (true
   for any realistic hidden dimension, e.g. the reference driver's
   default N = 128). *)
inline_for_extraction noextract
fn epilogue
  (#m #n #n2 : szp)
  (#_ : squash (SZ.v n2 == 2 * SZ.v n))
  (#_ : squash (SZ.v n % pair_group_n == 0))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  (gP : array2 half (rm m n2) { is_global gP })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#fP : perm)
  (#vP : chest2 half (SZ.v m) (SZ.v n2))
  (#vOut0 : chest2 half (SZ.v m) (SZ.v n))
  requires
    gpu ** gP |-> Frac fP vP ** gOut |-> vOut0
  ensures
    gpu ** gP |-> Frac fP vP **
    (exists* (vOut : chest2 half (SZ.v m) (SZ.v n)).
      gOut |-> vOut **
      pure (
        forall (r : natlt (SZ.v m)) (c : natlt (SZ.v n)).
          acc2 vOut r c ==
            epilogue_cell
              (acc2 vP r (gate_col_idx (SZ.v n) c))
              (acc2 vP r (up_col_idx (SZ.v n) c))))
