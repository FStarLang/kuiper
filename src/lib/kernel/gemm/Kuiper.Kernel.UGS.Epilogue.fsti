module Kuiper.Kernel.UGS.Epilogue
#lang-pulse

(* Foundation-level witness for the SwiGLU epilogue: one conceptual thread
   per output element, reading corresponding cells from separate FP16 gate
   and up projection matrices and reproducing the CUTLASS arithmetic exactly:

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
let real_swiglu (g u : real) : real =
  (g /. (1.0R +. exp (0.0R -. g))) *. u

(* [epilogue_cell] is a numerically faithful (admit/assume/magic-free)
   approximation of [real_swiglu]: if the FP16 [gate_h]/[up_h]
   approximate ([Kuiper.Approximates]'s [%~]) real numbers [gR]/[uR],
   then the exact device computation [epilogue_cell gate_h up_h]
   approximates [real_swiglu gR uR]. This is proved purely by
   composing this codebase's existing floating-point/real approximation
   algebra ([Kuiper.Approximates.Base]'s [a_add]/[a_mul]/[sub_approx]/
   [exp_approx]/[div_approx], plus [Kuiper.Float.Casts.Base]'s
   [cast_f16_to_f32_ok]/[cast_f32_to_f16_ok]), using
   [FStar.Math.Exp.exp_positive] to justify that the sigmoid's
   denominator is never zero. *)
val epilogue_cell_approx (gate_h up_h : half) (gR uR : real)
  : Lemma (requires gate_h %~ gR /\ up_h %~ uR)
          (ensures epilogue_cell gate_h up_h %~ real_swiglu gR uR)

(* Matrix-level lifting of [epilogue_cell_approx]. *)
val epilogue_matrix_approx
  (#m #n : nat)
  (vGate vUp : chest2 half m n)
  (rGate rUp : chest2 real m n)
  (eOut : chest2 half m n)
  : Lemma (requires
             vGate %~ rGate /\
             vUp %~ rUp /\
             (forall (r : natlt m) (c : natlt n).
                acc2 eOut r c ==
                  epilogue_cell (acc2 vGate r c) (acc2 vUp r c)))
          (ensures eOut %~ chest_comb real_swiglu rGate rUp)

(* Overwrites [gOut] with the pointwise SwiGLU of [gGate] and [gUp]. *)
inline_for_extraction noextract
fn epilogue
  (#m #n : szp)
  (#_ : squash (SZ.fits (m * n)))
  (gGate : array2 half (rm m n) { is_global gGate })
  (gUp : array2 half (rm m n) { is_global gUp })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#fGate #fUp : perm)
  (#vGate #vUp #vOut0 : chest2 half m n)
  requires
    gpu **
    gGate |-> Frac fGate vGate **
    gUp |-> Frac fUp vUp **
    gOut |-> vOut0
  ensures
    gpu **
    gGate |-> Frac fGate vGate **
    gUp |-> Frac fUp vUp **
    (exists* (vOut : chest2 half m n).
      gOut |-> vOut **
      pure (
        forall (r : natlt m) (c : natlt n).
          acc2 vOut r c ==
            epilogue_cell (acc2 vGate r c) (acc2 vUp r c)))
