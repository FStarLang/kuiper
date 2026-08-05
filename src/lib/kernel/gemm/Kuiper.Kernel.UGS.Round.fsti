module Kuiper.Kernel.UGS.Round
#lang-pulse

(* Rounds a FP32 array2 to a FP16 array2, cell by cell, using the exact
   round-to-nearest-even conversion CUDA's [__float2half_rn] performs (see
   [Kuiper.Float.Casts.Base.cast_f32_to_f16], which extracts to exactly
   that intrinsic).

   This is the second half of the wmma_ugs.cu [projection_wmma_kernel]
   witness: that kernel accumulates the WMMA matmul into a FP32
   [__shared__] tile and then has each lane convert 256/32 = 8 elements of
   that tile via [__float2half_rn] into the FP16 output [P]. Here the FP32
   accumulator instead lives in a (global) scratch array2 (see
   [Klas.UGS.Projection] for why), and this module performs the
   elementwise conversion step from that scratch buffer into the final
   FP16 output. The numerical content of the conversion (round-to-nearest
   FP32->FP16 per cell) is identical either way. *)

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

module SZ = Kuiper.SizeT

(* Overwrites every cell of [gP] with [cast_f32_to_f16] of the
   corresponding cell of [gPf32], while leaving [gPf32] untouched. A
   plain single-threaded Pulse [fn], meant to be launched directly via
   [Kuiper.Kernel.launch_kernel_1]. *)
inline_for_extraction noextract
fn round
  (#m #n : szp)
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  (gPf32 : array2 float (rm m n) { is_global gPf32 })
  (gP    : array2 half  (rm m n) { is_global gP })
  (#fP32 : perm)
  (#vPf32 : chest2 float (SZ.v m) (SZ.v n))
  (#vP0 : chest2 half (SZ.v m) (SZ.v n))
  requires
    gpu ** gPf32 |-> Frac fP32 vPf32 ** gP |-> vP0
  ensures
    gpu ** gPf32 |-> Frac fP32 vPf32 ** gP |-> chest_map cast_f32_to_f16 vPf32

(* Numeric-correctness bridge for [round]: if the FP32 accumulator [vPf32]
   as a whole "approximates" (in the sense of [Kuiper.Approximates]'s [%~])
   some real-valued matrix [rM] -- e.g. because it is the result of
   [Klas.GEMM.TensorCore2D.Inst.spec], whose own postcondition proves
   exactly this against [Kuiper.Spec.GEMM.matmul] -- then the FP16 result
   of rounding every cell via [cast_f32_to_f16] *also* approximates that
   very same real matrix [rM]. This lets [Klas.UGS.Projection] propagate a
   genuine numeric-correctness guarantee through the FP32->FP16 rounding
   step performed by [round], rather than stopping at "this is whatever
   [round] computed" (an equality with no numeric content). *)
val round_preserves_approx
  (#m #n : nat)
  (vPf32 : chest2 float m n)
  (rM : chest2 real m n)
  : Lemma (requires vPf32 %~ rM)
          (ensures chest_map cast_f32_to_f16 vPf32 %~ rM)
