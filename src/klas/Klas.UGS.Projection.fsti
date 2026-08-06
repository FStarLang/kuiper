module Klas.UGS.Projection
#lang-pulse

(* See Klas.UGS.Projection.fst for the full design rationale. This module
   is the projection-side ("Pass 1" / [projection_wmma_kernel]) witness
   foundation for wmma_reference/wmma_ugs.cu:

     FP16 row-major A[M,K] @ a logical FP16 W[K,N] via m16n16k16 WMMA,
     FP32 accumulation initialized to zero, K tiles visited in increasing
     order, result rounded to FP16 via __float2half_rn.

   [projection] reuses the existing, unmodified, *functionally verified*
   [Klas.GEMM.TensorCore2D.Inst.spec] for the tensor-core accumulation
   (unlike the numerically-opaque [Klas.GEMM.TensorCore.Inst.specialize_gpu]
   this module used previously, [spec]'s own postcondition proves its FP32
   output approximates [Kuiper.Spec.GEMM.matmul] of the real-number
   images of its inputs -- see the "NUMERIC CORRECTNESS" paragraph below),
   instantiated with a 64x64 block tile, 16x16x16 WMMA fragments, and a
   single 4x4-fragment warp per block (so exactly one CUDA block computes
   each 64x64 output tile, with the full (dynamic) K handled by that
   warp's own [K/16]-iteration loop, structurally analogous to
   wmma_ugs.cu's per-warp `for (k = 0; k < K; k += 16)` loop, just body
   handling a 4x4 grid of 16x16 WMMA tiles instead of a single one),
   followed by this witness's own [Kuiper.Kernel.UGS.Round.round] for the
   FP32 -> FP16 rounding step.

   CALLER REQUIREMENTS (see [projection]'s signature below for the exact
   formal statement):
     - [m] must be a multiple of 64, [k] must be a multiple of 16, and
       [n2] must be a multiple of 64 (the 64x64x16 block tile of the
       chosen [spec] instantiation; stricter than the underlying WMMA
       tile's own 16-multiple requirement, adopted here so that [spec]'s
       internal runtime divisibility guards -- see
       [Klas.GEMM.TensorCore2D.Inst.fst]'s [dguard] calls -- are honestly
       reflected in this wrapper's own contract).
     - [(m / 64) * (n2 / 64) <= max_blocks] (2^21): the projection uses
       one CUDA block per 64x64 output tile, so this bounds the output
       size.
     - [gA] must be 16-byte aligned. [gW] may use any executable layout;
       this wrapper materializes it into internal row-major scratch
       before invoking the vectorized tensor-core implementation.
     - [gPf32], the FP32 scratch accumulator array, and [gP], the FP16
       output array, are caller-provided scratch: their contents on entry
       are irrelevant, since [spec] zero-initializes and *overwrites*
       (rather than accumulates into) its output.

   NUMERIC CORRECTNESS: this module's postcondition states, with no
   admits, both that the FP32 accumulator [eC'] approximates
   ([Kuiper.Approximates]'s [%~]) the real-number matmul of [gA]'s and
   [gW]'s real images, *and* that the FP16-rounded result stored into
   [gP] approximates that very same real matmul (via
   [Kuiper.Kernel.UGS.Round.round_preserves_approx], which lifts the
   FP32-level approximation across the [__float2half_rn] rounding step
   using [Kuiper.Float.Casts.Base.cast_f32_to_f16_ok]). This is a
   genuinely stronger numeric guarantee than a previous version of this
   module carried (which only proved the FP16 output was *some* correctly
   rounded copy of *whatever* the tensor-core primitive computed, with no
   tie back to the abstract real-number matmul). *)

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
fn projection
  (m k n2 : szp)
  (#lW : layout2 k n2) {| ctlayout lW |}
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n2)))
  (#_ : squash (SZ.fits (m * n2)))
  (#_ : squash (SZ.v m % 64 == 0))
  (#_ : squash (SZ.v k % 16 == 0))
  (#_ : squash (SZ.v n2 % 64 == 0))
  (gA : array2 half (rm m k) { is_global gA })
  (gW : array2 half lW { is_global gW })
  (gPf32 : array2 float (rm m n2) { is_global gPf32 })
  (gP : array2 half (rm m n2) { is_global gP })
  (#_ : squash (aligned 16 (core gA)))
  (#eA : chest2 half m k)
  (#eW : chest2 half k n2)
  (#eC0 : chest2 float m n2)
  (#eP0 : chest2 half m n2)
  (#fA #fW : perm)
  preserves
    cpu **
    pure ((SZ.v m / 64) * (SZ.v n2 / 64) <= SZ.v max_blocks) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gW |-> Frac fW eW)
  requires
    on gpu_loc (gPf32 |-> eC0) **
    on gpu_loc (gP |-> eP0)
  ensures
    (exists* (eC' : chest2 float (SZ.v m) (SZ.v n2)).
      on gpu_loc (gPf32 |-> eC') **
      on gpu_loc (gP |-> chest_map cast_f32_to_f16 eC') **
      pure (
        eC' %~ MS.matmul (to_real_matrix eA) (to_real_matrix eW) /\
        chest_map cast_f32_to_f16 eC' %~ MS.matmul (to_real_matrix eA) (to_real_matrix eW)))
