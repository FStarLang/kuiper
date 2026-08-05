module Klas.UGS.SwiGLU
#lang-pulse

(* Concrete, extractable public entry point for wmma_reference/wmma_ugs.cu:
   composes [Klas.UGS.Projection.projection] ("Pass 1" -
   [projection_wmma_kernel]: FP16 A[M,K] @ FP16 W[K,2N] via m16n16k16
   WMMA, FP32 zero-initialized accumulation, K tiles increasing, FP16
   round-to-nearest-even) with [Kuiper.Kernel.UGS.Epilogue.epilogue]
   ("Pass 2" - [swiglu_epilogue_kernel]: SwiGLU over the interleaved
   gate/up halves of the projection output), matching wmma_ugs.cu's
   two-kernel pipeline end to end.

   Unlike [Klas.UGS.Projection]/[Kuiper.Kernel.UGS.Round]/
   [Kuiper.Kernel.UGS.Epilogue] (all [inline_for_extraction noextract],
   meant to be inlined at a concrete instantiation site), this [swiglu]
   function is a plain, dynamically-sized (M/K/N are runtime [szp]
   parameters, not compile-time constants) [fn] -- following the same
   style as [KWitness1.fst]'s [matmul_f32] -- so it extracts to a real,
   callable CUDA/C++ function.

   CALLER REQUIREMENTS (see [swiglu]'s signature below for the exact
   formal statement):
     - [m] must be a multiple of 64 and [k] must be a multiple of 16.
       The derived projection width [2 * n] must be a multiple of 64
       (inherited from
       [Klas.UGS.Projection.projection]'s 64x64x16 block-tiled
       [Klas.GEMM.TensorCore2D.Inst.spec] instantiation).
     - [n % 64 == 0] (the SwiGLU gate/up interleave
       groups -- [kPairGroupN = 64] in wmma_ugs.cu -- must tile [N]
       exactly; true for any realistic hidden dimension, e.g. the
       reference driver's default N = 128). This also makes the derived
       [2 * n] projection width a multiple of 64.
     - [(m / 64) * ((2 * n) / 64) <= max_blocks] (2^21): the projection
       uses one CUDA block per 64x64 output tile.
     - [gA]/[gW] must be 16-byte aligned (true of any cudaMalloc'd
       pointer).
     - The FP32 accumulator and FP16 rounded-projection buffers are
       implementation details allocated and freed inside [swiglu], so
       neither enlarges the public API or its memory-ownership contract.
     - [gOut], the final FP16 SwiGLU output (M-by-N), is caller-provided.

   NUMERIC CORRECTNESS: this module's postcondition propagates
   [Klas.UGS.Projection.projection]'s strengthened guarantee -- that the
   rounded FP16 projection output approximates
   ([Kuiper.Approximates]'s [%~]) the real-number matmul of [gA]'s and
   [gW]'s real images -- all the way through to [gOut], composed with
   [Kuiper.Kernel.UGS.Epilogue]'s real-valued SwiGLU specification
   ([real_epilogue_cell]/[real_epilogue]) via
   [epilogue_matrix_approx]. The public postcondition states only the
   final, fully-composed guarantee: [gOut] approximates the
   *mathematical* SwiGLU of the real matmul,
   [real_epilogue n (MS.matmul (to_real_matrix eA) (to_real_matrix eW))],
   i.e. [gOut]'s FP16 bits are a numerically faithful approximation of
   [g * sigmoid(g) * u] evaluated exactly on the true real-number
   product of [gA]'s and [gW]'s real images. Exact intermediate
   projection and epilogue facts remain proved in the component
   contracts but are intentionally hidden from this minimal audit
   surface. This whole chain (WMMA matmul -> FP32 accumulate -> FP16
   round -> SwiGLU epilogue) is proved admit/assume/magic-free. *)

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

open Klas.UGS.Projection
open Kuiper.Kernel.UGS.Epilogue

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

fn swiglu
  (m k n : szp)
  (#_ : squash (n % pair_group_n == 0))
  (#_ : squash (SZ.fits (2 * n)))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * (2 * n))))
  (#_ : squash (SZ.fits (m * (2 * n))))
  (#_ : squash (m % 64 == 0))
  (#_ : squash (k % 16 == 0))
  (#_ : squash ((m / 64) * ((2 * n) / 64) <= max_blocks))
  (gA : array2 half (rm m k) { is_global gA })
  (gW : array2 half (rm k (2 * n)) { is_global gW })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gW)))
  (#eA : chest2 half m k)
  (#eW : chest2 half k (2 * n))
  (#eOut0 : chest2 half m n)
  (#fA #fW : perm)
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gW |-> Frac fW eW)
  requires
    on gpu_loc (gOut |-> eOut0)
  ensures
    (exists* (eOut : chest2 half m n).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ real_epilogue n
        (MS.matmul (to_real_matrix eA) (to_real_matrix eW))))
