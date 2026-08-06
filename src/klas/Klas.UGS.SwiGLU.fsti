module Klas.UGS.SwiGLU
#lang-pulse

(* Concrete, extractable public entry point for wmma_reference/wmma_ugs.cu:
   composes two applications of [Klas.UGS.Projection.projection],
   computing FP16 A[M,K] @ FP16 W_gate[K,N] and A[M,K] @ W_up[K,N],
   with [Kuiper.Kernel.UGS.Epilogue.epilogue], which applies SwiGLU
   pointwise to the two projections.

   The verified computation and its mathematical specification contain
   three ordinary matrices: A, W_gate, and W_up.

   Unlike [Klas.UGS.Projection]/[Kuiper.Kernel.UGS.Round]/
   [Kuiper.Kernel.UGS.Epilogue] (all [inline_for_extraction noextract],
   meant to be inlined at a concrete instantiation site), this [swiglu]
   function is a plain, dynamically-sized (M/K/N are runtime [szp]
   parameters, not compile-time constants) [fn] -- following the same
   style as [KWitness1.fst]'s [matmul_f32] -- so it extracts to a real,
   callable CUDA/C++ function.

   CALLER REQUIREMENTS (see [swiglu]'s signature below for the exact
   formal statement):
     - [m] and [n] must be multiples of 64, and [k] a multiple of 16.
     - [(m / 64) * (n / 64) <= max_blocks] (2^21): each projection uses
       one CUDA block per 64x64 output tile.
     - [gA] must be 16-byte aligned (true of any cudaMalloc'd pointer).
     - The FP32 accumulator and FP16 rounded-projection buffers are
       implementation details allocated and freed inside [swiglu], so
       neither enlarges the public API or its memory-ownership contract.
     - [gOut], the final FP16 SwiGLU output (M-by-N), is caller-provided.

   NUMERIC CORRECTNESS: this module's postcondition propagates
   [Klas.UGS.Projection.projection]'s strengthened guarantee -- that the
   rounded FP16 projection outputs approximate
   ([Kuiper.Approximates]'s [%~]) the real-number matmuls of [gA] with
   [gWGate] and [gWUp] -- all the way through to [gOut], composed with
   [Kuiper.Kernel.UGS.Epilogue]'s real-valued SwiGLU specification
   ([real_swiglu]/[chest_comb]) via
   [epilogue_matrix_approx]. The public postcondition states only the
   final, fully-composed guarantee: [gOut] approximates the
   *mathematical* SwiGLU of the real matmul,
   [chest_comb real_swiglu
      (MS.matmul A W_gate) (MS.matmul A W_up)],
   i.e. [gOut]'s FP16 bits are a numerically faithful approximation of
   [g * sigmoid(g) * u] evaluated exactly on the true real-number
   products. Exact intermediate
   projection and epilogue facts remain proved in the component
   contracts but are intentionally hidden from this minimal audit
   surface. This whole chain (WMMA matmul -> FP32 accumulate -> FP16
   round -> SwiGLU epilogue) is proved admit/assume/magic-free. *)

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

open Klas.UGS.Projection
open Kuiper.Kernel.UGS.Epilogue

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
fn swiglu
  (m k n : szp)
  (#lGate #lUp : layout2 k n)
  {| ctlayout lGate, ctlayout lUp |}
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (m % 64 == 0))
  (#_ : squash (k % 16 == 0))
  (#_ : squash (n % 64 == 0))
  (#_ : squash ((m / 64) * (n / 64) <= max_blocks))
  (gA : array2 half (rm m k) { is_global gA })
  (gWGate : array2 half lGate { is_global gWGate })
  (gWUp : array2 half lUp { is_global gWUp })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#_ : squash (aligned 16 (core gA)))
  (#eA : chest2 half m k)
  (#eWGate #eWUp : chest2 half k n)
  (#eOut0 : chest2 half m n)
  (#fA #fWGate #fWUp : perm)
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gWGate |-> Frac fWGate eWGate) **
    on gpu_loc (gWUp |-> Frac fWUp eWUp)
  requires
    on gpu_loc (gOut |-> eOut0)
  ensures
    (exists* (eOut : chest2 half m n).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ chest_comb real_swiglu
        (MS.matmul (to_real_matrix eA) (to_real_matrix eWGate))
        (MS.matmul (to_real_matrix eA) (to_real_matrix eWUp))))
