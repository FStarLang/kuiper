module KWitness1

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp1.cu.

   imp1.cu computes, for each output cell (row, col):

       float sum = 0.0f;
       for (int i = 0; i < k; i++)
         sum += a[row * k + i] * b[i * n + col];
       c[row * n + col] = sum;

   i.e. a *forward*, left-associated accumulation starting from 0.0f. This is
   exactly MS.matmul_single, the forward left-associated sum from `zero` computed
   by the library's naive GEMM kernel (Kuiper.Kernel.GEMM.Naive). Instantiated at
   f32, row-major, it extracts to the same per-cell arithmetic as imp1.cu, hence
   is bit-equivalent. (The grid shape differs -- one block per output cell -- but
   that does not affect the per-cell floating-point result.)

   The spec below is stated over the *reals*: the f32 result approximates
   MS.matmul rA rB, the order-independent mathematical matmul. This is the SAME
   spec as KWitness2.matmul_f32 (the witness for imp2's reverse accumulation),
   differing only in the size precondition -- which is exactly the statement that
   imp1 and imp2 are "algebraically equivalent when ignoring floating-point
   error". *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Array2
module SZ = Kuiper.SizeT

(* Concrete, monomorphic spec: f32, row-major A, B and C.
   - requires SZ.v m * SZ.v n <= SZ.v max_blocks: the naive kernel launches one
     block per output cell.
   - requires the f32 inputs approximate the real matrices rA, rB.
   - ensures the resulting f32 matrix approximates the exact real matmul. *)
fn matmul_f32
  (m n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major m n) { M.is_global gC })
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (#eA : ematrix f32 m k)
  (#eB : ematrix f32 k n)
  (#eC : ematrix f32 m n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.v m * SZ.v n <= SZ.v max_blocks) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix f32 m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB))
