module KWitness2

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp2.cu.

   imp2.cu computes, for each output cell (row, col):

       float sum = 0.0f;
       for (int i = k-1; i >= 0; i--)
         sum += a[row * k + i] * b[i * n + col];
       c[row * n + col] = sum;

   i.e. a *reverse*, left-associated accumulation starting from 0.0f. No existing
   Kuiper kernel accumulates in this order, so the implementation (see
   KWitness2.fst) builds one: a SINGLE block of a SINGLE thread performs the whole
   matmul, accumulating each cell's dot product from index k-1 down to 0. That one
   thread owns all of gA, gB, gC outright, so there is no permission splitting, no
   block decomposition, and no setup/teardown ghost code.

   The spec below is the SAME real-valued spec as KWitness1.matmul_f32 (the imp1
   witness): the f32 result approximates MS.matmul rA rB, the order-independent
   mathematical matmul. It differs only in the size precondition. Because real
   addition is associative and commutative, the reverse accumulation order is
   irrelevant at the mathematical level -- which is exactly the statement that
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
   - requires only SZ.fits (SZ.v m * SZ.v n): the single thread loops over the
     m*n output cells.
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
    pure (SZ.fits (SZ.v m * SZ.v n)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix f32 m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB))
