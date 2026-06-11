module KWitness3

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp3.cu.

   imp3.cu computes, for each output cell (row, col), a *tiled* accumulation with
   tile width 16:

       float sum = 0.0f;
       for (int k0 = 0; k0 < k; k0 += 16) {
         float acc = 0.0f;
         for (int k1 = 0; k1 < 16 && k0 + k1 < k; k1++)
           acc += a[row * k + k0 + k1] * b[(k0 + k1) * n + col];
         sum += acc;
       }
       c[row * n + col] = sum;

   i.e. a forward, left-associated sum *within* each 16-wide tile (from 0.0f),
   and a forward, left-associated sum *across* the tile accumulators. This is yet
   another floating-point bracketing -- different from imp1 (flat forward) and
   imp2 (flat reverse). The implementation (see KWitness3.fst) reproduces this
   exact bracketing with a SINGLE block of a SINGLE thread, so there is no
   permission splitting or block decomposition.

   The spec below is the SAME real-valued spec as KWitness1/KWitness2: the f32
   result approximates MS.matmul rA rB, the order-independent mathematical
   matmul. It differs only in the size precondition. Because real addition is
   associative and commutative, the tiling (regrouping of the sum) is irrelevant
   at the mathematical level -- which is exactly the statement that imp1, imp2 and
   imp3 are "algebraically equivalent when ignoring floating-point error". *)

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
