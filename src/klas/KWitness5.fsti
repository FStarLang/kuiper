module KWitness5

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp5.cu.

   imp5.cu computes, for each output cell (row, col), a TILED matmul (tile = 16)
   whose cross-tile reduction is KAHAN-compensated:

       float acc = 0.0f, comp = 0.0f;
       for (int k0 = 0; k0 < k; k0 += 16) {
         float part = 0.0f;                       // forward sum within the tile
         for (int k1 = k0; k1 < min(k0 + 16, k); k1++)
           part += a[row * k + k1] * b[k1 * n + col];
         float yc = part - comp;                  // Kahan-combine the tile partial
         float t  = acc + yc;
         comp = (t - acc) - yc;
         acc  = t;
       }
       c[row * n + col] = acc;

   i.e. "imp3 (tiled, forward within a tile) but with a Kahan-compensated sum
   across the tiles". No existing Kuiper kernel reduces in this order, so the
   implementation (see KWitness5.fst) builds one: a SINGLE block of a SINGLE
   thread performs the whole matmul, summing each tile forward and combining the
   tile partials with the library Kahan combinator (Kuiper.Kahan.kahan_sum). That
   one thread owns all of gA, gB, gC outright, so there is no permission
   splitting, no block decomposition, and no setup/teardown ghost code.

   The spec below is the SAME real-valued spec as KWitness1.matmul_f32 (the imp1
   witness): the f32 result approximates MS.matmul rA rB, the order-independent
   mathematical matmul. It differs only in the size precondition. Because real
   addition is associative and commutative, neither the tiling nor the Kahan
   compensation changes the result at the mathematical level -- which is exactly
   the statement that imp5 is "algebraically equivalent, ignoring floating-point
   error" to imp1 / imp2 / imp3 / imp4. *)

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
