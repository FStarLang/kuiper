module KWitness7

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp7.cu.

   imp7.cu is the shared-memory tiled GEMM of imp6.cu with ONE change: the final
   store writes the result TRANSPOSED,

       C[c * n + r] = sum;     // imp6 writes C[r * n + c]

   so the cell (r, c) of A*B is stored at physical cell (c, r) of C. With the
   `c * n + r` indexing this is a clean transpose precisely when the output is
   SQUARE (m == n); hence this witness is parameterized over a single dimension n
   (A is n x k, B is k x n, C is n x n). The per-cell accumulation is the same
   plain forward, left-associated dot product as imp1/imp6 (the tiling moves data
   but keeps a single running `sum` across tiles), so each computed value is
   MS.matmul_single -- it is only the placement that is transposed.

   The witness builds a SINGLE block of a SINGLE thread that performs the whole
   matmul: for each output cell it accumulates the forward dot product and writes
   it to the transposed position. That one thread owns all of gA, gB, gC outright,
   so there is no permission splitting, no block decomposition, and no
   setup/teardown ghost code.

   The spec below states that the f32 result approximates `mtranspose (matmul rA rB)`,
   i.e. the transpose of the order-independent real matmul. This is the SAME
   underlying matmul spec as KWitness1..6 -- imp7 is "imp6 composed with a
   transpose" -- made precise by the `mtranspose`. (mtranspose preserves the
   matrix type only when it is square, which is why the spec is stated over the
   square n x n output.) *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Array2
module SZ = Kuiper.SizeT

(* Concrete, monomorphic spec: f32, row-major A (n x k), B (k x n), C (n x n).
   - requires only SZ.fits (SZ.v n * SZ.v n): the single thread loops over the
     n*n output cells.
   - requires the f32 inputs approximate the real matrices rA, rB.
   - ensures the resulting f32 matrix approximates the TRANSPOSE of the exact
     real matmul. *)
fn matmul_f32
  (n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major n k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major n n) { M.is_global gC })
  (rA : ematrix real n k)
  (rB : ematrix real k n)
  (#eA : ematrix f32 n k)
  (#eB : ematrix f32 k n)
  (#eC : ematrix f32 n n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.fits (SZ.v n * SZ.v n)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix f32 n n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ mtranspose (MS.matmul rA rB)))
