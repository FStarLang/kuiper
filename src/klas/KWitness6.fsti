module KWitness6

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp6.cu.

   imp6.cu is a SHARED-MEMORY tiled (tile = 16) matmul: each block cooperatively
   stages 16x16 tiles of A and B into __shared__ memory and the threads read from
   there. The tiling is a *memory-locality* optimization only -- crucially, each
   thread accumulates into a single running `sum` straight across all the tiles,
   WITHOUT a per-tile partial:

       float sum = 0.0f;
       for (int kk = 0; kk < k; kk += 16) {
         ... stage cA, cB ...
         for (int t = 0; t < 16; t++)
           sum += cA[sr][t] * cB[t][sc];     // same `sum`, no per-tile reset
       }
       C[r * n + col] = sum;

   So although the *data movement* is tiled, the floating-point accumulation order
   is plain forward, left-associated from 0.0f over t = 0 .. k-1 -- exactly the
   order of imp1.cu, and exactly MS.matmul_single, the forward left-associated sum
   from `zero` computed by the library's naive GEMM kernel
   (Kuiper.Kernel.GEMM.Naive). (Contrast imp3.cu, which resets a per-tile partial
   `acc = 0` and does `sum += acc`, giving a genuinely tiled bracketing.)

   The witness therefore reuses the SAME forward kernel as KWitness1: instantiated
   at f32, row-major, it extracts to the same per-cell arithmetic as imp6.cu, hence
   is bit-equivalent. (The grid/shared-memory shape differs -- one block per output
   cell, no shared staging -- but that does not affect the per-cell floating-point
   result.) This is the point of the relational-validation suite: tiling for memory
   locality does not change the result, so a memory-optimized kernel can be
   bit-equivalent to the naive one.

   The spec below is stated over the *reals*: the f32 result approximates
   MS.matmul rA rB, the order-independent mathematical matmul. This is the SAME
   spec as KWitness1..5.matmul_f32, differing only in the size precondition. *)

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
