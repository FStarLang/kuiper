module KWitness8

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp8.cu.

   imp8.cu is the classic "2D block-tiling" SGEMM (Simon Boehm's optimized kernel).
   Despite its shared-memory blocking and register tiling, the arithmetic it
   performs per output cell (r, c) is:

       float acc = 0.0f;                       // = threadResults
       for (int kk = 0; kk < K; kk++)          // forward, across all tiles in order
         acc += A[r*K+kk] * B[kk*N+c];
       C[r*N+c] = alpha * acc + beta * C[r*N+c];

   i.e. a forward, left-associated dot product (exactly MS.matmul_single, the order
   computed by imp1/imp6 and the library's naive GEMM kernel), followed by the
   general-SGEMM affine combine `C := alpha * (A*B) + beta * C`. The block/register
   tiling only changes data movement and which thread owns which cell -- it does not
   change the per-cell floating-point operations -- so this witness reproduces imp8
   with the EXISTING verified naive matmul kernel (Kuiper.Kernel.GEMM.Naive), whose
   per-cell combine step `comb (C_old) (dot)` is instantiated to
   `alpha * dot + beta * C_old`. Instantiated at f32, row-major, it extracts to the
   same per-cell arithmetic as imp8.cu, hence is bit-equivalent. (The grid shape
   differs -- one block per output cell, no shared staging -- but that does not
   affect the per-cell result.)

   The spec is stated over the reals: the f32 result approximates
   `alpha * (matmul rA rB) + beta * C_old`, where rA, rB are the real matrices the
   f32 inputs approximate and alpha, beta are taken exactly (to_real). With
   alpha = 1, beta = 0 this degenerates to KWitness1's pure-matmul spec. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Array2
module SZ = Kuiper.SizeT

(* The real-valued SGEMM result: cell (i, j) is alpha*(A*B)[i][j] + beta*C[i][j]. *)
let gemm_real
  (#m #k #n : nat)
  (alpha beta : real)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  : ematrix real m n
  = mkM (fun i j -> (alpha *. macc (MS.matmul rA rB) i j) +. (beta *. macc rC i j))

(* Concrete, monomorphic spec: f32, row-major A, B and C, with runtime scalars
   alpha and beta.
   - requires SZ.v m * SZ.v n <= SZ.v max_blocks: the naive kernel launches one
     block per output cell.
   - requires the f32 inputs approximate the real matrices rA, rB.
   - ensures the resulting f32 matrix approximates alpha*(A*B) + beta*C_old. *)
fn matmul_f32
  (m n k : szp)
  (alpha beta : f32)
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
      pure (eC' %~ gemm_real (to_real alpha) (to_real beta) rA rB (to_real_matrix eC)))
