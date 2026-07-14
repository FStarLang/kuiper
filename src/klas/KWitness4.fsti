module KWitness4

(* A fully-Kuiper, verified witness that is bit-for-bit equivalent to imp4.cu.

   imp4.cu computes, for each output cell (row, col), a KAHAN-compensated sum of
   the products a[row,i]*b[i,col]:

       float acc = 0.0f, comp = 0.0f;
       for (int i = 0; i < k; i++) {
         float yc = a[row * k + i] * b[i * n + col] - comp;
         float t  = acc + yc;
         comp = (t - acc) - yc;
         acc  = t;
       }
       c[row * n + col] = acc;

   The compensation term `comp` recovers low-order bits lost in `acc + yc`, so the
   running sum is far more accurate than the naive forward sum (imp1) -- but the
   bit pattern differs from imp1/imp2/imp3.

   The library already provides a verified Kahan GEMM kernel
   (Kuiper.Kernel.GEMM.Naive3, via Kuiper.DotProd.kahan_dotprod). This witness is
   a thin f32, row-major instantiation of it -- exactly as KWitness1 instantiates
   the naive forward kernel. Extracted, it produces the same per-cell Kahan
   arithmetic as imp4.cu, hence is bit-equivalent. (The grid differs.)

   The spec below is the SAME real-valued spec as KWitness1/2/3: the f32 result
   approximates MS.matmul rA rB, the order-independent mathematical matmul. Kahan
   summation still satisfies this approximation (it is a sum of the same products,
   just with error compensation) -- which is exactly the statement that imp1, imp2,
   imp3 and imp4 are all "algebraically equivalent when ignoring floating-point
   error". *)

#lang-pulse
open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Tensor
module SZ = Kuiper.SizeT

(* Concrete, monomorphic spec: f32, row-major A, B and C.
   - requires SZ.v m * SZ.v n <= SZ.v max_blocks * SZ.v max_threads: the Kahan
     kernel launches one thread per output cell (1024-thread blocks).
   - requires the f32 inputs approximate the real matrices rA, rB.
   - ensures the resulting f32 matrix approximates the exact real matmul. *)
fn matmul_f32
  (m n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major m n) { M.is_global gC })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#eA : chest2 f32 m k)
  (#eB : chest2 f32 k n)
  (#eC : chest2 f32 m n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.v m * SZ.v n <= SZ.v max_blocks * SZ.v max_threads) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 f32 m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB))
