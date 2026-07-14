module KWitness4

(* Implementation of the imp4.cu witness (Kahan-compensated K-accumulation). See
   KWitness4.fsti for the specification and the bit-equivalence rationale.

   This is a thin f32, row-major instantiation of the library's verified Kahan
   GEMM kernel Kuiper.Kernel.GEMM.Naive3 (which uses Kuiper.DotProd.kahan_dotprod
   internally), mirroring how KWitness1 instantiates the naive forward kernel. *)

#lang-pulse
open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Tensor
module SZ = Kuiper.SizeT
module K = Kuiper.Kernel.GEMM.Naive3

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
{
  M.tensor_pts_to_ref_located gA;
  M.tensor_pts_to_ref_located gB;
  M.tensor_pts_to_ref_located gC;

  K.mmcomb_gpu_approx
    (fun _o n -> n) (fun _o n -> n)
    gA gB gC
    rA rB (to_real_matrix eC);
  ()
}
