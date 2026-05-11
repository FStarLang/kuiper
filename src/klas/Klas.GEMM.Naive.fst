module Klas.GEMM.Naive

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix
open Kuiper.Array2
module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.Naive

inline_for_extraction noextract
fn spec
  (et : Type0) {| scalar et, real_like et |}
  (repA repB repC : trepr2)
  {| ctrepr2 repA, ctrepr2 repB, ctrepr2 repC |}
  (m n k : szp)
  (gA : array2 et (repA m k) { is_global gA })
  (gB : array2 et (repB k n) { is_global gB })
  (gC : array2 et (repC m n) { is_global gC })
  (rA rB : ematrix real _ _)
  (#eA #eB #eC : ematrix _ _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (K.size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures (
    (exists* (eC' : ematrix et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB)))
{
  pts_to_ref_located gA;
  pts_to_ref_located gB;
  pts_to_ref_located gC;

  K.mmcomb_gpu_approx
    (fun _o n -> n) (fun _o n -> n)
    gA gB gC
    rA rB (to_real_matrix eC);
  ()
}

let g_matmul_f32_rrr = spec f32 l2_row_major l2_row_major l2_row_major
let g_matmul_f64_rrr = spec f64 l2_row_major l2_row_major l2_row_major
let g_matmul_u32_rrr = spec u32 l2_row_major l2_row_major l2_row_major
let g_matmul_u64_rrr = spec u64 l2_row_major l2_row_major l2_row_major

let g_matmul_f32_ccc = spec f32 l2_col_major l2_col_major l2_col_major
let g_matmul_f64_ccc = spec f64 l2_col_major l2_col_major l2_col_major
let g_matmul_u32_ccc = spec u32 l2_col_major l2_col_major l2_col_major
let g_matmul_u64_ccc = spec u64 l2_col_major l2_col_major l2_col_major