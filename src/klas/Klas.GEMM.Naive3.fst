module Klas.GEMM.Naive3

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.Naive3

inline_for_extraction noextract
fn spec
  (et : Type0) {| floating et, real_like et, floating_real_like et |}
  (repA repB repC : trepr2)
  {| ctrepr2 repA, ctrepr2 repB, ctrepr2 repC |}
  (m n k : szp)
  (gA : tensor et (repA m k) { is_global gA })
  (gB : tensor et (repB k n) { is_global gB })
  (gC : tensor et (repC m n) { is_global gC })
  (rA rB : chest2 real _ _)
  (#eA #eB #eC : chest2 _ _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (K.size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures (
    (exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB)))
{
  tensor_pts_to_ref_located gA;
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;

  K.mmcomb_gpu_approx
    (fun _o n -> n) (fun _o n -> n)
    gA gB gC
    rA rB (Kuiper.Chest.to_real_chest eC);
  ()
}

let g_matmul_f32_rrr = spec f32 l2_row_major l2_row_major l2_row_major
let g_matmul_f64_rrr = spec f64 l2_row_major l2_row_major l2_row_major

let g_matmul_f32_ccc = spec f32 l2_col_major l2_col_major l2_col_major
let g_matmul_f64_ccc = spec f64 l2_col_major l2_col_major l2_col_major
