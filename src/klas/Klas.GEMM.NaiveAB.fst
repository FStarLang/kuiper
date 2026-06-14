module Klas.GEMM.NaiveAB

(* Naive GEMM with alpha/beta scaling:  C := alpha * A * B + beta * C,
   for an m x k matrix A, a k x n matrix B and an m x n matrix C.

   The existing Klas.GEMM.Naive only computes C := A * B (comb2); this is the
   general linear-combination (gemm) version using the verified naive kernel,
   which - unlike the tiled variants - has no tile-divisibility constraint on
   the dimensions.

   Setting n = 1 makes B a single-column vector and yields cuBLAS gemv
   (y := alpha * A * x + beta * y); the generated C entry point then takes the
   usual (alpha, beta, A, x/B, y/C) pointers. Transpose is selected via the
   chosen matrix layout (row- vs column-major). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.Naive

inline_for_extraction noextract
fn spec
  (et : Type0) {| scalar et, real_like et |}
  (repA repB repC : trepr2)
  {| ctrepr2 repA, ctrepr2 repB, ctrepr2 repC |}
  (m n k : szp)
  (alpha beta : et)
  (gA : tensor et (repA m k) { is_global gA })
  (gB : tensor et (repB k n) { is_global gB })
  (gC : tensor et (repC m n) { is_global gC })
  (rA rB : chest2 real _ _)
  (#eA #eB #eC : chest2 _ _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures (
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb (MS.rlincomb (to_real alpha) (to_real beta))
                            (Kuiper.Chest.to_real_chest eC) rA rB))
{
  tensor_pts_to_ref_located gA;
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;

  K.mmcomb_gpu_approx
    (MS.lincomb alpha beta) (MS.rlincomb (to_real alpha) (to_real beta))
    gA gB gC
    rA rB (Kuiper.Chest.to_real_chest eC);
  ()
}

(* gemm with alpha/beta; gemv is the n = 1 instance. *)
let gemm_f32_rrr = spec f32 l2_row_major l2_row_major l2_row_major
let gemm_f64_rrr = spec f64 l2_row_major l2_row_major l2_row_major
let gemm_u32_rrr = spec u32 l2_row_major l2_row_major l2_row_major
let gemm_u64_rrr = spec u64 l2_row_major l2_row_major l2_row_major

let gemm_f32_ccc = spec f32 l2_col_major l2_col_major l2_col_major
let gemm_f64_ccc = spec f64 l2_col_major l2_col_major l2_col_major
