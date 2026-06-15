module Klas.Gemv

(* cuBLAS gemv (no-transpose):  y := alpha * A * x + beta * y,
   with A an m x k matrix, x a k-vector, y an m-vector.

   This is the alpha/beta GEMM (Klas.GEMM.NaiveAB) with the output width n = 1:
   the m x k matrix A times the k x 1 column x is the m x 1 column A*x, scaled
   by alpha and accumulated into y with beta. We reuse the verified naive GEMM
   kernel and only specialize the output width, so the spec is exact at the
   element type. (The transpose variant is obtained by passing A in column-major
   layout, exactly as for gemm; here we fix the natural row-major layout.)

   Generated C entry: gemv_f32/f64/u32/u64(m, k, alpha, beta, A, x, y). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module NB = Klas.GEMM.NaiveAB

inline_for_extraction noextract
fn gemv_spec
  (et : Type0) {| scalar et, real_like et |}
  (repA repB repC : trepr2)
  {| ctrepr2 repA, ctrepr2 repB, ctrepr2 repC |}
  (m k : szp)
  (alpha beta : et)
  (gA : tensor et (repA m k) { is_global gA })      (* A : m x k *)
  (gx : tensor et (repB k 1sz) { is_global gx })    (* x : k x 1 *)
  (gy : tensor et (repC m 1sz) { is_global gy })    (* y : m x 1 *)
  (rA rB : chest2 real _ _)
  (#eA #eB #eC : chest2 _ _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gx |-> Frac fB eB)
  requires
    pure (m * 1sz <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gy |-> eC)
  ensures (
    exists* (eC' : chest2 et m 1).
      on gpu_loc (gy |-> eC') **
      pure (eC' %~ MS.mmcomb (MS.rlincomb (to_real alpha) (to_real beta))
                            (Kuiper.Chest.to_real_chest eC) rA rB))
{
  NB.spec et repA repB repC m 1sz k alpha beta gA gx gy rA rB;
}

let gemv_f32 = gemv_spec f32 l2_row_major l2_row_major l2_row_major
let gemv_f64 = gemv_spec f64 l2_row_major l2_row_major l2_row_major
let gemv_u32 = gemv_spec u32 l2_row_major l2_row_major l2_row_major
let gemv_u64 = gemv_spec u64 l2_row_major l2_row_major l2_row_major
