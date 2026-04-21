module Kuiper.GEMM.Naive3

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMGPU.Type
module M = Kuiper.Array2
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
module P = Kuiper.Poly.GEMM.Naive3

(* As matmul, row-major for now. *)
inline_for_extraction noextract
fn spec
  (et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m n k : szp)
  (gA : M.array2 et (l2_row_major m k) { M.is_global gA })
  (gB : M.array2 et (l2_row_major k n) { M.is_global gB })
  (gC : M.array2 et (l2_row_major m n) { M.is_global gC })
  (rA rB : ematrix real _ _)
  (#eA #eB #eC : ematrix _ _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (P.size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures (
    (exists* (eC' : ematrix et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB)))
{
  map_loc gpu_loc (fun () -> M.pts_to_ref gA);
  map_loc gpu_loc (fun () -> M.pts_to_ref gB);
  map_loc gpu_loc (fun () -> M.pts_to_ref gC);

  P.mmcomb_gpu_approx
    (fun _o n -> n) (fun _o n -> n)
    gA gB gC
    rA rB (to_real_matrix eC);
  ()
}

let g_matmul_f32_rrr = spec f32
let g_matmul_f64_rrr = spec f64
