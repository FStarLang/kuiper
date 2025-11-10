module Kuiper.MatMulTranspose

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMGPU.Type
module M = Kuiper.Matrix
module R = Kuiper.Matrix.Reprs
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
module GT = Kuiper.Ghost.Transpose

(* An example of computing tr(AB) by just shifting a view.
Basically:
  - Instantiating rA=rB=row_major, rC=col_major
  - Do the product, we get C = AB (in col-major)
  - View-shift C to get tr(AB) in row-major

  TODO: It would be nicer to do this just over matmul,
  but there is no view-like interface for CPU arrays.
*)
inline_for_extraction noextract
fn matmul_transpose_gpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (#et : Type0) {| scalar et |}
  (#rows : szp)
  (#shared : szp)
  (#cols : szp)
  (gA : M.gpu_matrix et (R.row_major rows shared) { M.is_global_matrix gA })
  (gB : M.gpu_matrix et (R.row_major shared cols) { M.is_global_matrix gB })
  (gC : M.gpu_matrix et (R.row_major cols rows) { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et cols rows)
  preserves
    cpu **
    on gpu_loc (gA |-> eA) **
    on gpu_loc (gB |-> eB)
  requires
    pure (size_req rows shared cols) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> mtranspose (MS.matmul eA eB))
{
  (* Recall that the lengths fit. We don't get a good error without this,
     but the problem is that we cannot call the crepr instance for gA/gB/gC
     without this fact. *)
  M.gpu_matrix_pts_to_ref_located gA;
  M.gpu_matrix_pts_to_ref_located gB;
  M.gpu_matrix_pts_to_ref_located gC;
  map_loc gpu_loc (fun () -> GT.ghost_transpose1 gC);
  mmcomb_gpu MS.comb2 gA gB (GT.row2col gC);
  map_loc gpu_loc (fun () -> GT.ghost_transpose1_back gC);
}

let matmul_transpose_gpu_f32_ff #rows #shared #cols =
  matmul_transpose_gpu
    Kuiper.Poly.GEMM.Naive.mmcomb_gpu
    #f32 #_
    #rows #shared #cols
