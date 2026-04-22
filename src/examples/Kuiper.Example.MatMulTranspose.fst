module Kuiper.Example.MatMulTranspose

#lang-pulse
open Kuiper
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg

module M = Kuiper.Array2
module MS = Kuiper.Spec.GEMM
module TGT = Kuiper.Ghost.TensorTranspose

inline_for_extraction noextract
fn matmul_transpose_gpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (#et : Type0) {| scalar et |}
  (#m #n #k : szp)
  (gA : M.t et (l2_row_major m k) { M.is_global gA })
  (gB : M.t et (l2_row_major k n) { M.is_global gB })
  (gC : M.t et (l2_row_major n m) { M.is_global gC })
  (#eA #eB #eC : ematrix et _ _)
  preserves
    cpu **
    on gpu_loc (gA |-> eA ** gB |-> eB)
  requires
    pure (size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> mtranspose (MS.matmul eA eB))
{
  (* Recall that the lengths fit. We don't get a good error without this,
     but the problem is that we cannot call the crepr instance for gA/gB/gC
     without this fact. *)
  map_loc gpu_loc (fun () -> M.pts_to_ref gA);
  map_loc gpu_loc (fun () -> M.pts_to_ref gB);
  map_loc gpu_loc (fun () -> M.pts_to_ref gC);

  map_loc gpu_loc (fun () -> TGT.ghost_transpose1 gC);
  mmcomb_gpu MS.comb2 gA gB (TGT.row2col gC);
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1_back gC);
}

[@@Comment
"An example of computing tr(AB) by just shifting a view. Basically:
  - Instantiating rA=rB=row_major, rC=col_major
  - Do the product, we get C = AB (in col-major)
  - View-shift C to get tr(AB) in row-major

TODO: It would be nicer to do this just over a CPU-side matmul, but there is no
view-like interface for CPU arrays."]
let matmul_transpose_gpu_f32_ff #m #n #k =
  matmul_transpose_gpu
    Kuiper.Kernel.GEMM.Naive.mmcomb_gpu_exact
    #f32 #_
    #m #n #k
