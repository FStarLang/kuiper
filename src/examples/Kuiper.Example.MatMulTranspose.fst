module Kuiper.Example.MatMulTranspose

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg

module MS = Kuiper.Spec.GEMM
module TGT = Kuiper.Ghost.TensorTranspose

[@@Comment
"An example of computing tr(AB) by just shifting a view. Basically:
  - Instantiating rA=rB=row_major, rC=col_major
  - Do the product, we get C = AB (in col-major)
  - View-shift C to get tr(AB) in row-major"]
inline_for_extraction noextract
fn matmul_transpose_gpu
  (#et : Type0) {| scalar et |}
  (#m #n #k : szp)
  (gA : tensor et (l2_row_major m k) { is_global gA })
  (gB : tensor et (l2_row_major k n) { is_global gB })
  (gC : tensor et (l2_row_major n m) { is_global gC })
  (#eA #eB #eC : chest2 et _ _)
  preserves
    cpu **
    on gpu_loc (gA |-> eA ** gB |-> eB)
  requires
    pure (m * n <= max_blocks) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> mtranspose (MS.matmul eA eB))
{
  (* Recall that the lengths fit. We don't get a good error without this,
     but the problem is that we cannot call the crepr instance for gA/gB/gC
     without this fact. *)
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gA);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gB);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gC);

  map_loc gpu_loc (fun () -> TGT.ghost_transpose1 gC);
  Kuiper.Kernel.GEMM.Naive2.mmcomb_gpu_exact
    MS.comb2 gA gB (TGT.row2col gC);

  map_loc gpu_loc (fun () -> TGT.ghost_transpose1_back gC);
  ();
}
