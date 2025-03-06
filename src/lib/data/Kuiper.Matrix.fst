module Kuiper.Matrix
#lang-pulse

open Kuiper
module T = FStar.Tactics.V2

let gpu_matrix (et:Type0) (rows cols : nat) : Type0 =
  gpu_array et (rows * cols)

let gpu_matrix_pts_to
  (#et:Type) (#rows #cols : nat)
  ([@@@mkey] gm : gpu_matrix et rows cols)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop
  = gpu_pts_to_array gm #f em.s

fn gpu_matrix_read
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (#f:perm)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm #f em
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to gm #f em **
    pure (v == macc em i j)
{
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm #f em;
  gpu_pts_to_ref gm;
  let idx = i *^ cols +^ j;
  let v = gpu_array_read #et #(rows * cols) #0 #(rows * cols) gm idx;
  fold gpu_matrix_pts_to gm #f em;
  v;
}

fn gpu_matrix_write
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t) (* NOTE! This is a concrete argument *)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (vv : et)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em i j vv)
{
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm em;
  gpu_pts_to_ref gm;
  let idx = i *^ cols +^ j;
  gpu_array_write #et #(rows * SZ.v cols) #0 #(rows * SZ.v cols) gm idx vv;
  fold gpu_matrix_pts_to gm (mupd em i j vv);
}

let gpu_matrix_pts_to_cell
  (#et:Type) (#rows #cols : nat)
  ([@@@mkey] gm : gpu_matrix et rows cols)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et)
  : slprop
  = gpu_pts_to_slice gm #f (i * cols + j) (i * cols + j + 1) seq![v]

fn gpu_matrix_read_cell
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm #f i j v0
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm #f i j v **
    pure (v == v0)
{
  open FStar.SizeT;
  unfold gpu_matrix_pts_to_cell gm #f i j v0;
  gpu_pts_to_slice_ref #et #(rows * cols) #f gm _ _ #(seq![reveal v0]);
  let idx = i *^ cols +^ j;
  let v = gpu_array_read #et #(rows * cols) #idx #(idx+1) gm idx;
  fold gpu_matrix_pts_to_cell gm #f i j v0;
  v;
}

fn gpu_matrix_write_cell
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t) (* NOTE! This is a concrete argument *)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm i j v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm i j v1
{
  open FStar.SizeT;
  unfold gpu_matrix_pts_to_cell gm i j v0;
  gpu_pts_to_slice_ref #et #(rows * cols) gm _ _ #(seq![reveal v0]);
  let idx = i *^ cols +^ j;
  assert (gpu_pts_to_slice gm idx (idx+1) seq![reveal v0]);
  gpu_array_write #et #(rows * cols) #idx #(idx+1) gm idx v1;
  with s'. assert (gpu_pts_to_slice gm idx (idx+1) s');
  Kuiper.Seq.Common.lem_one_elem s' v1;
  assert (gpu_pts_to_slice gm idx (idx+1) seq![v1]);
  fold gpu_matrix_pts_to_cell gm i j v1;
}
