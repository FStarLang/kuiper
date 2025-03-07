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

inline_for_extraction noextract
fn gpu_matrix_alloc
  (#et:Type) {| scalar et |}
  (rows cols : szp)
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    gm : gpu_matrix et rows cols
  ensures
    exists* em. gm |-> em
{
  open FStar.SizeT;
  let gm = gpu_array_alloc #et (rows *^ cols);
  with s. assert (gpu_pts_to_array gm #1.0R s);
  let em = M #et #rows #cols s;
  // let em = M <| seq![zero #et];
  // assert (gpu_pts_to_array gm #1.0R em.s);
  fold (gpu_matrix_pts_to gm em);
  gm;
}

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type) {| scalar et |}
  (#rows #cols : erased nat)
  (gm : gpu_matrix et rows cols)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp
{
  unfold gpu_matrix_pts_to gm em;
  gpu_array_free gm;
}

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type) {| scalar et |}
  (#rows #cols : szp)
  (a : vec et)
  (gA : gpu_matrix et rows cols)
  (#s : erased (seq et){ len s == rows * cols })
  preserves
    (a |-> s) **
    cpu
  requires
    (gA |-> 'm0) **
    pure (SZ.fits (rows * cols))
  ensures
    gA |-> M s
{
  unfold gpu_matrix_pts_to gA 'm0;
  Kuiper.Array.gpu_memcpy_host_to_device gA a (SZ.mul rows cols);
  fold gpu_matrix_pts_to gA (M s);
  ();
}

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type) {| scalar et |}
  (#rows #cols : szp)
  (a : vec et)
  (gA : gpu_matrix et rows cols)
  (#s : ematrix et rows cols)
  preserves
    (gA |-> s) **
    cpu
  requires
    (a |-> 's0) **
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols)
  ensures
    a |-> M?.s s
{
  Pulse.Lib.Vec.pts_to_len a;
  unfold gpu_matrix_pts_to gA s;
  Kuiper.Array.gpu_memcpy_device_to_host a gA (SZ.mul rows cols);
  fold gpu_matrix_pts_to gA s;
  ();
}

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
{
  admit(); // just tedious
}

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
  ensures
    gpu_matrix_pts_to gm #f em
{
  admit(); // just tedious
}

inline_for_extraction noextract
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

inline_for_extraction noextract
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

inline_for_extraction noextract
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

inline_for_extraction noextract
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

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    bigstar #uid 0 (rows * cols) (fun i ->
      gpu_matrix_pts_to_cell gm #f (i / cols) (i % cols) (macc em (i / cols) (i % cols)))
{
  admit(); // just tedious
}

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    bigstar #uid 0 (rows * cols) (fun i ->
      gpu_matrix_pts_to_cell gm #f (i / cols) (i % cols) (macc em (i / cols) (i % cols)))
  ensures
    gpu_matrix_pts_to gm #f em
{
  admit(); // just tedious
}
