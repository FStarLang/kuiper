module Kuiper.Matrix
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.GhostMap
open Kuiper.EMatrix
module A = Kuiper.ArrayView
module T = FStar.Tactics.V2

inline_for_extraction noextract
let cview_from_clayout_ff
  (et : Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (c : clayout l)
  : szlt rows & szlt cols -> szlt (rows * cols)
  = fun (i, j) -> c.c_to i j <: szlt (rows * cols)

inline_for_extraction noextract
let cview_from_clayout_gg
  (et : Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (c : clayout l)
  : szlt (rows * cols) -> szlt rows & szlt cols
  = fun x -> (c.c_from1 x, c.c_from2 x)

inline_for_extraction noextract
instance cview_from_clayout
  (et : Type)
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (c : clayout l)
  : A.cview (aview_from_mlayout et l) =
{
  lenfits = ();
  cit = szlt rows & szlt cols;
  cibij = {
    ff = cview_from_clayout_ff et c;
    gg = cview_from_clayout_gg et c;
    ff_gg = ez;
    gg_ff = ez;
  };
}

let gpu_matrix (et:Type0) (#rows #cols : nat) (l : mlayout rows cols) : Type0 =
  A.varray (aview_from_mlayout et #rows #cols l)

let core g = A.core g
let core_match g1 g2 = A.core_match g1 g2

let gpu_matrix_pts_to
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop
  = A.varray_pts_to gm #f em

inline_for_extraction noextract
ghost
fn gpu_matrix_concr
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#em : ematrix et rows cols)
  requires
    g |-> em
  ensures
    core g |-> to_seq l em
{
  unfold gpu_matrix_pts_to g #1.0R em;
  let a' = A.varray_concr g;
  assert (pure (Seq.equal (to_seq l em) (A.to_seq (aview_from_mlayout et #rows #cols l) em)));
  a'
}

inline_for_extraction noextract
fn gpu_matrix_abs
  (#et:Type)
  (#rows #cols : erased nat) (l : mlayout rows cols)
  (p : gpu_array et (mlayout_size l))
  (#em : ematrix et rows cols)
  requires
    p |-> to_seq l em
  returns
    g' : gpu_matrix et l
  ensures
    pure (core g' == p) **
    (g' |-> em)
{
  assert (pure (Seq.equal (to_seq l em) (A.to_seq (aview_from_mlayout et l) em)));
  rewrite each to_seq l em as A.to_seq (aview_from_mlayout et l) em;
  let g' = A.varray_abs (aview_from_mlayout et l) p;
  fold gpu_matrix_pts_to g' em;
  g'
}

inline_for_extraction noextract
fn gpu_matrix_alloc0
  (#et:Type) {| sized et |}
  (rows cols : szp)
  (l : mlayout rows cols)
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    gm : gpu_matrix et l
  ensures
    exists* em. gm |-> em
{
  open FStar.SizeT;
  let gm = A.varray_alloc0 (rows *^ cols) (aview_from_mlayout et l);
  with s. assert (A.varray_pts_to gm #1.0R s);
  fold gpu_matrix_pts_to gm s;
  gm;
}

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp
{
  unfold gpu_matrix_pts_to gm em;
  A.varray_free gm;
}

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_share_n gm k;
  ghost
  fn aux (i:natlt k)
    requires A.varray_pts_to gm #(f /. k) em
    ensures  gpu_matrix_pts_to gm #(f /. k) em
  {
    fold gpu_matrix_pts_to gm #(f /. k) em;
  };
  bigstar_map #0 #uid #0 #k aux;
}

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
  ensures
    gpu_matrix_pts_to gm #f em
{
  ghost
  fn aux (i:natlt k)
    requires gpu_matrix_pts_to gm #(f /. k) em
    ensures  A.varray_pts_to gm #(f /. k) em
  {
    unfold gpu_matrix_pts_to gm #(f /. k) em;
  };
  bigstar_map #uid #0 #0 #k aux;
  A.varray_gather_n gm k;
  fold gpu_matrix_pts_to gm #f em;
}

inline_for_extraction noextract
fn gpu_matrix_read
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| cl : clayout l |}
  (gm : gpu_matrix et l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (#f : perm)
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
  unfold gpu_matrix_pts_to gm #f em;
  let r = A.varray_read gm (i,j);
  fold gpu_matrix_pts_to gm #f em;
  r
}

inline_for_extraction noextract
fn gpu_matrix_write
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (v : et)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em i j v)
{
  unfold gpu_matrix_pts_to gm em;
  A.varray_write gm (i,j) v;
  assert (pure (
    mupd em i j v
    `Kuiper.EMatrix.equal`
    (aview_from_mlayout et #rows #cols l).igm.upd em (A.cit_to_it _ (i,j)) v));
  fold gpu_matrix_pts_to gm (mupd em i j v);
}

let gpu_matrix_pts_to_cell
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et)
  : slprop
  = A.varray_pts_to_cell gm #f (i,j) v

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt cols)
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
  unfold gpu_matrix_pts_to_cell gm #f i j v0;
  (* very awkward *)
  rewrite
    each Mktuple2 #(natlt rows) #(natlt cols) (SZ.v i) (SZ.v j)
      as A.cit_to_it (aview_from_mlayout et l) (i, j);
  let v = A.varray_read_cell gm (i,j);
  with ai. assert (A.varray_pts_to_cell gm #f ai v0);
  rewrite each ai as Mktuple2 #(natlt rows) #(natlt cols) i j;
  fold gpu_matrix_pts_to_cell gm #f i j v0;
  v;
}

inline_for_extraction noextract
fn gpu_matrix_write_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt cols)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm i j v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm i j v1
{
  unfold gpu_matrix_pts_to_cell gm i j v0;
  (* very awkward *)
  rewrite
    each Mktuple2 #(natlt rows) #(natlt cols) (SZ.v i) (SZ.v j)
      as A.cit_to_it (aview_from_mlayout et l) (i, j);
  A.varray_write_cell gm (i,j) v1;
  with ai. assert (A.varray_pts_to_cell gm #1.0R ai v1);
  rewrite each ai as Mktuple2 #(natlt rows) #(natlt cols) i j;
  fold gpu_matrix_pts_to_cell gm i j v1;
}

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_explode gm;
  (* Change the type... convince pulse. *)
  with (ty:Type0) d ff. assert forevery ty #d ff;
  rewrite forevery ty #d ff as forevery (natlt rows & natlt cols) #d ff;

  forevery_ext #(natlt rows & natlt cols)
    (fun i -> A.varray_pts_to_cell gm #f i ((aview_from_mlayout et l).igm.acc em i))
    (fun i -> gpu_matrix_pts_to_cell gm #f i._1 i._2 (macc em i._1 i._2));
  forevery_unflatten #(natlt rows) #_ #(natlt cols)
    (fun r c -> gpu_matrix_pts_to_cell gm #f r c (macc em r c));
}

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
  ensures
    gpu_matrix_pts_to gm #f em
{
  forevery_flatten #(natlt rows) #_ #(natlt cols)
    (fun r c -> gpu_matrix_pts_to_cell gm #f r c (macc em r c));
  forevery_ext #(natlt rows & natlt cols)
    (fun i -> gpu_matrix_pts_to_cell gm #f i._1 i._2 (macc em i._1 i._2))
    (fun i -> A.varray_pts_to_cell gm #f i ((aview_from_mlayout et l).igm.acc em i));
  A.varray_implode gm;
  fold gpu_matrix_pts_to gm #f em;
}

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type0) {| sized et |}
  (#rows #cols : SZ.t)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    (a |-> s) **
    cpu
  requires
    (gm |-> em)
  ensures
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols) **
    (gm |-> from_seq l s)
{
  Pulse.Lib.Vec.pts_to_len a;
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm #1.0R em;
  A.varray_from_array #_ #_ #(rows *^ cols) gm a;
  from_seq_rel l s;
  fold gpu_matrix_pts_to gm #1.0R (from_seq l s);
}

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type0) {| sized et |}
  (#rows #cols : SZ.t)
  (#l : mlayout rows cols)
  (a : vec et)
  (gm : gpu_matrix et l)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    (gm |-> em) **
    cpu
  requires
    (a |-> s)
  ensures
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols) **
    (a |-> to_seq l em)
{
  Pulse.Lib.Vec.pts_to_len a;
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm #1.0R em;
  A.varray_to_array #_ #_ #(rows *^ cols) a gm;
  to_seq_rel l em;
  fold gpu_matrix_pts_to gm #1.0R em;
}
