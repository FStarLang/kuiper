module Kuiper.ArrayView
#lang-pulse

open Kuiper
open Kuiper.Bijection
module B = Kuiper.Array (* base *)
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

let varray #a #len #vt vw =
  B.gpu_array a len

let core a = a

let varray_pts_to
  (#et:Type) (#len : erased nat) (#vt:_) (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : vt)
  : slprop
  =
    B.gpu_pts_to_array a #f (vw.bij.gg v)

inline_for_extraction noextract
fn varray_concr
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0)
  (#vw : aview t len vt)
  (a : varray vw)
  (#v : vt)
  requires
    a |-> v
  ensures
    core a |-> vw.bij.gg v
{
  unfold varray_pts_to a v;
}

inline_for_extraction noextract
fn varray_abs
  (#t:Type0)
  (#len0 : erased nat) (#vt0:Type0) (#vw0 : aview t len0 vt0)
  (a : varray vw0)
  (#len : erased nat) (#vt:Type0) (vw : aview t len vt)
  (#v : vt)
  requires
    core a |-> vw.bij.gg v
  returns
    a' : varray vw
  ensures
    pure (len0 == len /\ core a == core a') **
    (a' |-> v)
{
  gpu_pts_to_ref (core a);
  let a' : varray vw = core a;
  rewrite each core a as a';
  fold varray_pts_to #t #len #vt #vw a' #1.0R v;
  a'
}

inline_for_extraction noextract
fn varray_alloc0
  (#et:Type) {| sized et |}
  (len : SZ.t) (#vt:Type0) (vw : aview et len vt)
  preserves
    cpu
  requires
    pure (SZ.fits len)
  returns
    a : varray vw
  ensures
    exists* v. a |-> v
{
  let a = B.gpu_array_alloc #et len;
  with v.
    assert (a |-> v);
  rewrite each v as vw.bij.gg (vw.bij.ff v);
  fold varray_pts_to #et #len #vt #vw a #1.0R (vw.bij.ff v);
  a
}

// inline_for_extraction noextract
// fn varray_alloc1
//   (#et:Type) {| sized et |}
//   (#len : SZ.t) (#vt:Type0) (vw : aview et len vt)
//   (v0 : vt)
//   preserves
//     cpu
//   requires
//     pure (SZ.fits len)
//   returns
//     a : varray vw
//   ensures
//     a |-> v0
// {
//   let a = varray_alloc0 #et len vw;
//   (* fill? *)
//   admit();
// }

inline_for_extraction noextract
fn varray_free
  (#et:Type) {| sized et |}
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : varray vw)
  (#v : vt)
  preserves
    cpu
  requires
    a |-> v
  ensures emp
{
  unfold varray_pts_to a v;
  B.gpu_array_free a;
}

ghost
fn varray_share_n
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (#[T.exact (`0)] uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    varray_pts_to a #f v
  ensures
    bigstar #uid 0 k (fun _ -> varray_pts_to a #(f /. k) v)
{
  unfold varray_pts_to a #f v;
  B.gpu_slice_share #uid a 0 len k;
}

ghost
fn varray_gather_n
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (#uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    bigstar #uid 0 k (fun _ -> varray_pts_to a #(f /. k) v)
  ensures
    varray_pts_to a #f v
{
  B.gpu_slice_gather #uid a 0 len k;
  fold varray_pts_to a #f v;
}

inline_for_extraction noextract
fn varray_read
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : varray vw)
  (i : vw.it)
  (#f : perm)
  (#v : vt)
  requires
    gpu **
    varray_pts_to a #f v
  returns
    e : et
  ensures
    gpu **
    varray_pts_to a #f v **
    pure (e == vw.acc i v)
{
  let ci = cidx vw i;
  unfold varray_pts_to a #f v;
  let r = B.gpu_array_read #et #len #0 #len a #f ci;
  vw.galois1 v i;
  fold varray_pts_to a #f v;
  r
}

inline_for_extraction noextract
fn varray_write
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : varray vw)
  (i : vw.it)
  (e : et)
  (#f : perm)
  (#v0 : vt)
  requires
    gpu **
    varray_pts_to a v0
  ensures
    gpu **
    varray_pts_to a (vw.upd i e v0)
{
  let ci = cidx vw i;
  unfold varray_pts_to a v0;
  B.gpu_array_write #et #len #0 #len a ci e;
  vw.galois2 v0 i e;
  fold varray_pts_to a (vw.upd i e v0);
  ()
}

let varray_pts_to_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.it)
  (v : et)
  : slprop
  = gpu_pts_to_slice a #f (vw.ibij.gg i) (vw.ibij.gg i + 1) seq![v]

inline_for_extraction noextract
fn varray_read_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : varray vw)
  (i : vw.it)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    varray_pts_to_cell a #f i v0
  returns
    v : et
  ensures
    gpu **
    varray_pts_to_cell a #f i v **
    pure (v == v0)
{
  let ci = cidx vw i;
  unfold varray_pts_to_cell a #f i v0;
  let r = B.gpu_array_read #et #len #ci #(ci+1) a #f ci;
  fold varray_pts_to_cell a #f i v0;
  r
}

inline_for_extraction noextract
fn varray_write_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : varray vw)
  (i : vw.it)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    varray_pts_to_cell a i v0
  ensures
    gpu **
    varray_pts_to_cell a i v1
{
  let ci = cidx vw i;
  unfold varray_pts_to_cell a i v0;
  B.gpu_array_write #_ #_ #ci #(ci+1) a ci v1;
  with s'. assert (B.gpu_pts_to_slice a ci (ci+1) s');
  Kuiper.Seq.Common.lem_one_elem s' v1;
  fold varray_pts_to_cell a i v1;
  ()
}

ghost
fn varray_explode
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    varray_pts_to a #f v
  ensures
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.acc i v)
{
  unfold varray_pts_to a #f v;
  B.gpu_array_slice_1 a;
  rewrite
    bigstar 0 len
     (fun i -> gpu_pts_to_slice a #f i (i+1) seq![vw.bij.gg v @! i])
  as
    bigstar 0 (Enumerable.cardinal (natlt len))
      (fun i -> gpu_pts_to_slice a #f i (i+1) seq![vw.bij.gg v @! i]);
  forevery_fromstar #(natlt len)
      (fun i -> gpu_pts_to_slice a #f i (i+1) seq![vw.bij.gg v @! i]);
  forevery_iso vw.ibij _;
  assert
    forall+ (i:vw.it).
      gpu_pts_to_slice a #f (vw.ibij.gg i) (vw.ibij.gg i + 1) seq![vw.bij.gg v @! vw.ibij.gg i];

  Classical.forall_intro_2 vw.galois1;
  forevery_ext #vw.it
    (fun i -> gpu_pts_to_slice a #f (vw.ibij.gg i) (vw.ibij.gg i + 1) seq![vw.bij.gg v @! vw.ibij.gg i])
    (fun i -> gpu_pts_to_slice a #f (vw.ibij.gg i) (vw.ibij.gg i + 1) seq![vw.acc i v]);
}

ghost
fn varray_implode
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.acc i v)
  ensures
    varray_pts_to a #f v
{
  Classical.forall_intro_2 vw.galois1;
  forevery_ext #vw.it
    (fun i -> gpu_pts_to_slice a #f (vw.ibij.gg i) (vw.ibij.gg i + 1) seq![vw.acc i v])
    (fun i -> gpu_pts_to_slice a #f (vw.ibij.gg i) (vw.ibij.gg i + 1) seq![vw.bij.gg v @! vw.ibij.gg i]);
  forevery_iso_back vw.ibij
      (fun i -> gpu_pts_to_slice a #f i (i+1) seq![vw.bij.gg v @! i]);
  forevery_tostar #(natlt len)
      (fun i -> gpu_pts_to_slice a #f i (i+1) seq![vw.bij.gg v @! i]);
  B.gpu_array_unslice_1 a;
  fold varray_pts_to a #f v;
}
