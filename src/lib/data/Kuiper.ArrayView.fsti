module Kuiper.ArrayView
#lang-pulse

open Kuiper
open Kuiper.Bijection
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

// [@@erasable]
noeq
type aview (a : Type) (len : nat) (vt : Type) = {
  it : Type0; (* index type *)
  bij : lseq a len =~ vt; (* bijection to sequences *)
  (* the view type is a map from index type into element type *)
  acc : it -> vt -> GTot a;
  upd : it -> a -> vt -> GTot vt;
  ibij : natlt len =~ it; (* bijection of indexes. *)

  cidx : a:it -> c:sz{SZ.v c == ibij.gg a};
  aidx : c:sz{c < len} -> a:it{a == ibij.ff c};

  galois1 : (a:vt -> i:it -> Lemma (acc i a == (bij.gg a) @! (ibij.gg i)));
  galois2 : (abs:vt -> i:it -> x:a -> Lemma (bij.gg (upd i x abs) == Seq.upd (bij.gg abs) (ibij.gg i) x));
}

(* Needed to check the spec of explode/implode, as we iterate over the indices. *)
instance enumerable_view_it (#a:Type) (#len:nat) (#vt:Type) (vw : aview a len vt)
  : Enumerable.enumerable vw.it =
{
  _cardinal = len;
  bij = bij_sym vw.ibij;
}

let cidx (#a : Type) (#len : erased nat) (#vt : Type)
         (vw : aview a len vt)
  : (a:vw.it -> c:sz{SZ.v c == vw.ibij.gg a})
  = match vw with {cidx} -> cidx

inline_for_extraction noextract
val varray (#a : Type0) (#len : nat) (#vt : Type) (vw : aview a len vt) : Type0

inline_for_extraction noextract
val core
  (#a : Type)
  (#len : erased nat)
  (#vt : Type) (#vw : aview a len vt)
  (g : varray vw)
  : Kuiper.Array.gpu_array a len

val core_match
  (#et : Type)
  (#len : erased nat)
  (#vt : Type)
  (#vw : aview et len vt)
  (a1 a2 : varray vw)
  : Lemma (requires core a1 == core a2)
          (ensures  a1 == a2)

val varray_pts_to
  (#a:Type) (#len : erased nat) (#vt:_) (#vw : aview a len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : vt)
  : slprop

unfold
instance has_pts_to (#a:Type) (#len : nat) (#vt:Type) (#vw : aview a len vt)
  : has_pts_to (varray vw) vt = {
  pts_to = varray_pts_to;
}

(* These are really ghost steps only... but
since the varray type encodes the view as an argument,
we return a new array (with the same core). This is so we do not
expose that the varray type does not really use the layout
argument. Exposing that may bring in some dangers wrt typeclass resolution
picking the wrong view.

But the current setting means we cannot do these shifts in ghost code...
so maybe that's a bullet we should bite. *)
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

(* Ownership over a single index. *)
val varray_pts_to_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.it)
  (v : et)
  : slprop

(* Ownership over a single index. *)

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
