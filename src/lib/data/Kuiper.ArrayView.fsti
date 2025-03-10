module Kuiper.ArrayView
#lang-pulse

open Kuiper
open Kuiper.Bijection
module B = Kuiper.Array (* base *)
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

// [@@erasable]
noeq
type aview (a : Type) (len : nat) (vt : Type) = {
  it : Type0; (* index type *)
  bij : lseq a len =~ vt; (* bijection to sequences *)
  (* the view type is a map from index type into element type *)
  acc : it -> vt -> a;
  upd : it -> a -> vt -> vt;
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

// inline_for_extraction
// type mrepr = #len:nat -> alayout a len vt rows

inline_for_extraction noextract
val gpu_array (#a : Type0) (#len : nat) (#vt : Type) (vw : aview a len vt) : Type0

inline_for_extraction noextract
val core
  (#a : Type)
  (#len : erased nat)
  (#vt : Type) (#vw : aview a len vt)
  (g : gpu_array vw)
  : Kuiper.Array.gpu_array a len

val gpu_array_pts_to
  (#a:Type) (#len : erased nat) (#vt:_) (#vw : aview a len vt)
  ([@@@mkey] a : gpu_array vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : vt)
  : slprop

unfold
instance has_pts_to (#a:Type) (#len : nat) (#vt:Type) (#vw : aview a len vt)
  : has_pts_to (gpu_array vw) vt = {
  pts_to = gpu_array_pts_to;
}

inline_for_extraction noextract
fn gpu_array_concr
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0)
  (#vw : aview t len vt)
  (a : gpu_array vw)
  (#v : vt)
  requires
    a |-> v
  ensures
    core a |-> vw.bij.gg v

inline_for_extraction noextract
fn gpu_array_abs
  (#t:Type0)
  (#len0 : erased nat) (#vt0:Type0) (#vw0 : aview t len0 vt0)
  (a : gpu_array vw0)
  (#len : erased nat) (#vt:Type0) (vw : aview t len vt)
  (#v : vt)
  requires
    core a |-> vw.bij.gg v
  returns
    a' : gpu_array vw
  ensures
    pure (len0 == len /\ core a == core a') **
    (a' |-> v)

inline_for_extraction noextract
fn gpu_array_alloc0
  (#et:Type) {| sized et |}
  (len : SZ.t) (#vt:Type0) (vw : aview et len vt)
  preserves
    cpu
  requires
    pure (SZ.fits len)
  returns
    a : gpu_array vw
  ensures
    exists* v. a |-> v

inline_for_extraction noextract
fn gpu_array_free
  (#et:Type) {| sized et |}
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : gpu_array vw)
  (#v : vt)
  preserves
    cpu
  requires
    a |-> v
  ensures emp

ghost
fn gpu_array_share_n
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (#[T.exact (`0)] uid: int)
  (a : gpu_array vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    gpu_array_pts_to a #f v
  ensures
    bigstar #uid 0 k (fun _ -> gpu_array_pts_to a #(f /. k) v)

ghost
fn gpu_array_gather_n
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (#uid: int)
  (a : gpu_array vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    bigstar #uid 0 k (fun _ -> gpu_array_pts_to a #(f /. k) v)
  ensures
    gpu_array_pts_to a #f v

inline_for_extraction noextract
fn gpu_array_read
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : gpu_array vw)
  (i : vw.it)
  (#f : perm)
  (#v : vt)
  requires
    gpu **
    gpu_array_pts_to a #f v
  returns
    e : et
  ensures
    gpu **
    gpu_array_pts_to a #f v **
    pure (e == vw.acc i v)

inline_for_extraction noextract
fn gpu_array_write
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : gpu_array vw)
  (i : vw.it)
  (e : et)
  (#f : perm)
  (#v0 : vt)
  requires
    gpu **
    gpu_array_pts_to a v0
  ensures
    gpu **
    gpu_array_pts_to a (vw.upd i e v0)

(* Ownership over a single index. *)
val gpu_array_pts_to_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  ([@@@mkey] a : gpu_array vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.it)
  (v : et)
  : slprop

(* Ownership over a single index. *)

inline_for_extraction noextract
fn gpu_array_read_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : gpu_array vw)
  (i : vw.it)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    gpu_array_pts_to_cell a #f i v0
  returns
    v : et
  ensures
    gpu **
    gpu_array_pts_to_cell a #f i v **
    pure (v == v0)

inline_for_extraction noextract
fn gpu_array_write_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : gpu_array vw)
  (i : vw.it)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_array_pts_to_cell a i v0
  ensures
    gpu **
    gpu_array_pts_to_cell a i v1

ghost
fn gpu_array_explode
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : gpu_array vw)
  (#f : perm)
  (#v : vt)
  requires
    gpu_array_pts_to a #f v
  ensures
    forall+ (i : vw.it).
      gpu_array_pts_to_cell #et #len #vt #vw a #f i (vw.acc i v)

ghost
fn gpu_array_implode
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (vw : aview et len vt)
  (a : gpu_array vw)
  (#f : perm)
  (#v : vt)
  requires
    forall+ (i : vw.it).
      gpu_array_pts_to_cell #et #len #vt #vw a #f i (vw.acc i v)
  ensures
    gpu_array_pts_to a #f v
