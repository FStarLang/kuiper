module Kuiper.ArrayView
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.GhostMap { is_ghost_map }
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

(* The view type is an array of length len, with elements of type a,
   and index type it. The view type is a map from index type into element
   type. *)

[@@erasable]
noeq
type aview (a : Type) (len : nat) (vt : Type) = {
  (* abstract index type *)
  it : Type0;
  (* the view is essentially a map ... *)
  igm : is_ghost_map vt it a;
  (* ... from an enumerable type *)
  ibij : it =~ natlt len;
}

inline_for_extraction noextract
class cview (#a : Type) (#len : erased nat) (#vt : Type) (avw : aview a len vt) = {
  (* the length is actually realizable. *)
  lenfits : squash (SZ.fits len);

  (* a concrete index type *)
  cit : Type0;
  (* with translation to/from machine integers *)
  cibij : cit =~ szlt len;
  (* this also implies it =~ cit *)
}

(* hm.... the choice of bijections above makes these a bit awkward *)

let it_to_nat
  (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (i : vw.it)
  : GTot (natlt len)
  = i |~> vw.ibij

let it_of_nat
  (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (i: natlt len)
  : GTot vw.it
  = i <~| vw.ibij

let cit_to_it
  (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt) {| cw : cview vw |}
  (i : cw.cit)
  : GTot vw.it
  = (SZ.v (i |~> cw.cibij)) <~| vw.ibij

let cit_of_it
  (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt) {| cw : cview vw |}
  (i: vw.it)
  : GTot cw.cit
  = SZ.uint_to_t (i |~> vw.ibij) <~| cw.cibij

let to_seq
  (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (v : vt)
  : GTot (lseq a len)
  = Seq.init_ghost len (fun i -> vw.igm.acc v (it_of_nat vw i))

let from_seq
  (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (s : lseq a len)
  : GTot vt
  = vw.igm.bij.gg (F.on_g vw.it <| fun i -> s @! it_to_nat vw i)

val to_from (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (s : lseq a len)
  : Lemma (ensures to_seq vw (from_seq vw s) == s)
          [SMTPat (to_seq vw (from_seq vw s))]

(* Avoid ghost effect when using projector. *)
inline_for_extraction noextract
let cidx
  (#a : Type) (#len : erased nat) (#vt : Type)
  (#vw : aview a len vt) (cw : cview vw)
  (cit : cw.cit)
  : c:sz{c == cw.cibij.ff cit}
  = match cw with {cibij} -> cibij.ff cit

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
  (#a:Type) (#len : nat) (#vt:_) (#vw : aview a len vt)
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
  (#v : erased vt)
  requires
    a |-> v
  ensures
    core a |-> to_seq vw v

inline_for_extraction noextract
fn varray_abs
  (#t:Type0)
  (#len0 : erased nat) (#vt0:Type0) (#vw0 : aview t len0 vt0)
  (a : varray vw0)
  (#len : erased nat) (#vt:Type0) (vw : aview t len vt)
  (#v : erased vt)
  requires
    core a |-> to_seq vw v
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
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  (a : varray vw)
  (#v : erased vt)
  preserves
    cpu
  requires
    a |-> v
  ensures emp

ghost
fn varray_share_n
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
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
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
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
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (#f : perm)
  (#v : erased vt)
  requires
    gpu **
    varray_pts_to a #f v
  returns
    e : et
  ensures
    gpu **
    varray_pts_to a #f v **
    pure (e == vw.igm.acc v (cit_to_it vw i))

inline_for_extraction noextract
fn varray_write
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (e : et)
  (#v0 : erased vt)
  requires
    gpu **
    (a |-> v0)
  ensures
    gpu **
    (a |-> vw.igm.upd v0 (cit_to_it vw i) e)

(* Ownership over a single index. *)
val varray_pts_to_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.it)
  (v : et)
  : slprop

inline_for_extraction noextract
fn varray_read_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    varray_pts_to_cell a #f (cit_to_it vw i) v0
  returns
    v : et
  ensures
    gpu **
    varray_pts_to_cell a #f (cit_to_it vw i) v **
    pure (v == v0)

inline_for_extraction noextract
fn varray_write_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    varray_pts_to_cell a (cit_to_it vw i) v0
  ensures
    gpu **
    varray_pts_to_cell a (cit_to_it vw i) v1

(* Note: the functions below take a constraint for enumerable vw.it,
   even if there is an enumeration in vw.ibij. We do this since it's
   not necessary for that enumeration to match the one in the typeclass system.
   For example, for a matrix view, that enumeration can be anything
   depending on the layout chosen, but the enumeration we want for the
   **abstract indices** is just lexicographic. *)

ghost
fn varray_explode
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  {| Enumerable.enumerable vw.it |}
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    varray_pts_to a #f v
  ensures
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)

ghost
fn varray_implode
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  {| Enumerable.enumerable vw.it |}
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)
  ensures
    varray_pts_to a #f v

(* memcpys *)

inline_for_extraction noextract
fn varray_from_array
  (#et:Type) {| sized et |}
  (#len : SZ.t) (#vt:Type0)
  (#vw : aview et len vt)
  (va : varray vw)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == len})
  (#v : erased vt)
  preserves
    (a |-> s) **
    cpu
  requires
    (va |-> v)
  ensures
    pure (SZ.fits len /\ Pulse.Lib.Vec.length a == len) **
    (va |-> from_seq vw s)

inline_for_extraction noextract
fn varray_to_array
  (#et:Type) {| sized et |}
  (#len : SZ.t) (#vt:Type0)
  (#vw : aview et len vt)
  (a : vec et)
  (va : varray vw)
  (#s : erased (seq et){Seq.length s == len})
  (#v : erased vt)
  preserves
    (va |-> v) **
    cpu
  requires
    (a |-> s)
  ensures
    pure (SZ.fits len /\ Pulse.Lib.Vec.length a == len) **
    (a |-> to_seq vw v)
