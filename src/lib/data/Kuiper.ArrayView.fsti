module Kuiper.ArrayView
#lang-pulse

include Kuiper.View

open Kuiper
open Kuiper.Bijection
open Kuiper.GhostMap { is_ghost_map }
open Kuiper.View
open FStar.FunctionalExtensionality { (^->>) }
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

inline_for_extraction noextract
val varray (#a : Type0) (#len : nat) (#vt : Type) (vw : aview a len vt) : Type0

inline_for_extraction noextract
val from_array
  (#a : Type0)
  (#len : erased nat)
  (#vt : Type)
  (vw : aview a len vt)
  (arr : gpu_array a len)
  : varray vw

inline_for_extraction noextract
val core
  (#a : Type)
  (#len : erased nat)
  (#vt : Type) (#vw : aview a len vt)
  (g : varray vw)
  : arr : Kuiper.Array.gpu_array a len { from_array vw arr == g }

val lem_from_array_core
  (#a : Type)
  (#len : erased nat)
  (#vt : Type) (#vw : aview a len vt)
  (arr : varray vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]

val lem_core_from_array
  (#a : Type)
  (#len : erased nat)
  (#vt : Type) (#vw : aview a len vt)
  (p : gpu_array a len)
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]

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

ghost
fn varray_pts_to_ref
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0)
  (#vw : aview t len vt)
  (a : varray vw)
  (#f : perm)
  (#v : erased vt)
  preserves
    varray_pts_to a #f v
  ensures
    pure (SZ.fits len)

ghost
fn varray_concr
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0)
  (#vw : aview t len vt)
  (a : varray vw)
  (#f : perm)
  (#v : erased vt)
  requires
    a |-> Fraction f v
  ensures
    core a |-> Fraction f (to_seq vw v)

ghost
fn varray_abs
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0) (vw : aview t len vt)
  (a : gpu_array t len)
  (#f : perm)
  (#v : vt)
  requires
    a |-> Fraction f (to_seq vw v)
  ensures
    from_array vw a |-> Fraction f v

ghost
fn varray_abs'
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0) (vw : aview t len vt)
  (a : gpu_array t len)
  (#f : perm)
  (#v : lseq t len)
  requires
    a |-> Fraction f v
  ensures
    from_array vw a |-> Fraction f (from_seq vw v)

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
  preserves
    gpu **
    varray_pts_to a #f v
  returns
    e : et
  ensures
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
  preserves
    gpu
  requires
    a |-> v0
  ensures
    a |-> vw.igm.upd v0 (cit_to_it vw i) e

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
  preserves
    gpu
  requires
    varray_pts_to_cell a #f (cit_to_it vw i) v0
  returns
    v : et
  ensures
    varray_pts_to_cell a #f (cit_to_it vw i) v **
    pure (v == v0)

(* This variant helps to avoid having to rewrite the pts_to
   into the proper shape at then call _write_cell, and then rewrite
   it back. *)
inline_for_extraction noextract
fn varray_read_cell'
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (ai : erased vw.it)
  (#f : perm)
  (#v0 : erased et)
  preserves
    gpu
  requires
    varray_pts_to_cell a #f ai v0 **
    pure (ai == cit_to_it vw i)
  returns
    v : et
  ensures
    varray_pts_to_cell a #f ai v **
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
  preserves
    gpu
  requires
    varray_pts_to_cell a (cit_to_it vw i) v0
  ensures
    varray_pts_to_cell a (cit_to_it vw i) v1

(* This variant helps to avoid having to rewrite the pts_to
   into the proper shape at then call _write_cell, and then rewrite
   it back. *)
inline_for_extraction noextract
fn varray_write_cell'
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (ai : erased vw.it)
  (v1 : et)
  (#v0 : erased et)
  preserves
    gpu
  requires
    varray_pts_to_cell a ai v0 **
    pure (ai == cit_to_it vw i)
  ensures
    varray_pts_to_cell a ai v1

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
