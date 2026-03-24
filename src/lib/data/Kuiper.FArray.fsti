module Kuiper.FArray
#lang-pulse

(* One-dimensional arrays with a layout, analogous to gpu_matrix but
   with a single index. Spec type: lseq et len. *)

open Kuiper
open Kuiper.Injection
open FStar.Tactics.Typeclasses { no_method }
module V = Kuiper.View
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

(* ============ LAYOUT ============ *)

[@@erasable]
noeq
type flayout (len : nat) = {
  flen : nat;
  fmap : (natlt len @~> natlt flen);
}

let flayout_size (#len : nat) (l : flayout len) : GTot nat = l.flen

inline_for_extraction
class cflayout (#len : erased nat) (l : flayout len) = {
  [@@@no_method] cf_len : (x:SZ.t { SZ.v x == l.flen });
  [@@@no_method] cf_sz  : (x:SZ.t { SZ.v x == reveal len });
  [@@@no_method] cf_to  : (i:SZ.t{i < len}) -> r:SZ.t{SZ.v r == l.fmap.f (SZ.v i)};
}

(* ============ VIEW ============ *)

let farray_aview (et : Type) (#len : nat) (l : flayout len)
  : V.aview et (lseq et len)
  = {
      iview = {
        len = l.flen;
        ait = natlt len;
        step = { imap = l.fmap; };
      };
      ctn = solve;
    }

(* ============ TYPE ============ *)

inline_for_extraction noextract
val farray (et : Type0) (#len : nat) (l : flayout len) : Type0

val is_global_farray (#et : Type0) (#len : nat) (#l : flayout len) (a : farray et l) : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#len : erased nat)
  (l : flayout len)
  (a : gpu_array et (flayout_size l))
  : farray et l

inline_for_extraction noextract
val core
  (#et : Type0) (#len : erased nat) (#l : flayout len)
  (a : farray et l)
  : gpu_array et (flayout_size l)

val lem_core_from_array
  (#et : Type) (#len : erased nat) (#l : flayout len)
  (a : farray et l)
  : Lemma (ensures from_array l (core a) == a /\ (is_global_array (core a) <==> is_global_farray a))
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#len : erased nat)
  (l : flayout len)
  (p : gpu_array et (flayout_size l))
  : Lemma (ensures core (from_array l p) == p /\ (is_global_farray (from_array l p) <==> is_global_array p))
          [SMTPat (from_array l p)]

(* ============ OWNERSHIP ============ *)

val farray_pts_to
  (#et : Type) (#len : nat) (#l : flayout len)
  ([@@@mkey] a : farray et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : lseq et len)
  : slprop

instance
val is_send_across_global_farray
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l { is_global_farray a })
  (#f : perm) (s : lseq et len)
  : is_send_across gpu_of (farray_pts_to a #f s)

unfold
instance has_pts_to_farray (et : Type) (len : erased nat) (l : _)
  : has_pts_to (farray et l) (lseq et len) = {
  pts_to = farray_pts_to;
}

ghost
fn farray_pts_to_ref
  (#et : Type) (#len : nat) (#l : flayout len)
  (a : farray et l)
  (#f : perm) (#s : erased (lseq et len))
  preserves a |-> Frac f s
  ensures pure (SZ.fits (flayout_size l))

ghost
fn farray_share_n
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires a |-> Frac f s
  ensures  forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn farray_gather_n
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures  a |-> Frac f s

(* ============ CONCRETE VIEW INSTANCE ============ *)

inline_for_extraction noextract
let cflayout_imap
  (#len : erased nat)
  (#l : flayout len)
  (c : cflayout l)
  : szlt len -> szlt l.flen
  = fun i -> c.cf_to i

inline_for_extraction noextract
instance cfarray_ciview
  (et : Type)
  (#len : erased nat)
  (l : flayout len)
  (c : cflayout l)
  : Kuiper.IView.ciview (farray_aview et l).iview =
{
  clen = c.cf_len;
  sch = {
    cit = szlt len;
    bij = Kuiper.Bijection.natural;
  };
  step = {
    cimap = mk_cinj (cflayout_imap c);
    compat = ez;
  };
}

(* ============ READ / WRITE ============ *)

inline_for_extraction noextract
fn farray_read
  (#et : Type0)
  (#len : erased nat)
  (#l : flayout len) {| cflayout l |}
  (a : farray et l)
  (i : szlt len)
  (#f : perm)
  (#s : erased (lseq et len))
  preserves a |-> Frac f s
  returns v : et
  ensures pure (v == Seq.index s i)

inline_for_extraction noextract
fn farray_write
  (#et : Type0)
  (#len : erased nat)
  (#l : flayout len) {| cflayout l |}
  (a : farray et l)
  (i : szlt len)
  (v : et)
  (#s : erased (lseq et len))
  requires a |-> s
  ensures  a |-> (Seq.upd s i v <: lseq et len)

(* ============ CELL-LEVEL OPERATIONS ============ *)

val farray_pts_to_cell
  (#et : Type) (#len : nat) (#l : flayout len)
  ([@@@mkey] a : farray et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : natlt len)
  (v : et)
  : slprop

val farray_pts_to_cell_eq
  (#et : Type) (#len : nat) (#l : flayout len)
  (a : farray et l) (i : natlt len) (f : perm) (v : et)
  : Lemma (farray_pts_to_cell a #f i v
           ==
           gpu_pts_to_cell (core a) #f (l.fmap.f i) v)

ghost
fn farray_explode
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l)
  (#f : perm)
  (#s : lseq et len)
  requires a |-> Frac f s
  ensures
    forall+ (i : natlt len).
      farray_pts_to_cell a #f i (Seq.index s i)

ghost
fn farray_implode
  (#et : Type0) (#len : nat) (#l : flayout len)
  (a : farray et l)
  (#f : perm)
  (#s : lseq et len)
  requires
    pure (SZ.fits (flayout_size l))
  requires
    forall+ (i : natlt len).
      farray_pts_to_cell a #f i (Seq.index s i)
  ensures
    a |-> Frac f s
