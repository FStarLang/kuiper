module Kuiper.Array1

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Index
open FStar.Tactics.Typeclasses { no_method }
module V = Kuiper.View
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

[@@erasable]
noeq
type layout (len : nat) = {
  ulen : nat;
  imap : (natlt len @~> natlt ulen);
}

let layout_size (#len : nat) (l : layout len) : GTot nat = l.ulen

inline_for_extraction noextract
class clayout (#len : erased nat) (l : layout len) = {
  [@@@no_method]
  culen : (x:SZ.t { SZ.v x == l.ulen });

  [@@@no_method]
  all_fit : squash (SZ.fits len);

  [@@@no_method]
  cimap :
    i:szlt len ->
    r:SZ.t{SZ.v r == l.imap.f (SZ.v i)};
}

let aview (et : Type) (#len : nat) (l : layout len)
  : V.aview et (lseq et len)
  = {
      iview = {
        len = l.ulen;
        ait = natlt len;
        step = { imap = l.imap; };
      };
      ctn = solve;
    }

inline_for_extraction noextract
val t (et : Type0) (#len : nat) (l : layout len) : Type0

unfold let array1 = t

val is_global (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#len : erased nat)
  (l : layout len)
  (a : gpu_array et (layout_size l))
  : t et l

inline_for_extraction noextract
val core
  (#et : Type0) (#len : erased nat) (#l : layout len)
  (a : t et l)
  : gpu_array et (layout_size l)

val lem_core_from_array
  (#et : Type) (#len : erased nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#len : erased nat)
  (l : layout len)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#len : nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val pts_to
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : lseq et len)
  : slprop

instance
val is_send_across_global
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l { is_global a })
  (#f : perm) (s : lseq et len)
  : is_send_across gpu_of (pts_to a #f s)

unfold
instance has_pts_to_inst (et : Type) (len : erased nat) (l : _)
  : has_pts_to (t et l) (lseq et len)
  = { pts_to }

ghost
fn pts_to_ref
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm) (#s : erased (lseq et len))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))

ghost
fn share_n
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn gather_n
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

inline_for_extraction noextract
let clayout_imap
  (#len : erased nat)
  (#l : layout len)
  (c : clayout l)
  : szlt len -> szlt l.ulen
  = fun i -> c.cimap i

inline_for_extraction noextract
instance cfarray_ciview
  (et : Type)
  (#len : erased nat)
  (l : layout len)
  (c : clayout l)
  : Kuiper.IView.ciview (aview et l).iview =
{
  clen = c.culen;
  sch = {
    cit = szlt len;
    bij = Kuiper.Bijection.natural;
  };
  step = {
    cimap = mk_cinj (clayout_imap c);
    compat = ez;
  };
}

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| clayout l |}
  (a : t et l)
  (i : szlt len)
  (#f : perm)
  (#s : erased (lseq et len))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == Seq.index s i)

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| clayout l |}
  (a : t et l)
  (i : szlt len)
  (v : et)
  (#s : erased (lseq et len))
  requires
    a |-> s
  ensures
    a |-> (Seq.upd s i v <: lseq et len)

val pts_to_cell
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] i : natlt len)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#len : nat) (#l : layout len)
  : has_pts_to (cell (t et l) (natlt len)) et
= {
  pts_to = (fun (Cell ar i) #f v -> pts_to_cell ar #f i v);
}

val pts_to_cell_eq
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l) (i : natlt len) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f i) v)

ghost
fn explode
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm)
  (#s : lseq et len)
  requires a |-> Frac f s
  ensures
    forall+ (i : natlt len).
      Cell a i |-> Frac f (Seq.index s i)

ghost
fn implode
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm)
  (#s : lseq et len)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (i : natlt len).
      Cell a i |-> Frac f (Seq.index s i)
  ensures
    a |-> Frac f s
