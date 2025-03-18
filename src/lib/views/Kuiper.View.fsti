module Kuiper.View

#lang-pulse

open Kuiper
open Kuiper.GhostMap { is_ghost_map }
open Kuiper.Bijection
module F = FStar.FunctionalExtensionality
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
  (* with a concrete translation to/from machine integers *)
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

val to_seq_upd (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (v : vt)
  (i : vw.it)
  (x : a)
  : Lemma (ensures to_seq vw (vw.igm.upd v i x) == Seq.upd (to_seq vw v) (it_to_nat vw i) x)
          [SMTPat (to_seq vw (vw.igm.upd v i x))]
