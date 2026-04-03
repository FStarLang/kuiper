module Kuiper.Array3

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Index
open FStar.Tactics.Typeclasses { no_method }
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let ait (d0 d1 d2 : nat) = natlt d0 & natlt d1 & natlt d2

let raw_cit = sz & sz & sz

let cit_fits (d0 d1 d2 : nat) (idx : raw_cit) : prop =
  pi_3_0 idx < d0 /\ pi_3_1 idx < d1 /\ pi_3_2 idx < d2

[@@erasable]
noeq
type layout (d0 d1 d2 : nat) = {
  ulen : nat;
  imap : ait d0 d1 d2 @~> natlt ulen;
}

let layout_size (#d0 #d1 #d2 : nat) (l : layout d0 d1 d2) : GTot nat = l.ulen

inline_for_extraction noextract
class clayout (#d0 #d1 #d2 : erased nat) (l : layout d0 d1 d2) = {
  [@@@no_method]
  culen : (x:SZ.t { SZ.v x == l.ulen });

  [@@@no_method]
  all_fit : squash (SZ.fits d0 /\ SZ.fits d1 /\ SZ.fits d2);

  [@@@no_method]
  cimap :
    i:szlt d0 ->
    j:szlt d1 ->
    k:szlt d2 ->
    r:SZ.t{SZ.v r == l.imap.f (SZ.v i, SZ.v j, SZ.v k)};
}

inline_for_extraction noextract
val t (et : Type0) (#d0 #d1 #d2 : nat) (l : layout d0 d1 d2) : Type0

unfold let array3 = t

val is_global (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (a : gpu_array et (layout_size l))
  : t et l

inline_for_extraction noextract
val core
  (#et : Type0) (#d0 #d1 #d2 : erased nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : gpu_array et (layout_size l)

val lem_core_from_array
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val pts_to
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : EMatrix3.t et d0 d1 d2)
  : slprop

instance
val is_send_across_global
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l { is_global a })
  (#f : perm) (s : EMatrix3.t et d0 d1 d2)
  : is_send_across gpu_of (pts_to a #f s)

unfold
instance has_pts_to_inst (et : Type) (d0 d1 d2 : erased nat) (l : _)
  : has_pts_to (t et l) (EMatrix3.t et d0 d1 d2)
  = { pts_to }

ghost
fn pts_to_ref
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm) (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))

ghost
fn share_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn gather_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

inline_for_extraction noextract
let clayout_imap
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (c : clayout l)
  : (szlt d0 & szlt d1 & szlt d2 -> szlt c.culen)
  = fun (i, j, k) -> c.cimap i j k

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| clayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (#f : perm)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| clayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (v : et)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  requires
    a |-> s
  ensures
    a |-> EMatrix3.mupd s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk) v

val pts_to_cell
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ijk : ait d0 d1 d2)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  : has_pts_to (cell (t et l) (ait d0 d1 d2)) et
= {
  pts_to = (fun (Cell ar ijk) #f v -> pts_to_cell ar #f ijk v);
}

val pts_to_cell_eq
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (ijk : ait d0 d1 d2) (f : perm) (v : et)
  : Lemma (Cell a ijk |-> Frac f v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f ijk) v)

ghost
fn explode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires a |-> Frac f s
  ensures
    forall+ (ijk : ait d0 d1 d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))

ghost
fn implode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ijk : ait d0 d1 d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))
  ensures
    a |-> Frac f s

(* Syntax, in lieu of a typeclass *)
unfold let op_Array_Access
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| clayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (#f : perm)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  = read #et #d0 #d1 #d2 #l a ijk #f #s

unfold let op_Array_Assignment
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| clayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (v : et)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  = write #et #d0 #d1 #d2 #l a ijk v #s
