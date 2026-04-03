module Kuiper.Array2

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Index
open Kuiper.EMatrix
open Kuiper.Matrix.Common
open FStar.Tactics.Typeclasses { no_method }
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let ait (rows cols : nat) = natlt rows & natlt cols

let raw_cit = sz & sz

let cit_fits (rows cols : nat) (idx : raw_cit) : prop =
  pi_2_0 idx < rows /\ pi_2_1 idx < cols

[@@erasable]
noeq
type layout (rows cols : nat) = {
  ulen : nat;
  imap : ait rows cols @~> natlt ulen;
}

let layout_size (#rows #cols : nat) (l : layout rows cols) : GTot nat = l.ulen

inline_for_extraction noextract
class clayout (#rows #cols : erased nat) (l : layout rows cols) = {
  [@@@no_method]
  culen : (x:SZ.t { SZ.v x == l.ulen });

  [@@@no_method]
  all_fit : squash (SZ.fits rows /\ SZ.fits cols);

  [@@@no_method]
  cimap :
    i:szlt rows  ->
    j:szlt cols  ->
    r:SZ.t{SZ.v r == l.imap.f (SZ.v i, SZ.v j)};
}

inline_for_extraction noextract
val t (et : Type0) (#rows #cols : nat) (l : layout rows cols) : Type0

unfold let array2 = t

val is_global (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#rows #cols : erased nat)
  (l : layout rows cols)
  (a : gpu_array et (layout_size l))
  : t et l

inline_for_extraction noextract
val core
  (#et : Type0) (#rows #cols : erased nat) (#l : layout rows cols)
  (a : t et l)
  : gpu_array et (layout_size l)

val lem_core_from_array
  (#et : Type) (#rows #cols : erased nat)
  (#l : layout rows cols)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#rows #cols : erased nat)
  (l : layout rows cols)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val pts_to
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : ematrix et rows cols)
  : slprop

instance
val is_send_across_global
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l { is_global a })
  (#f : perm) (s : ematrix et rows cols)
  : is_send_across gpu_of (pts_to a #f s)

unfold
instance has_pts_to_inst (et : Type) (rows cols : erased nat) (l : _)
  : has_pts_to (t et l) (ematrix et rows cols)
  = { pts_to }

ghost
fn pts_to_ref
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm) (#s : erased (ematrix et rows cols))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))

ghost
fn share_n
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (k : pos)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn gather_n
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (k : pos)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

inline_for_extraction noextract
let clayout_imap
  (#rows #cols : erased nat)
  (#l : layout rows cols)
  (c : clayout l)
  : (szlt rows & szlt cols) -> szlt l.ulen
  = fun (i, j) -> c.cimap i j

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| clayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (#f : perm)
  (#s : erased (ematrix et rows cols))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == macc s (pi_2_0 ij) (pi_2_1 ij))

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| clayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (v : et)
  (#s : erased (ematrix et rows cols))
  requires
    a |-> s
  ensures
    a |-> mupd s (pi_2_0 ij) (pi_2_1 ij) v

val pts_to_cell
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ij : ait rows cols)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  : has_pts_to (cell (t et l) (ait rows cols)) et
= {
  pts_to = (fun (Cell ar ij) #f v -> pts_to_cell ar #f ij v);
}

val pts_to_cell_eq
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (ij : ait rows cols) (f : perm) (v : et)
  : Lemma (Cell a ij |-> Frac f v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f ij) v)

ghost
fn explode
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires a |-> Frac f s
  ensures
    forall+ (ij : ait rows cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))

ghost
fn implode
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ij : ait rows cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))
  ensures
    a |-> Frac f s

(* Syntax, in lieu of a typeclass *)
unfold let op_Array_Access
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| clayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (#f : perm)
  (#s : erased (ematrix et rows cols))
  = read #et #rows #cols #l a ij #f #s

unfold let op_Array_Assignment
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| clayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (v : et)
  (#s : erased (ematrix et rows cols))
  = write #et #rows #cols #l a ij v #s
