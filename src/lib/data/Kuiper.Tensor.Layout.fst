module Kuiper.Tensor.Layout

open Kuiper
open Kuiper.Injection
open Kuiper.Index
open Kuiper.Chest
module SZ = Kuiper.SizeT
open Kuiper.Enumerable

let rec enumerable_abs (#r : nat) (d : idesc r) : enumerable (abs d) =
  match d with
  | INil -> solve <: enumerable unit
  | ICons h t ->
    enumerable_prod (natlt h) (abs t) #solve #(enumerable_abs t)

instance enumerable_abs' (#r : nat) (d : idesc r) : enumerable (abs d) = enumerable_abs d

let rec abs_d_cardinal (#r : nat) (d : idesc r)
  : Lemma (ensures cardinal (abs d) #_ == sizeof d)
          [SMTPat (cardinal (abs d))]
  =
  match d with
  | INil -> ()
  | ICons h t ->
    let _ = abs_d_cardinal t in
    ()

(* The underlying array must be large enough to hold all the elements of the tensor. *)
let full_layout_size_lt (#r : nat) (#d : idesc r) (l : tlayout d)
  : Lemma (ensures  l.ulen >= sizeof d)
          [SMTPat (has_type l (tlayout d))]
  = injection_implies_lte_cardinal (abs d) (natlt l.ulen) l.imap;
    ()

(* The underlying array is exactly the size of the tensor if and only if the layout is full. *)
let full_layout_size (#r : nat) (#d : idesc r) (l : tlayout d)
  : Lemma (requires is_full l)
          (ensures  l.ulen == sizeof d)
          [SMTPat (is_full l)]
  = let b : Kuiper.Bijection.bijection (abs d) (natlt l.ulen) = Kuiper.Bijection.bij_inj' l.imap in
    bijection_implies_equal_cardinal (abs d) (natlt l.ulen) b;
    ()

let full_layout_size' (#r : nat) (#d : idesc r) (l : tlayout d)
  : Lemma (requires l.ulen == sizeof d)
          (ensures is_full l)
          [SMTPat (is_full l)]
  = injection_equal_cardinal_implies_bijection (abs d) (natlt l.ulen) l.imap;
    ()

let ctlayout_must_fit (#r : nat) (#d : idesc r) (#l : tlayout d)
  (c : ctlayout l)
  : Lemma (ensures SZ.fits (sizeof d))
          [SMTPat (has_type c (ctlayout #r #d l))]
  = ()
