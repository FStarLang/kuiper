module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Bijection
open Kuiper.Injection

let layout_bij
  (#r1 : nat) (#s1 : shape r1)
  (#r2 : nat) (#s2 : shape r2)
  (b : abs s1 =~ abs s2)
  (l : tlayout s1)
  : tlayout s2 = {
    ulen = l.ulen;
    imap = {
      f = (fun x -> l.imap.f (b.gg x))
    };
  }

let bij_up
  (#r1 : nat) (#s1 : shape r1)
  (#r2 : nat) (#s2 : shape r2)
  (#_ : all_fit s1) (#_ : all_fit s2)
  (cb : conc s1 ==~ conc s2)
  : (abs s1 =~ abs s2) = {
    ff = (fun x -> up (cb.cff (down x)));
    gg = (fun x -> up (cb.cgg (down x)));
  }

// FIXME: cbij should be a typeclass
inline_for_extraction noextract
instance clayout_bij
  (#r1 : nat) (#s1 : shape r1)
  (#r2 : nat) (#s2 : shape r2)
  (#_ : all_fit s1) (#_ : all_fit s2)
  (cb : conc s1 ==~ conc s2)
  (l : tlayout s1) {| cl : ctlayout l |}
  : ctlayout (layout_bij (bij_up cb) l)
  = {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun x -> cl.cimap (cb.cgg x));
  }

let chest_bij
  (#et : Type0)
  (#r1 : nat) (#s1 : shape r1)
  (#r2 : nat) (#s2 : shape r2)
  (b : abs s1 =~ abs s2)
  (c : chest s1 et)
  : chest s2 et =
    mk s2 (fun i -> acc c (b.gg i))

ghost
fn tensor_abij
  (#et : Type0)
  (#r1 : nat) (#s1 : shape r1)
  (#r2 : nat) (#s2 : shape r2)
  (b : abs s1 =~ abs s2)
  (#l : tlayout s1)
  (a : tensor et l)
  (#s : chest s1 et)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (layout_bij b l) |-> Frac f (chest_bij b s)

let idx1_to_idx2 (#len : nat)
  (i : abs (len @| INil))
  : abs (1 @| len @| INil) =
    let (j, ()) = i in
    (0, (j, ()))

let idx2_to_idx1 (#len : nat)
  (i : abs (1 @| len @| INil))
  : abs (len @| INil) =
    let (0, (j, ())) = i in
    (j, ())

let bij12 (len : nat)
  : (abs (len @| INil) =~ abs (1 @| len @| INil)) = {
    ff = idx1_to_idx2;
    gg = idx2_to_idx1;
  }

let l1_to_l2 (#len : nat) (l : layout1 len)
  : layout2 1 len
  = layout_bij (bij12 len) l

let c1_to_c2
  (#et : Type0) (#len : nat)
  (c1 : chest1 et len)
  : chest2 et 1 len =
  mk2 fun _ i -> acc1 c1 i

ghost
fn t1_to_t2
  (#et : Type0)
  (#len : nat)
  (#l : layout1 len)
  (a : array1 et l)
  (#s : chest1 et len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (l1_to_l2 l) |-> Frac f (c1_to_c2 s)

let l2_to_l1
  (#len : nat) (l : layout2 1 len)
  : layout1 len
  = layout_bij (bij_sym (bij12 len)) l

let c2_to_c1
  (#et : Type0) (#len : nat)
  (c2 : chest2 et 1 len)
  : chest1 et len =
  mk1 fun i -> acc2 c2 0 i

ghost
fn t2_to_t1
  (#et : Type0)
  (#len : nat)
  (#l : layout2 1 len)
  (a : array2 et l)
  (#s : chest2 et 1 len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (l2_to_l1 l) |-> Frac f (c2_to_c1 s)
