module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Bijection
open Kuiper.Injection

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

let l1_to_l2 (#len : nat)
  (l : layout1 len)
  : layout2 1 len = {
    ulen = l.ulen;
    imap = {
      f = (fun x -> l.imap.f (idx2_to_idx1 x));
    };
  }

let c1_to_c2
  (#et : Type0) (#len : nat)
  (c1 : chest1 et len)
  : chest2 et 1 len =
  mk2 fun _ i -> acc1 c1 i

ghost
fn t1_to_t2
  (#et : Type0)
  (len : nat)
  (#l : layout1 len)
  (a : array1 et l)
  (#s : chest1 et len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    from_array (l1_to_l2 l) (core a) |-> Frac f (c1_to_c2 s)

let l2_to_l1 (#len : nat) (l : layout2 1 len) : layout1 len = {
  ulen = l.ulen;
  imap = {
    f = (fun x -> l.imap.f (idx1_to_idx2 x));
  };
}

let c2_to_c1
  (#et : Type0) (#len : nat)
  (c2 : chest2 et 1 len)
  : chest1 et len =
  mk1 fun i -> acc2 c2 0 i

ghost
fn t2_to_t1
  (#et : Type0)
  (len : nat)
  (#l : layout2 1 len)
  (a : array2 et l)
  (#s : chest2 et 1 len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    from_array (l2_to_l1 l) (core a) |-> Frac f (c2_to_c1 s)
