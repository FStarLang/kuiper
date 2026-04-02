module Kuiper.Index

open Kuiper.Bijection
open Kuiper.Common
open Kuiper.SizeT

let rec up_down #n (#d : idesc n) (v : abs d) :
  Lemma (ensures all_fit d ==> up (down v) == v)
        [SMTPat (up (down v))]
=
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: natlt t & abs ts in
    up_down is

let rec down_up #n (#d : idesc n) (v : conc d) :
  Lemma (ensures all_fit d ==> down (up v) == v)
        [SMTPat (down (up v))]
=
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: szlt t & conc ts in
    down_up is

#push-options "--warn_error -271"
let rec insert_modulo (#n:nat) (i : natlt n) (d : idesc n)
  : Lemma (insert_i #(n-1) i (d @! i) (modulo_i i d) == d)
          [SMTPat (insert_i #(n-1) i (d @! i) (modulo_i i d))]
  = match d with
    | INil -> ()
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i -> insert_modulo (i-1) ts

let rec modulo_insert (#n:nat) (i : natlt (n+1)) (k : nat) (d : idesc n)
  : Lemma (ensures modulo_i i (insert_i i k d) == d)
          [SMTPat (modulo_i i (insert_i i k d))]
  = match i with
    | 0 -> ()
    | i ->
      match d with
      | INil -> assert false
      | ICons t ts -> modulo_insert (i-1) k ts

let rec modulo_size_lemma (#n:nat) (i : natlt n) (d : idesc n)
  : Lemma (sizeof (modulo_i i d) * (d @! i) == sizeof d)
          [SMTPat (sizeof (modulo_i i d)); SMTPat (sizeof d)]
  = match d with
    | INil -> ()
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i -> modulo_size_lemma (i-1) ts

let rec insert_size_lemma (#n:nat) (i : natlt (n+1)) (k : nat{SizeT.fits k}) (d : idesc n)
  : Lemma (sizeof (insert_i i k d) == sizeof d * k)
          [SMTPat (sizeof (insert_i i k d)); SMTPat (sizeof d)]
  = match i with
    | 0 -> ()
    | i ->
      match d with
      | INil -> assert false
      | ICons t ts -> insert_size_lemma (i-1) k ts
#pop-options

let rec bring_forward_commute (#n:nat) (i : natlt n) (d : idesc n{all_fit d})
  (idx : abs d)
  : Lemma (down2 i d ((abs_bring_forward_bij i d).ff idx) ==
          (conc_bring_forward_bij i d).ff (down idx))
  = match d with
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i ->
        let idx1, idx_mod = idx <: natlt t & abs ts in
        bring_forward_commute (i-1) ts idx_mod

let rec bring_forward_commute2 (#n:nat) (i : natlt n) (d : idesc n)
  (j : szlt (d @! i)) (idx : conc (modulo_i i d))
  : Lemma (up ((conc_bring_forward_bij i d).gg (j, idx))
           == (abs_bring_forward_bij i d).gg (SizeT.v j, up idx))
  = match d with
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i ->
        let hh, tt = idx <: szlt (d @! 0) & conc (modulo_i (i-1) ts) in
        bring_forward_commute2 (i-1) ts j tt
