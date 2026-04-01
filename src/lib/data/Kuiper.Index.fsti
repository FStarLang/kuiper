module Kuiper.Index

open Kuiper.Bijection
open Kuiper.Common
open Kuiper.SizeT

(* An idesc represents a tensor type, where every ICons adds a dimension.
   MxN matrix = ICons m (ICons n INil) *)
[@@erasable]
noeq
type idesc : nat -> Type =
  | INil : idesc 0
  | ICons : #n:nat -> w:nat{SizeT.fits w} -> tl:(idesc n) -> idesc (n+1)

[@@strict_on_arguments [1]]
let rec ( @! ) (#n:nat) (d : idesc n) (i : natlt n) : GTot (x:nat{SizeT.fits x}) =
  match d with
  | ICons t ts ->
    match i with
    | 0 -> t
    | i -> ts @! (i - 1)

[@@strict_on_arguments [1]]
let rec sizeof (#r : nat) (d : idesc r) : GTot nat =
  match d with
  | INil -> 1
  | ICons t ts -> t * sizeof ts

(* Abstract index type for a tensor *)
[@@strict_on_arguments [1]]
let rec abs #n (i : idesc n) : eqtype =
  match i with
  | INil -> unit
  | ICons h ts -> natlt h & abs ts

(* Concrete index type for a tensor. This could also be eqtype, but I don't
think that is needed and would be bad at runtime. *)
[@@strict_on_arguments [1]]
let rec conc #n (i : idesc n) : Type0 =
  match i with
  | INil -> unit
  | ICons h ts -> szlt h & conc ts

[@@strict_on_arguments [1]]
let rec up #n (#d : idesc n) (v : conc d) : GTot (abs d) =
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: szlt t & conc ts in
    ((FStar.SizeT.v i1 <: natlt t), up is)

[@@strict_on_arguments [1]]
let rec down #n (#d : idesc n) (v : abs d) : GTot (conc d) =
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: natlt t & abs ts in
    ((FStar.SizeT.uint_to_t i1 <: szlt t), down is)

val up_down #n (#d : idesc n) (v : abs d) :
  Lemma (up (down v) == v)
        [SMTPat (up (down v))]

val down_up #n (#d : idesc n) (v : conc d) :
  Lemma (down (up v) == v)
        [SMTPat (down (up v))]

(* Remove (fix) a given dimension *)
let rec modulo_i (#n:nat) (i : natlt n) (d : idesc n) : idesc (n-1) =
  (* Cannot match on d and i simultaneously *)
  match d with
  | INil -> assert False; INil
  | ICons t ts ->
    match i with
    | 0 -> ts
    | i -> ICons t (modulo_i (i-1) ts)

(* Insert a dimension. Note the n+1, one can insert at the very end. *)
let rec insert_i (#n:nat) (i : natlt (n+1)) (k : nat{SizeT.fits k}) (d : idesc n) : idesc (n+1) =
  match i with
  | 0 -> ICons k d
  | i -> ICons (d @! 0) (insert_i (i-1) k (modulo_i 0 d))

(* Silence warning about using '-' in patterns. It's in an implicit,
not much to do, and it works. *)
#push-options "--warn_error -271"
val insert_modulo (#n:nat) (i : natlt n) (d : idesc n)
  : Lemma (insert_i #(n-1) i (d @! i) (modulo_i i d) == d)
          [SMTPat (insert_i #(n-1) i (d @! i) (modulo_i i d))]

val modulo_insert (#n:nat) (i : natlt (n+1)) (k : nat{SizeT.fits k}) (d : idesc n)
  : Lemma (ensures modulo_i i (insert_i i k d) == d)
          [SMTPat (modulo_i i (insert_i i k d))]

val modulo_size_lemma (#n:nat) (i : natlt n) (d : idesc n)
  : Lemma (sizeof (modulo_i i d) * (d @! i) == sizeof d)
          [SMTPat (sizeof (modulo_i i d)); SMTPat (sizeof d)]

val insert_size_lemma (#n:nat) (i : natlt (n+1)) (k : nat{SizeT.fits k}) (d : idesc n)
  : Lemma (sizeof (insert_i i k d) == sizeof d * k)
          [SMTPat (sizeof (insert_i i k d)); SMTPat (sizeof d)]
#pop-options

let rec abs_bring_forward_bij (#n:nat) (i : natlt n) (d : idesc n)
  : (abs d =~ natlt (d @! i) & abs (modulo_i i d))
  = match i with
    | 0 -> bij_self _
    | _ ->
      bij_prod (bij_self _) (abs_bring_forward_bij (i-1) (ICons?.tl d))
      `bij_comp`
      bij_push_tuple3 #(natlt (d @! 0))

let rec conc_bring_forward_bij (#n:nat) (i : natlt n) (d : idesc n)
  : (conc d =~ szlt (d @! i) & conc (modulo_i i d))
  = match i with
    | 0 -> bij_self _
    | _ ->
      bij_prod (bij_self _) (conc_bring_forward_bij (i-1) (ICons?.tl d))
      `bij_comp`
      bij_push_tuple3 #(szlt (d @! 0))

(* A computationally relevant version of the above, for use in cimap. *)
inline_for_extraction noextract
let rec c_conc_bring_forward_bij (#n : Ghost.erased nat) (i : szlt n) (d : idesc n)
  : cb : (conc d ==~ szlt (d @! i) & conc (modulo_i i d)) { cb.bij == conc_bring_forward_bij i d }
  = match i with
    | 0sz -> cbij_self _
    | _ ->
      cbij_prod (cbij_self _) (c_conc_bring_forward_bij (i-^1sz) (ICons?.tl d))
      `cbij_comp`
      cbij_push_tuple3 #(szlt (d @! 0))

(*
   abs d --abs_bring_forward--> natlt (d @! i) & abs (modulo_i i d)
    |                                         |
    |                                         |
   down                                  down x down
    |                                         |
    v                                         v
   conc d --conc_bring_forward--> szlt (d @! i) & conc (modulo_i i d)
*)
let down2 (#n:nat) (i : natlt n) (d : idesc n)
  (tup : natlt (d @! i) & abs (modulo_i i d))
  : GTot (szlt (d @! i) & conc (modulo_i i d))
  = match tup with
    | (j, abs_mod) -> ((FStar.SizeT.uint_to_t j <: szlt (d @! i)), down abs_mod)

val bring_forward_commute (#n:nat) (i : natlt n) (d : idesc n)
  (idx : abs d)
  : Lemma (down2 i d ((abs_bring_forward_bij i d).ff idx) ==
          (conc_bring_forward_bij i d).ff (down idx))

val bring_forward_commute2 (#n:nat) (i : natlt n) (d : idesc n)
  (j : szlt (d @! i)) (idx : conc (modulo_i i d))
  : Lemma (up ((conc_bring_forward_bij i d).gg (j, idx))
           == (abs_bring_forward_bij i d).gg (SizeT.v j, up idx))

let abs_conc_bij (#n:nat) (d : idesc n)
  : (abs d =~ conc d)
  = {
    ff = down;
    gg = up;
    ff_gg = (fun v -> down_up v);
    gg_ff = (fun v -> up_down v);
  }
