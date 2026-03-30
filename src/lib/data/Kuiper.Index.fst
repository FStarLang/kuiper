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
  | ICons : #n:nat -> nat -> tl:(idesc n) -> idesc (n+1)

[@@strict_on_arguments [1]]
let rec ( @! ) (#n:nat) (d : idesc n) (i : natlt n) : GTot nat =
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
  | ICons h ts -> natlt h & abs #(n-1) ts

(* Concrete index type for a tensor. This could also be eqtype, but I don't
think that is needed and would be bad at runtime. *)
[@@strict_on_arguments [1]]
let rec conc #n (i : idesc n) : Type =
  match i with
  | INil -> unit
  | ICons h ts -> szlt h & conc #(n-1) ts

[@@strict_on_arguments [1]]
let rec up #n (#d : idesc n) (v : conc d) : GTot (abs d) =
  match d with
  | INil -> ()
  | ICons t ts ->
    assert_norm (conc (ICons t ts) == szlt t & conc ts);
    let i1, is = v <: szlt t & conc ts in
    ((FStar.SizeT.v i1 <: natlt t), up is)

(* Remove (fix) a given dimension *)
let rec modulo_i (#n:nat) (i : natlt n) (d : idesc n) : idesc (n-1) =
  (* Cannot match on d and i simultaneously *)
  match d with
  | INil -> assert False; INil
  | ICons t ts ->
    match i with
    | 0 -> ts
    | i -> ICons t (modulo_i (i-1) ts)

let abs_bring_forward_bij (#n:nat) (i : natlt n) (d : idesc n)
  : (abs d =~ natlt (d @! i) & abs (modulo_i i d))
  = magic()

let conc_bring_forward_bij (#n:nat) (i : natlt n) (d : idesc n)
  : (conc d =~ natlt (d @! i) & conc (modulo_i i d))
  = magic()

let abs_conc_bij (#n:nat) (d : idesc n)
  : (abs d =~ conc d)
  = {
    ff = magic();
    gg = up;
    ff_gg = magic();
    gg_ff = magic();
  }
