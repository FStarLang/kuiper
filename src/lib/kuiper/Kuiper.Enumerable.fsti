module Kuiper.Enumerable

#lang-pulse
open Kuiper.Common
open Kuiper.Bijection
open FStar.Tactics.V2
open FStar.Tactics.Typeclasses

class enumerable (a:Type) = {
  [@@@no_method]_cardinal : nat;
  bij : bijection a (natlt _cardinal);
}

let cardinal (a:Type) {| d:enumerable a |} : nat = d._cardinal

let to_nat (#a:Type) {| d : enumerable a |} (x:a) : natlt (cardinal a) =
  d.bij.ff x

let of_nat (#a:Type) {| d : enumerable a |} (x:natlt (cardinal a)) : a =
  d.bij.gg x

let to_of_pat (#a:Type) {| d:enumerable a |} (x : natlt (cardinal a))
  : Lemma (to_nat (of_nat x) == x)
          [SMTPat (to_nat (of_nat x))]
  =
  d.bij.ff_gg x

let of_to_pat (#a:Type) {| d:enumerable a |} (x : a)
  : Lemma (of_nat (to_nat x) == x)
          [SMTPat (of_nat (to_nat x))]
  =
  d.bij.gg_ff x

instance enumerable_natlt (n:nat) : enumerable (natlt n) = {
  _cardinal = n;
  bij = bij_self _;
}

instance enumerable_prod (t1 t2 : Type)
  {| d1 : enumerable t1 |} {| d2 : enumerable t2 |}
  : enumerable (t1 & t2)
= {
  _cardinal = cardinal t1 * cardinal t2;
  bij = bij_prod d1.bij d2.bij `bij_comp` bij_nat_prod;
}

let bijection_implies_equal_cardinal
  (a b : Type) {| d1 : enumerable a |} {| d2 : enumerable b |}
  (bij : bijection a b)
  : Lemma (cardinal a == cardinal b)
  =
    let bij' : (natlt (cardinal a) =~ natlt (cardinal b)) =
      bij_sym d1.bij `bij_comp` bij `bij_comp` d2.bij
    in
    __bij_cardinal (cardinal a) (cardinal b) bij'


(* sigh, need hoisting or proofs fail. *)
let ff_unit_natlt1 : unit -> natlt 1 = fun _ -> 0
let gg_unit_natlt1 : natlt 1 -> unit = fun _ -> ()

let ff_gg_unit_natlt1 (x : natlt 1)
  : Lemma (ff_unit_natlt1 (gg_unit_natlt1 x) == x)
  = ()

let gg_ff_unit_natlt1 (x : unit)
  : Lemma (gg_unit_natlt1 (ff_unit_natlt1 x) == x)
  = ()

let bij_unit : bijection unit (natlt 1) = {
  ff = ff_unit_natlt1;
  gg = gg_unit_natlt1;
  ff_gg = ff_gg_unit_natlt1;
  gg_ff = gg_ff_unit_natlt1;
}

instance enumerable_unit : enumerable unit = {
  _cardinal = 1;
  bij = bij_unit;
}
