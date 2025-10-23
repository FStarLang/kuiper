module Kuiper.Enumerable

#lang-pulse
open Kuiper.Common
open Kuiper.Bijection
open FStar.Tactics.Typeclasses

let to_of_pat (#a:Type) {| d:enumerable a |} (x : natlt (cardinal a #_))
  : Lemma (to_nat (of_nat x) == x)
          [SMTPat (to_nat (of_nat x))]
  =
  d.bij.ff_gg x

let of_to_pat (#a:Type) {| d:enumerable a |} (x : a)
  : Lemma (of_nat (to_nat x) == x)
          [SMTPat (of_nat (to_nat x))]
  =
  d.bij.gg_ff x

let bijection_implies_equal_cardinal
  (a b : Type) {| d1 : enumerable a |} {| d2 : enumerable b |}
  (bij : bijection a b)
  : Lemma (cardinal a #_ == cardinal b #_)
  =
    let bij' : (natlt (cardinal a #_) =~ natlt (cardinal b #_)) =
      bij_sym d1.bij `bij_comp` bij `bij_comp` d2.bij
    in
    __bij_cardinal (cardinal a #_) (cardinal b #_) bij'

let no_inj_to_smaller_nat (n1 n2 : nat{n2 < n1})
  (f : natlt n1 -> GTot (natlt n2))
  : Lemma (exists (x y : natlt n1). x <> y /\ f x == f y)
  = admit()

let injection_implies_lte_cardinal
  (a b : Type) {| d1 : enumerable a |} {| d2 : enumerable b |}
  (inj : injection a b)
  : Lemma (cardinal a #_ <= cardinal b #_)
  = let inj' : injection (natlt (cardinal a #_)) (natlt (cardinal b #_)) =
      inj_bij (bij_sym d1.bij) `inj_comp` inj `inj_comp` inj_bij d2.bij
    in
    if cardinal a #_ > cardinal b #_ then
      no_inj_to_smaller_nat (cardinal a #_) (cardinal b #_) inj'.f

let injection_equal_cardinal_implies_bijection
  (a b : Type) {| d1 : enumerable a |} {| d2 : enumerable b |}
  (inj : injection a b)
  : Lemma (requires cardinal a #_ == cardinal b #_)
          (ensures  FStar.Functions.is_surj inj.f)
  = let contra (y : b)
      : Lemma (requires ~(exists (x : a). inj.f x == y))
              (ensures False)
      = let f' : a -> GTot (y':b{y' =!= y}) = (fun x -> inj.f x) in
        assert (Functions.is_inj f');
        let inj' : injection a (y':b{y' =!= y}) = { f = f'; is_inj = ez } in
        // injection_implies_lte_cardinal a (y':b{y' =!= y}) inj';
        // ^ hmm, need enumerable instance for the subtype
        admit()
    in
    Classical.forall_intro (Classical.move_requires contra)
