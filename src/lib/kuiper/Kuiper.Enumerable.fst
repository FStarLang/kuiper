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

let injection_implies_lte_cardinal
  (a b : Type) {| d1 : enumerable a |} {| d2 : enumerable b |}
  (inj : injection a b)
  : Lemma (cardinal a #_ <= cardinal b #_)
  = let aux () : Lemma (requires cardinal a #_ > cardinal b #_)
                       (ensures False)
      = admit()
    in
    Classical.move_requires aux ();
    ()