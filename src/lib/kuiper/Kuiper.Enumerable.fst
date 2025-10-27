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
  = Kuiper.Functions.pigeon n1 n2 f

let injection_implies_lte_cardinal
  (a b : Type) {| d1 : enumerable a |} {| d2 : enumerable b |}
  (inj : injection a b)
  : Lemma (cardinal a #_ <= cardinal b #_)
  = let inj' : injection (natlt (cardinal a #_)) (natlt (cardinal b #_)) =
      inj_bij (bij_sym d1.bij) `inj_comp` inj `inj_comp` inj_bij d2.bij
    in
    if cardinal a #_ > cardinal b #_ then
      no_inj_to_smaller_nat (cardinal a #_) (cardinal b #_) inj'.f


let distinct_seq_greater_than_cardinal_impossible
  (#a:Type)
  {| d : enumerable a |}
  (s:Seq.seq a {(forall (x y: natlt (Seq.length s)). x <> y ==> Seq.index s x =!= Seq.index s y)})
: Lemma
  (requires cardinal a #_ < Seq.length s)
  (ensures False)
= let n = cardinal a #_ in
  if n = 0 then let _ = to_nat (Seq.index s 0) in ()
  else (
    let s' : Seq.seq (FStar.Fin.under n) = Seq.init_ghost (Seq.length s) (fun i -> to_nat (Seq.index s i)) in
    let i1, i2 = FStar.Fin.pigeonhole s' in
    assert (of_nat #a i1 == of_nat #a i2)
  )

let distinct_seq_enumerates_all (#a:Type) {| d : enumerable a |} (s:Seq.seq a {Seq.length s == cardinal a #_}) (n:a)
: Lemma
  (requires forall (x y: natlt (cardinal a #_)). x <> y ==> Seq.index s x =!= Seq.index s y)
  (ensures exists (i:natlt (cardinal a #_)). Seq.index s i == n)
= introduce (forall (i:natlt (cardinal a #_)). Seq.index s i =!= n) ==> False
  with _ . (
    distinct_seq_greater_than_cardinal_impossible (Seq.snoc s n)
  )

let injection_equal_cardinal_implies_bijection
  (a b : Type) {| d1 : enumerable a |} {| d2 : enumerable b |}
  (inj : injection a b)
  : Lemma (requires cardinal a #_ == cardinal b #_)
          (ensures  FStar.Functions.is_surj inj.f)
  = let n = cardinal a #_ in
    let s0 : Seq.seq b = Seq.init_ghost n (fun i -> inj.f (of_nat i)) in
    assert (forall (x y : natlt n). x <> y ==> Seq.index s0 x =!= Seq.index s0 y);
    introduce forall (y : b). exists (x:a). inj.f x == y
    with (
      distinct_seq_enumerates_all s0 y
    )
