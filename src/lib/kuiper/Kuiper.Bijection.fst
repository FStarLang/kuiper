module Kuiper.Bijection

#lang-pulse

open Kuiper.Common
open Kuiper.Functions

let galois_nopat (#a #b : _) (d : a =~ b) (x:a) (y:b)
  : Lemma (d.ff x == y <==> x == d.gg y)
  = d.ff_gg y;
    d.gg_ff x

let galois (#a #b : _) (d : a =~ b) (x:a) (y:b)
  : Lemma (d.ff x == y <==> x == d.gg y)
          [SMTPat (d.ff x); SMTPat (d.gg y)]
  = galois_nopat d x y

#push-options "--warn_error -288"
let galois_forall (#a #b : _) (d : a =~ b)
  : Lemma (forall (x:a) (y:b). d.ff x == y <==> x == d.gg y)
          [SMTPat (has_type d (a =~ b))] // OK? Useful?
  = Classical.forall_intro_2 (galois_nopat d)
#pop-options

let __bij_cardinal (n1 n2 : nat) (bij : natlt n1 =~ natlt n2)
  : Lemma (n1 == n2) =
  if n1 > n2 then pigeon n1 n2 bij.ff;
  if n1 < n2 then pigeon n2 n1 bij.gg;
  ()

let bij_cardinal (n1 n2 : nat)
  : Lemma (requires exists (b : natlt n1 =~ natlt n2). True)
          (ensures n1 == n2)
  = Classical.forall_intro (__bij_cardinal n1 n2)
