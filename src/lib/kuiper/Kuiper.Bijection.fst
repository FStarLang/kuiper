module Kuiper.Bijection

#lang-pulse

open Kuiper.Common
open FStar.Tactics.V2

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
  let auxf (x y : natlt n1) : Lemma (bij.ff x == bij.ff y ==> x == y) =
    bij.gg_ff x;
    bij.gg_ff y
  in
  Classical.forall_intro_2 auxf;
  let auxg (x y : natlt n2) : Lemma (bij.gg x == bij.gg y ==> x == y) =
    bij.ff_gg x;
    bij.ff_gg y
  in
  Classical.forall_intro_2 auxg;
  (* clearly true, can't be bothered to prove right now *)
  assume (n1 > n2 ==> exists x y. bij.ff x == bij.ff y /\ x =!= y);
  assume (n1 < n2 ==> exists x y. bij.gg x == bij.gg y /\ x =!= y);
  ()

let bij_cardinal (n1 n2 : nat)
  : Lemma (requires exists (b : natlt n1 =~ natlt n2). True)
          (ensures n1 == n2)
  = Classical.forall_intro (__bij_cardinal n1 n2)
