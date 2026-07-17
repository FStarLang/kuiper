module Kuiper.Shareable

#lang-pulse
open Pulse
open Kuiper.ForEvery
open Kuiper.Common
open FStar.Tactics.Typeclasses

class shareable (res : perm -> slprop) = {
  [@@@no_method]
  _share_n :
    ghost fn (n : pos) (#p : perm)
      requires res p
      ensures forall+ (_ : natlt n). res (p /. Real.of_int n);
  [@@@no_method]
  _gather_n :
    ghost fn (n : pos) (#p : perm)
      requires forall+ (_ : natlt n). res (p /. Real.of_int n)
      ensures  res p;
}

ghost
fn share_n
  (res : perm -> slprop)
  {| d : shareable res |}
  (n : pos) (#p : perm)
  requires res p
  ensures forall+ (_ : natlt n). res (p /. Real.of_int n)

ghost
fn gather_n
  (res : perm -> slprop)
  {| d : shareable res |}
  (n : pos) (#p : perm)
  requires forall+ (_ : natlt n). res (p /. Real.of_int n)
  ensures  res p

instance val emp_shareable: shareable (fun _ -> emp)

instance val double_shareable
  (res1 : perm -> slprop) {| shareable res1 |}
  (res2 : perm -> slprop) {| shareable res2 |}
  (ffr1 ffr2: perm):
  (shareable (fun fr -> res1 (ffr1 *. fr) ** res2 (ffr2 *. fr)))
