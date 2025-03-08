module Kuiper.ForEvery

#lang-pulse
open Pulse
open Kuiper.Common
open Kuiper.Bijection
open Kuiper.Enumerable
open Pulse.Lib.BigStar

val forevery
  (a:Type) {| enumerable a |}
  (f : a -> slprop)
  : slprop

(* We can use this... but eta matters, so for now at least,
   let's just stick with forevery. *)
unfold
let ( forall+ )
  (#a:Type) {| enumerable a |}
  (f : a -> slprop)
  : slprop = forevery a f

ghost
fn forevery_flatten
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). f x y
  ensures
    forall+ (xy : a & b). f xy._1 xy._2

ghost
fn forevery_iso
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (bij : (a =~ b))
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (y:b). p (bij.gg y)

ghost
fn forevery_tostar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    bigstar 0 (cardinal a) (fun i -> p (of_nat i))

ghost
fn forevery_fromstar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    bigstar 0 (cardinal a) (fun i -> p (of_nat i))
  ensures
    forall+ (x:a). p x

ghost
fn forevery_unit_intro
  (p : slprop)
  requires
    p
  ensures
    forall+ (_:unit). p

ghost
fn forevery_unit_elim
  (p : slprop)
  requires
    forall+ (_:unit). p
  ensures
    p

(* SHOULD NOT BE NEEDED!
   1) We should mark the p argument of forevery as extensional,
      and have the checker do the work for us.
   2) Using forall+, everything should be uniformly eta-expanded.
 *)
ghost
fn forevery_eta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a p
  ensures
    forevery a (fun x -> p x)

ghost
fn forevery_uneta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a (fun x -> p x)
  ensures
    forevery a p

ghost
fn forevery_rw_size
  (n1 : nat)
  (n2 : nat{n1 == n2})
  (#p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n1). p i
  ensures
    forall+ (i : natlt n2). p i
