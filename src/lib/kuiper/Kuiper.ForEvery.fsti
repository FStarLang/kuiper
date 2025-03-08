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
    forevery a (fun x ->
      forevery b (fun y -> f x y))
  ensures
    forevery (a & b) (fun (x, y) -> f x y)

ghost
fn forevery_iso
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (bij : (a =~ b))
  (p : a -> slprop)
  requires
    forevery a p
  ensures
    forevery b (fun y -> p (bij.gg y))

ghost
fn forevery_tostar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a p
  ensures
    bigstar 0 (cardinal a) (fun i -> p (of_nat i))

ghost
fn forevery_fromstar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    bigstar 0 (cardinal a) (fun i -> p (of_nat i))
  ensures
    forevery a p
