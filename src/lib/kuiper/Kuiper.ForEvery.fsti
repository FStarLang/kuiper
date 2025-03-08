module Kuiper.ForEvery

#lang-pulse
open Pulse
open Kuiper.Common
open Kuiper.Bijection
open Kuiper.Enumerable

val forevery
  (a:Type) {| enumerable a |}
  (f : a -> slprop)
  : slprop

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
