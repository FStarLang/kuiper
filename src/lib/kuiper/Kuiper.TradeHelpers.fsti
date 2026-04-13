module Kuiper.TradeHelpers

#lang-pulse
open Pulse
open Pulse.Lib.Trade
open Pulse.Lib.Forall

[@@allow_ambiguous]
ghost
fn ambig_trade_elim
  (#p #q : slprop)
  ()
  requires
    p ** (p @==> q)
  ensures q

unfold
let factored (p q : slprop) : slprop = p ** trade p q

ghost
fn map_forall u#a
  (#a:Type u#a)
  (p1 p2 : a -> slprop)
  (f : ghost fn (x:a)
         requires p1 x
         ensures  p2 x)
  requires
    forall* (x:a). p1 x
  ensures
    forall* (x:a). p2 x

ghost
fn vmap_forall u#a u#b
  (#a : Type u#a)
  (#b : Type u#b)
  (p : a -> slprop)
  (f : b -> a)
  requires
    forall* (x:a). p x
  ensures
    forall* (x:b). p (f x)
