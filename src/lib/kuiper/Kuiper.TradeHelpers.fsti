module Kuiper.TradeHelpers

#lang-pulse
open Pulse
open Pulse.Lib.Trade

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
