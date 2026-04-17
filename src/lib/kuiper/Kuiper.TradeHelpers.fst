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
{
  elim_trade _ _;
}

// move to library?
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
{
  intro_forall #_ #p2 (forall* x. p1 x) fn x {
    elim_forall x;
    f x;
  };
}

// move to library?
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
{
  intro_forall #_ #(fun x -> p (f x)) (forall* x. p x) fn x {
    elim_forall (f x);
  };
}
