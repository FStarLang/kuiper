module Kuiper.Epoch
#lang-pulse

open Pulse

val epoch_live (n:nat) : slprop
val epoch_done (n:nat) : slprop

ghost
fn get_epoch ()
  requires emp
  returns e : erased nat
  ensures epoch_live e

ghost
fn done_lower (e f :nat)
  preserves epoch_done e
  requires pure (f <= e)
  ensures  epoch_done f
