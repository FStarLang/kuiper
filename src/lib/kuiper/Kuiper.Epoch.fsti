module Kuiper.Epoch
#lang-pulse

open Pulse

type epoch_t = erased nat

val epoch_live (n:epoch_t) : slprop
val epoch_done (n:epoch_t) : slprop

ghost
fn get_epoch ()
  requires emp
  returns e : epoch_t
  ensures epoch_live e

ghost
fn done_lower (e f : epoch_t)
  preserves epoch_done e
  requires pure (f <= e)
  ensures  epoch_done f
