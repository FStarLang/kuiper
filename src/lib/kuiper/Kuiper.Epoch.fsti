module Kuiper.Epoch
#lang-pulse

open Pulse
open Kuiper.Kernel.Stream

type epoch_t (s: stream_t) = erased nat

val epoch_live (#s: stream_t) (n:epoch_t s) : slprop
val epoch_done (#s: stream_t) (n:epoch_t s) : slprop

ghost
fn get_epoch (s: stream_t) ()
  returns e : epoch_t s
  ensures epoch_live #s e

ghost
fn done_lower (s: stream_t) (e f : epoch_t s)
  preserves epoch_done #s e
  requires pure (f <= e)
  ensures  epoch_done #s f