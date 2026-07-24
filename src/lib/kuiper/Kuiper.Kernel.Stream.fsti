module Kuiper.Kernel.Stream
#lang-pulse

open Pulse

val stream_t: Type0

val stream_live (s: stream_t) : slprop

noextract
fn fresh_stream ()
  returns s:stream_t 
  ensures stream_live s

noextract
fn destroy_stream
  (s: stream_t)
  requires stream_live s