module Kuiper.Kernel.Stream
#lang-pulse

open Pulse

val stream_t: Type0

val stream_live (s: stream_t) : slprop

(* An exclusive witness that no work has ever been enqueued on [s]. It is
produced only by [fresh_stream] and is consumed by [Kuiper.Epoch.init_epoch] to
mint the stream's (unique) epoch counter. Threading the counter through it is
what makes epochs trustworthy: if this token could be obtained twice for one
stream, a client could mint two independent counters for it, and each would
under-count the work the other enqueued. See Kuiper.Epoch. *)
val stream_fresh (s: stream_t) : slprop

noextract
fn fresh_stream ()
  returns s:stream_t
  ensures stream_live s ** stream_fresh s

noextract
fn destroy_stream
  (s: stream_t)
  requires stream_live s