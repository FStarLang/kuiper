module Kuiper.Epoch
#lang-pulse

open Pulse
open Kuiper.Kernel.Stream

(* An epoch is a *position in a stream's work queue*, not a synchronization
round: every operation enqueued on a stream (e.g. a kernel launch) advances the
position by one. This is what lets a launch name the set of operations that
precede it, and hence observe their results without a host synchronization. *)
type epoch_t = erased nat

unfold
let epoch_next (e:epoch_t) : epoch_t = hide (reveal e + 1)

(* Exclusive: the tail of [s]'s queue is at position [n], i.e. the next
operation enqueued on [s] takes slot [n]. Exclusivity is the whole point -- it
is what makes the position an accurate account of the queue -- so this is
obtained only once per stream, from [init_epoch], and threaded linearly
thereafter. *)
val epoch_live (s: stream_t) (n:epoch_t) : slprop

(* Duplicable. Stream-order arrival: every operation enqueued on [s] at a
position < [n] has completed.

   This is the trigger under which the results of *stream-ordered* operations
   are pledged. Crucially it is available to a subsequent launch on [s] without
   any host synchronization, since CUDA already guarantees that work enqueued
   on a stream does not begin until all previously enqueued work on that stream
   has finished.

   There is deliberately no way for a client to introduce this at the current
   tail of the queue: the only introduction rules are [flushed_lower] and
   [done_flushed]. If a client could conjure [epoch_flushed s n] for the
   position it is about to enqueue at, it could redeem a kernel's result on the
   host before the kernel ran, and then, say, hand that result to a kernel on a
   *different* stream -- a genuine race. *)
val epoch_flushed (s: stream_t) (n:epoch_t) : slprop

(* Duplicable. Host-visible arrival: the host has observed the completion of
every operation enqueued on [s] at a position < [n], via [sync_stream] or
[sync_device].

   This is strictly weaker than [epoch_flushed] and is the trigger to use for
   effects that escape the stream's queue abstraction -- notably the
   host-visible half of a `cudaMemcpyAsync`, which is ordered with respect to
   the host only at a real synchronization point, not with respect to
   subsequently enqueued stream work. *)
val epoch_done (s: stream_t) (n:epoch_t) : slprop

(* Mint a stream's epoch counter. Consuming [stream_fresh] is what guarantees
this happens exactly once per stream, and that the position it returns really
is the (empty) queue's tail. The position itself is kept abstract. *)
ghost
fn init_epoch (s: stream_t) ()
  requires stream_fresh s
  returns e : epoch_t
  ensures epoch_live s e

ghost
fn done_lower (s: stream_t) (e f : epoch_t)
  preserves epoch_done s e
  requires pure (f <= e)
  ensures  epoch_done s f

ghost
fn flushed_lower (s: stream_t) (e f : epoch_t)
  preserves epoch_flushed s e
  requires pure (f <= e)
  ensures  epoch_flushed s f

(* Host-observed completion implies stream-order arrival: if the host has waited
for the work, it has certainly finished.

   The converse does NOT hold, and that asymmetry is the entire safety argument
   for operations that bypass the stream queue. A stream-ordered producer hands
   back an [epoch_flushed]-triggered pledge, which a later launch on the same
   stream can consume directly; a non-stream-ordered producer hands back an
   [epoch_done]-triggered pledge, which cannot be converted into one, and so
   remains unusable until the client actually synchronizes. *)
ghost
fn done_flushed (s: stream_t) (e : epoch_t)
  preserves epoch_done s e
  ensures  epoch_flushed s e
