module Kuiper.Epoch
#lang-pulse

open Pulse
open Kuiper.Kernel.Stream

(* An epoch is a position in a stream's work queue, not a synchronization
round: every operation enqueued on a stream advances the position by one. This
lets a launch name the operations that precede it, and hence consume their
results without host synchronization. *)
type epoch_t = erased nat

unfold
let epoch_next (e:epoch_t) : epoch_t = e + 1

(* Exclusive ownership of the current tail of [s]'s work queue. The next
operation enqueued on [s] takes slot [n]. *)
val epoch_live (s: stream_t) (n:epoch_t) : slprop

(* Duplicable evidence that all operations before queue position [n] have
completed. Host code obtains this witness through stream or device
synchronization.

A pledge guarded by [epoch_done s n] can also feed a subsequent operation on
the same stream. The launch primitive uses stream ordering to justify this
dependency without exposing a completion witness or the pledged resources to
its caller. A pledge is not itself evidence of completion, so chaining launches
does not let the host redeem their results before synchronization.

One completion predicate suffices for both uses: enqueuing only returns a
pledge and advances the unique epoch counter. There is no rule deriving
[epoch_done s n] from [epoch_live s n], and a previously obtained completion
witness cannot redeem a newly enqueued result at a later position. *)
val epoch_done (s: stream_t) (n:epoch_t) : slprop

(* Mint the unique epoch counter for a newly created stream. Consuming
[stream_fresh] guarantees this happens exactly once. *)
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
