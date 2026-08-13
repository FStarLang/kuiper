module Kuiper.Kernel.Base
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.Base
open Kuiper.Array
open Kuiper.Epoch
open Pulse.Lib.Pledge
open Kuiper.Kernel.Desc
open Kuiper.Kernel.Stream
open Kuiper.Seq.Common
open FStar.Seq
open Kuiper.ForEvery


(* This is the single primitive for launching kernels, with the most general
type and capabilities. There are many simpler versions in the Kuiper.Kernel module,
all implemented using this one and without any extra assumptions.

The precondition is not owned outright but pledged at the launch's own queue
position: [pledge0 (epoch_flushed s e) p] is "p will be available to whatever
runs at position [e] on [s]". Since the launch takes slot [e], everything it
depends on is enqueued strictly earlier, so the stream's own ordering guarantee
suffices and no host synchronization is needed. The result is pledged one
position later, so consecutive dependent launches on a single stream chain
directly:

    launch_kernel_full k1 s;   (* e     ~> e + 1 *)
    launch_kernel_full k2 s;   (* e + 1 ~> e + 2, sees k1's output *)

To start such a chain from a resource you own on the host, inject it with
[return_pledge] (see [launch_kernel_full_owned] in Kuiper.Kernel). To finish it,
synchronize and use [pledge_flushed_done] / [done_flushed]. *)
noextract
fn launch_kernel_full
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (s: stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    pledge0 (epoch_flushed s e) (on gpu_loc full_pre)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_flushed s (epoch_next e)) (on gpu_loc full_post)

(* Synchronization enqueues nothing, so it leaves the queue position alone; it
only lets the host observe that everything already enqueued has finished. *)
noextract
fn sync_stream
  (s: stream_t)
  (#e:epoch_t)
  preserves
    cpu ** stream_live s ** epoch_live s e
  ensures
    epoch_done s e

val sync_token: slprop

ghost fn sync_stream_ghost
  (s: stream_t)
  (#e:epoch_t)
  preserves
    sync_token ** stream_live s ** epoch_live s e
  ensures
    epoch_done s e

noextract
fn sync_device ()
  (frame: erased slprop)
  (p: erased slprop)
  (q: erased slprop)
  (justif:
    ghost fn ()
      preserves sync_token
      requires frame ** p
      ensures frame ** q)
  preserves
    frame ** cpu
  requires p
  ensures q