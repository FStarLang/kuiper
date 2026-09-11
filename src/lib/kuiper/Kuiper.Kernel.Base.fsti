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
type and capabilities. The input is pledged at the launch's queue position:
[pledge0 (epoch_done s e) p] means that [p] is available to the operation
that takes position [e] on [s]. Everything that produced it is earlier on the
same stream, so CUDA stream ordering makes it available without host
synchronization. The output is pledged one position later, allowing dependent
launches to chain directly. This consumes and produces pledges; it does not
give the caller an [epoch_done] witness or ownership of the pledged result. *)
noextract
fn launch_kernel_full
  (#full_pre #full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (s: stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    pledge0 (epoch_done s e) (on gpu_loc full_pre)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_done s (epoch_next e)) (on gpu_loc full_post)

(* Synchronization enqueues no work, so it leaves the queue position alone. *)
noextract
[@@FStar.Attributes.custard_extern "KPR_MUST_stream_sync";
   FStar.Attributes.custard_c_header "kuiper.h"]
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
  (frame p q : erased slprop)
  (justif:
    ghost fn ()
      preserves sync_token
      requires frame ** p
      ensures frame ** q)
  preserves
    frame ** cpu
  requires p
  ensures q
