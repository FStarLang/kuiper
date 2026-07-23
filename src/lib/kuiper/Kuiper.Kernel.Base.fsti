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

module SZ = Kuiper.SizeT

(* This is the single primitive for launching kernels, with the most general
type and capabilities. There are many simpler versions in the Kuiper.Kernel module,
all implemented using this one and without any extra assumptions. *)
noextract
fn launch_kernel_full
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (s: stream_t)
  (#e : epoch_t s)
  preserves cpu ** stream_live s ** epoch_live e
  requires
    on gpu_loc full_pre
  ensures
    pledge0 (epoch_done e) (on gpu_loc full_post)

noextract
fn sync_stream
  (s: stream_t)
  (#e:epoch_t s)
  preserves
    cpu ** stream_live s
  requires
    epoch_live e
  returns
    e' : epoch_t s
  ensures
    epoch_done e **
    epoch_live e' **
    pure (e' >= e)

val sync_token: slprop

ghost fn sync_stream_ghost
  (s: stream_t)
  (#e:epoch_t s)
  preserves
    sync_token ** stream_live s
  requires
    epoch_live e
  returns
    e' : epoch_t s
  ensures
    epoch_done e **
    epoch_live e' **
    pure (e' >= e)

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

