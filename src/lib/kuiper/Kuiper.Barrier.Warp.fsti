module Kuiper.Barrier.Warp

#lang-pulse

open Kuiper.Common
open Pulse.Lib.Pervasives
open Kuiper.ForEvery
open FStar.Tactics.V2
open Kuiper.Base
open Kuiper.SizeT


let warp_size = 32

(* Barrier_wait but for warps.  (__syncwarp())
  TODO: implement properly using a contract that goes in the kernel description,
  instead of this inline proof that could be thread-dependent and thus unsound
  (not all threads agree on the barrier). *)
fn warp_barrier_wait
  ()
  (p q: natlt warp_size -> slprop)
  (proof: stt_ghost unit emp_inames
    (requires forall+ (i:natlt warp_size). p i)
    (ensures  fun _ -> forall+ (i:natlt warp_size). q i))
  (#n: erased nat)
  (#tid : enatlt n)
  preserves thread_id n tid
  requires p (tid % warp_size)
  ensures  q (tid % warp_size)
