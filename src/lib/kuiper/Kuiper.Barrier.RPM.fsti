module Kuiper.Barrier.RPM

#lang-pulse

open Pulse
open Pulse.Lib.BigStar
open Kuiper.Base
module SZ = FStar.SizeT

[@@no_mkeys]
val mbarrier_tok
  (n:nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (it : nat)
  (tid : nat { 0 <= tid /\ tid < n })
  : slprop

ghost
fn mk_mbarrier
  (n: SZ.t { 0 < n /\ n <= max_threads })
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  requires block_setup n
  ensures  block_setup n ** bigstar 0 n (mbarrier_tok n p 0)

// __syncthreads()
fn mbarrier_wait
  (#n : erased nat)
  (#p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (#it : erased nat)
  (#tid : erased nat { tid < n })
  requires mbarrier_tok n p  it    tid ** bigstar 0 n (p it tid)
  ensures  mbarrier_tok n p (it+1) tid ** bigstar 0 n (fun (from: nat { 0 <= from /\ from < n }) -> p it from tid)

ghost
fn drop_mbarrier
  (#n : nat)
  (#p : (it:nat -> from: nat { from < n } -> to: nat { to < n } -> slprop))
  (#it: nat)
  requires bigstar 0 n (mbarrier_tok n p it)
  ensures  emp
