module Kuiper.Barrier.RPM

#lang-pulse

open Pulse
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Common
module B = Kuiper.Barrier

(* A resource-passing matrix. *)
let rpm_t (n:nat) =
  it:nat ->
  from: natlt n ->
  to: natlt n ->
  slprop

let row
  (#n:nat) (p : rpm_t n)
  (it : nat)
  (i : natlt n)
  : slprop =
  forall+ (j: natlt n). p it i j

let col
  (#n:nat) (p : rpm_t n)
  (it : nat)
  (j : natlt n)
  : slprop =
  forall+ (i: natlt n). p it i j

[@@no_mkeys]
val mbarrier_tok
  (n:nat)
  (p : rpm_t n)
  : slprop

instance val mbarrier_tok_sendable
  (n:nat)
  (p : rpm_t n)
: is_send_across block_of (mbarrier_tok n p)

ghost
fn mk_mbarrier
  (n: nat { 0 < n /\ n <= max_threads })
  (p : rpm_t n)
  requires can_create_barrier n
  ensures  consumed_can_create_barrier
  ensures forall+ (i : natlt n). mbarrier_tok n p ** B.barrier_state 0

// NB: reusing the same barrier_state token
inline_for_extraction noextract
fn mbarrier_wait
  ()
  (#n : erased nat)
  (#p : rpm_t n)
  (#it : erased nat)
  (#tid : enatlt n)
  preserves mbarrier_tok n p
  preserves thread_id n tid
  requires B.barrier_state it     ** row p it tid
  ensures  B.barrier_state (it+1) ** col p it tid
