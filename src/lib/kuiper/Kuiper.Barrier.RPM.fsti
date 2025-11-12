module Kuiper.Barrier.RPM

#lang-pulse

open Pulse
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Common

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
  (it : nat)
  (tid : natlt n)
  : slprop

instance val mbarrier_tok_sendable 
  (n:nat)
  (p : rpm_t n)
  (it : nat)
  (tid : natlt n)
: is_send_across block_of (mbarrier_tok n p it tid)

ghost
fn mk_mbarrier
  (n: nat { 0 < n /\ n <= max_threads })
  (p : rpm_t n)
  requires can_create_barrier n
  ensures  consumed_can_create_barrier
  ensures forall+ i. mbarrier_tok n p 0 i

inline_for_extraction noextract
fn mbarrier_wait
  ()
  (#n : erased nat)
  (#p : rpm_t n)
  (#it : erased nat)
  (#tid : enatlt n)
  requires mbarrier_tok n p  it    tid ** row p it tid
  ensures  mbarrier_tok n p (it+1) tid ** col p it tid
