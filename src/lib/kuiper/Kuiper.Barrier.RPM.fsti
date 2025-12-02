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

(* Trade a row of p for a column of p. *)
let mbarrier_contract (#n:nat) (p : rpm_t n) : B.contract n = {
  B.rin = row p;
  B.rout = col p;
}

[@@no_mkeys]
unfold
let mbarrier_tok (n : nat) (p : rpm_t n) : slprop =
  B.barrier_tok #n (mbarrier_contract p)

val mbarrier_transform 
  (#n : nat)
  (p : rpm_t n)
  : B.barrier_transform #n (mbarrier_contract p)

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
