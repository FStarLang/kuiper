module Kuiper.Barrier.RPM

#lang-pulse

open Pulse
open Pulse.Lib.BigStar
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
  bigstar 0 n (fun j -> p it i j)

let col
  (#n:nat) (p : rpm_t n)
  (it : nat)
  (j : natlt n)
  : slprop =
  bigstar 0 n (fun i -> p it i j)

[@@no_mkeys]
val mbarrier_tok
  (n:nat)
  (p : rpm_t n)
  (it : nat)
  (tid : natlt n)
  : slprop

ghost
fn mk_mbarrier
  (n: nat { 0 < n /\ n <= max_threads })
  (p : rpm_t n)
  requires block_setup n
  ensures  block_setup n ** bigstar 0 n (mbarrier_tok n p 0)

fn mbarrier_wait
  ()
  (#n : erased nat)
  (#p : rpm_t n)
  (#it : erased nat)
  (#tid : erased (natlt n))
  requires mbarrier_tok n p  it    tid ** row p it tid
  ensures  mbarrier_tok n p (it+1) tid ** col p it tid
