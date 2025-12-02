module Kuiper.Barrier

#lang-pulse

open Kuiper.Common
open Pulse.Lib.Pervasives
open Kuiper.ForEvery
open FStar.Tactics.V2
open Kuiper.Base
open Kuiper.SizeT

(* A barrier over nthreads. This is for specification only,
there is no runtime representation for it, nor a handle-like
type, as this models the builtin CUDA __syncthreads() function. *)

type barrier_side (n:nat) =
  it : nat ->
  tid : natlt n ->
  slprop

[@@erasable]
noeq
type contract (n:nat) = {
  rin  : barrier_side n;
  rout : barrier_side n;
}

let empty_contract (n:nat) : contract n = {
  rin  = (fun _it _tid -> emp);
  rout = (fun _it _tid -> emp);
}

type barrier_transform (#n:nat) (c : contract n) =
  it:nat ->
  stt_ghost unit emp_inames
           (requires forall+ (i:natlt n). c.rin it i)
           (ensures  fun _ -> forall+ (i:natlt n). c.rout it i)

val empty_barrier_transform (n:nat)
  : barrier_transform (empty_contract n)

(* A token representing that there is a barrier in scope.
   This is a ghost token, and is not used at runtime. *)

(* A token representing that there is a barrier in scope with contract (p,q).
This does not change as we wait on the barrier. *)
[@@no_mkeys]
val barrier_tok (#n:nat) (c : contract n) : slprop

(* A token that we are at iteration `it` of the current barrier. *)
[@@no_mkeys]
val barrier_state (it : nat) : slprop

(* Note: none of the tokens above are sendable at all. *)

(* Wait on the barrier. This function blocks until all threads call it
   simultaneously. Each thread provides the current p
   and gets the current q. The iteration counter is incremented. *)
fn barrier_wait
  ()
  (#n : erased nat)
  (#c : contract n)
  (#it : erased nat)
  (#tid : enatlt n)
  preserves barrier_tok c
  preserves thread_id n tid
  requires barrier_state it     ** c.rin it tid
  ensures  barrier_state (it+1) ** c.rout it tid
