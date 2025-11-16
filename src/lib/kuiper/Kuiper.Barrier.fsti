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

type barrier_transform (#n:nat) (p q : barrier_side n) =
  it:nat ->
  stt_ghost unit emp_inames
           (requires forall+ (i:natlt n). p it i)
           (ensures  fun _ -> forall+ (i:natlt n). q it i)

(* A token representing that there is a barrier in scope.
   This is a ghost token, and is not used at runtime. *)

(* A token representing that there is a barrier in scope with contract (p,q).
This does not change as we wait on the barrier. *)
[@@no_mkeys]
val barrier_tok (#n:nat) (p q : barrier_side n) : slprop

(* A token that we are at iteration `it` of the current barrier. *)
[@@no_mkeys]
val barrier_state (it : nat) : slprop

instance val barrier_tok_sendable
  (n:nat)
  (p q : barrier_side n)
: is_send_across block_of (barrier_tok #n p q)

instance val barrier_state_sendable
  (it : nat)
: is_send_across block_of (barrier_state it)

(* Creating a barrier for n threads. Note how this is a
   ghost function!. The 'pf' argument is a proof, once and
   for all, that the resource passing in each barrier_wait is
   correct, since it shows how all of the p's give all of the
   q's at each iteration. *)
ghost
fn mk_barrier
  (n: nat)
  (p q : barrier_side n)
  (pf : barrier_transform p q)
  requires can_create_barrier n
  ensures  consumed_can_create_barrier
  ensures  forall+ (i:natlt n). barrier_tok p q ** barrier_state 0

(* Wait on the barrier. This function blocks until all threads call it
   simultaneously. Each thread provides the current p
   and gets the current q. The iteration counter is incremented. *)
fn barrier_wait
  ()
  (#n : erased nat)
  (#p #q : barrier_side n)
  (#it : erased nat)
  (#tid : enatlt n)
  preserves barrier_tok p q
  preserves thread_id n tid
  requires barrier_state it     ** p it tid
  ensures  barrier_state (it+1) ** q it tid
