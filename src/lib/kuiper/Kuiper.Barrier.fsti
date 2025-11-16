module Kuiper.Barrier

#lang-pulse

open FStar.Tactics.V2
open Kuiper.Base
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.SizeT
open Kuiper.StateMachine
open Pulse.Lib.Pervasives

(* A barrier over nthreads. This is for specification only,
there is no runtime representation for it, nor a handle-like
type, as this models the builtin CUDA __syncthreads() function. *)

type barrier_side (stm : stm_t) (n:nat) =
  s : stm.state0 ->
  tid : natlt n ->
  slprop

type barrier_transform (#n:nat) (#stm:stm_t) (p q : barrier_side stm n) =
  s : stm.state0 ->
  stt_ghost unit emp_inames
           (requires forall+ (i:natlt n). p s i)
           (ensures  fun _ -> forall+ (i:natlt n). q s i)

(* A token representing that there is a barrier in scope.
   This is a ghost token, and is not used at runtime. *)

(* A token representing that there is a barrier in scope with contract (p,q).
This does not change as we wait on the barrier. *)
[@@no_mkeys]
val barrier_tok (#n:nat) (#stm : stm_t) (p q : barrier_side stm n) : slprop

(* A token that we are at iteration `it` of the current barrier. *)
[@@no_mkeys]
val barrier_state (#stm:stm_t) (s : st stm.state0) : slprop

instance val barrier_tok_sendable
  (n:nat)
  (stm : stm_t)
  (p q : barrier_side stm n)
  : is_send_across block_of (barrier_tok #n #stm p q)

instance val barrier_state_sendable
  (#stm:stm_t)
  (s : st stm.state0)
  : is_send_across block_of (barrier_state s)

(* Creating a barrier for n threads. Note how this is a
   ghost function!. The 'pf' argument is a proof, once and
   for all, that the resource passing in each barrier_wait is
   correct, since it shows how all of the p's give all of the
   q's at each iteration. *)
ghost
fn mk_barrier
  (n: nat)
  (#stm : stm_t)
  (p q : barrier_side stm n)
  (pf : barrier_transform p q)
  requires can_create_barrier n
  ensures  consumed_can_create_barrier
  ensures  forall+ (i:natlt n). barrier_tok p q ** barrier_state (S stm.init)

(* Wait on the barrier. This function blocks until all threads call it
   simultaneously. Each thread provides the current p
   and gets the current q. The iteration counter is incremented. *)
fn barrier_wait
  (#stm : stm_t)
  (#n : erased nat)
  ()
  (#p #q : barrier_side stm n)
  (#it : erased stm.state0)
  (#tid : enatlt n)
  preserves barrier_tok p q
  preserves thread_id n tid
  requires barrier_state (S it)        ** p it tid
  ensures  barrier_state (stm.next it) ** q it tid
