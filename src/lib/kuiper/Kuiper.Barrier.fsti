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

(* A token representing that
   1) There is a barrier in scope
   2) We are thread 'tid' in the group
   3) We are in interation 'it'

   p and q represent the pre- and postconditions
   to the barrier_wait for a given thread at a given iteration.
*)
val barrier_tok
  (#n:nat)
  ([@@@mkey] p [@@@mkey] q : barrier_side n)
  (it : nat)
  ([@@@mkey] tid : natlt n)
  : slprop

instance val barrier_tok_sendable
  (n:nat)
  (p q : barrier_side n)
  (it : nat)
  (tid : natlt n)
: is_send_across block_of (barrier_tok #n p q it tid)

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
  ensures  forall+ (i:natlt n). barrier_tok p q 0 i

(* Wait on the barrier. This function blocks until all threads call it
   simultaneously. Each thread provides the current p
   and gets the current q. The iteration counter is incremented. *)
fn barrier_wait
  ()
  (#n : erased nat)
  (#p #q : barrier_side n)
  (#it : erased nat)
  (#tid : enatlt n)
  requires barrier_tok p q  it    tid ** p it tid
  ensures  barrier_tok p q (it+1) tid ** q it tid
