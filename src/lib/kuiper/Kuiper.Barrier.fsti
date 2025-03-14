module Kuiper.Barrier

#lang-pulse

open Kuiper.Common
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open Kuiper.Base
open Kuiper.SizeT

(* A barrier over nthreads. This is for specification only,
there is no runtime representation for it, nor a handle-like
type, as this models the builtin CUDA __syncthreads() function. *)

(* A token representing that
   1) There is a barrier in scope
   2) We are thread 'tid' in the group
   3) We are in interation 'it'

   p and q represent the pre- and postconditions
   to the barrier_wait for a given thread at a given iteration.
*)
val barrier_tok
  (#n:nat)
  (p : (it:nat -> tid:natlt n -> slprop))
  (q : (it:nat -> tid:natlt n -> slprop))
  (it : nat)
  (tid : natlt n)
  : slprop

(* Creating a barrier for n threads. Note how this is a
   ghost function!. The 'pf' argument is a proof, once and
   for all, that the resource passing in each barrier_wait is
   correct, since it shows how all of the p's give all of the
   q's at each iteration. *)
ghost
fn mk_barrier
  (n: nat { 0 < n /\ n <= max_threads })
  (p : (it:nat -> tid:natlt n -> slprop))
  (q : (it:nat -> tid:natlt n -> slprop))
  (pf : (it:nat -> stt_ghost unit emp_inames
                  (requires bigstar 0 n (p it))
                  (ensures  fun _ -> bigstar 0 n (q it))))
  requires block_setup_tok n
  ensures  block_setup_tok n ** bigstar 0 n (barrier_tok p q 0)

(* Wait on the barrier. This function blocks until all threads call it
   simultaneously. Each thread provides the current p
   and gets the current q. The iteration counter is incremented. *)
fn barrier_wait
  ()
  (#n : erased nat)
  (#p : (it:nat -> tid:natlt n -> slprop))
  (#q : (it:nat -> tid:natlt n -> slprop))
  (#it : erased nat)
  (#tid : enatlt n)
  requires barrier_tok p q  it    tid ** p it tid
  ensures  barrier_tok p q (it+1) tid ** q it tid
