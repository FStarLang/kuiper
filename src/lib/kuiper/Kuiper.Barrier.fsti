module Kuiper.Barrier

#lang-pulse

open Kuiper.Common
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open Kuiper.Base
open Kuiper.SizeT
module SZ = FStar.SizeT

(* A barrier over nthreads. This is for specification only,
there is no runtime representation for, as this models
the builtin CUDA __syncthreads() function. *)
[@@erasable]
val barrier
  (n:nat)
  : Type0

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
  (b : barrier n)
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
  (n: SZ.t { 0 < n /\ n <= max_threads })
  (p : (it:nat -> tid:natlt n -> slprop))
  (q : (it:nat -> tid:natlt n -> slprop))
  (pf : (it:nat -> stt_ghost unit emp_inames
                  (requires bigstar 0 n (p it))
                  (ensures  fun _ -> bigstar 0 n (q it))))
  requires block_setup n
  returns  b : erased (barrier n)
  ensures  block_setup n ** bigstar 0 n (barrier_tok p q b 0)

(* Wait on the barrier. This function blocks until all threads call it
   simultaneously. Each thread provides the current p
   and gets the current q. The iteration counter is incremented. *)
fn barrier_wait
  (#n : erased nat)
  (#p : (it:nat -> tid:natlt n -> slprop))
  (#q : (it:nat -> tid:natlt n -> slprop))
  (b : barrier n)
  (#it : erased nat)
  (#tid : erased (natlt n))
  requires barrier_tok p q b  it    tid ** p it tid
  ensures  barrier_tok p q b (it+1) tid ** q it tid

(* I don't think this is really useful, nor even desirable. The API
should probably enforce we always return a barrier token back, so
there are never two conflicting specs. So why even have the barrier type...? *)

// ghost
// fn drop_barrier
//   (#n : nat)
//   (#p : (it:nat -> tid:natlt n -> slprop))
//   (#q : (it:nat -> tid:natlt n -> slprop))
//   (#b : barrier n)
//   (#it: nat)
//   requires bigstar 0 n (barrier_tok p q b it)
//   ensures  emp
