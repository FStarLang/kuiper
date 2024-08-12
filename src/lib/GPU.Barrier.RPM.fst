module GPU.Barrier.RPM

#lang-pulse

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open GPU.Base
module B = GPU.Barrier
open GPU.SizeT
module SZ = FStar.SizeT

let mbarrier_tok
  (n:nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (it : nat)
  (tid : nat { 0 <= tid /\ tid < n })
  : slprop = exists* b. B.barrier_tok #n
    (fun it (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
    (fun it (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to)) b it tid

ghost
fn mk_mbarrier_proof
  (n : nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (it: nat)
  requires bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
  ensures  bigstar 0 n (fun (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to))
{
  bigstar_map #0 #0 #0 #n #(fun (from : nat { 0 <= from /\ from < n }) -> bigstar #0 0 n (p it from)) #_
    (fun (from : nat { 0 <= from /\ from < n }) -> bigstar_eta _);
  bigstar_commute #0 #0 0 n 0 n (fun (from : nat { 0 <= from /\ from < n }) -> fun (to : nat { 0 <= to /\ to < n }) -> p it from to);
}

// TODO: remove
ghost fn fold_mbarrier_tok
  (#n:nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (b : B.barrier n)
  (it : nat)
  (tid : nat { 0 <= tid /\ tid < n })
  requires B.barrier_tok #n
    (fun it (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
    (fun it (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to)) b it tid
  ensures mbarrier_tok n p it tid
{
  fold (mbarrier_tok n p it tid)
}

ghost
fn mk_mbarrier
  (n: SZ.t { 0 < n /\ n <= max_threads })
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  requires block_setup n
  ensures  block_setup n ** bigstar 0 n (mbarrier_tok n p 0)
{
  let b = B.mk_barrier n
    (fun it (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
    (fun it (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to))
    (mk_mbarrier_proof n p);
  bigstar_map #0 #0 #0 #n (fold_mbarrier_tok #n p b 0);
}

// __syncthreads()
fn mbarrier_wait
  (#n : erased nat)
  (#p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (#it : erased nat)
  (#tid : erased nat { tid < n })
  requires mbarrier_tok n p  it    tid ** bigstar 0 n (p it tid)
  ensures  mbarrier_tok n p (it+1) tid ** bigstar 0 n (fun (from: nat { 0 <= from /\ from < n }) -> p it from tid)
{
  unfold mbarrier_tok;
  B.barrier_wait #n #(fun it (from: nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from)) #_ _;
  fold (mbarrier_tok n p (it+1) tid);
}

ghost
fn drop_mbarrier
  (#n : nat)
  (#p : (it:nat -> from: nat { from < n } -> to: nat { to < n } -> slprop))
  (#it: nat)
  requires bigstar 0 n (mbarrier_tok n p it)
  ensures  emp
{
  (* should use drop_barrier.. but not a big deal really. *)
  drop_ (bigstar 0 n (mbarrier_tok n p it))
}
