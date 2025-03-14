module Kuiper.Barrier.RPM

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open Kuiper.Base
module B = Kuiper.Barrier
open Kuiper.SizeT

let mbarrier_tok
  (n : nat)
  (p : rpm_t n)
  (it : nat)
  (tid : natlt n)
  : slprop
  =
  (* Trade a row of p for a column of p. *)
  B.barrier_tok #n
    (row p)
    (col p)
    it tid

ghost
fn mk_mbarrier_proof
  (n : nat)
  (p : rpm_t n)
  (it: nat)
  requires bigstar 0 n (row p it)
  ensures  bigstar 0 n (col p it)
{
  (* very nice. *)
  bigstar_commute 0 n 0 n (p it);
}

ghost
fn mk_mbarrier
  (n: nat { 0 < n /\ n <= max_threads })
  (p : rpm_t n)
  requires block_setup_tok n
  ensures  block_setup_tok n ** bigstar 0 n (mbarrier_tok n p 0)
{
  B.mk_barrier n (row p) (col p) (mk_mbarrier_proof n p);
  (* Need to intro an exists in every component of the bigstar. *)
  ghost
  fn aux (i : natlt n)
    requires B.barrier_tok #n (row p) (col p) 0 i
    ensures  mbarrier_tok n p 0 i
  {
    fold (mbarrier_tok n p 0 i);
  };
  bigstar_map #0 #0 #0 #n aux;
}

fn mbarrier_wait
  ()
  (#n : erased nat)
  (#p : rpm_t n)
  (#it : erased nat)
  (#tid : enatlt n)
  requires mbarrier_tok n p  it    tid ** row p it tid
  ensures  mbarrier_tok n p (it+1) tid ** col p it tid
{
  unfold mbarrier_tok;
  B.barrier_wait ();
  fold mbarrier_tok;
}
