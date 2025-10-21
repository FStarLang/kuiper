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
fn mk_mbarrier
  (n: nat { 0 < n /\ n <= max_threads })
  (p : rpm_t n)
  preserves block_setup_tok n
  ensures forall+ i. mbarrier_tok n p 0 i
{
  B.mk_barrier n (row p) (col p) fn it {
    (* very nice. *)
    forevery_commute (p it);
  };
  (* Need to intro an exists in every component of the bigstar. *)
  forevery_map
    (B.barrier_tok #n (row p) (col p) 0)
    (mbarrier_tok n p 0)
    fn i { fold (mbarrier_tok n p 0 i) };
}

inline_for_extraction noextract
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
