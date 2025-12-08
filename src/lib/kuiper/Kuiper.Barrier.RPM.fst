module Kuiper.Barrier.RPM

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open Kuiper.Base
module B = Kuiper.Barrier
open Kuiper.SizeT


(* Cool. *)
fn mbarrier_transform (#n:nat) (p:rpm_t n)
: B.barrier_transform #n (mbarrier_contract p) = it {
  forevery_commute (p it);
}

inline_for_extraction noextract
fn mbarrier_wait
  ()
  (#n : erased nat)
  (#p : rpm_t n)
  (#it : erased nat)
  (#tid : enatlt n)
  preserves mbarrier_tok n p
  preserves thread_id n tid
  requires B.barrier_state it     ** row p it tid
  ensures  B.barrier_state (it+1) ** col p it tid
{
  unfold mbarrier_tok;
  rewrite row p it tid as (mbarrier_contract p).B.rin it tid;
  B.barrier_wait ();
  rewrite (mbarrier_contract p).B.rout it tid as col p it tid;
  fold mbarrier_tok;
}
