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
  : slprop
  =
  (* Trade a row of p for a column of p. *)
  B.barrier_tok #n
    (row p)
    (col p)

instance mbarrier_tok_sendable
  (n:nat)
  (p : rpm_t n)
: is_send_across block_of (mbarrier_tok n p)
= solve

ghost
fn mk_mbarrier
  (n: nat { 0 < n /\ n <= max_threads })
  (p : rpm_t n)
  requires can_create_barrier n
  ensures  consumed_can_create_barrier
  ensures forall+ (i : natlt n). mbarrier_tok n p ** B.barrier_state 0
{
  B.mk_barrier n (row p) (col p) fn it {
    (* very nice. *)
    forevery_commute (p it);
  };
  (* Need to intro an exists in every component of the bigstar. *)
  forevery_map
    (fun (i: natlt n) -> B.barrier_tok #n (row p) (col p) ** B.barrier_state 0)
    (fun (i: natlt n) -> mbarrier_tok n p ** B.barrier_state 0)
    fn i { fold (mbarrier_tok n p) };
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
  B.barrier_wait ();
  fold mbarrier_tok;
}
