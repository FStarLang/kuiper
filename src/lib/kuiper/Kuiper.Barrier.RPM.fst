module Kuiper.Barrier.RPM

open Pulse
open Kuiper.ForEvery
open Kuiper.Common
#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open Kuiper.Base
module B = Kuiper.Barrier
open Kuiper.SizeT


(* Cool. *)
let unfold_rpm () : Tac unit =
  norm [delta_only [`%mbarrier_contract; `%row; `%col]; iota; primops];
  slprop_equiv_norm ()

fn mbarrier_transform (#n:nat) (p:rpm_t n)
: B.barrier_transform #n (mbarrier_contract p) = it {
  rewrite (forall+ (i:natlt n). (mbarrier_contract p).B.rin it i)
       as (forall+ (x:natlt n) (y:natlt n). p it x y)
       by unfold_rpm ();
  forevery_commute (p it);
  rewrite (forall+ (y:natlt n) (x:natlt n). p it x y)
       as (forall+ (i:natlt n). (mbarrier_contract p).B.rout it i)
       by unfold_rpm ();
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
