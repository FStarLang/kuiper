module Kuiper.IntApprox

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Approximates.U32
open Kuiper.Seq.Common
module U32 = FStar.UInt32

(* Strengthen the spec of a reduction *)

noextract
fn reduce (#et:Type0) {| scalar et, real_like et |}
  (len : erased nat) (a : larray u32 len) (#s : erased (seq u32))
  (#r : erased (seq real))
  preserves a |-> s ** pure (s %~ r)
  returns   res : u32
  ensures   pure (res %~ seq_fold_left (+.) 0.0R r)
{
  admit(); // Intentional, to not import HReduce here.
}

(* A stronger exact spec for reduce on u32s, proven from the approximate spec. *)
noextract
fn reduce_u32 (len : erased nat) (a : larray u32 len) (#s : erased (seq u32))
  preserves a |-> s
  returns   res : u32
  ensures   pure (U32.v res == seq_fold_left add zero s)
{
  to_real_seq_is_approx s;
  let res = reduce #u32 len a #s #(seq_map to_real s);
  let rr = seq_fold_left (+.) 0.0R (seq_map to_real s);
  assert pure (res %~ rr);
  sum_is_approx s (seq_map to_real s);
  v_approximates_inj res (seq_fold_left add zero s) rr;
  res
}

(* Force extraction of something *)
let x = 1ul
