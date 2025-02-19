module Kuiper.DotProduct3

#lang-pulse

open Kuiper
module U64 = FStar.UInt64
module HR = Kuiper.HReduceU64Plus

(* calling it size means name resolution confusion with Kuiper.Sized.size *)
let dp2_size : sz = 1024sz

(* pointwise mul of sequences *)
let pmul (s1 s2: seq u64)
  : Ghost (seq u64)
          (requires len s1 == len s2)
          (ensures fun _ -> True)
  = Seq.init_ghost (len s1)
      (fun i -> U64.mul_mod (Seq.index s1 i) (Seq.index s2 i))

let sum = Kuiper.Seq.Common.seq_fold_left HR.op HR.neu

fn main
  (a1 a2: vec u64)
  (v1 v2: erased (seq u64))
  (#_: squash (len v1 = dp2_size /\ len v2 = dp2_size))
  preserves cpu ** (a1 |-> v1) ** (a2 |-> v2)
  requires emp
  returns  dp: u64
  ensures  pure (dp == sum (pmul v1 v2))

