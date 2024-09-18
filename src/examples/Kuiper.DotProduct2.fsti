module Kuiper.DotProduct2

#lang-pulse

open Kuiper
module A = Pulse.Lib.Array
module U64 = FStar.UInt64
module HR = Kuiper.HReduceU64Plus

(* calling it size means name resolution confusion with Kuiper.Sized.size *)
let dp2_size : sz = 1024sz

(* pointwise mul of sequences *)
let pmul (s1 s2: seq u64)
  : Ghost (seq u64)
          (requires Seq.length s1 == Seq.length s2)
          (ensures fun _ -> True)
  = Seq.init_ghost (Seq.length s1)
      (fun i -> U64.mul_mod (Seq.index s1 i) (Seq.index s2 i))

let sum = Kuiper.Seq.Common.seq_fold_left HR.op HR.neu
// NB: HR.op=U64.add_mod, HR.neu=U64.zero, but the brittle
// SMT encoding breaks if we use that instead of exactly the same term
// as appears in HR

fn main
  (a1 a2: array u64)
  (v1 v2: erased (seq u64))
  (#_: squash (Seq.length v1 = dp2_size /\ Seq.length v2 = dp2_size))
  requires cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2
  returns  dp: u64
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** pure (dp == sum (pmul v1 v2))
