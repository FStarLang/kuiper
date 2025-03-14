module Kuiper.Poly.DotProduct

#lang-pulse

open Kuiper
module U64 = FStar.UInt64

(* pointwise mul of sequences *)
let pmul
  (#et:Type0) {| scalar et |}
  (s1 s2: seq et)
  : Ghost (seq et)
          (requires len s1 == len s2)
          (ensures fun _ -> True)
  = Seq.init_ghost (len s1)
      (fun i -> Seq.index s1 i `mul` Seq.index s2 i)

let sum 
  (#et:Type0) {| scalar et |}
  (s : seq et)
  : GTot et
  = Kuiper.Seq.Common.seq_fold_left add zero s

fn dotprod
  (#et:Type0) {| scalar et |}
  (lena : szp{lena <= max_threads})
  (a1 a2: vec et)
  (v1 v2: erased (seq et))
  (#_: squash (len v1 == lena /\ len v2 == lena))
  preserves
    cpu **
    (a1 |-> v1) **
    (a2 |-> v2)
  requires
    pure (is_comm_semigroup #et zero add)
  returns 
    dp : et
  ensures
    pure (dp == sum (pmul v1 v2))
