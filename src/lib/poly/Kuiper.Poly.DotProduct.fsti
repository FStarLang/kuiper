module Kuiper.Poly.DotProduct

#lang-pulse

open Kuiper

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

inline_for_extraction noextract
type dotprod_ty
  (et:Type0) {| scalar et, real_like et |} =
  fn (lena : szp{lena <= max_threads})
     (a1 : vec et)
     (a2 : vec et)
     (#v1 #v2 : erased (lseq et lena))
     (vr1 vr2: erased (lseq real lena) { v1 %~ vr1 /\ v2 %~ vr2 })
  preserves
    cpu ** a1 |-> v1 ** a2 |-> v2
  returns
    dp : et
  ensures
    pure (dp %~ sum (pmul vr1 vr2))

inline_for_extraction noextract
val dotprod
  (#et:Type0) {| scalar et, real_like et |}
  : dotprod_ty et
