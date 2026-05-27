module Kuiper.Example.Mul
#lang-pulse

open Kuiper
module U64 = FStar.UInt64

unfold let op_String_Access = Seq.index

let smul (s1 : seq u64) (s2 : seq u64 { len s2 == len s1 })
  : GTot (sr : seq u64 { len sr == len s1 })
  = Seq.init (len s1) (fun i -> U64.mul_mod s1.[i] s2.[i])

[@@CPrologue "__device__"] // no KrmlPrivate, example
fn kf (#size : erased nat)
  (a1 a2 ar : larray u64 size)
  (s1 s2 : erased (seq u64))
  (#_ : squash (len s1 == size /\ len s2 == size))
  (bid : szlt size)
  preserves
    pts_to_slice a1 #(1.0R /. size) 0 size s1 **
    pts_to_slice a2 #(1.0R /. size) 0 size s2
  requires
    pts_to_slice ar bid (bid + 1) 's
  ensures
    pts_to_slice ar bid (bid + 1) seq![(smul s1 s2).[bid]]
{
  let v1 = slice_read a1 bid;
  let v2 = slice_read a2 bid;
  let v = FStar.UInt64.(v1 *%^ v2);
  slice_write ar bid v;
  (**)with sr. assert pts_to_slice ar bid (bid + 1) sr;
  (**)Seq.lemma_eq_intro sr seq![(smul s1 s2).[bid]];
  ()
}
