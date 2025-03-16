module Kuiper.Mul
#lang-pulse

open Kuiper
module U64 = FStar.UInt64
module SZ = FStar.SizeT

unfold let op_String_Access = Seq.index

let smul (s1 : seq u64) (s2 : seq u64 { len s2 == len s1 })
  : GTot (sr : seq u64 { len sr == len s1 })
  = Seq.init (len s1) (fun i -> U64.mul_mod s1.[i] s2.[i])

[@@CPrologue "__device__"] // no KrmlPrivate, example
fn kf (#size : erased nat{size > 0}) (* do NOT use erased pos, inference suffers *)
  (a1 a2 ar : gpu_array u64 size)
  (s1 s2 : erased (seq u64))
  (#_ : squash (len s1 == size /\ len s2 == size))
  (ebid : enatlt size)
  preserves
    gpu ** block_id size ebid **
    pts_to a1 #(1.0R /. size) s1 **
    pts_to a2 #(1.0R /. size) s2 **
    emp
  requires
    gpu_pts_to_slice ar ebid (ebid + 1) 's
  ensures
    gpu_pts_to_slice ar ebid (ebid + 1) seq![(smul s1 s2).[ebid]]
{
  let bid = get_bid ();
  rewrite each ebid as SZ.v bid;
  assert (pure (bid >= 0));
  assert (pure (bid < size));
  let v1 = gpu_array_read #_ #_ #0 #size a1 bid;
  let v2 = gpu_array_read #_ #_ #0 #size a2 bid;
  let v = FStar.UInt64.(v1 *%^ v2);
  gpu_array_write #_ #_ #bid #(bid + 1) ar bid v;
  (**)with sr. assert gpu_pts_to_slice ar bid (bid + 1) sr;
  (**)Seq.lemma_eq_intro sr seq![(smul s1 s2).[ebid]];
  rewrite each SZ.v bid as ebid;
  ()
}
