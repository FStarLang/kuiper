module Kuiper.Mul

open Kuiper
module U32 = FStar.UInt32
module U64 = FStar.UInt64

unfold let op_String_Access = Seq.index

let smul (s1 : seq u64) (s2 : seq u64 { len s2 == len s1 })
  : GTot (sr : seq u64 { len sr == len s1 })
  = Seq.init (len s1) (fun i -> U64.mul_mod s1.[i] s2.[i])

#lang-pulse

[@@ CPrologue "__global__"]
fn kernel (#size : erased nat)
  (a1 a2 ar : gpu_array u64 size)
  (s1 s2 : erased (seq u64))
  (etid : erased tid_t)
  (#_ : squash (len s1 == size /\ len s2 == size /\ gdim_x etid * bdim_x etid == size))
  preserves
    gpu ** thread_id etid **
    pts_to a1 #(1.0R /. size) s1 **
    pts_to a2 #(1.0R /. size) s2
  requires
    gpu_pts_to_slice ar (thread_index etid) (thread_index etid + 1) 's
  ensures
    gpu_pts_to_slice ar (thread_index etid) (thread_index etid + 1) seq![(smul s1 s2).[thread_index etid]]
{
  let tid = thread_idx_all ();
  rewrite each thread_index etid as tid;
  let v1 = gpu_array_read #_ #_ #0 #size a1 tid;
  let v2 = gpu_array_read #_ #_ #0 #size a2 tid;
  let v = FStar.UInt64.(v1 *%^ v2);
  gpu_array_write #_ #_ #tid #(tid + 1) ar tid v;
  (**)with sr. assert gpu_pts_to_slice ar tid (tid + 1) sr;
  (**)Seq.lemma_eq_intro sr seq![(smul s1 s2).[thread_index etid]];
  rewrite each FStar.SizeT.v tid as thread_index etid;
  ()
}
