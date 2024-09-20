module Kuiper.Mul

open FStar.Mul
open FStar.Seq

open Kuiper
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

unfold let len = Seq.length

unfold let op_String_Access = Seq.index

let smul (s1 : seq u64) (s2 : seq u64 { len s2 == len s1 })
  : GTot (sr : seq u64 { len sr == len s1 })
  = Seq.init (len s1) (fun i -> U64.mul_mod s1.[i] s2.[i])

#lang-pulse

[@@ CPrologue "__global__"]
fn kernel (#size : erased nat)
  (a1 a2 ar : gpu_array u64 size)
  (s1 s2 : (s: erased (seq u64) { len s == size }))
  (etid : erased tid_t { gdim_x etid * bdim_x etid == size })
  requires gpu ** thread_id etid ** 
    gpu_pts_to_array a1 #(1.0R /. Real.of_int size) s1 **
    gpu_pts_to_array a2 #(1.0R /. Real.of_int size) s2 **
    gpu_pts_to_array_slice ar (thread_index etid) (thread_index etid + 1) 's
  ensures  gpu ** thread_id etid ** 
    gpu_pts_to_array a1 #(1.0R /. Real.of_int size) s1 **
    gpu_pts_to_array a2 #(1.0R /. Real.of_int size) s2 **
    gpu_pts_to_array_slice ar (thread_index etid) (thread_index etid + 1) seq![(smul s1 s2).[thread_index etid]]
{
  let id = thread_idx_all ();
  
  (**)unfold gpu_pts_to_array a1 #(1.0R /. Real.of_int size) s1;
  let v1 = gpu_array_read #_ #_ #0 #size a1 id;
  (**)fold gpu_pts_to_array a1 #(1.0R /. Real.of_int size) s1;

  (**)unfold gpu_pts_to_array a2 #(1.0R /. Real.of_int size) s2;
  let v2 = gpu_array_read #_ #_ #0 #size a2 id;
  (**)fold gpu_pts_to_array a2 #(1.0R /. Real.of_int size) s2;

  let v = U64.mul_mod v1 v2;
  gpu_array_write #_ #_ #id #(id + 1) ar id v;
  (**)with sr. assert gpu_pts_to_array_slice ar id (id + 1) sr;
  (**)Seq.lemma_eq_intro sr seq![(smul s1 s2).[thread_index etid]];
  ()
}
