module Kuiper.Mul

open FStar.Mul
open FStar.Seq

open Kuiper
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

unfold let op_String_Access = Seq.index

let smul (s1 : seq u64) (s2 : seq u64 { len s2 == len s1 })
  : GTot (sr : seq u64 { len sr == len s1 })
  = Seq.init (len s1) (fun i -> U64.mul_mod s1.[i] s2.[i])

#lang-pulse

[@@pulse_unfold]
unfold
let pts_to_slice
  (#a:Type u#0)
  (#sz:nat)
  (x:gpu_array a sz)
  (#[Tactics.exact (`1.0R)] f : perm)
  (i:nat) (j:nat)
  (v : seq a)
: slprop
= gpu_pts_to_slice #a #sz x #f i j v

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
    pts_to_slice ar (thread_index etid) (thread_index etid + 1) 's
  ensures
    pts_to_slice ar (thread_index etid) (thread_index etid + 1) seq![(smul s1 s2).[thread_index etid]]
{
  let id = thread_idx_all ();
  let v1 = gpu_array_read #_ #_ #0 #size a1 id;
  let v2 = gpu_array_read #_ #_ #0 #size a2 id;
  let v = U64.(v1 *%^ v2);
  gpu_array_write #_ #_ #id #(id + 1) ar id v;
  (**)with sr. assert gpu_pts_to_slice ar id (id + 1) sr;
  (**)Seq.lemma_eq_intro sr seq![(smul s1 s2).[thread_index etid]];
  ()
}
