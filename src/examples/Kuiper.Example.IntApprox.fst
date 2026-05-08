module Kuiper.Example.IntApprox

#lang-pulse

open Kuiper
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module U32 = FStar.UInt32
module Array1 = Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }

(* Strengthen the spec of a reduction *)

noextract
fn reduce (#et:Type0) {| scalar et, real_like et |}
  (len : szp)
  (a : Array1.t u32 (l1_forward len) { Array1.is_global a })
  (#s : erased (lseq u32 len))
  (r : erased (lseq real len))
  preserves cpu
  preserves on gpu_loc (a |-> s)
  requires  pure (s %~ r) **
            pure (SZ.fits (len + 1024sz)) // Almost impossible to falsify
  returns   res : u32
  ensures   pure (res %~ rsum r)
{
  assert pure (Seq.equal (seq_map id r) r);
  Kuiper.Kernel.HReduce.reduce id id 1024sz len a r;
}

(* A stronger exact spec for reduce on u32s, proven from the approximate spec. *)
noextract
fn reduce_u32
  (len : szp)
  (a : Array1.t u32 (l1_forward len) { Array1.is_global a })
  (#s : erased (lseq u32 len))
  preserves cpu
  preserves on gpu_loc (a |-> s)
  requires  pure (SZ.fits (len + 1024sz)) // Almost impossible to falsify
  returns   res : u32
  ensures   pure (U32.v res == seq_fold_left add zero s)
{
  to_real_seq_is_approx s;
  let res = reduce #u32 len a #s (seq_map to_real s);
  let rr = rsum (seq_map to_real s);
  assert pure (res %~ rr);
  sum_is_approx s (seq_map to_real s);
  v_approximates_inj res (seq_fold_left add zero s) rr;
  res
}
