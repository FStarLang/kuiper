module Klas.Reduce

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout.Alg
open Kuiper.Seq.Common
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module HReduce = Kuiper.Kernel.HReduce

fn mean
  (n : szp { SZ.fits (n + max_threads) })
  (a : Array1.t f32 (l1_forward n) { Array1.is_global a })
  (#sa : erased (lseq f32 n))
  preserves cpu
  preserves on gpu_loc (a |-> sa)
  returns res : f32
  ensures pure (res %~ (rsum (to_real_seq (reveal sa)) /. Real.of_int n))
{
  let vr : erased (lseq real n) = hide (to_real_seq (reveal sa));
  to_real_seq_is_approx (reveal sa);
  let sum : f32 =
    HReduce.reduce #f32 id id 1024sz n #(l1_forward n) #(c_l1_forward _) a vr;
  assert pure (Seq.equal (seq_map id (reveal vr)) (reveal vr));
  let n_u32 : FStar.UInt32.t = SZ.sizet_to_u32 n;
  let n_i64 : FStar.Int64.t = FStar.Int.Cast.uint32_to_int64 n_u32;
  let nf : f32 = of_int #f32 n_i64;
  of_int_approx #f32 n_i64;
  div sum nf
}
