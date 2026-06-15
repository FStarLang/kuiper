module Klas.Arange

#lang-pulse
open Kuiper

module SZ = Kuiper.SizeT
module U64 = FStar.UInt64
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast
module Array1 = Kuiper.Array1
module Arange = Kuiper.Kernel.Arange
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }

inline_for_extraction noextract
let arange_step_i64
  (n : szp { SZ.v n < pow2 32 })
  (start step : u64)
  (_ : squash (U64.v start + U64.v step * SZ.v n < pow2 64))
  (i : SZ.t { SZ.v i < SZ.v n })
  : u64
  = let i_u32 : U32.t = SZ.sizet_to_u32 i in
    let i_u64 : U64.t = Cast.uint32_to_uint64 i_u32 in
    U64.add start (U64.mul step i_u64)

fn arange_i64
  (n : szp { n <= max_blocks * max_threads })
  (start step : u64)
  (_ : squash (U64.v start + U64.v step * n < pow2 64))
  preserves cpu
  returns out : Array1.t u64 (l1_forward n)
  ensures
    on gpu_loc (out |-> Arange.arange_seq n (arange_step_i64 n start step ())) **
    pure (is_global out)
{
  Arange.arange_gpu n (arange_step_i64 n start step ())
}
