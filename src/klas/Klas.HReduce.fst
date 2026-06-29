module Klas.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout.Alg
module K = Kuiper.Kernel.Reduce

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et, real_like et |}
  (lay : (len:nat -> layout1 len))
  {| (len:szp -> ctlayout (lay len)) |}
  (nth : szp { nth <= max_threads })
  (lena : szp { SZ.fits (lena + nth) })
  (a : array1 et (lay lena) { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena)
  norewrite // no purification on fsti
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr)
  returns
    res : et
  ensures
    pure (res %~ chest1_rsum vr)
{
  let lena64 = SZ.sizet_to_uint64 lena;
  let nth64 = SZ.sizet_to_uint64 nth;
  dassert (not (FStar.UInt64.(lena64 +%^ nth64 <^ lena64)));
  assert pure (equal (chest_map id vr) vr);
  K.reduce1 id id lena nth a vr;
}

let reduce_f16_plus : reduce_ty f16 l1_forward = inst _ _
let reduce_f32_plus : reduce_ty f32 l1_forward = inst _ _
let reduce_f64_plus : reduce_ty f64 l1_forward = inst _ _
let reduce_u32_plus : reduce_ty u32 l1_forward = inst _ _
let reduce_u64_plus : reduce_ty u64 l1_forward = inst _ _
