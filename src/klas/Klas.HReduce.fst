module Klas.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout.Alg
module K = Kuiper.Kernel.HReduce

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et, real_like et |}
  (lay : (len:nat -> Array1.layout len))
  {| (len:szp -> ctlayout (lay len)) |}
  (lena : szp { lena <= max_threads })
  (a : Array1.t et (lay lena) { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  norewrite // no purification on fsti
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr)
  returns
    res : et
  ensures
    pure (res %~ rsum vr)
{
  K.reduce lena a vr;
}

let reduce_f16_plus : reduce_ty f16 l1_forward = inst _ _
let reduce_f32_plus : reduce_ty f32 l1_forward = inst _ _
let reduce_f64_plus : reduce_ty f64 l1_forward = inst _ _
let reduce_u32_plus : reduce_ty u32 l1_forward = inst _ _
let reduce_u64_plus : reduce_ty u64 l1_forward = inst _ _
