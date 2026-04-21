module Klas.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout.Alg
module P = Kuiper.Poly.HReduce

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et, real_like et |}
  (lay : (len:nat -> Array1.layout len))
  {| (len:szp -> ctlayout (lay len)) |}
  (lena : szp { lena <= max_threads })
  (a : Array1.t et (lay lena) { Array1.is_global a })
  (#va : erased (lseq et lena))
  (vr : erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ vr)
  ensures (
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure ((va' @! 0) %~ seq_fold_left (+.) 0.0R vr)
  )
{
  P.reduce lena a vr;
}

let reduce_f16_plus : reduce_ty f16 l1_forward = inst _ _
let reduce_f32_plus : reduce_ty f32 l1_forward = inst _ _
let reduce_f64_plus : reduce_ty f64 l1_forward = inst _ _
let reduce_u32_plus : reduce_ty u32 l1_forward = inst _ _
let reduce_u64_plus : reduce_ty u64 l1_forward = inst _ _
