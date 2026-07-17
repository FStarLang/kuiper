module Klas.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Seq.Common
module SZ = FStar.SizeT

(* Type of reduction over a family of layouts. Note:
pre_map is specialized to the identity here. *)
inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |}
  (lay : (len:nat -> layout1 len))
=
  fn (nth : szp { nth <= max_threads })
     (len : szp { SZ.fits (len + nth) })
     (a : array1 et (lay len) { is_global a })
     (#va : chest1 et len)
     (vr : chest1 real len)
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr)
  returns
    res : et
  ensures
    pure (res %~ chest1_rsum vr)

val reduce_f16_plus : reduce_ty f16 l1_forward
val reduce_f32_plus : reduce_ty f32 l1_forward
val reduce_f64_plus : reduce_ty f64 l1_forward
val reduce_u32_plus : reduce_ty u32 l1_forward
val reduce_u64_plus : reduce_ty u64 l1_forward
