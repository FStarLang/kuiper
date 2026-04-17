module Kuiper.HReduce

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Seq.Common

(* Type of reduction over a family of layouts. *)
inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |}
  (lay : (len:nat -> Array1.layout len))
=
  fn (len : szp { len <= max_threads })
     (a : Array1.t et (lay len) { Array1.is_global a })
     (#va : erased (lseq et len))
     (vr : erased (lseq real len))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ vr)
  ensures (
    exists* (va' : lseq et len).
      on gpu_loc (a |-> va') **
      pure ((va' @! 0) %~ seq_fold_left (+.) 0.0R vr)
  )

val reduce_f16_plus : reduce_ty f16 l1_forward
val reduce_f32_plus : reduce_ty f32 l1_forward
val reduce_f64_plus : reduce_ty f64 l1_forward
val reduce_u32_plus : reduce_ty u32 l1_forward
val reduce_u64_plus : reduce_ty u64 l1_forward
