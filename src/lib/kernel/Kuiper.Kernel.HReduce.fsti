module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }
module Array1 = Kuiper.Array1

inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |} =
  fn (len : szp { len <= max_threads })
     (#l : Array1.layout len) {| ctlayout l |}
     (a : Array1.t et l { Array1.is_global a })
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

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et, real_like et |} : reduce_ty et
