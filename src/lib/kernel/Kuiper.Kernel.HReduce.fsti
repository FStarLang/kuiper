module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
module Array1 = Kuiper.Array1

// TODO: generalize operation? It currently always uses `add`
// from the scalar class.

inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |} =
  fn (len : szp { len <= max_threads })
     (#l : Array1.layout len) {| ctlayout l |}
     (a : Array1.t et l { Array1.is_global a })
     (#va : erased (lseq et len))
     (vr : erased (lseq real len))
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr)
  returns
    res : et
  ensures
    pure (res %~ rsum vr)

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et, real_like et |} : reduce_ty et
