module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1

// TODO: generalize operation? It currently always uses `add`
// from the scalar class.

inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |} =
  fn (nth : szp { nth <= max_threads })
     (lena : sz)
     (#l : Array1.layout lena) {| ctlayout l |}
     (a : Array1.t et l { Array1.is_global a })
     (#va : erased (lseq et lena))
     (vr : erased (lseq real lena))
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr) **
    pure (SZ.fits (lena + nth)) // Almost impossible to falsify
  returns
    res : et
  ensures
    pure (res %~ rsum vr)

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et, real_like et |} : reduce_ty et
