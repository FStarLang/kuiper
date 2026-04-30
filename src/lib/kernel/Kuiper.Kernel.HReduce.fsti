module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
open Kuiper.Seq.Common { seq_map }
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1

// TODO: generalize operation? It currently always uses `add`
// from the scalar class.

(* Could we use this instead of approx2? *)
instance approx_function_can_approximate
  (dom1 dom2 cod1 cod2 : Type)
  {| can_approximate dom1 dom2, can_approximate cod1 cod2 |}
  : can_approximate (dom1 -> cod1) (dom2 -> cod2) = {
  approximates = (fun f g -> forall x y. x %~ y ==> f x %~ g y);
}

inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |} =
  fn (pre_map : et -> et)
     (pre_map_r : real -> real { pre_map %~ pre_map_r })
     (nth : szp { nth <= max_threads })
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
    pure (res %~ rsum (seq_map pre_map_r vr))

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et, real_like et |} : reduce_ty et
