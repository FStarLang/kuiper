module Kuiper.Kernel.HReduce.Max

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
open Kuiper.Seq.Common { seq_map }
open Kuiper.Math.OnlineSoftmax { seq_max }
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1

(* Function approximation, needed to state `pre_map %~ pre_map_r`. *)
instance approx_function_can_approximate
  (dom1 dom2 cod1 cod2 : Type)
  {| can_approximate dom1 dom2, can_approximate cod1 cod2 |}
  : can_approximate (dom1 -> cod1) (dom2 -> cod2) = {
  approximates = (fun f g -> forall x y. x %~ y ==> f x %~ g y);
}

(* Parallel single-block max reduction.

   Same shape as Kuiper.Kernel.HReduce.reduce, but:
   - it reduces with fmax/seq_max instead of add/rsum, and
   - it requires `0 < nth /\ nth <= lena`, guaranteeing every strided bucket
     is non-empty (max has no real-number identity). *)

inline_for_extraction noextract
type reduce_max_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (pre_map : et -> et)
     (pre_map_r : real -> real { pre_map %~ pre_map_r })
     (nth : szp { nth <= max_threads })
     (lena : szp)
     (#l : Array1.layout lena) {| ctlayout l |}
     (a : Array1.t et l { Array1.is_global a })
     (#va : erased (lseq et lena))
     (vr : erased (lseq real lena))
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr) **
    pure (0 < SZ.v nth /\ SZ.v nth <= lena) **
    pure (SZ.fits (lena + nth)) // Almost impossible to falsify
  returns
    res : et
  ensures
    pure (res %~ seq_max (seq_map pre_map_r vr))

inline_for_extraction noextract
val reduce_max (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  : reduce_max_ty et
