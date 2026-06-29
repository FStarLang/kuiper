module Kuiper.Kernel.Reduce

#lang-pulse

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT

// Duplicate from Kuiper.Kernel.HReduce. Move to approx?
instance approx_function_can_approximate
  (dom1 dom2 cod1 cod2 : Type)
  {| can_approximate dom1 dom2, can_approximate cod1 cod2 |}
  : can_approximate (dom1 -> cod1) (dom2 -> cod2) = {
  approximates = (fun f g -> forall x y. x %~ y ==> f x %~ g y);
}

(* Simple version for a 1D array. Returns the result. *)
inline_for_extraction noextract
fn reduce1
  (#et : Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (len : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (len + nth) })
  (#l : layout1 len) {| ctlayout l |}
  (x  : array1 et l { is_global x })
  (#sx   : chest1 et len)
  (vr    : chest1 real len)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    pure (sx %~ vr)
  returns
    res : et
  ensures
    pure (res %~ chest1_rsum (chest_map pre_map_r vr))
