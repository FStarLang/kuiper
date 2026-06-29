module Kuiper.Kernel.HReduce.Max

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Math.OnlineSoftmax { seq_max }
module SZ = Kuiper.SizeT

(* Parallel single-block max reduction.

   Same shape as Kuiper.Kernel.HReduce.reduce, but:
   - it reduces with fmax/seq_max instead of add/rsum, and
   - it requires `0 < nth /\ nth <= lena`, guaranteeing every strided bucket
     is non-empty (max has no real-number identity). *)

inline_for_extraction noextract
fn reduce_max
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : szp)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena)
  norewrite // sigh... spec in fsti is not purified
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr) **
    pure (0 < SZ.v nth /\ SZ.v nth <= lena) **
    pure (SZ.fits (lena + nth))
  returns
    res : et
  ensures
    pure (res %~ seq_max (chest1_to_seq (chest_map pre_map_r vr)))
