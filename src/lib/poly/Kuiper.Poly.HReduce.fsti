module Kuiper.Poly.HReduce

#lang-pulse

open Kuiper

inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et |} =
  (lena : szp { lena < max_threads }) ->
  (a : gpu_array et lena) ->
  (#va : erased (seq et)) ->
  stt unit
  (requires
    cpu **
    gpu_pts_to_array a va)
  (ensures fun _ ->
    cpu **
    Kuiper.IsReduction.gpu_pts_to_slice_sum a 0 lena va)

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et |} :
  reduce_ty et
