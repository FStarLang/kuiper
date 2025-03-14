module Kuiper.HReduce

#lang-pulse

open Kuiper

inline_for_extraction noextract
type reduce_ty (et : Type0) =
  (lena : szp { lena < max_threads }) ->
  (a : gpu_array et lena) ->
  (#va : erased (seq et)) ->
  stt unit
  (requires
    cpu **
    gpu_pts_to_array a va)
  (ensures fun _ ->
    cpu **
    (exists* va'. gpu_pts_to_array a va'))

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et |} :
  reduce_ty et
