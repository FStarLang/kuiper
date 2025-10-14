module Kuiper.Poly.HReduce

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Seq.Common

inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |} =
  (lena : szp { lena < max_threads }) ->
  (a : gpu_array et lena) ->
  (#va : erased (seq et)) ->
  (#vr : erased (seq real){seq_approximates va vr}) ->
  stt unit
  (requires
    cpu **
    gpu_pts_to_array a va)
  (ensures fun _ ->
    cpu **
    exists* (va' : seq et{Seq.length va' > 0}).
      gpu_pts_to_array a va' **
      pure ((va' @! 0) `approximates` seq_fold_left (+.) 0.0R vr))

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et, real_like et |} :
  reduce_ty et
