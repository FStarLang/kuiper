module Kuiper.Poly.Softmax

#lang-pulse
open Kuiper
open Kuiper.Approximates

module Vec = Pulse.Lib.Vec

unfold
type softmax_gpu_ty (et : Type0) {| floating et, real_like et |} =
  (#lena : szp { lena < max_threads }) ->
  (a : gpu_array et lena) ->
  (#va : erased (seq et)) ->
  stt unit
  (requires
    cpu **
    (gpu_pts_to_array a va **
     pure (lena > 0 /\ lena <= max_blocks)))
  (ensures fun _ ->
    cpu **
     (exists* v'. gpu_pts_to_array a v'))

inline_for_extraction noextract
val softmax_gpu (#et:Type0) {| floating et, real_like et |}
  : softmax_gpu_ty et

unfold
type softmax_ty (et : Type0) {| floating et, real_like et |} =
  (#lena : szp { lena < max_threads }) ->
  (a : Vec.lvec et lena) ->
  (#va : erased (seq et)) ->
  stt unit
  (requires
    cpu **
    (a |-> va **
     pure (lena > 0 /\ lena <= max_blocks)))
  (ensures fun _ ->
    cpu **
    (exists* v'. a |-> v'))

inline_for_extraction noextract
val softmax (#et : Type0) {| floating et, real_like et |}
: softmax_ty et
