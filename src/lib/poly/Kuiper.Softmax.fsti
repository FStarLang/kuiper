module Kuiper.Softmax

#lang-pulse
open Kuiper

module Vec = Pulse.Lib.Vec

unfold
type k_pointwise_exp_ty
  (et:Type0) {| floating et |} =
  (#lena : erased nat) ->
  (a : gpu_array et lena) ->
  (etid : tid_t { gdim_x etid == reveal lena /\ bdim_x etid == 1 }) ->
  stt unit
  (requires
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid))
  (ensures fun _ ->
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid))

inline_for_extraction noextract
val k_pointwise_exp (#et : Type0) {| floating et |} : k_pointwise_exp_ty et

(* we're underspecifying.. so these coincide. *)
unfold
type k_pointwise_div_ty
  (et:Type0) {| floating et |} =
  (#lena : erased nat) ->
  (a : gpu_array et lena) ->
  (d : et) ->
  (etid : tid_t { gdim_x etid == reveal lena /\ bdim_x etid == 1 }) ->
  stt unit
  (requires
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid))
  (ensures fun _ ->
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid))

inline_for_extraction noextract
val k_pointwise_div (#et : Type0) {| floating et |} : k_pointwise_div_ty et

unfold
type softmax_gpu_ty (et : Type0) {| floating et |} =
  (#lena : szp { lena < max_threads }) ->
  (a : gpu_array et lena) ->
  (#va : erased (seq et)) ->
  stt unit
  (requires
    cpu **
    gpu_pts_to_array a va **
    pure (lena > 0 /\ lena <= max_blocks))
  (ensures fun _ ->
    cpu **
    (exists* v'. gpu_pts_to_array a v'))

inline_for_extraction noextract
val softmax_gpu (#et:Type0) {| floating et |}
  (kexp : k_pointwise_exp_ty et #_)
  (kdiv : k_pointwise_div_ty et #_)
  (kreduce : Kuiper.HReduce.k_reduce_ty et #_)
  : softmax_gpu_ty et

unfold type softmax_ty (et : Type0) {| floating et |} =
  (#lena : szp { lena < max_threads }) ->
  (a : Vec.lvec et lena) ->
  (#va : erased (seq et)) ->
  stt unit
  (requires
    cpu **
    (a |-> va) **
    pure (lena > 0 /\ lena <= max_blocks))
  (ensures fun _ ->
    cpu **
    (exists* v'. a |-> v'))

inline_for_extraction noextract
val softmax (#et : Type0) {| floating et |}
  (kexp : k_pointwise_exp_ty et #_)
  (kdiv : k_pointwise_div_ty et #_)
  (kreduce : Kuiper.HReduce.k_reduce_ty et #_)
: softmax_ty et
