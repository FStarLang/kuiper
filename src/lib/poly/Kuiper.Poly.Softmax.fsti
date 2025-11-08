module Kuiper.Poly.Softmax

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.RealExpDiv
module KS = Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec

let sum (#et:Type0) {| scalar et |} (s:seq et) =
  KS.seq_fold_left add zero s

val sum_non_zero
    (s:seq real { forall (i:natlt (Seq.length s)). Seq.index s i >. 0.0R })
    (acc:real)
: Lemma
  (requires Seq.length s > 0)
  (ensures KS.seq_fold_left add acc s >. acc)

let softmax_real (s:Seq.seq real { Seq.length s > 0 }) =
  let open KS in
  let exps = seq_map rexp s in
  let avg : real = sum exps in
  sum_non_zero exps zero;
  seq_map FStar.Real.(fun x -> x /. avg) exps

unfold
type softmax_gpu_ty (et : Type0) {| floating et, real_like et |} =
  (#lena : szp { lena < max_threads }) ->
  (a : gpu_global_array et lena) ->
  (#va : erased (seq et)) ->
  (#ra : erased (seq real) { Seq.length ra == SizeT.v lena /\ va %~ ra /\ lena > 0 }) ->
  stt unit
  (requires
    cpu **
    (gpu_pts_to_array a va **
     pure (lena <= max_blocks)))
  (ensures fun _ ->
    cpu **
     (exists* (v':seq et). gpu_pts_to_array a v' **
        pure (v' %~ softmax_real ra)))

inline_for_extraction noextract
val softmax_gpu (#et:Type0) {| floating et, real_like et |}
  : softmax_gpu_ty et

unfold
type softmax_ty (et : Type0) {| floating et, real_like et |} =
  (#lena : szp { lena < max_threads }) ->
  (a : Vec.lvec et lena) ->
  (#va : erased (seq et)) ->
  (#ra : erased (seq real) { Seq.length ra == SizeT.v lena /\ va %~ ra /\ lena > 0 }) ->
  stt unit
  (requires
    cpu **
    (a |-> va **
     pure (lena <= max_blocks)))
  (ensures fun _ ->
    cpu **
    (exists* (v':seq et). a |-> v' **
        pure (v' %~ softmax_real ra)))

inline_for_extraction noextract
val softmax (#et : Type0) {| floating et, real_like et |}
: softmax_ty et
