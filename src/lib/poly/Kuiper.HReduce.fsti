module Kuiper.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.IsReduction


let size : sz = 1024sz

inline_for_extraction
val k_reduce_ty (et:Type0) {| scalar et |} : Type0

inline_for_extraction noextract
val d_reduce (#et:Type0) {| scalar et |} : k_reduce_ty et

(* FIXME!!!!!!! Type must unfold or we get weird extracted C.
e.g. this
  void ( *Kuiper_HReduceU64Plus2_reduce_u64(size_t lena, uint64_t *a))(size_t x0, uint64_t *x1)
instead of
  void Kuiper_HReduceU64Plus2_reduce_u64(size_t lena, uint64_t *a)
*)
unfold
type reduce_ty (et:Type0) {| scalar et |} =
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
val reduce (#et:Type0) {| scalar et |} (kk : k_reduce_ty et #_) : reduce_ty et
