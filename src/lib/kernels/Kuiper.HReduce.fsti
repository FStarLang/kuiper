module Kuiper.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.IsReduction

module SZ = FStar.SizeT

let size : sz = 1024sz

(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
unfold
let gpu_pts_to_slice_sum_inner
  (#et:Type0) {| simple_scalar et |}
  (#sz:nat)
  (r : gpu_array et sz)
  (i j :nat)
  (v : seq et)
  (s : seq et)
: slprop
= gpu_pts_to_slice r i j s
  ** pure (i < j /\ j <= sz /\
           len v = sz /\
           len s = j - i /\
           squash (is_reduction zero add (Seq.slice v i j) (s @! 0))) // SQUASH VERY IMPORTANT!!

(* Not easy to mark this unfold as it has a lambda (in the exists) *)
let gpu_pts_to_slice_sum
  (#et:Type0) {| simple_scalar et |}
  (#sz:nat)
  ([@@@mkey] r: gpu_array et sz)
  ([@@@mkey] i : nat)
  (j:nat)
  (v: seq et)
: slprop
= if_ (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner r i j v s)

// Barrier

val barrier_matrix
  (#et:Type0) {| simple_scalar et |}
  (nth: nat) (r : gpu_array et nth)
  (v: seq et)
  (it from to: nat)
  : slprop

unfold
let kpre
  (#et:Type0) {| simple_scalar et |}
  (nth: nat) (a : gpu_array et nth) (s : erased (seq et))
  (#_: squash (len s == nth)) (tid:nat{tid < nth})
  : slprop =
    gpu_pts_to_slice a tid (tid+1) seq![Seq.index s tid]

unfold
let kpost
  (#et:Type0) {| simple_scalar et |}
  (nth: nat) (a : gpu_array et nth) (s : erased (seq et))
  (#_: squash (len s == nth)) (tid:nat{tid < nth})
  : slprop =
    if_ (tid = 0) (gpu_pts_to_slice_sum a 0 nth s)

unfold
type k_reduce_ty (et:Type0) {| simple_scalar et |} =
  (nth : szp { nth <= 1024 }) ->
  (a : gpu_array et nth) ->
  (#s :  erased (seq et)) ->
  (#_: squash (Seq.length s == SZ.v nth)) ->
  (etid : tid_t { (gdim_x etid <: nat) == 1 /\ (bdim_x etid <: nat) == SZ.v nth }) ->
  stt unit
  (requires
    gpu **
    thread_id etid **
    mbarrier_tok nth (barrier_matrix nth a s) 0 (tidx_x etid) **
    kpre nth a s (thread_index etid))
  (ensures fun _ ->
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nth (barrier_matrix nth a s) it (tidx_x etid)) **
    kpost nth a s (thread_index etid))

inline_for_extraction noextract
val d_reduce (#et:Type0) {| simple_scalar et |} : k_reduce_ty et
inline_for_extraction noextract
val k_reduce (#et:Type0) {| simple_scalar et |} : k_reduce_ty et

(* FIXME!!!!!!! Type must unfold or we get weird extracted C.
e.g. this
  void ( *Kuiper_HReduceU64Plus2_reduce_u64(size_t lena, uint64_t *a))(size_t x0, uint64_t *x1)
instead of
  void Kuiper_HReduceU64Plus2_reduce_u64(size_t lena, uint64_t *a)
*)
unfold
type reduce_ty (et:Type0) {| simple_scalar et |} =
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
val reduce (#et:Type0) {| simple_scalar et |} (kk : k_reduce_ty et #_) : reduce_ty et
