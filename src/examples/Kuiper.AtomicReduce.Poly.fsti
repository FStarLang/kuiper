module Kuiper.AtomicReduce.Poly

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics
open Kuiper.AtomicReduce.Poly.Kernel

module SZ = FStar.SizeT

unfold
type reduce_ty
  (et : Type0) {| scalar et |} {| d : has_atomic_add et |} =
  (n : sz) ->
  (a : gpu_array et n) ->
  (#f : perm) ->
  (#v_a : erased (seq et)) ->
  stt et
  (requires
    cpu **
    pure (f == 1.0R) **
    gpu_pts_to_array a #f v_a **
    pure (SZ.v n > 0 /\ SZ.v n <= 1024))
  (ensures fun r ->
    cpu **
    gpu_pts_to_array a #f v_a **
    pure (r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a))

inline_for_extraction noextract
val reduce
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (k : kernel_ty et #_ #_)
  : reduce_ty et #_ #_
