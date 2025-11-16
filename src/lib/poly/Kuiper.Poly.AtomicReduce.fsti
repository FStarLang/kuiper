module Kuiper.Poly.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics

module SZ = Kuiper.SizeT

inline_for_extraction noextract
type reduce_ty
  (et : Type0) {| scalar et |} {| d : has_atomic_add et |} =
  (n : szp{n < max_blocks}) ->
  (a : gpu_array et n { is_global_array a }) ->
  (#f : perm) ->
  (#v_a : erased (seq et)) ->
  stt et
  (requires
    cpu **
    pure (f == 1.0R) **
    on gpu_loc (a |-> Frac f v_a) **
    pure (SZ.v n > 0 /\ SZ.v n <= 1024))
  (ensures fun r ->
    cpu **
    on gpu_loc (a |-> Frac f v_a) **
    pure (r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a))

inline_for_extraction noextract
val reduce
  (#et : Type0) {| scalar et |} {| has_atomic_add et |}
  : reduce_ty et
