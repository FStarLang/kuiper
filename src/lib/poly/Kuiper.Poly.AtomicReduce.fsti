module Kuiper.Poly.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics


noeq
type is_ac_w (#t:Type) (f: t -> t -> t) = {
  comm : (x:t -> y:t -> Lemma (f x y == f y x));
  assoc : (x:t -> y:t -> z:t -> Lemma (f (f x y) z == f x (f y z)));
}

inline_for_extraction noextract
type reduce_ty
  (et : Type0) {| scalar et |} {| d : has_atomic_add et |} =
  fn (ac : is_ac_w d.pure_op)
     (n : szp{n < max_blocks})
     (a : gpu_array et n { is_global_array a })
     (#v_a : erased (seq et))
  preserves cpu ** on gpu_loc (a |-> v_a)
  returns   r : et
  ensures   pure (r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a)

inline_for_extraction noextract
val reduce
  (#et : Type0) {| scalar et |} {| has_atomic_add et |}
  : reduce_ty et
