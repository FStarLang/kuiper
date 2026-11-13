module KPlay.Proof.Base.LocalReductionMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module Seq = FStar.Seq
module FP = Kuiper.Float32
module Approx = Kuiper.Approximates.Base
module A = KPlay.Proof.Shared.Attention
module Local = KPlay.Proof.Base.LocalArray
module Reduction = KPlay.Proof.Base.ReductionNative
module Norm = KPlay.Proof.Base.Norm
module Math = KPlay.Proof.Base.AttentionAlgebra

inline_for_extraction noextract
let partial_rep (#capacity:pos) (before:Local.snapshot capacity)
  (count:nat{count <= capacity /\ count <= 256}) (logical:A.vector 256) : prop =
  forall (i:nat{i < count}). exists (value:FP.t).
    Seq.index before i == Some value /\ Approx.v_approximates value (logical i)

inline_for_extraction noextract
let stored (#capacity:pos) (scope:prop) (before:Local.snapshot capacity)
  (count:nat{count < capacity /\ count < 256}) (logical:A.vector 256) (value:FP.t)
  : Lemma
      (requires Local.ready #capacity before count /\
        (scope ==> partial_rep #capacity before count logical /\ Approx.v_approximates value (logical count)))
      (ensures Local.ready #capacity (Seq.upd before count (Some value)) (count + 1) /\
        (scope ==> partial_rep #capacity (Seq.upd before count (Some value)) (count + 1) logical)) = ()

inline_for_extraction noextract
let reduce_approx (#capacity:pos) (scope:prop) (before:Local.snapshot capacity)
  (count:pos{count <= capacity /\ count <= 256 /\ Norm.tree_size count})
  (ready:squash (Local.ready before count)) (logical:A.vector 256) (linear:bool) (value:FP.t)
  : Lemma
      (requires (scope ==> partial_rep before count logical) /\
        value == (if linear then Reduction.linear_value (Local.prefix_values before count ready) count
          else Norm.tree count (Reduction.values (Local.prefix_values before count ready))))
      (ensures scope ==> Approx.v_approximates value (Math.partial_sum count logical)) =
  let under (_:scope) : Lemma (Approx.v_approximates value (Math.partial_sum count logical)) =
    let at_index (i:nat{i < count})
      : Lemma (Approx.v_approximates (Seq.index (Local.prefix_values before count ready) i) (logical i)) = ()
    in
    FStar.Classical.forall_intro at_index;
    if linear then Reduction.linear_approx (Local.prefix_values before count ready) logical count
    else Norm.tree_approx count (Reduction.values (Local.prefix_values before count ready)) logical
  in
  FStar.Classical.impl_intro under
