module KPlay.Proof.Base.AttentionHeadStages

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module Model = KPlay.Proof.Shared.Model
module Pipeline = KPlay.Proof.Base.AttentionPipeline
module Stages = KPlay.Proof.Base.AttentionStages
module Algebra = KPlay.Proof.Base.AttentionAlgebra
module Projection = KPlay.Proof.Base.AttentionProjection
module Cache = KPlay.Proof.Base.CacheNative
module CacheMath = KPlay.Proof.Base.CacheMath
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Transfer = KPlay.Proof.Base.Transfer
module CacheProjection = KPlay.Proof.Base.CacheProjection
module Schedule = KPlay.Proof.Base.Schedule
module Layout = KPlay.Proof.Base.MatvecLayout
module Math = KPlay.Proof.Base.ScratchMath
module Writes = KPlay.Proof.Base.MatvecWrites
module SIMT = KPlay.Proof.Base.SIMT
module SM = KPlay.Proof.Base.SIMTMath
module Softmax = KPlay.Proof.Base.Softmax

type phase = p:nat{3 <= p /\ p < 19}

inline_for_extraction noextract
let head (phase:phase) : P.index 8 = (phase - 3) / 2

inline_for_extraction noextract
let score (phase:phase) : bool = phase % 2 = 1

inline_for_extraction noextract
let input (#position:nat) (s:Pipeline.stages position) (phase:phase)
  : A.vector (CacheMath.columns position (score phase)) =
  if score phase then s.Pipeline.queries (head phase)
  else Algebra.attention_probabilities position s.Pipeline.queries s.Pipeline.history (head phase)

inline_for_extraction noextract
let output (#position:nat) (s:Pipeline.stages position) (phase:phase)
  : A.vector (CacheMath.rows position (score phase)) =
  if score phase then Algebra.score_dots position s.Pipeline.queries s.Pipeline.history (head phase)
  else s.Pipeline.context (head phase)

inline_for_extraction noextract
let dimensions (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:phase)
  : Lemma (
      (Schedule.projection_at call position layer phase).Schedule.rows == CacheMath.rows position (score phase) /\
      (Schedule.projection_at call position layer phase).Schedule.columns == CacheMath.columns position (score phase) /\
      (Schedule.projection_at call position layer phase).Schedule.input_offset + CacheMath.columns position (score phase) <= 16000 /\
      (Schedule.projection_at call position layer phase).Schedule.output_offset + CacheMath.rows position (score phase) <= 16000) =
  Schedule.bounds call position layer phase

inline_for_extraction noextract
let source_input (scope:prop) (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:phase)
  (c:Pipeline.logical_call{c.Pipeline.call.A.position == position})
  (s:Pipeline.stages position) (before:Transfer.scratch)
  : Lemma
      (requires scope ==> Stages.live c s before (Stages.matrix_input phase))
      (ensures scope ==> Math.represents #(CacheMath.columns position (score phase)) before
        (Schedule.projection_at call position layer phase).Schedule.input_offset (input s phase)) =
  dimensions call position layer phase;
  if score phase then (
    let at_index (i:P.index 256)
      : Lemma (Algebra.flat_heads 8 s.Pipeline.queries (head phase * 256 + i) == s.Pipeline.queries (head phase) i) = ()
    in
    FStar.Classical.forall_intro at_index
  ) else (
    assert (Stages.matrix_input phase == 7 + 3 * head phase);
    assert (7 <= Stages.matrix_input phase /\ Stages.matrix_input phase <= 28);
    assert (Stages.matrix_input phase % 3 == 1 /\ (Stages.matrix_input phase - 6) / 3 == head phase);
    assert (scope ==> Math.represents #(position + 1) before 13500
      (Algebra.attention_probabilities position s.Pipeline.queries s.Pipeline.history (head phase)))
  )

inline_for_extraction noextract
let output_linear (position:nat) (phase:phase)
  (c:Pipeline.logical_call{c.Pipeline.call.A.position == position})
  (s:Pipeline.stages position) (row:P.index (CacheMath.rows position (score phase)))
  : Lemma
      (requires s == Pipeline.build c)
      (ensures A.linear (CacheMath.rows position (score phase)) (CacheMath.columns position (score phase))
        (CacheMath.matrix position s.Pipeline.history (head phase) (score phase)) (input s phase) row ==
        output s phase row) =
  let canonical = Pipeline.build c in
  if score phase then Algebra.score_linear position s.Pipeline.queries s.Pipeline.history (head phase) row
  else Algebra.context_linear position s.Pipeline.queries s.Pipeline.history (head phase) row

inline_for_extraction noextract
let projection_live (scope:prop) (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:phase)
  (c:Pipeline.logical_call{c.Pipeline.call.A.position == position})
  (s:Pipeline.stages position) (before after:Transfer.scratch)
  : Lemma
      (requires Math.outside before after
          (Schedule.projection_at call position layer phase).Schedule.output_offset
          (Schedule.projection_at call position layer phase).Schedule.rows /\
        (scope ==> Stages.live c s before (Stages.matrix_input phase) /\
          Math.represents #(CacheMath.rows position (score phase)) after
            (Schedule.projection_at call position layer phase).Schedule.output_offset (output s phase)))
      (ensures scope ==> Stages.live c s after (Stages.matrix_output phase)) =
  dimensions call position layer phase;
  let d = Schedule.projection_at call position layer phase in
  Math.frame_vector #1024 scope before after d.Schedule.output_offset d.Schedule.rows 0 c.Pipeline.input;
  Math.frame_vector #2048 scope before after d.Schedule.output_offset d.Schedule.rows 2048
    (Algebra.flat_heads 8 s.Pipeline.queries);
  Math.frame_vector #2048 scope before after d.Schedule.output_offset d.Schedule.rows 8192 s.Pipeline.gates;
  Math.frame_vector #512 scope before after d.Schedule.output_offset d.Schedule.rows 6144
    (Algebra.flat_heads 2 s.Pipeline.keys);
  Math.frame_vector #512 scope before after d.Schedule.output_offset d.Schedule.rows 6656
    (Algebra.flat_heads 2 s.Pipeline.values);
  if score phase then ()
  else (
    assert (Stages.matrix_output phase == 8 + 3 * head phase);
    assert (Stages.matrix_output phase % 3 == 2 /\ (Stages.matrix_output phase - 5) / 3 == head phase + 1);
    let at_head (h:P.index 8)
      : Lemma (scope ==> h < head phase + 1 ==>
          Math.represents #256 after (10400 + h * 256) (s.Pipeline.context h)) =
      if h < head phase then
        Math.frame_vector #256 scope before after (10400 + head phase * 256) 256 (10400 + h * 256)
          (s.Pipeline.context h)
      else ()
    in
    FStar.Classical.forall_intro at_head
  )

inline_for_extraction noextract
let mma (scope:prop) (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:phase) (cache:Cache.snapshot)
  (ready:squash (Cache.initialized cache layer (position + 1))) (before:Hidden.snapshot)
  (c:Pipeline.logical_call{c.Pipeline.call.A.position == position})
  (s:Pipeline.stages position)
  : Lemma
      (requires s == Pipeline.build c /\
        (scope ==> CacheMath.history_approximates cache layer s.Pipeline.history /\
          Stages.live c s (Native.hidden_values before) (Stages.matrix_input phase)))
      (ensures scope ==> Stages.live c s
        (Native.hidden_values (Writes.after (CacheProjection.geometry call position layer phase)
          (CacheProjection.tensor_contents cache call position layer phase ready) before
          (Schedule.projection_at call position layer phase).Schedule.rows))
        (Stages.matrix_output phase)) =
  dimensions call position layer phase;
  let g = CacheProjection.geometry call position layer phase in
  let values = CacheProjection.tensor_contents cache call position layer phase ready in
  let result = Native.hidden_values (Writes.after g values before (Schedule.projection_at call position layer phase).Schedule.rows) in
  source_input scope call position layer phase c s (Native.hidden_values before);
  CacheMath.matrix_approx scope cache call position layer phase s.Pipeline.history ready;
  Math.mma_approx scope g values before (CacheMath.matrix position s.Pipeline.history (head phase) (score phase)) (input s phase);
  Math.mma_frame g values before;
  let under (_:scope)
    : Lemma (Math.represents #(CacheMath.rows position (score phase)) result
        (Schedule.projection_at call position layer phase).Schedule.output_offset (output s phase)) =
    let at_row (row:P.index (CacheMath.rows position (score phase)))
      : Lemma (Approx.v_approximates
          (Seq.index result ((Schedule.projection_at call position layer phase).Schedule.output_offset + row))
          (output s phase row)) =
      Projection.linear_equation (CacheMath.rows position (score phase)) (CacheMath.columns position (score phase))
        (CacheMath.matrix position s.Pipeline.history (head phase) (score phase)) (input s phase) row;
      output_linear position phase c s row
    in
    FStar.Classical.forall_intro at_row
  in
  FStar.Classical.impl_intro under;
  projection_live scope call position layer phase c s (Native.hidden_values before) result

inline_for_extraction noextract
let simt (scope:prop) (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:phase)
  (c:Pipeline.logical_call{c.Pipeline.call.A.position == position})
  (s:Pipeline.stages position) (before after:Transfer.scratch)
  : Lemma
      (requires s == Pipeline.build c /\
        SIMT.frame (CacheProjection.geometry call position layer phase) before after
          (Schedule.projection_at call position layer phase).Schedule.rows /\
        (scope ==> Stages.live c s before (Stages.matrix_input phase) /\
          SIMT.produced (CacheProjection.geometry call position layer phase) after
            (CacheMath.matrix position s.Pipeline.history (head phase) (score phase)) (input s phase)
            (Schedule.projection_at call position layer phase).Schedule.rows))
      (ensures scope ==> Stages.live c s after (Stages.matrix_output phase)) =
  dimensions call position layer phase;
  FStar.Classical.forall_intro (output_linear position phase c s);
  projection_live scope call position layer phase c s before after

inline_for_extraction noextract
let softmax_equation (position:nat) (s:Pipeline.stages position) (head:P.index 8) (i:P.index (position + 1))
  : Lemma (A.softmax (position + 1)
      (Algebra.scaled_scores (position + 1) (Algebra.score_dots position s.Pipeline.queries s.Pipeline.history head)) i ==
      Algebra.attention_probabilities position s.Pipeline.queries s.Pipeline.history head i) =
  Algebra.softmax_congr (position + 1)
    (Algebra.scaled_scores (position + 1) (Algebra.score_dots position s.Pipeline.queries s.Pipeline.history head))
    (Algebra.attention_scores position s.Pipeline.queries s.Pipeline.history head) i

inline_for_extraction noextract
let softmax_input (scope:prop) (position:nat) (phase:phase{phase % 2 = 0})
  (c:Pipeline.logical_call{c.Pipeline.call.A.position == position}) (s:Pipeline.stages position)
  (before:Transfer.scratch)
  : Lemma
      (requires scope ==> Stages.live c s before (Stages.checkpoint phase))
      (ensures scope ==> Math.represents #(position + 1) before 13500
        (Algebra.score_dots position s.Pipeline.queries s.Pipeline.history (head phase))) =
  assert (Stages.checkpoint phase == 6 + 3 * head phase);
  assert (Stages.checkpoint phase % 3 == 0 /\ (Stages.checkpoint phase - 6) / 3 == head phase)

inline_for_extraction noextract
let softmax (scope:prop) (call:Model.invocation) (position:nat{position < Model.calls call})
  (phase:phase{phase % 2 = 0})
  (c:Pipeline.logical_call{c.Pipeline.call.A.position == position})
  (s:Pipeline.stages position) (before after:Transfer.scratch)
  : Lemma
      (requires Softmax.frame before after 13500 (position + 1) /\
        (scope ==> Stages.live c s before (Stages.checkpoint phase) /\
          Softmax.produced (position + 1) 13500 after
            (Algebra.scaled_scores (position + 1)
              (Algebra.score_dots position s.Pipeline.queries s.Pipeline.history (head phase))) (position + 1)))
      (ensures scope ==> Stages.live c s after (Stages.matrix_input phase)) =
  FStar.Classical.forall_intro (softmax_equation position s (head phase))
