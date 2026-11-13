module KPlay.Proof.Base.AttentionBook

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module Seq = FStar.Seq
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Pipeline = KPlay.Proof.Base.AttentionPipeline
module Stages = KPlay.Proof.Base.AttentionStages
module Heads = KPlay.Proof.Base.AttentionHeadStages
module Projection = KPlay.Proof.Base.AttentionProjection
module Algebra = KPlay.Proof.Base.AttentionAlgebra
module Math = KPlay.Proof.Base.ScratchMath
module Scratch = KPlay.Proof.Base.LinearBook
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Cache = KPlay.Proof.Base.CacheNative
module CacheMath = KPlay.Proof.Base.CacheMath
module CacheProjection = KPlay.Proof.Base.CacheProjection
module Prepare = KPlay.Proof.Base.AttentionPrepare
module PrepareMath = KPlay.Proof.Base.AttentionPrepareMath
module RMS = KPlay.Proof.Base.RMS
module RMSMath = KPlay.Proof.Base.RMSMath
module Config = KPlay.Proof.Base.RMSConfigs
module Scalar = KPlay.Proof.Base.Scalar
module Softmax = KPlay.Proof.Base.Softmax
module SIMT = KPlay.Proof.Base.SIMT
module SM = KPlay.Proof.Base.SIMTMath
module Source = KPlay.Proof.Base.SourceProjection
module Schedule = KPlay.Proof.Base.Schedule
module Writes = KPlay.Proof.Base.MatvecWrites
module Phase = KPlay.Proof.Base.PhaseMMA

noeq type root (extra:Type0) = {
  payload:extra;
  logical:Pipeline.logical_call;
  stages:Pipeline.stages logical.Pipeline.call.A.position;
  hidden:Transfer.scratch;
  cache:Cache.snapshot;
}

noeq type metadata (extra:Type0) = {
  root:root extra;
  prepared:Transfer.scratch;
}

inline_for_extraction noextract
let initial_numeric (#extra:Type0) (r:root extra) : prop =
  Math.represents #1024 r.hidden 0 r.logical.Pipeline.input /\
  Stages.rope r.logical.Pipeline.call.A.position r.hidden /\
  CacheMath.optional_history r.cache r.logical.Pipeline.call.A.layer
    (r.logical.Pipeline.before.A.kv r.logical.Pipeline.call.A.layer)

inline_for_extraction noextract
let rope_preserved (scope:prop) (position:nat) (before after:Transfer.scratch)
  : Lemma
      (requires Scratch.same_rope before after /\ (scope ==> Stages.rope position before))
      (ensures scope ==> Stages.rope position after) =
  let at_index (i:P.index 32)
    : Lemma (scope ==>
        Approx.v_approximates (Seq.index after (15600 + i)) (A.rope_cos position i) /\
        Approx.v_approximates (Seq.index after (15632 + i)) (A.rope_sin position i)) =
    assert (Seq.index after (15600 + i) == Seq.index before (15600 + i));
    assert (Seq.index after (15600 + (32 + i)) == Seq.index before (15600 + (32 + i)))
  in
  FStar.Classical.forall_intro at_index

[@@"opaque_to_smt"]
inline_for_extraction noextract
let valid (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (point:Stages.point) (m:metadata extra) (h:Hidden.snapshot) (cache:Cache.snapshot) : prop =
  let r = m.root in
  anchor r /\
  r.logical.Pipeline.call == Model.attention_call call position layer /\
  r.logical.Pipeline.weight == O.model_full ((Inputs.parameter_values weights).O.full layer) /\
  r.stages == Pipeline.build r.logical /\
  Cache.initialized r.cache layer position /\
  (point = 0 ==> Native.hidden_values h == r.hidden) /\
  Scratch.same_rope r.hidden (Native.hidden_values h) /\
  (if point < 5 then cache == r.cache
   else cache == Cache.appended r.cache m.prepared layer position 512 /\
     Cache.initialized cache layer (position + 1)) /\
  (scope ==> initial_numeric r /\
    Stages.live r.logical r.stages (Native.hidden_values h) point /\
    (5 <= point ==>
      Stages.live r.logical r.stages m.prepared 5 /\
      CacheMath.history_approximates cache layer r.stages.Pipeline.history))

inline_for_extraction noextract
let hold (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (point next:Stages.point{next > 0 /\ (point < 5 <==> next < 5)})
  (m:metadata extra) (before after:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor point m before cache /\
        Scratch.same_rope (Native.hidden_values before) (Native.hidden_values after) /\
        (scope ==> Stages.live m.root.logical m.root.stages (Native.hidden_values after) next))
      (ensures valid call position layer weights scope anchor next m after cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor point m before cache);
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor next m after cache)

inline_for_extraction noextract
let prepare_rope (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  : Lemma (Scratch.same_rope before (Prepare.prepared weights before layer)) =
  let at_index (i:P.index 64)
    : Lemma (Seq.index (Prepare.prepared weights before layer) (15600 + i) == Seq.index before (15600 + i)) =
    PrepareMath.outside_frame weights before layer (15600 + i)
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let with_prepared (#extra:Type0) (m:metadata extra) (prepared:Transfer.scratch) : metadata extra =
  {m with prepared}

inline_for_extraction noextract
let prepare_step (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (before:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor 4 m before cache)
      (ensures (
        let result = Prepare.prepared weights (Native.hidden_values before) layer in
        valid call position layer weights scope anchor 5
          (with_prepared m result) (Scratch.tensor result) (Cache.appended cache result layer position 512))) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor 4 m before cache);
  let c = m.root.logical in
  let s = m.root.stages in
  let result = Prepare.prepared weights (Native.hidden_values before) layer in
  let appended = Cache.appended cache result layer position 512 in
  Scratch.tensor_values result;
  rope_preserved scope c.Pipeline.call.A.position m.root.hidden (Native.hidden_values before);
  Stages.prepared weights (Native.hidden_values before) c s scope;
  Cache.append_extends cache result layer position;
  prepare_rope weights (Native.hidden_values before) layer;
  let under (_:scope)
    : Lemma (CacheMath.history_approximates appended layer s.Pipeline.history) =
    CacheMath.prepared_matrix result s.Pipeline.keys s.Pipeline.values;
    let canonical = Pipeline.build c in
    CacheMath.append cache result layer position (c.Pipeline.before.A.kv layer) s.Pipeline.keys s.Pipeline.values
  in
  FStar.Classical.impl_intro under;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor 5
      (with_prepared m result) (Scratch.tensor result) appended)

inline_for_extraction noextract
let weight_step (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Projection.weight_phase) (weights:C.chest2 BF.t 752393024 1) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (cache:Cache.snapshot) (before:Hidden.snapshot)
  : Lemma (
      valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_input phase) m before cache ==>
      valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_output phase) m
        (Writes.after (Source.weight_geometry call position layer phase) weights before
          (Schedule.projection_at call position layer phase).Schedule.rows) cache) =
  let result = Writes.after (Source.weight_geometry call position layer phase) weights before
    (Schedule.projection_at call position layer phase).Schedule.rows in
  let under (_:valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_input phase) m before cache)
    : Lemma (valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_output phase) m result cache) =
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
      (valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_input phase) m before cache);
    Stages.weight_projection call position layer phase weights before m.root.logical m.root.stages scope;
    Math.mma_frame (Source.weight_geometry call position layer phase) weights before;
    Schedule.bounds call position layer phase;
    assert (Scratch.same_rope (Native.hidden_values before) (Native.hidden_values result));
    hold call position layer (Native.packed_values weights) scope anchor
      (Stages.matrix_input phase) (Stages.matrix_output phase) m before result cache
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let weight_post (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Projection.weight_phase) (weights:C.chest2 BF.t 752393024 1) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (h:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma (
      Phase.weight_post call position layer phase weights
        (valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_input phase)) m h cache ==>
      valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_output phase) m h cache) =
  FStar.Classical.forall_intro (weight_step call position layer phase weights scope anchor m cache)

inline_for_extraction noextract
let initialized (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop) (point:Stages.point{5 <= point})
  (m:metadata extra) (h:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma (valid call position layer weights scope anchor point m h cache ==> Cache.initialized cache layer (position + 1)) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor point m h cache)

inline_for_extraction noextract
let cache_step (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Heads.phase) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (cache:Cache.snapshot) (before:Hidden.snapshot)
  (ready:squash (Cache.initialized cache layer (position + 1)))
  : Lemma (
      valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache ==>
      valid call position layer weights scope anchor (Stages.matrix_output phase) m
        (Writes.after (CacheProjection.geometry call position layer phase)
          (CacheProjection.tensor_contents cache call position layer phase ready) before
          (Schedule.projection_at call position layer phase).Schedule.rows) cache) =
  let result = Writes.after (CacheProjection.geometry call position layer phase)
    (CacheProjection.tensor_contents cache call position layer phase ready) before
    (Schedule.projection_at call position layer phase).Schedule.rows in
  let under (_:valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache)
    : Lemma (valid call position layer weights scope anchor (Stages.matrix_output phase) m result cache) =
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
      (valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache);
    Heads.mma scope call position layer phase cache ready before m.root.logical m.root.stages;
    Math.mma_frame (CacheProjection.geometry call position layer phase)
      (CacheProjection.tensor_contents cache call position layer phase ready) before;
    Schedule.bounds call position layer phase;
    assert (Scratch.same_rope (Native.hidden_values before) (Native.hidden_values result));
    hold call position layer weights scope anchor (Stages.matrix_input phase) (Stages.matrix_output phase) m before result cache
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let cache_post (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Heads.phase) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (h:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma (
      Phase.cache_post call position layer phase
        (valid call position layer weights scope anchor (Stages.matrix_input phase)) m h cache ==>
      valid call position layer weights scope anchor (Stages.matrix_output phase) m h cache) =
  let each (before:Hidden.snapshot)
    : Lemma (forall (ready:squash (Cache.initialized cache layer (position + 1))).
        valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache ==>
        valid call position layer weights scope anchor (Stages.matrix_output phase) m
          (Writes.after (CacheProjection.geometry call position layer phase)
            (CacheProjection.tensor_contents cache call position layer phase ready) before
            (Schedule.projection_at call position layer phase).Schedule.rows) cache) =
    FStar.Classical.forall_intro (cache_step call position layer phase weights scope anchor m cache before)
  in
  FStar.Classical.forall_intro each

inline_for_extraction noextract
let simt_step (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Heads.phase) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra{m.root.logical.Pipeline.call.A.position == position})
  (before:Hidden.snapshot) (cache:Cache.snapshot) (after:Transfer.scratch)
  : Lemma
      (requires valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache /\
        SIMT.frame (CacheProjection.geometry call position layer phase) (Native.hidden_values before) after
          (Schedule.projection_at call position layer phase).Schedule.rows /\
        (scope ==> SIMT.produced (CacheProjection.geometry call position layer phase) after
          (CacheMath.matrix position m.root.stages.Pipeline.history (Heads.head phase) (Heads.score phase))
          (Heads.input m.root.stages phase) (Schedule.projection_at call position layer phase).Schedule.rows))
      (ensures valid call position layer weights scope anchor (Stages.matrix_output phase) m (Scratch.tensor after) cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache);
  Scratch.tensor_values after;
  Heads.simt scope call position layer phase m.root.logical m.root.stages (Native.hidden_values before) after;
  Schedule.bounds call position layer phase;
  assert (Scratch.same_rope (Native.hidden_values before) after);
  hold call position layer weights scope anchor (Stages.matrix_input phase) (Stages.matrix_output phase) m before (Scratch.tensor after) cache

inline_for_extraction noextract
let softmax_step (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Heads.phase{phase % 2 = 0}) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra{m.root.logical.Pipeline.call.A.position == position})
  (before:Hidden.snapshot) (cache:Cache.snapshot) (after:Transfer.scratch)
  : Lemma
      (requires valid call position layer weights scope anchor (Stages.checkpoint phase) m before cache /\
        Softmax.frame (Native.hidden_values before) after 13500 (position + 1) /\
        (scope ==> Softmax.produced (position + 1) 13500 after
          (Algebra.scaled_scores (position + 1)
            (Algebra.score_dots position m.root.stages.Pipeline.queries m.root.stages.Pipeline.history (Heads.head phase)))
          (position + 1)))
      (ensures valid call position layer weights scope anchor (Stages.matrix_input phase) m (Scratch.tensor after) cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor (Stages.checkpoint phase) m before cache);
  Scratch.tensor_values after;
  Heads.softmax scope call position phase m.root.logical m.root.stages (Native.hidden_values before) after;
  assert (Scratch.same_rope (Native.hidden_values before) after);
  hold call position layer weights scope anchor (Stages.checkpoint phase) (Stages.matrix_input phase) m before (Scratch.tensor after) cache

inline_for_extraction noextract
let simt_ready (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Heads.phase) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra{m.root.logical.Pipeline.call.A.position == position})
  (before:Hidden.snapshot) (cache:Cache.snapshot)
  (ready:squash (Cache.initialized cache layer (position + 1)))
  : Lemma
      (requires valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache)
      (ensures scope ==>
        SM.matrix_rep (CacheProjection.geometry call position layer phase)
          (CacheProjection.contents cache call position layer phase ready)
          (CacheMath.matrix position m.root.stages.Pipeline.history (Heads.head phase) (Heads.score phase)) /\
        SM.input_rep (CacheProjection.geometry call position layer phase) (Native.hidden_values before)
          (Heads.input m.root.stages phase)) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor (Stages.matrix_input phase) m before cache);
  CacheMath.simt_matrix scope cache call position layer phase m.root.stages.Pipeline.history ready;
  Heads.source_input scope call position layer phase m.root.logical m.root.stages (Native.hidden_values before)

type simple_phase = p:Stages.phase{p <> 3 /\ (p < 3 \/ 19 <= p \/ p % 2 = 1)}

inline_for_extraction noextract
let prelude (layer:A.full_attention_layer) (phase:simple_phase)
  (weights:Transfer.packed) (before:Transfer.scratch) : GTot Transfer.scratch =
  if phase = 0 then RMS.after (Config.model_norm layer false) before weights 1
  else if phase = 19 then Scalar.writes Scalar.AttentionGate before 2048
  else if phase = 20 then RMS.after (Config.model_norm layer true) (Scalar.writes Scalar.Residual before 1024) weights 1
  else if phase = 22 then Scalar.writes Scalar.Mlp before 3584
  else before

inline_for_extraction noextract
let prelude_rope (layer:A.full_attention_layer) (phase:simple_phase)
  (weights:Transfer.packed) (before:Transfer.scratch)
  : Lemma (Scratch.same_rope before (prelude layer phase weights before)) =
  let at_index (i:P.index 64)
    : Lemma (Seq.index (prelude layer phase weights before) (15600 + i) == Seq.index before (15600 + i)) =
    if phase = 0 then RMSMath.outside_frame (Config.model_norm layer false) before weights 1 (15600 + i)
    else if phase = 19 then Scalar.outside_frame Scalar.AttentionGate before (15600 + i)
    else if phase = 20 then (
      Scalar.outside_frame Scalar.Residual before (15600 + i);
      RMSMath.outside_frame (Config.model_norm layer true) (Scalar.writes Scalar.Residual before 1024) weights 1 (15600 + i)
    ) else if phase = 22 then Scalar.outside_frame Scalar.Mlp before (15600 + i)
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let prelude_valid (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:simple_phase) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (before:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor (Stages.checkpoint phase) m before cache)
      (ensures valid call position layer weights scope anchor (Stages.matrix_input phase)
        m (Scratch.tensor (prelude layer phase weights (Native.hidden_values before))) cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor (Stages.checkpoint phase) m before cache);
  let h = Native.hidden_values before in
  let result = prelude layer phase weights h in
  Scratch.tensor_values result;
  if phase = 0 then Stages.first_norm scope weights h m.root.logical m.root.stages
  else if phase = 19 then Stages.gate scope h m.root.logical m.root.stages
  else if phase = 20 then Stages.post_attention scope weights h m.root.logical m.root.stages
  else if phase = 22 then Stages.product scope h m.root.logical m.root.stages;
  prelude_rope layer phase weights h;
  hold call position layer weights scope anchor (Stages.checkpoint phase) (Stages.matrix_input phase) m before (Scratch.tensor result) cache

inline_for_extraction noextract
let postlude (phase:Stages.phase) (before:Transfer.scratch) : GTot Transfer.scratch =
  if phase = 22 then Scalar.writes Scalar.Residual before 1024 else before

inline_for_extraction noextract
let postlude_valid (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:A.full_attention_layer)
  (phase:Stages.phase) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (before:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor (Stages.matrix_output phase) m before cache)
      (ensures valid call position layer weights scope anchor (Stages.checkpoint (phase + 1))
        m (Scratch.tensor (postlude phase (Native.hidden_values before))) cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (valid call position layer weights scope anchor (Stages.matrix_output phase) m before cache);
  let h = Native.hidden_values before in
  let result = postlude phase h in
  Scratch.tensor_values result;
  if phase = 22 then (
    Stages.final_residual scope h m.root.logical m.root.stages;
    Math.scalar_frame Scalar.Residual h
  ) else Stages.phase_finished phase;
  assert (Scratch.same_rope h result);
  hold call position layer weights scope anchor (Stages.matrix_output phase) (Stages.checkpoint (phase + 1)) m before (Scratch.tensor result) cache
