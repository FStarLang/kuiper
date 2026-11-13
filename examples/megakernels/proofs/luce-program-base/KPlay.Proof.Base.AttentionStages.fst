module KPlay.Proof.Base.AttentionStages

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module SZ = Kuiper.SizeT
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Pipeline = KPlay.Proof.Base.AttentionPipeline
module Projection = KPlay.Proof.Base.AttentionProjection
module Algebra = KPlay.Proof.Base.AttentionAlgebra
module Math = KPlay.Proof.Base.ScratchMath
module Prepare = KPlay.Proof.Base.AttentionPrepare
module PrepareMath = KPlay.Proof.Base.AttentionPrepareMath
module RMS = KPlay.Proof.Base.RMS
module RMSMath = KPlay.Proof.Base.RMSMath
module Config = KPlay.Proof.Base.RMSConfigs
module Scalar = KPlay.Proof.Base.Scalar
module Inputs = KPlay.Proof.Base.Inputs
module Native = KPlay.Proof.Base.NativeProjection
module Transfer = KPlay.Proof.Base.Transfer
module Hidden = KPlay.Proof.Base.WarpHidden
module Schedule = KPlay.Proof.Base.Schedule
module Source = KPlay.Proof.Base.SourceProjection
module Writes = KPlay.Proof.Base.MatvecWrites

type phase = p:nat{p < 23}
type point = p:nat{p <= 37}

inline_for_extraction noextract
let checkpoint (phase:nat{phase <= 23}) : point =
  if phase = 0 then 0 else if phase <= 3 then phase + 1
  else if phase < 19 then
    if phase % 2 = 1 then 5 + 3 * ((phase - 3) / 2) else 6 + 3 * ((phase - 3) / 2)
  else if phase = 19 then 29 else if phase = 20 then 31 else if phase = 21 then 33
  else if phase = 22 then 34 else 37

inline_for_extraction noextract
let matrix_input (phase:phase) : point =
  if phase = 0 then 1 else if phase < 3 then phase + 1
  else if phase < 19 then (if phase % 2 = 1 then 5 else 7) + 3 * ((phase - 3) / 2)
  else if phase = 19 then 30 else if phase = 20 then 32 else if phase = 21 then 33 else 35

inline_for_extraction noextract
let matrix_output (phase:phase) : point =
  if phase < 3 then phase + 2
  else if phase < 19 then (if phase % 2 = 1 then 6 else 8) + 3 * ((phase - 3) / 2)
  else if phase = 19 then 31 else if phase = 20 then 33 else if phase = 21 then 34 else 36

inline_for_extraction noextract
let phase_finished (phase:phase{phase < 22})
  : Lemma (matrix_output phase == checkpoint (phase + 1)) =
  if phase < 3 then () else if phase < 18 then (
    if phase % 2 = 1 then () else ()
  ) else if phase = 18 then () else if phase = 19 then () else if phase = 20 then () else assert (phase = 21)

inline_for_extraction noextract
let contexts (position:nat) (s:Pipeline.stages position) (h:Transfer.scratch) (count:nat{count <= 8}) : prop =
  forall (head:P.index 8). head < count ==>
    Math.represents #256 h (10400 + head * 256) (s.Pipeline.context head)

inline_for_extraction noextract
let live (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position)
  (h:Transfer.scratch) (point:point) : prop =
  Math.represents #1024 h 0
    (if point < 32 then c.Pipeline.input else if point < 37 then s.Pipeline.residual else s.Pipeline.final_hidden) /\
  (1 <= point /\ point <= 4 ==> Math.represents #1024 h 1024 s.Pipeline.normalized) /\
  (32 <= point /\ point <= 36 ==> Math.represents #1024 h 1024 s.Pipeline.postnorm) /\
  (2 <= point /\ point <= 4 ==> Math.represents #4096 h 2048 s.Pipeline.projected_qg) /\
  (3 <= point /\ point <= 4 ==> Math.represents #512 h 6144 s.Pipeline.projected_k) /\
  (point = 4 ==> Math.represents #512 h 6656 s.Pipeline.projected_v) /\
  (5 <= point /\ point <= 29 ==>
    Math.represents #2048 h 2048 (Algebra.flat_heads 8 s.Pipeline.queries) /\
    Math.represents #2048 h 8192 s.Pipeline.gates /\
    Math.represents #512 h 6144 (Algebra.flat_heads 2 s.Pipeline.keys) /\
    Math.represents #512 h 6656 (Algebra.flat_heads 2 s.Pipeline.values) /\
    contexts c.Pipeline.call.A.position s h ((point - 5) / 3)) /\
  (6 <= point /\ point <= 28 /\ point % 3 = 0 ==>
    Math.represents #(c.Pipeline.call.A.position + 1) h 13500
      (Algebra.score_dots c.Pipeline.call.A.position s.Pipeline.queries s.Pipeline.history ((point - 6) / 3))) /\
  (6 <= point /\ point <= 28 /\ point % 3 = 1 ==>
    Math.represents #(c.Pipeline.call.A.position + 1) h 13500
      (Algebra.attention_probabilities c.Pipeline.call.A.position s.Pipeline.queries s.Pipeline.history ((point - 6) / 3))) /\
  (30 <= point /\ point <= 31 ==> Math.represents #2048 h 10400 s.Pipeline.gated) /\
  (point = 31 ==> Math.represents #1024 h 12500 s.Pipeline.mixed) /\
  (33 <= point /\ point <= 34 ==> Math.represents #3584 h 2048 s.Pipeline.mlp_gate) /\
  (point = 34 ==> Math.represents #3584 h 5632 s.Pipeline.mlp_up) /\
  (point = 35 ==> Math.represents #3584 h 2048 s.Pipeline.product) /\
  (point = 36 ==> Math.represents #1024 h 12500 s.Pipeline.down)

inline_for_extraction noextract
let source_input (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:Projection.weight_phase) (c:Pipeline.logical_call)
  (s:Pipeline.stages c.Pipeline.call.A.position) (before:Transfer.scratch) (scope:prop)
  : Lemma
      (requires scope ==> live c s before (matrix_input phase))
      (ensures scope ==> Math.represents #(Projection.columns phase) before
        (Schedule.projection_at call position layer phase).Schedule.input_offset (Projection.input s phase)) =
  Projection.source_dimensions call position layer phase;
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then ()
  else if phase = 19 then () else if phase = 20 then () else if phase = 21 then () else assert (phase = 22)

inline_for_extraction noextract
let projection_live (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:Projection.weight_phase) (c:Pipeline.logical_call)
  (s:Pipeline.stages c.Pipeline.call.A.position) (before after:Transfer.scratch) (scope:prop)
  : Lemma
      (requires Math.outside before after (Schedule.projection_at call position layer phase).Schedule.output_offset
          (Schedule.projection_at call position layer phase).Schedule.rows /\
        (scope ==> live c s before (matrix_input phase) /\
          Math.represents #(Projection.rows phase) after
            (Schedule.projection_at call position layer phase).Schedule.output_offset (Projection.output s phase)))
      (ensures scope ==> live c s after (matrix_output phase)) =
  Projection.source_dimensions call position layer phase;
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then ()
  else if phase = 19 then () else if phase = 20 then () else if phase = 21 then () else assert (phase = 22)

inline_for_extraction noextract
let weight_projection (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:Projection.weight_phase) (weights:C.chest2 BF.t 752393024 1)
  (before:Hidden.snapshot) (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position) (scope:prop)
  : Lemma
      (requires s == Pipeline.build c /\
        c.Pipeline.weight == O.model_full ((Inputs.parameter_values (Native.packed_values weights)).O.full layer) /\
        (scope ==> live c s (Native.hidden_values before) (matrix_input phase)))
      (ensures scope ==> live c s
        (Native.hidden_values (Writes.after (Source.weight_geometry call position layer phase) weights before
          (Schedule.projection_at call position layer phase).Schedule.rows)) (matrix_output phase)) =
  Projection.source_dimensions call position layer phase;
  source_input call position layer phase c s (Native.hidden_values before) scope;
  let g = Source.weight_geometry call position layer phase in
  let result = Native.hidden_values (Writes.after g weights before (Schedule.projection_at call position layer phase).Schedule.rows) in
  Projection.weights_approx call position layer phase weights;
  Math.mma_approx scope g weights before (Projection.weight c.Pipeline.weight phase) (Projection.input s phase);
  Math.mma_frame g weights before;
  let under (_:scope)
    : Lemma (Math.represents #(Projection.rows phase) result
        (Schedule.projection_at call position layer phase).Schedule.output_offset (Projection.output s phase)) =
    let at_index (i:P.index (Projection.rows phase))
      : Lemma (Approx.v_approximates
          (Seq.index result ((Schedule.projection_at call position layer phase).Schedule.output_offset + i))
          (Projection.output s phase i)) =
      Projection.stage_equation c s phase i;
      Projection.linear_equation (Projection.rows phase) (Projection.columns phase)
        (Projection.weight c.Pipeline.weight phase) (Projection.input s phase) i
    in
    FStar.Classical.forall_intro at_index
  in
  FStar.Classical.impl_intro under;
  projection_live call position layer phase c s (Native.hidden_values before) result scope

inline_for_extraction noextract
let rope (position:nat) (before:Transfer.scratch) : prop =
  forall (i:P.index 32).
    Approx.v_approximates (Seq.index before (15600 + i)) (A.rope_cos position i) /\
    Approx.v_approximates (Seq.index before (15632 + i)) (A.rope_sin position i)

inline_for_extraction noextract
let prepared_inputs (weights:Transfer.packed) (before:Transfer.scratch)
  (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position) (scope:prop)
  : Lemma
      (requires c.Pipeline.weight == O.model_full ((Inputs.parameter_values weights).O.full c.Pipeline.call.A.layer) /\
        (scope ==> live c s before 4 /\ rope c.Pipeline.call.A.position before))
      (ensures scope ==> PrepareMath.represented weights before c.Pipeline.call.A.layer c.Pipeline.call.A.position
        c.Pipeline.weight s.Pipeline.projected_qg s.Pipeline.projected_k s.Pipeline.projected_v) =
  Inputs.layer_extent c.Pipeline.call.A.layer;
  let weight_cell (i:P.index 752393024)
    : Lemma (Approx.v_approximates (Seq.index weights i) (Approx.to_real (Seq.index weights i))) =
    Approx.to_real_ok (Seq.index weights i)
  in
  FStar.Classical.forall_intro weight_cell

inline_for_extraction noextract
let prepared (weights:Transfer.packed) (before:Transfer.scratch)
  (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position) (scope:prop)
  : Lemma
      (requires s == Pipeline.build c /\
        c.Pipeline.weight == O.model_full ((Inputs.parameter_values weights).O.full c.Pipeline.call.A.layer) /\
        (scope ==> live c s before 4 /\ rope c.Pipeline.call.A.position before))
      (ensures scope ==> live c s (Prepare.prepared weights before c.Pipeline.call.A.layer) 5) =
  let canonical = Pipeline.build c in
  prepared_inputs weights before c s scope;
  let under (_:scope)
    : Lemma (live c s (Prepare.prepared weights before c.Pipeline.call.A.layer) 5) =
    let queries (i:P.index 2048)
      : Lemma (Approx.v_approximates (Seq.index (Prepare.prepared weights before c.Pipeline.call.A.layer) (2048 + i))
          (Algebra.flat_heads 8 s.Pipeline.queries i) /\
        Approx.v_approximates (Seq.index (Prepare.prepared weights before c.Pipeline.call.A.layer) (8192 + i))
          (s.Pipeline.gates i)) =
      PrepareMath.query_approx weights before c.Pipeline.call.A.layer c.Pipeline.call.A.position c.Pipeline.weight
        s.Pipeline.projected_qg s.Pipeline.projected_k s.Pipeline.projected_v (i / 256) (i % 256);
      PrepareMath.gates_approx weights before c.Pipeline.call.A.layer c.Pipeline.call.A.position c.Pipeline.weight
        s.Pipeline.projected_qg s.Pipeline.projected_k s.Pipeline.projected_v (i / 256) (i % 256)
    in
    FStar.Classical.forall_intro queries;
    let key_values (i:P.index 512)
      : Lemma (Approx.v_approximates (Seq.index (Prepare.prepared weights before c.Pipeline.call.A.layer) (6144 + i))
          (Algebra.flat_heads 2 s.Pipeline.keys i) /\
        Approx.v_approximates (Seq.index (Prepare.prepared weights before c.Pipeline.call.A.layer) (6656 + i))
          (Algebra.flat_heads 2 s.Pipeline.values i)) =
      PrepareMath.key_approx weights before c.Pipeline.call.A.layer c.Pipeline.call.A.position c.Pipeline.weight
        s.Pipeline.projected_qg s.Pipeline.projected_k s.Pipeline.projected_v (i / 256) (i % 256);
      PrepareMath.value_approx weights before c.Pipeline.call.A.layer c.Pipeline.call.A.position c.Pipeline.weight
        s.Pipeline.projected_qg s.Pipeline.projected_k s.Pipeline.projected_v (i / 256) (i % 256)
    in
    FStar.Classical.forall_intro key_values;
    let residual (i:P.index 1024)
      : Lemma (Seq.index (Prepare.prepared weights before c.Pipeline.call.A.layer) i == Seq.index before i) =
      PrepareMath.outside_frame weights before c.Pipeline.call.A.layer i
    in
    FStar.Classical.forall_intro residual
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let norm_weight (layer:A.full_attention_layer) (weights:Transfer.packed) (post:bool) : A.vector 1024 =
  let w = O.model_full ((Inputs.parameter_values weights).O.full layer) in
  if post then w.A.post_attention_norm else w.A.input_norm

inline_for_extraction noextract
let norm_weights (layer:A.full_attention_layer) (weights:Transfer.packed) (post:bool)
  : Lemma (O.vector_approximates
      (RMSMath.weight_values 1024 weights (SZ.v (Config.model_norm layer post).RMS.weight_offset))
      (norm_weight layer weights post)) =
  Inputs.layer_extent layer;
  let at_index (i:P.index 1024)
    : Lemma (Approx.v_approximates
        (Seq.index weights (Inputs.layer_base layer + (if post then 1024 else 0) + i))
        (norm_weight layer weights post i)) =
    Approx.to_real_ok (Seq.index weights (Inputs.layer_base layer + (if post then 1024 else 0) + i))
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let normalized (scope:prop) (layer:A.full_attention_layer) (weights:Transfer.packed) (before:Transfer.scratch)
  (post:bool) (input:A.vector 1024)
  : Lemma
      (requires scope ==> Math.represents #1024 before 0 input)
      (ensures scope ==> Math.represents #1024 (RMS.after (Config.model_norm layer post) before weights 1) 1024
        (A.rms_norm 1024 (norm_weight layer weights post) input)) =
  norm_weights layer weights post;
  Math.rms_approx scope (Config.model_norm layer post) weights before 0 input (norm_weight layer weights post);
  FStar.Classical.forall_intro (PrepareMath.rms_bridge 1024 (norm_weight layer weights post) input)

inline_for_extraction noextract
let first_norm (scope:prop) (weights:Transfer.packed) (before:Transfer.scratch)
  (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position)
  : Lemma
      (requires s == Pipeline.build c /\
        c.Pipeline.weight == O.model_full ((Inputs.parameter_values weights).O.full c.Pipeline.call.A.layer) /\
        (scope ==> live c s before 0))
      (ensures scope ==> live c s (RMS.after (Config.model_norm c.Pipeline.call.A.layer false) before weights 1) 1) =
  let canonical = Pipeline.build c in
  normalized scope c.Pipeline.call.A.layer weights before false c.Pipeline.input;
  Math.rms_frame (Config.model_norm c.Pipeline.call.A.layer false) weights before

inline_for_extraction noextract
let activation_equation (silu:bool) (x y:real)
  : Lemma (Scalar.logical (if silu then Scalar.Mlp else Scalar.AttentionGate) x y ==
      Algebra.activated_product silu x y) = ()

inline_for_extraction noextract
let gate (scope:prop) (before:Transfer.scratch)
  (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position)
  : Lemma
      (requires s == Pipeline.build c /\ (scope ==> live c s before 29))
      (ensures scope ==> live c s (Scalar.writes Scalar.AttentionGate before 2048) 30) =
  let canonical = Pipeline.build c in
  Math.scalar_approx scope Scalar.AttentionGate before s.Pipeline.gates (Algebra.flat_heads 8 s.Pipeline.context);
  let at_index (i:P.index 2048)
    : Lemma (Scalar.logical Scalar.AttentionGate (s.Pipeline.gates i) (Algebra.flat_heads 8 s.Pipeline.context i) ==
        s.Pipeline.gated i) =
    activation_equation false (s.Pipeline.gates i) (Algebra.flat_heads 8 s.Pipeline.context i)
  in
  FStar.Classical.forall_intro at_index;
  Math.scalar_frame Scalar.AttentionGate before

inline_for_extraction noextract
let post_attention (scope:prop) (weights:Transfer.packed) (before:Transfer.scratch)
  (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position)
  : Lemma
      (requires s == Pipeline.build c /\
        c.Pipeline.weight == O.model_full ((Inputs.parameter_values weights).O.full c.Pipeline.call.A.layer) /\
        (scope ==> live c s before 31))
      (ensures scope ==> live c s
        (RMS.after (Config.model_norm c.Pipeline.call.A.layer true) (Scalar.writes Scalar.Residual before 1024) weights 1) 32) =
  let canonical = Pipeline.build c in
  Math.scalar_approx scope Scalar.Residual before c.Pipeline.input s.Pipeline.mixed;
  let residual = Scalar.writes Scalar.Residual before 1024 in
  let result = RMS.after (Config.model_norm c.Pipeline.call.A.layer true) residual weights 1 in
  normalized scope c.Pipeline.call.A.layer weights residual true s.Pipeline.residual;
  Math.rms_frame (Config.model_norm c.Pipeline.call.A.layer true) weights residual;
  Math.frame_vector #1024 scope residual result 1024 1024 0 s.Pipeline.residual;
  assert (scope ==> Math.represents #1024 result 1024 s.Pipeline.postnorm)

inline_for_extraction noextract
let product (scope:prop) (before:Transfer.scratch)
  (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position)
  : Lemma
      (requires s == Pipeline.build c /\ (scope ==> live c s before 34))
      (ensures scope ==> live c s (Scalar.writes Scalar.Mlp before 3584) 35) =
  let canonical = Pipeline.build c in
  Math.scalar_approx scope Scalar.Mlp before s.Pipeline.mlp_gate s.Pipeline.mlp_up;
  let at_index (i:P.index 3584)
    : Lemma (Scalar.logical Scalar.Mlp (s.Pipeline.mlp_gate i) (s.Pipeline.mlp_up i) == s.Pipeline.product i) =
    activation_equation true (s.Pipeline.mlp_gate i) (s.Pipeline.mlp_up i)
  in
  FStar.Classical.forall_intro at_index;
  Math.scalar_frame Scalar.Mlp before

inline_for_extraction noextract
let final_residual (scope:prop) (before:Transfer.scratch)
  (c:Pipeline.logical_call) (s:Pipeline.stages c.Pipeline.call.A.position)
  : Lemma
      (requires s == Pipeline.build c /\ (scope ==> live c s before 36))
      (ensures scope ==> live c s (Scalar.writes Scalar.Residual before 1024) 37) =
  let canonical = Pipeline.build c in
  Math.scalar_approx scope Scalar.Residual before s.Pipeline.residual s.Pipeline.down
