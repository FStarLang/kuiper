module KPlay.Proof.Base.LinearStages

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module SZ = Kuiper.SizeT
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Prior = KPlay.Proof.Base.RecurrentPipeline
module LP = KPlay.Proof.Base.LinearProjection
module Source = KPlay.Proof.Base.SourceProjection
module Schedule = KPlay.Proof.Base.Schedule
module Layout = KPlay.Proof.Base.MatvecLayout
module Writes = KPlay.Proof.Base.MatvecWrites
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Math = KPlay.Proof.Base.ScratchMath
module RMS = KPlay.Proof.Base.RMS
module RMSMath = KPlay.Proof.Base.RMSMath
module Config = KPlay.Proof.Base.RMSConfigs
module Scalar = KPlay.Proof.Base.Scalar
module Mixer = KPlay.Proof.Base.RecurrentMixer
module Heads = KPlay.Proof.Base.RecurrentHeads

type phase = n:nat{n < 8}
type point = n:nat{n <= 13}

inline_for_extraction noextract
let checkpoint (phase:nat{phase <= 8}) : point =
  if phase = 0 then 0 else if phase <= 4 then phase + 1
  else if phase = 5 then 7 else if phase = 6 then 9 else if phase = 7 then 10 else 13

inline_for_extraction noextract
let matrix_input (phase:phase) : point =
  if phase = 0 then 1 else if phase < 4 then phase + 1
  else if phase = 4 then 6 else if phase = 5 then 8 else if phase = 6 then 9 else 11

inline_for_extraction noextract
let matrix_output (phase:phase) : point =
  if phase < 4 then phase + 2 else if phase = 4 then 7
  else if phase = 5 then 9 else if phase = 6 then 10 else 12

inline_for_extraction noextract
let live (call:Prior.logical_call) (stages:Prior.decoder_stages) (hidden:Transfer.scratch) (point:point) : prop =
  Math.represents #1024 hidden 0
    (if point < 8 then call.Prior.input else if point < 13 then stages.Prior.residual else stages.Prior.final_hidden) /\
  (1 <= point /\ point <= 7 ==> Math.represents #1024 hidden 1024 stages.Prior.normalized) /\
  (8 <= point /\ point <= 12 ==> Math.represents #1024 hidden 1024 stages.Prior.postnorm) /\
  (2 <= point /\ point <= 5 ==> Math.represents #6144 hidden 2048 stages.Prior.raw) /\
  (3 <= point /\ point <= 5 ==> Math.represents #2048 hidden 8192 stages.Prior.projected_z) /\
  (4 <= point /\ point <= 5 ==> Math.represents #16 hidden 10240 stages.Prior.projected_b) /\
  (point = 5 ==> Math.represents #16 hidden 10256 stages.Prior.projected_a) /\
  (6 <= point /\ point <= 7 ==> Math.represents #2048 hidden 10400 stages.Prior.gated_values) /\
  (point = 7 ==> Math.represents #1024 hidden 12500 stages.Prior.mixed) /\
  (9 <= point /\ point <= 10 ==> Math.represents #3584 hidden 2048 stages.Prior.mlp_gate) /\
  (point = 10 ==> Math.represents #3584 hidden 5632 stages.Prior.mlp_up) /\
  (point = 11 ==> Math.represents #3584 hidden 2048 stages.Prior.product) /\
  (point = 12 ==> Math.represents #1024 hidden 12500 stages.Prior.down)

inline_for_extraction noextract
let source_input (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.linear_layer) (phase:phase) (logical:Prior.logical_call) (stages:Prior.decoder_stages)
  (before:Transfer.scratch) (scope:prop)
  : Lemma
      (requires scope ==> live logical stages before (matrix_input phase))
      (ensures scope ==> Math.represents #(LP.columns phase) before
        (Schedule.projection_at call position layer phase).Schedule.input_offset (LP.input stages phase)) =
  LP.source_dimensions call position layer phase;
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then () else if phase = 3 then ()
  else if phase = 4 then () else if phase = 5 then () else if phase = 6 then () else assert (phase = 7)

inline_for_extraction noextract
let projection_live (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.linear_layer) (phase:phase) (logical:Prior.logical_call) (stages:Prior.decoder_stages)
  (before after:Transfer.scratch) (scope:prop)
  : Lemma
      (requires
        Math.outside before after (Schedule.projection_at call position layer phase).Schedule.output_offset
          (Schedule.projection_at call position layer phase).Schedule.rows /\
        (scope ==> live logical stages before (matrix_input phase) /\
          Math.represents #(LP.rows phase) after (Schedule.projection_at call position layer phase).Schedule.output_offset
            (LP.output stages phase)))
      (ensures scope ==> live logical stages after (matrix_output phase)) =
  LP.source_dimensions call position layer phase;
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then () else if phase = 3 then ()
  else if phase = 4 then () else if phase = 5 then () else if phase = 6 then () else assert (phase = 7)

inline_for_extraction noextract
let projection (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.linear_layer) (phase:phase) (weights:C.chest2 BF.t 752393024 1) (before:Hidden.snapshot)
  (logical:Prior.logical_call) (stages:Prior.decoder_stages) (scope:prop)
  : Lemma
      (requires stages == Prior.build_stages logical /\
        logical.Prior.weight == O.model_linear ((Inputs.parameter_values (Native.packed_values weights)).O.linear layer) /\
        (scope ==> live logical stages (Native.hidden_values before) (matrix_input phase)))
      (ensures scope ==> live logical stages
        (Native.hidden_values (Writes.after (Source.weight_geometry call position layer phase) weights before
          (Schedule.projection_at call position layer phase).Schedule.rows))
        (matrix_output phase)) =
  LP.source_dimensions call position layer phase;
  source_input call position layer phase logical stages (Native.hidden_values before) scope;
  let g = Source.weight_geometry call position layer phase in
  let result = Native.hidden_values (Writes.after g weights before (Schedule.projection_at call position layer phase).Schedule.rows) in
  LP.weights_approx call position layer phase weights;
  Math.mma_approx scope g weights before (LP.weight logical.Prior.weight phase) (LP.input stages phase);
  Math.mma_frame g weights before;
  let under (_:scope)
    : Lemma (Math.represents #(LP.rows phase) result (Schedule.projection_at call position layer phase).Schedule.output_offset
        (LP.output stages phase)) =
    let at_index (i:P.index (LP.rows phase))
      : Lemma (Approx.v_approximates
          (Seq.index result ((Schedule.projection_at call position layer phase).Schedule.output_offset + i))
          (LP.output stages phase i)) =
      LP.stage_equation logical stages phase i;
      LP.project_equation phase logical.Prior.weight (LP.input stages phase) i
    in
    FStar.Classical.forall_intro at_index
  in
  FStar.Classical.impl_intro under;
  projection_live call position layer phase logical stages (Native.hidden_values before) result scope

inline_for_extraction noextract
let rms_equation (weights input:R.vector 1024) (i:P.index 1024)
  : Lemma (P.normalized_for_logits P.reference_epsilon input weights i ==
      R.centered_rms_norm weights input i) =
  assert (P.normalized_for_logits P.reference_epsilon input weights i ==
      R.centered_rms_norm weights input i) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Proof.Shared.Projection.normalized_for_logits"; "KPlay.Proof.Shared.Projection.rms_argument";
      "KPlay.Proof.Shared.Projection.real_sum"; "KPlay.Proof.Shared.Projection.reference_epsilon";
      "KPlay.Proof.Shared.Recurrent.centered_rms_norm"; "KPlay.Proof.Shared.Recurrent.rms_scale";
      "KPlay.Proof.Shared.Recurrent.sum_squares"; "KPlay.Proof.Shared.Recurrent.epsilon"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let norm_weight (layer:Model.linear_layer) (weights:Transfer.packed) (post:bool) : GTot (R.vector 1024) =
  let w = O.model_linear ((Inputs.parameter_values weights).O.linear layer) in
  if post then w.R.post_attention_norm else w.R.input_norm

inline_for_extraction noextract
let norm_weights (layer:Model.linear_layer) (weights:Transfer.packed) (post:bool)
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
let normalized (scope:prop) (layer:Model.linear_layer) (weights:Transfer.packed) (before:Transfer.scratch)
  (post:bool) (input:R.vector 1024)
  : Lemma
      (requires scope ==> Math.represents #1024 before 0 input)
      (ensures scope ==> Math.represents #1024 (RMS.after (Config.model_norm layer post) before weights 1) 1024
        (R.centered_rms_norm (norm_weight layer weights post) input)) =
  norm_weights layer weights post;
  Math.rms_approx scope (Config.model_norm layer post) weights before 0 input (norm_weight layer weights post);
  FStar.Classical.forall_intro (rms_equation (norm_weight layer weights post) input)

inline_for_extraction noextract
let first_norm (scope:prop) (layer:Model.linear_layer) (weights:Transfer.packed) (before:Transfer.scratch)
  (logical:Prior.logical_call) (stages:Prior.decoder_stages)
  : Lemma
      (requires stages == Prior.build_stages logical /\
        logical.Prior.weight == O.model_linear ((Inputs.parameter_values weights).O.linear layer) /\
        (scope ==> live logical stages before 0))
      (ensures scope ==> live logical stages (RMS.after (Config.model_norm layer false) before weights 1) 1) =
  let canonical = Prior.build_stages logical in
  normalized scope layer weights before false logical.Prior.input;
  Math.rms_frame (Config.model_norm layer false) weights before

inline_for_extraction noextract
let post_attention (scope:prop) (layer:Model.linear_layer) (weights:Transfer.packed) (before:Transfer.scratch)
  (logical:Prior.logical_call) (stages:Prior.decoder_stages)
  : Lemma
      (requires stages == Prior.build_stages logical /\
        logical.Prior.weight == O.model_linear ((Inputs.parameter_values weights).O.linear layer) /\
        (scope ==> live logical stages before 7))
      (ensures scope ==> live logical stages
        (RMS.after (Config.model_norm layer true) (Scalar.writes Scalar.Residual before 1024) weights 1) 8) =
  let canonical = Prior.build_stages logical in
  Math.scalar_approx scope Scalar.Residual before logical.Prior.input stages.Prior.mixed;
  normalized scope layer weights (Scalar.writes Scalar.Residual before 1024) true stages.Prior.residual;
  let residual = Scalar.writes Scalar.Residual before 1024 in
  let result = RMS.after (Config.model_norm layer true) residual weights 1 in
  Math.rms_frame (Config.model_norm layer true) weights residual;
  Math.frame_vector #1024 scope residual result 1024 1024 0 stages.Prior.residual;
  assert (scope ==> Math.represents #1024 result 0 stages.Prior.residual);
  assert (scope ==> Math.represents #1024 result 1024 stages.Prior.postnorm)

inline_for_extraction noextract
let product (scope:prop) (before:Transfer.scratch) (logical:Prior.logical_call) (stages:Prior.decoder_stages)
  : Lemma
      (requires stages == Prior.build_stages logical /\ (scope ==> live logical stages before 10))
      (ensures scope ==> live logical stages (Scalar.writes Scalar.Mlp before 3584) 11) =
  let canonical = Prior.build_stages logical in
  Math.scalar_approx scope Scalar.Mlp before stages.Prior.mlp_gate stages.Prior.mlp_up;
  Math.scalar_frame Scalar.Mlp before

inline_for_extraction noextract
let final_residual (scope:prop) (before:Transfer.scratch) (logical:Prior.logical_call) (stages:Prior.decoder_stages)
  : Lemma
      (requires stages == Prior.build_stages logical /\ (scope ==> live logical stages before 12))
      (ensures scope ==> live logical stages (Scalar.writes Scalar.Residual before 1024) 13) =
  let canonical = Prior.build_stages logical in
  Math.scalar_approx scope Scalar.Residual before stages.Prior.residual stages.Prior.down

inline_for_extraction noextract
let mixer (scope:prop) (before after:Transfer.scratch) (state:FStar.Seq.lseq FP.t 4718592)
  (slot:nat{slot < 18}) (logical:Prior.logical_call) (stages:Prior.decoder_stages)
  : Lemma
      (requires stages == Prior.build_stages logical /\ Mixer.outside before after /\
        (scope ==> live logical stages before 5 /\
          Heads.completed state after slot (Mixer.head_models stages) 16))
      (ensures scope ==> live logical stages after 6) =
  let under (_:scope) : Lemma (live logical stages after 6) =
    Mixer.output logical stages state after slot
  in
  FStar.Classical.impl_intro under
