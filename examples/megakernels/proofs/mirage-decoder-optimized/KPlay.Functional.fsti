module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Shape
open FStar.Real
open Kuiper.Approximates.Base

module Shared = KPlay.SharedSpec
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module Casts = Kuiper.Float.Casts
module Seq = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64
module ScoreChest = Kuiper.Chest
module ScoreMMA = Kuiper.TensorCore.MMA
module ScoreF = Kuiper.Floating.Base

(* Numerical hypotheses refer to one consumed invocation boundary. The native
   producer is recorded by ownership, not reconstructed by a pure FTZ model.
   Output initialization has no numerical gate. *)

inline_for_extraction noextract
let hidden_elements : pos = 8 * 1024

inline_for_extraction noextract
let weight_elements : pos = 15730944

inline_for_extraction noextract
let rotary_elements : pos = 2 * 4096 * 128

inline_for_extraction noextract
type stored_hidden = Shared.matrix 8 1024

inline_for_extraction noextract
type bf16_values (n:nat) = lseq BF16.t n

inline_for_extraction noextract
type bf16_array (n:nat) = larray BF16.t n

inline_for_extraction noextract
let hidden_index (token:Shared.index 8) (channel:Shared.index 1024)
  : i:nat{i < hidden_elements} =
  token * 1024 + channel

inline_for_extraction noextract
let rotary_index
  (position:nat{position <= 4089})
  (plane:Shared.index 2) (token:Shared.index 7) (lane:Shared.index 128)
  : i:nat{i < rotary_elements} =
  plane * (4096 * 128) + (position + token) * 128 + lane

inline_for_extraction noextract
let query_index
  (token:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 128)
  : i:nat{i < 8 * 4096} =
  token * 4096 + (head / 2) * 512 + (head % 2) * 128 + lane

inline_for_extraction noextract
let key_index
  (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : i:nat{i < 8 * 4096} =
  token * 4096 + head * 512 + 256 + lane

inline_for_extraction noextract
let value_index
  (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : i:nat{i < 8 * 4096} =
  token * 4096 + head * 512 + 384 + lane

inline_for_extraction noextract
let hidden_representation
  (storage:bf16_values hidden_elements) (logical:stored_hidden) : prop =
  forall (token:Shared.index 8) (channel:Shared.index 1024).
    v_approximates (Seq.index storage (hidden_index token channel))
      (logical token channel)

inline_for_extraction noextract
let weights_representation
  (storage:bf16_values weight_elements) (logical:Shared.parameters) : prop =
  (forall (channel:Shared.index 1024).
    v_approximates (Seq.index storage channel)
      (logical.input_layernorm channel) /\
    v_approximates (Seq.index storage (1024 + channel))
      (logical.post_attention_layernorm channel)) /\
  (forall (lane:Shared.index 128).
    v_approximates (Seq.index storage (6293504 + lane))
      (logical.q_norm lane) /\
    v_approximates (Seq.index storage (6293504 + 128 + lane))
      (logical.k_norm lane)) /\
  (forall (row:Shared.index 2048) (column:Shared.index 1024).
    v_approximates (Seq.index storage (2048 + row * 1024 + column))
      (logical.q_proj row column)) /\
  (forall (row column:Shared.index 1024).
    v_approximates (Seq.index storage (2048 + (2048 + row) * 1024 + column))
      (logical.k_proj row column) /\
    v_approximates (Seq.index storage (2048 + (3072 + row) * 1024 + column))
      (logical.v_proj row column)) /\
  (forall (row:Shared.index 1024) (column:Shared.index 2048).
    v_approximates (Seq.index storage (4196352 + row * 2048 + column))
      (logical.o_proj row column)) /\
  (forall (row:Shared.index 3072) (column:Shared.index 1024).
    v_approximates (Seq.index storage (6293760 + row * 1024 + column))
      (logical.gate_proj row column) /\
    v_approximates (Seq.index storage (6293760 + (3072 + row) * 1024 + column))
      (logical.up_proj row column)) /\
  (forall (row:Shared.index 1024) (column:Shared.index 3072).
    v_approximates (Seq.index storage (12585216 + row * 3072 + column))
      (logical.down_proj row column))

inline_for_extraction noextract
let rotary_representation
  (storage:bf16_values rotary_elements)
  (position:nat{position <= 4089}) (logical:Shared.attention_data) : prop =
  forall (token:Shared.index 7) (lane:Shared.index 128).
    v_approximates (Seq.index storage (rotary_index position 0 token lane))
      (logical.cosine token lane) /\
    v_approximates (Seq.index storage (rotary_index position 1 token lane))
      (logical.sine token lane)

[@@erasable]
noeq
type input_denotation = {
  hidden: stored_hidden;
  weights: Shared.parameters;
  attention: Shared.attention_data
}

inline_for_extraction noextract
let valid_denotation (logical:input_denotation) : prop =
  Shared.valid_attention_data logical.attention

inline_for_extraction noextract
let input_representation
  (hidden:bf16_values hidden_elements)
  (weights:bf16_values weight_elements)
  (rotary:bf16_values rotary_elements)
  (position:nat{position <= 4089})
  (logical:input_denotation) : prop =
  valid_denotation logical /\
  hidden_representation hidden logical.hidden /\
  weights_representation weights logical.weights /\
  rotary_representation rotary position logical.attention

inline_for_extraction noextract
let input_mapping
  (logical:input_denotation{valid_denotation logical})
  (layer:Shared.index 28)
  (before:Shared.cache_state{Shared.valid_cache_before layer before})
  : GTot Shared.prefill_input = {
  hidden = (fun token channel -> logical.hidden token channel);
  weight = logical.weights;
  data = logical.attention;
  layer = layer;
  cache_before = before
}

[@@erasable]
noeq
type output_denotation = {
  hidden: stored_hidden;
  keys: Shared.heads 8;
  values: Shared.heads 8
}

[@@erasable]
noeq
type cache_observation = {
  rotated_buffer: bf16_array (8 * 4096);
  rotated_values: bf16_values (8 * 4096)
}

[@@erasable]
noeq
type invocation_inputs = {
  native_hidden: bf16_values hidden_elements;
  native_weights: bf16_values weight_elements;
  native_rotary: bf16_values rotary_elements;
  native_position: nat;
  native_layer: Shared.index 28;
  native_before: Shared.cache_state
}

inline_for_extraction noextract
let native_inputs
  (x:bf16_values hidden_elements) (weights:bf16_values weight_elements)
  (table:bf16_values rotary_elements) (position:nat{position <= 4089})
  (layer:Shared.index 28) (before:Shared.cache_state) : invocation_inputs =
  { native_hidden = x; native_weights = weights; native_rotary = table;
    native_position = position; native_layer = layer; native_before = before }

inline_for_extraction noextract
let computed_cache_representation
  (observation:cache_observation) (logical:output_denotation) : prop =
  forall (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128).
    v_approximates (Seq.index observation.rotated_values (key_index token head lane))
      (logical.keys token head lane) /\
    v_approximates (Seq.index observation.rotated_values (value_index token head lane))
      (logical.values token head lane)

inline_for_extraction noextract
let source_query_operand
  (rotated:bf16_values (8 * 4096)) (head:Shared.index 8) (step:Shared.index 8)
  : GTot (ScoreChest.chest2 BF16.t 16 16) =
  ScoreChest.mk2 (fun row lane ->
    if row < 14 then
      Seq.index rotated
        ((row / 2) * 4096 + head * 512 + (row % 2) * 128 + step * 16 + lane)
    else ScoreF.zero)

inline_for_extraction noextract
let source_key_operand
  (rotated:bf16_values (8 * 4096)) (head:Shared.index 8) (step:Shared.index 8)
  : GTot (ScoreChest.chest2 BF16.t 16 8) =
  ScoreChest.mk2 (fun lane key ->
    if key < 7 then
      Seq.index rotated (key * 4096 + head * 512 + 256 + step * 16 + lane)
    else ScoreF.zero)

inline_for_extraction noextract
let rec source_mma_scores
  (rotated:bf16_values (8 * 4096)) (head:Shared.index 8) (steps:nat{steps <= 8})
  : GTot (ScoreChest.chest2 F32.t 16 8) (decreases steps) =
  if steps = 0 then ScoreChest.const (16 @| 8 @| INil) (ScoreF.zero #F32.t)
  else ScoreMMA.emma (source_mma_scores rotated head (steps - 1))
    (source_query_operand rotated head (steps - 1))
    (source_key_operand rotated head (steps - 1))

inline_for_extraction noextract
let source_mma_score_tile
  (rotated:bf16_values (8 * 4096)) (head:Shared.index 8)
  : GTot (lseq F32.t 128) =
  Seq.init_ghost 128 (fun i ->
    ScoreChest.acc2 (source_mma_scores rotated head 8) (i / 8) (i % 8))

inline_for_extraction noextract
let bilateral_mma_live_scores_finite
  (rotated:bf16_values (8 * 4096)) (head:Shared.index 8) : prop =
  forall (row:Shared.index 14) (col:Shared.index 7{col <= row / 2}).
    Finite? (kind (ScoreChest.acc2 (source_mma_scores rotated head 8) row col))

inline_for_extraction noextract
let observed_numerical_premises (observation:cache_observation) : prop =
  forall (head:Shared.index 8).
    bilateral_mma_live_scores_finite observation.rotated_values head

inline_for_extraction noextract
let rotated_history_representation
  (rotated:bf16_values (8 * 4096))
  (query:Shared.heads 16) (logical:output_denotation) : prop =
  (forall (token:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 128).
    v_approximates (Seq.index rotated (query_index token head lane))
      (query token head lane)) /\
  (forall (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128).
    v_approximates (Seq.index rotated (key_index token head lane))
      (logical.keys token head lane) /\
    v_approximates (Seq.index rotated (value_index token head lane))
      (logical.values token head lane))

inline_for_extraction noextract
let output_representation
  (storage:bf16_values hidden_elements) (logical:output_denotation) : prop =
  hidden_representation storage logical.hidden

inline_for_extraction noextract
let state_mapping (input:Shared.prefill_input) (logical:output_denotation)
  : GTot Shared.cache_state =
  Shared.update_cache input.cache_before input.layer logical.keys logical.values

inline_for_extraction noextract
let output_mapping (input:Shared.prefill_input) (logical:output_denotation)
  : GTot Shared.layer_result = {
  hidden_after = (fun token channel -> logical.hidden token channel);
  cache_after = state_mapping input logical
}

inline_for_extraction noextract
let padding_after (weight:Shared.parameters) (incoming:Shared.vector 1024)
  : GTot (Shared.vector 1024) =
  let normalized = Shared.rms_norm weight.post_attention_layernorm incoming in
  let gate = Shared.linear weight.gate_proj normalized in
  let up = Shared.linear weight.up_proj normalized in
  let activated : Shared.vector 3072 =
    fun channel -> Shared.silu (gate channel) *. up channel
  in
  let down = Shared.linear weight.down_proj activated in
  fun channel -> incoming channel +. down channel

inline_for_extraction noextract
let required_postcondition
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (storage:bf16_values hidden_elements) (observation:cache_observation) : prop =
  exists (logical:output_denotation).
    output_representation storage logical /\
    computed_cache_representation observation logical /\
    Shared.postcondition input (output_mapping input logical) /\
    (forall (channel:Shared.index 1024).
      logical.hidden 7 channel == padding_after input.weight padding channel)

(* Mathematical decomposition reused from the base representation proof. *)
[@@"opaque_to_smt"]
inline_for_extraction noextract
let decoder_parts (input:Shared.prefill_input) (padding:Shared.vector 1024)
  : GTot output_denotation =
  let weight = input.weight in
  let normalized = Shared.normalize_hidden weight.input_layernorm input.hidden in
  let query = Shared.normalize_heads weight.q_norm
    (Shared.split_heads #16 (Shared.linear_rows weight.q_proj normalized)) in
  let key = Shared.normalize_heads weight.k_norm
    (Shared.split_heads #8 (Shared.linear_rows weight.k_proj normalized)) in
  let value = Shared.split_heads #8 (Shared.linear_rows weight.v_proj normalized) in
  let rotated_query = Shared.rotary input.data query in
  let rotated_key = Shared.rotary input.data key in
  let probabilities = Shared.attention_probabilities input.data
    rotated_query (Shared.repeat_kv rotated_key) in
  let mixed = Shared.attention_values probabilities (Shared.repeat_kv value) in
  let projected = Shared.linear_rows weight.o_proj
    (Shared.concatenate_heads mixed) in
  let residual = Shared.residual_add input.hidden projected in
  let normalized_residual =
    Shared.normalize_hidden weight.post_attention_layernorm residual in
  let hidden = Shared.residual_add residual
    (Shared.feed_forward weight normalized_residual) in
  {
    hidden = (fun token channel ->
      if token < 7 then hidden token channel
      else padding_after weight padding channel);
    keys = rotated_key;
    values = value
  }

inline_for_extraction noextract
let decoder_parts_cache_at
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      ((decoder_parts input padding).keys token head lane ==
        Shared.rotary input.data
          (Shared.normalize_heads input.weight.k_norm
            (Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj
              (Shared.normalize_hidden input.weight.input_layernorm input.hidden)))) token head lane /\
       (decoder_parts input padding).values token head lane ==
        Shared.split_heads #8 (Shared.linear_rows input.weight.v_proj
          (Shared.normalize_hidden input.weight.input_layernorm input.hidden)) token head lane)
  =
  reveal_opaque (`%decoder_parts) (decoder_parts input padding)

inline_for_extraction noextract
let decomposition_refines_shared
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  : Lemma
      (Shared.postcondition input (output_mapping input (decoder_parts input padding)) /\
       (forall (channel:Shared.index 1024).
         (decoder_parts input padding).hidden 7 channel ==
           padding_after input.weight padding channel))
  = reveal_opaque (`%decoder_parts) (decoder_parts input padding)

inline_for_extraction noextract
let decoder_parts_hidden_at
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma ((decoder_parts input padding).hidden row col ==
      (Shared.decoder_layer input).hidden_after row col) =
  decomposition_refines_shared input padding

inline_for_extraction noextract
let decoder_parts_padding_at
  (input:Shared.prefill_input) (padding:Shared.vector 1024) (col:Shared.index 1024)
  : Lemma ((decoder_parts input padding).hidden 7 col == padding_after input.weight padding col) =
  decomposition_refines_shared input padding

inline_for_extraction noextract
let represented_parts_establish_postcondition
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (storage:bf16_values hidden_elements) (observation:cache_observation)
  : Lemma
      (requires
        output_representation storage (decoder_parts input padding) /\
        computed_cache_representation observation (decoder_parts input padding))
      (ensures required_postcondition input padding storage observation)
  =
  decomposition_refines_shared input padding;
  FStar.Classical.exists_intro
    (fun logical ->
      output_representation storage logical /\
      computed_cache_representation observation logical /\
      Shared.postcondition input (output_mapping input logical) /\
      (forall (channel:Shared.index 1024).
        logical.hidden 7 channel == padding_after input.weight padding channel))
    (decoder_parts input padding)

inline_for_extraction noextract
let mapped_cache_frame
  (input:Shared.prefill_input) (logical:output_denotation)
  (other:Shared.index 28{other <> input.layer})
  : Lemma ((state_mapping input logical) other == input.cache_before other) =
  Shared.cache_frame input.cache_before input.layer logical.keys logical.values other

inline_for_extraction noextract
let mapped_cache_target
  (input:Shared.prefill_input) (logical:output_denotation)
  : Lemma
      (Shared.cache_initialized ((state_mapping input logical) input.layer) /\
       Shared.cache_length ((state_mapping input logical) input.layer) == 7 /\
       Shared.same_cache_slot ((state_mapping input logical) input.layer)
         (Shared.Prefilled { Shared.keys = logical.keys; Shared.values = logical.values })) =
  Shared.cache_target input.cache_before input.layer logical.keys logical.values

inline_for_extraction noextract
let represented_output_at
  (input:Shared.prefill_input) (storage:bf16_values hidden_elements)
  (logical:output_denotation)
  (token:Shared.index 7) (channel:Shared.index 1024)
  : Lemma
      (requires output_representation storage logical)
      (ensures v_approximates (Seq.index storage (hidden_index token channel))
        ((output_mapping input logical).hidden_after token channel)) =
  ()

inline_for_extraction noextract
let prefilled_slot_at
  (left right:Shared.cached_kv)
  (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      (requires Shared.same_cache_slot (Shared.Prefilled left) (Shared.Prefilled right))
      (ensures
        left.keys token head lane == right.keys token head lane /\
        left.values token head lane == right.values token head lane)
  = ()

#push-options "--fuel 4 --ifuel 1 --z3rlimit 30"
inline_for_extraction noextract
let same_mapped_result_determines_history
  (input:Shared.prefill_input) (left right:output_denotation)
  (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      (requires Shared.same_result (output_mapping input left) (output_mapping input right))
      (ensures
        left.keys token head lane == right.keys token head lane /\
        left.values token head lane == right.values token head lane)
  =
  assert (Shared.same_cache_slot
    ((state_mapping input left) input.layer)
    ((state_mapping input right) input.layer));
  assert (Shared.same_cache_slot
    (Shared.Prefilled { Shared.keys = left.keys; Shared.values = left.values })
    (Shared.Prefilled { Shared.keys = right.keys; Shared.values = right.values }));
  prefilled_slot_at
    { Shared.keys = left.keys; Shared.values = left.values }
    { Shared.keys = right.keys; Shared.values = right.values }
    token head lane
#pop-options

inline_for_extraction noextract
let shared_postcondition_determines_history
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (logical:output_denotation)
  (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      (requires Shared.postcondition input (output_mapping input logical))
      (ensures
        logical.keys token head lane == (decoder_parts input padding).keys token head lane /\
        logical.values token head lane == (decoder_parts input padding).values token head lane)
  =
  decomposition_refines_shared input padding;
  Shared.postcondition_determines_result input
    (output_mapping input logical) (output_mapping input (decoder_parts input padding));
  same_mapped_result_determines_history input logical (decoder_parts input padding)
    token head lane

inline_for_extraction noextract
let required_postcondition_requires_observed_cache
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (storage:bf16_values hidden_elements) (observation:cache_observation)
  : Lemma
      (requires required_postcondition input padding storage observation)
      (ensures computed_cache_representation observation (decoder_parts input padding))
  =
  let from_logical (logical:output_denotation)
    : Lemma
        (requires
          computed_cache_representation observation logical /\
          Shared.postcondition input (output_mapping input logical))
        (ensures computed_cache_representation observation (decoder_parts input padding))
    =
    let at (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
      : Lemma
          (logical.keys token head lane == (decoder_parts input padding).keys token head lane /\
           logical.values token head lane == (decoder_parts input padding).values token head lane)
      = shared_postcondition_determines_history input padding logical token head lane
    in FStar.Classical.forall_intro_3 at
  in FStar.Classical.forall_intro (FStar.Classical.move_requires from_logical)

inline_for_extraction noextract
let mapped_cache_shape_and_frame
  (input:Shared.prefill_input) (logical:output_denotation)
  : Lemma
      (Shared.cache_initialized ((state_mapping input logical) input.layer) /\
       Shared.cache_length ((state_mapping input logical) input.layer) == 7 /\
       (forall (other:Shared.index 28). other <> input.layer ==>
         (state_mapping input logical) other == input.cache_before other))
  =
  mapped_cache_target input logical;
  let at (other:Shared.index 28)
    : Lemma (other <> input.layer ==>
        (state_mapping input logical) other == input.cache_before other)
    = if other <> input.layer then mapped_cache_frame input logical other
  in FStar.Classical.forall_intro at

inline_for_extraction noextract
let prefill_cache_is_erased_from_computation
  (logical:input_denotation{valid_denotation logical})
  (left_layer right_layer:Shared.index 28)
  (left:Shared.cache_state{Shared.valid_cache_before left_layer left})
  (right:Shared.cache_state{Shared.valid_cache_before right_layer right})
  (token:Shared.index 8) (channel:Shared.index 1024)
  (kv_token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      (let l = decoder_parts (input_mapping logical left_layer left) (logical.hidden 7) in
       let r = decoder_parts (input_mapping logical right_layer right) (logical.hidden 7) in
       l.hidden token channel == r.hidden token channel /\
       l.keys kv_token head lane == r.keys kv_token head lane /\
       l.values kv_token head lane == r.values kv_token head lane)
  =
  reveal_opaque (`%decoder_parts)
    (decoder_parts (input_mapping logical left_layer left) (logical.hidden 7));
  reveal_opaque (`%decoder_parts)
    (decoder_parts (input_mapping logical right_layer right) (logical.hidden 7))

[@@erasable]
inline_for_extraction noextract
val invocation (inputs:invocation_inputs) : Type0

val invocation_pending (#inputs:invocation_inputs) ([@@@mkey] run:invocation inputs) : slprop

val invocation_consumed (#inputs:invocation_inputs)
  ([@@@mkey] run:invocation inputs) (observation:cache_observation) : slprop

val invocation_result (#inputs:invocation_inputs)
  ([@@@mkey] run:invocation inputs) (observation:cache_observation)
  (result:bf16_values hidden_elements) : slprop

ghost
fn begin_invocation (#inputs:erased invocation_inputs)
  returns run:erased (invocation inputs)
  ensures invocation_pending #inputs run

ghost
fn invocation_observations_agree
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  (#actual #observed:erased cache_observation)
  (#result:erased (bf16_values hidden_elements))
  preserves invocation_consumed #inputs run actual ** invocation_result #inputs run observed result
  ensures pure (actual == observed)

inline_for_extraction noextract
let numerical_premises (#inputs:invocation_inputs)
  (run:invocation inputs) (observation:cache_observation) : slprop =
  invocation_consumed #inputs run observation **
  pure (observed_numerical_premises observation)

inline_for_extraction noextract
let completed_execution (#inputs:invocation_inputs)
  (run:invocation inputs) (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (storage:bf16_values hidden_elements) (observation:cache_observation) : slprop =
  invocation_result #inputs run observation storage **
  pure (observed_numerical_premises observation ==>
    required_postcondition input padding storage observation)

ghost
fn execution_no_overflow_refines_shared
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  (#input:erased Shared.prefill_input) (#padding:erased (Shared.vector 1024))
  (#result:erased (bf16_values hidden_elements))
  (#actual #observed:erased cache_observation)
  preserves
    completed_execution #inputs run input padding result observed **
    numerical_premises #inputs run actual
  ensures pure (required_postcondition input padding result observed)

inline_for_extraction noextract
ghost
fn snapshot_computed_cache
  (rotated:bf16_array (8 * 4096))
  (#values:erased (bf16_values (8 * 4096))) (#f:perm)
  preserves on gpu_loc (rotated |-> Frac f values)
  returns observation:erased cache_observation
  ensures pure ((reveal observation).rotated_buffer == rotated /\
    (reveal observation).rotated_values == reveal values)

(* FULL_HOST_OBLIGATION *)
fn kplay_run
  (rows:U64.t{U64.v rows == 8})
  (hidden_width:U64.t{U64.v hidden_width == 1024})
  (head_width:U64.t{U64.v head_width == 128})
  (positions:U64.t{U64.v positions == 4096})
  (weights_length:U64.t{U64.v weights_length == weight_elements})
  (rope_length:U64.t{U64.v rope_length == rotary_elements})
  (input0:bf16_array hidden_elements{is_global_array input0})
  (input1:bf16_array weight_elements{is_global_array input1})
  (input2:bf16_array rotary_elements{is_global_array input2})
  (input3:I64.t{0 <= I64.v input3 /\ I64.v input3 <= 4089})
  (output0:bf16_array hidden_elements{is_global_array output0})
  (#x:erased (bf16_values hidden_elements))
  (#weights:erased (bf16_values weight_elements))
  (#table:erased (bf16_values rotary_elements))
  (#fx #fw #ft:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  preserves
    cpu **
    on gpu_loc (
      input0 |-> Frac fx x **
      input1 |-> Frac fw weights **
      input2 |-> Frac ft table)
  requires
    on gpu_loc (Pulse.Lib.Array.pts_to_uninit output0 hidden_elements) **
    invocation_pending #inputs run **
    pure (input_representation x weights table (I64.v input3) logical /\
      inputs == native_inputs x weights table (I64.v input3) layer before)
  returns observation:erased cache_observation
  ensures
    exists* (result:bf16_values hidden_elements).
      on gpu_loc (output0 |-> result) **
      pure (computed_cache_representation (reveal observation)
        (decoder_parts (input_mapping logical layer before) ((reveal logical).hidden 7))) **
      invocation_consumed #inputs run (reveal observation) **
      completed_execution #inputs run (input_mapping logical layer before)
        ((reveal logical).hidden 7) result (reveal observation)

(* kplay_run itself has no finite-domain precondition. Dropping its numerical
   implication therefore does not weaken the hypotheses of this corollary. *)
inline_for_extraction noextract
fn kplay_run_safety
  (rows:U64.t{U64.v rows == 8})
  (hidden_width:U64.t{U64.v hidden_width == 1024})
  (head_width:U64.t{U64.v head_width == 128})
  (positions:U64.t{U64.v positions == 4096})
  (weights_length:U64.t{U64.v weights_length == weight_elements})
  (rope_length:U64.t{U64.v rope_length == rotary_elements})
  (input0:bf16_array hidden_elements{is_global_array input0})
  (input1:bf16_array weight_elements{is_global_array input1})
  (input2:bf16_array rotary_elements{is_global_array input2})
  (input3:I64.t{0 <= I64.v input3 /\ I64.v input3 <= 4089})
  (output0:bf16_array hidden_elements{is_global_array output0})
  (#x:erased (bf16_values hidden_elements))
  (#weights:erased (bf16_values weight_elements))
  (#table:erased (bf16_values rotary_elements))
  (#fx #fw #ft:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  preserves cpu ** on gpu_loc (
    input0 |-> Frac fx x ** input1 |-> Frac fw weights ** input2 |-> Frac ft table)
  requires on gpu_loc (Pulse.Lib.Array.pts_to_uninit output0 hidden_elements) **
    pure (input_representation x weights table (I64.v input3) logical)
  ensures exists* (result:bf16_values hidden_elements). on gpu_loc (output0 |-> result)
