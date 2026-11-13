module KPlay.Functional

#lang-pulse

open Kuiper
open FStar.Real
open Kuiper.Approximates.Base

module Shared = KPlay.SharedSpec
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module Casts = Kuiper.Float.Casts
module Seq = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64
module Pow = FStar.Math.Pow
module FP = Kuiper.Floating.Base

(* Bilateral numerical comparison excludes overflow/invalid arithmetic in
   either source. This stronger base refinement consumes no native numerical
   premises. Approximation and real positivity never imply native finiteness.
   kplay_run_safety exposes the independent resource guarantee below. *)

(* These are dimensions of the authorized logical model, not trace values. *)
inline_for_extraction noextract
let hidden_elements : pos = 7 * 1024

inline_for_extraction noextract
let rotary_elements : pos = 2 * 7 * 128

inline_for_extraction noextract
type bf16_values (n:nat) = lseq BF16.t n

inline_for_extraction noextract
type bf16_array (n:nat) = larray BF16.t n

[@@erasable]
noeq
type weight_offsets = {
  pre_norm: int;
  qkv: int;
  qk_norm: int;
  attention_output: int;
  post_norm: int;
  gate_up: int;
  down: int
}

inline_for_extraction noextract
let offsets_of_scalars
  (pre qkv qk output post gate down:I64.t) : GTot weight_offsets = {
  pre_norm = I64.v pre;
  qkv = I64.v qkv;
  qk_norm = I64.v qk;
  attention_output = I64.v output;
  post_norm = I64.v post;
  gate_up = I64.v gate;
  down = I64.v down
}

inline_for_extraction noextract
let valid_span (length:nat) (offset:int) (count:pos) : prop =
  0 <= offset /\ offset + count <= length

inline_for_extraction noextract
let valid_weight_offsets (length:nat) (offsets:weight_offsets) : prop =
  length <= 2147483647 /\
  valid_span length offsets.pre_norm 1024 /\
  valid_span length offsets.qkv (4096 * 1024) /\
  valid_span length offsets.qk_norm 256 /\
  valid_span length offsets.attention_output (1024 * 2048) /\
  valid_span length offsets.post_norm 1024 /\
  valid_span length offsets.gate_up (6144 * 1024) /\
  valid_span length offsets.down (1024 * 3072)

inline_for_extraction noextract
let hidden_index (token:Shared.index 7) (channel:Shared.index 1024)
  : i:nat{i < hidden_elements} =
  token * 1024 + channel

inline_for_extraction noextract
let rotary_index
  (plane:Shared.index 2) (token:Shared.index 7) (lane:Shared.index 128)
  : i:nat{i < rotary_elements} =
  plane * (7 * 128) + token * 128 + lane

inline_for_extraction noextract
let hidden_representation
  (storage:bf16_values hidden_elements) (logical:Shared.hidden_states) : prop =
  forall (token:Shared.index 7) (channel:Shared.index 1024).
    v_approximates (Seq.index storage (hidden_index token channel))
      (logical token channel)

inline_for_extraction noextract
let weights_representation
  (#length:nat)
  (storage:bf16_values length)
  (offsets:weight_offsets{valid_weight_offsets length offsets})
  (logical:Shared.parameters) : prop =
  (forall (channel:Shared.index 1024).
    v_approximates (Seq.index storage (offsets.pre_norm + channel))
      (logical.input_layernorm channel) /\
    v_approximates (Seq.index storage (offsets.post_norm + channel))
      (logical.post_attention_layernorm channel)) /\
  (forall (channel:Shared.index 128).
    v_approximates (Seq.index storage (offsets.qk_norm + channel))
      (logical.q_norm channel) /\
    v_approximates (Seq.index storage (offsets.qk_norm + 128 + channel))
      (logical.k_norm channel)) /\
  (forall (row:Shared.index 2048) (column:Shared.index 1024).
    v_approximates
      (Seq.index storage (offsets.qkv + row * 1024 + column))
      (logical.q_proj row column)) /\
  (forall (row column:Shared.index 1024).
    v_approximates
      (Seq.index storage (offsets.qkv + (2048 + row) * 1024 + column))
      (logical.k_proj row column) /\
    v_approximates
      (Seq.index storage (offsets.qkv + (3072 + row) * 1024 + column))
      (logical.v_proj row column)) /\
  (forall (row:Shared.index 1024) (column:Shared.index 2048).
    v_approximates
      (Seq.index storage (offsets.attention_output + row * 2048 + column))
      (logical.o_proj row column)) /\
  (forall (row:Shared.index 3072) (column:Shared.index 1024).
    v_approximates
      (Seq.index storage (offsets.gate_up + row * 1024 + column))
      (logical.gate_proj row column) /\
    v_approximates
      (Seq.index storage (offsets.gate_up + (3072 + row) * 1024 + column))
      (logical.up_proj row column)) /\
  (forall (row:Shared.index 1024) (column:Shared.index 3072).
    v_approximates
      (Seq.index storage (offsets.down + row * 3072 + column))
      (logical.down_proj row column))

inline_for_extraction noextract
let rotary_representation
  (storage:bf16_values rotary_elements) (logical:Shared.attention_data) : prop =
  forall (token:Shared.index 7) (lane:Shared.index 128).
    v_approximates (Seq.index storage (rotary_index 0 token lane))
      (logical.cosine token lane) /\
    v_approximates (Seq.index storage (rotary_index 1 token lane))
      (logical.sine token lane)

(* Value-level interpretation of the live embedding binding, not a second
   rotary computation in the decoder or its wrapper. *)
inline_for_extraction noextract
let source_rotary_angle (token:Shared.index 7) (lane:Shared.index 128) : F32.t =
  let theta = FP.of_int #F32.t 1000000L in
  let numerator = FP.of_int #F32.t (I64.int_to_t (lane % 64)) in
  let denominator = FP.of_int #F32.t 64L in
  let power = FP.pow theta (FP.div numerator denominator) in
  FP.mul (FP.of_int #F32.t (I64.int_to_t token)) (FP.div (FP.one #F32.t) power)

inline_for_extraction noextract
let source_rotary_angle_correct (token:Shared.index 7) (lane:Shared.index 128)
  : Lemma
      (v_approximates (source_rotary_angle token lane) (Shared.rope_angle token lane))
  =
  let theta = FP.of_int #F32.t 1000000L in
  let numerator = FP.of_int #F32.t (I64.int_to_t (lane % 64)) in
  let denominator = FP.of_int #F32.t 64L in
  let exponent = FP.div numerator denominator in
  let exponent_real = FStar.Real.of_int (lane % 64) /. 64.0R in
  of_int_approx #F32.t 1000000L;
  of_int_approx #F32.t (I64.int_to_t (lane % 64));
  of_int_approx #F32.t 64L;
  div_approx numerator denominator (FStar.Real.of_int (lane % 64)) 64.0R;
  F32.pow_approx theta exponent 1000000.0R exponent_real;
  let power = FP.pow theta exponent in
  div_approx (FP.one #F32.t) power 1.0R (Pow.pow 1000000.0R exponent_real);
  of_int_approx #F32.t (I64.int_to_t token);
  a_mul (FP.of_int #F32.t (I64.int_to_t token)) (FP.div (FP.one #F32.t) power)
    (FStar.Real.of_int token) (1.0R /. Pow.pow 1000000.0R exponent_real);
  assert (exponent_real == FStar.Real.of_int (2 * (lane % 64)) /. 128.0R)

inline_for_extraction noextract
let source_rotary_cell
  (plane:Shared.index 2) (token:Shared.index 7) (lane:Shared.index 128) : BF16.t =
  let angle = source_rotary_angle token lane in
  Casts.fcast #F32.t #BF16.t (if plane = 0 then FP.cos angle else FP.sin angle)

inline_for_extraction noextract
let source_rotary_cell_correct
  (plane:Shared.index 2) (token:Shared.index 7) (lane:Shared.index 128)
  : Lemma
      (v_approximates (source_rotary_cell plane token lane)
        (if plane = 0 then Shared.prefill_cos token lane else Shared.prefill_sin token lane))
  =
  source_rotary_angle_correct token lane;
  let angle = source_rotary_angle token lane in
  if plane = 0 then (
    F32.cos_approx angle (Shared.rope_angle token lane);
    Casts.fcast_approx #F32.t #BF16.t (FP.cos angle) (Shared.prefill_cos token lane)
  ) else (
    F32.sin_approx angle (Shared.rope_angle token lane);
    Casts.fcast_approx #F32.t #BF16.t (FP.sin angle) (Shared.prefill_sin token lane)
  )

inline_for_extraction noextract
let embedding_establishes_rotary_representation
  (storage:bf16_values rotary_elements)
  : Lemma
      (requires forall (plane:Shared.index 2) (token:Shared.index 7) (lane:Shared.index 128).
        Seq.index storage (rotary_index plane token lane) == source_rotary_cell plane token lane)
      (ensures rotary_representation storage (Shared.prefill_attention_data ()))
  =
  let point (token:Shared.index 7) (lane:Shared.index 128)
    : Lemma
        (v_approximates (Seq.index storage (rotary_index 0 token lane))
           (Shared.prefill_cos token lane) /\
         v_approximates (Seq.index storage (rotary_index 1 token lane))
           (Shared.prefill_sin token lane))
    =
    source_rotary_cell_correct 0 token lane;
    source_rotary_cell_correct 1 token lane
  in
  FStar.Classical.forall_intro_2 point

[@@erasable]
noeq
type input_denotation = {
  weights: Shared.parameters;
  incoming_residual: Shared.hidden_states;
  incoming_delta: Shared.hidden_states;
  attention: Shared.attention_data
}

inline_for_extraction noextract
let valid_denotation (logical:input_denotation) : prop =
  Shared.valid_attention_data logical.attention

inline_for_extraction noextract
let input_representation
  (#length:nat)
  (weights:bf16_values length)
  (residual delta:bf16_values hidden_elements)
  (rotary:bf16_values rotary_elements)
  (offsets:weight_offsets{valid_weight_offsets length offsets})
  (fused:bool)
  (logical:input_denotation) : prop =
  valid_denotation logical /\
  weights_representation weights offsets logical.weights /\
  hidden_representation residual logical.incoming_residual /\
  (if fused then hidden_representation delta logical.incoming_delta else True) /\
  rotary_representation rotary logical.attention

inline_for_extraction noextract
let input_mapping
  (logical:input_denotation{valid_denotation logical})
  (fused:bool)
  (layer:Shared.index 28)
  (before:Shared.cache_state{Shared.valid_cache_before layer before})
  : GTot Shared.prefill_input = {
  hidden =
    (if fused
     then Shared.residual_add logical.incoming_residual logical.incoming_delta
     else logical.incoming_residual);
  weight = logical.weights;
  data = logical.attention;
  layer = layer;
  cache_before = before
}

inline_for_extraction noextract
let input_branch_mapping
  (logical:input_denotation{valid_denotation logical})
  (fused:bool)
  (layer:Shared.index 28)
  (before:Shared.cache_state{Shared.valid_cache_before layer before})
  (token:Shared.index 7) (channel:Shared.index 1024)
  : Lemma
      ((input_mapping logical fused layer before).hidden token channel ==
        (if fused then
           logical.incoming_residual token channel +.
             logical.incoming_delta token channel
         else logical.incoming_residual token channel))
  = ()

inline_for_extraction noextract
let rounded_pair_denotes_sum
  (left right:BF16.t) (left_real right_real:real)
  : Lemma
      (requires
        v_approximates left left_real /\ v_approximates right right_real)
      (ensures
        v_approximates
          (Casts.fcast #F32.t #BF16.t
            (add #F32.t
              (Casts.fcast #BF16.t #F32.t left)
              (Casts.fcast #BF16.t #F32.t right)))
          (left_real +. right_real))
  =
  Casts.fcast_approx #BF16.t #F32.t left left_real;
  Casts.fcast_approx #BF16.t #F32.t right right_real;
  a_add (Casts.fcast #BF16.t #F32.t left)
    (Casts.fcast #BF16.t #F32.t right) left_real right_real;
  Casts.fcast_approx #F32.t #BF16.t
    (add #F32.t
      (Casts.fcast #BF16.t #F32.t left)
      (Casts.fcast #BF16.t #F32.t right))
    (left_real +. right_real)

(* The cache component is an erased prefill history. It is not an extra
   physical output or a persistent cache usable by a later decoding call. *)
[@@erasable]
noeq
type pending_output = {
  residual: Shared.hidden_states;
  delta: Shared.hidden_states;
  keys: Shared.heads 8;
  values: Shared.heads 8
}

[@@erasable]
noeq
type cache_observation = {
  qkv_values: bf16_values 28672;
  qk_values: bf16_values 21504
}

inline_for_extraction noextract
let computed_cache_representation
  (qkv:bf16_values 28672) (qk:bf16_values 21504) (logical:pending_output) : prop =
  forall (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128).
    v_approximates (Seq.index qk (row*3072+2048+head*128+lane))
      (logical.keys row head lane) /\
    v_approximates (Seq.index qkv (row*4096+3072+head*128+lane))
      (logical.values row head lane)

inline_for_extraction noextract
let output_representation
  (residual delta:bf16_values hidden_elements) (logical:pending_output) : prop =
  hidden_representation residual logical.residual /\
  hidden_representation delta logical.delta

inline_for_extraction noextract
let state_mapping (input:Shared.prefill_input) (logical:pending_output)
  : GTot Shared.cache_state =
  Shared.update_cache input.cache_before input.layer logical.keys logical.values

inline_for_extraction noextract
let output_mapping (input:Shared.prefill_input) (logical:pending_output)
  : GTot Shared.layer_result = {
  hidden_after = Shared.residual_add logical.residual logical.delta;
  cache_after = state_mapping input logical
}

(* A decomposition of Shared.decoder_layer, using only its mathematical
   operations. No machine reduction, rounding, or instruction trace is here. *)
inline_for_extraction noextract
let decoder_parts (input:Shared.prefill_input) : GTot pending_output =
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
  {
    residual = residual;
    delta = Shared.feed_forward weight normalized_residual;
    keys = rotated_key;
    values = value
  }

inline_for_extraction noextract
let same_hidden (left right:Shared.hidden_states) : prop =
  forall (token:Shared.index 7) (channel:Shared.index 1024).
    left token channel == right token channel

inline_for_extraction noextract
let same_pending (left right:pending_output) : prop =
  same_hidden left.residual right.residual /\
  same_hidden left.delta right.delta /\
  (forall (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128).
    left.keys token head lane == right.keys token head lane /\
    left.values token head lane == right.values token head lane)

(* Fixing the first summand rules out arbitrary cancelling output pairs.
   The shared postcondition then determines the second summand AND the K/V. *)
inline_for_extraction noextract
let pending_postcondition (input:Shared.prefill_input) (logical:pending_output)
  : prop =
  same_hidden logical.residual (decoder_parts input).residual /\
  Shared.postcondition input (output_mapping input logical)

inline_for_extraction noextract
let semantic_postcondition
  (input:Shared.prefill_input)
  (residual delta:bf16_values hidden_elements)
  (observation:cache_observation) : prop =
  exists (logical:pending_output).
    output_representation residual delta logical /\
    computed_cache_representation observation.qkv_values observation.qk_values logical /\
    pending_postcondition input logical

inline_for_extraction noextract
let decomposition_refines_shared (input:Shared.prefill_input)
  : Lemma
      (Shared.postcondition input (output_mapping input (decoder_parts input)) /\
       pending_postcondition input (decoder_parts input))
  = ()

inline_for_extraction noextract
let output_hidden_at
  (input:Shared.prefill_input) (logical:pending_output)
  (token:Shared.index 7) (channel:Shared.index 1024)
  : Lemma
      ((output_mapping input logical).hidden_after token channel ==
        logical.residual token channel +. logical.delta token channel)
  = ()

inline_for_extraction noextract
let pending_residual_add_refines_hidden
  (input:Shared.prefill_input) (logical:pending_output)
  (residual delta:bf16_values hidden_elements)
  (token:Shared.index 7) (channel:Shared.index 1024)
  : Lemma
      (requires output_representation residual delta logical)
      (ensures
        v_approximates
          (Casts.fcast #F32.t #BF16.t
            (add #F32.t
              (Casts.fcast #BF16.t #F32.t
                (Seq.index residual (hidden_index token channel)))
              (Casts.fcast #BF16.t #F32.t
                (Seq.index delta (hidden_index token channel)))))
          ((output_mapping input logical).hidden_after token channel))
  =
  output_hidden_at input logical token channel;
  rounded_pair_denotes_sum
    (Seq.index residual (hidden_index token channel))
    (Seq.index delta (hidden_index token channel))
    (logical.residual token channel) (logical.delta token channel)

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

#push-options "--fuel 4 --ifuel 1"
inline_for_extraction noextract
let output_mapping_determines_components
  (input:Shared.prefill_input) (left right:pending_output)
  : Lemma
      (requires
        same_hidden left.residual right.residual /\
        Shared.same_result (output_mapping input left) (output_mapping input right))
      (ensures same_pending left right)
  =
  let hidden_cell (token:Shared.index 7) (channel:Shared.index 1024)
    : Lemma (left.delta token channel == right.delta token channel)
    =
    assert ((output_mapping input left).hidden_after token channel ==
      (output_mapping input right).hidden_after token channel);
    output_hidden_at input left token channel;
    output_hidden_at input right token channel
  in
  FStar.Classical.forall_intro_2 hidden_cell;
  assert (Shared.same_cache_slot
    ((state_mapping input left) input.layer)
    ((state_mapping input right) input.layer));
  assert (Shared.same_cache_slot
    (Shared.Prefilled { Shared.keys = left.keys; Shared.values = left.values })
    (Shared.Prefilled { Shared.keys = right.keys; Shared.values = right.values }));
  let cache_cell
    (token:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
    : Lemma
        (left.keys token head lane == right.keys token head lane /\
         left.values token head lane == right.values token head lane)
    =
    prefilled_slot_at
      { Shared.keys = left.keys; Shared.values = left.values }
      { Shared.keys = right.keys; Shared.values = right.values }
      token head lane
  in
  FStar.Classical.forall_intro_3 cache_cell
#pop-options

inline_for_extraction noextract
let pending_components_are_determined
  (input:Shared.prefill_input) (logical:pending_output)
  : Lemma
      (requires pending_postcondition input logical)
      (ensures same_pending logical (decoder_parts input))
  =
  decomposition_refines_shared input;
  Shared.postcondition_determines_result input
    (output_mapping input logical) (output_mapping input (decoder_parts input));
  output_mapping_determines_components input logical (decoder_parts input)

inline_for_extraction noextract
let computed_cache_refines_shared
  (input:Shared.prefill_input) (logical:pending_output)
  (qkv:bf16_values 28672) (qk:bf16_values 21504)
  : Lemma
      (requires
        same_pending logical (decoder_parts input) /\
        computed_cache_representation qkv qk logical)
      (ensures computed_cache_representation qkv qk (decoder_parts input))
  = ()

inline_for_extraction noextract
let observed_outputs_establish_semantic_postcondition
  (input:Shared.prefill_input)
  (residual delta:bf16_values hidden_elements)
  (observation:cache_observation)
  : Lemma
      (requires
        output_representation residual delta (decoder_parts input) /\
        computed_cache_representation observation.qkv_values observation.qk_values (decoder_parts input))
      (ensures semantic_postcondition input residual delta observation)
  =
  decomposition_refines_shared input;
  FStar.Classical.exists_intro
    (fun logical ->
      output_representation residual delta logical /\
      computed_cache_representation observation.qkv_values observation.qk_values logical /\
      pending_postcondition input logical)
    (decoder_parts input)

inline_for_extraction noextract
let computed_outputs_establish_semantic_postcondition
  (input:Shared.prefill_input) (logical:pending_output)
  (residual delta:bf16_values hidden_elements)
  (observation:cache_observation)
  : Lemma
      (requires
        same_pending logical (decoder_parts input) /\
        output_representation residual delta logical /\
        computed_cache_representation observation.qkv_values observation.qk_values logical)
      (ensures semantic_postcondition input residual delta observation)
  =
  decomposition_refines_shared input;
  assert (pending_postcondition input logical);
  FStar.Classical.exists_intro
    (fun result ->
      output_representation residual delta result /\
      computed_cache_representation observation.qkv_values observation.qk_values result /\
      pending_postcondition input result)
    logical

inline_for_extraction noextract
let semantic_postcondition_requires_observed_cache
  (input:Shared.prefill_input)
  (residual delta:bf16_values hidden_elements)
  (observation:cache_observation)
  : Lemma
      (requires semantic_postcondition input residual delta observation)
      (ensures
        computed_cache_representation observation.qkv_values observation.qk_values (decoder_parts input))
  =
  let at (logical:pending_output)
    : Lemma
        (requires
          computed_cache_representation observation.qkv_values observation.qk_values logical /\
          pending_postcondition input logical)
        (ensures
          computed_cache_representation observation.qkv_values observation.qk_values (decoder_parts input))
    =
    pending_components_are_determined input logical;
    computed_cache_refines_shared input logical observation.qkv_values observation.qk_values
  in
  FStar.Classical.forall_intro (FStar.Classical.move_requires at)

inline_for_extraction noextract
let mapped_cache_frame
  (input:Shared.prefill_input) (logical:pending_output)
  (other:Shared.index 28{other <> input.layer})
  : Lemma
      ((state_mapping input logical) other == input.cache_before other)
  =
  Shared.cache_frame input.cache_before input.layer logical.keys logical.values other

inline_for_extraction noextract
let mapped_cache_target
  (input:Shared.prefill_input) (logical:pending_output)
  : Lemma
      (Shared.cache_initialized ((state_mapping input logical) input.layer) /\
       Shared.cache_length ((state_mapping input logical) input.layer) == 7 /\
       Shared.same_cache_slot ((state_mapping input logical) input.layer)
         (Shared.Prefilled { Shared.keys = logical.keys; Shared.values = logical.values }))
  =
  Shared.cache_target input.cache_before input.layer logical.keys logical.values

inline_for_extraction noextract
let mapped_cache_shape_and_frame
  (input:Shared.prefill_input) (logical:pending_output)
  : Lemma
      (Shared.cache_initialized ((state_mapping input logical) input.layer) /\
       Shared.cache_length ((state_mapping input logical) input.layer) == 7 /\
       (forall (other:Shared.index 28). other <> input.layer ==>
         (state_mapping input logical) other == input.cache_before other))
  =
  mapped_cache_target input logical;
  let point (other:Shared.index 28)
    : Lemma
        (other <> input.layer ==>
          (state_mapping input logical) other == input.cache_before other)
    =
    if other <> input.layer then mapped_cache_frame input logical other
  in
  FStar.Classical.forall_intro point

inline_for_extraction noextract
let prefill_cache_is_erased_from_computation
  (logical:input_denotation{valid_denotation logical})
  (fused:bool)
  (left_layer right_layer:Shared.index 28)
  (left:Shared.cache_state{Shared.valid_cache_before left_layer left})
  (right:Shared.cache_state{Shared.valid_cache_before right_layer right})
  : Lemma
      (same_pending
        (decoder_parts (input_mapping logical fused left_layer left))
        (decoder_parts (input_mapping logical fused right_layer right)))
  = ()

inline_for_extraction noextract
ghost
fn snapshot_computed_cache
  (qkv:bf16_array 28672) (qk:bf16_array 21504)
  (#qkv_values:erased (bf16_values 28672))
  (#qk_values:erased (bf16_values 21504))
  (#fqkv #fqk:perm)
  preserves
    on gpu_loc (qkv |-> Frac fqkv qkv_values ** qk |-> Frac fqk qk_values)
  returns observation:erased cache_observation
  ensures pure (
    (reveal observation).qkv_values == reveal qkv_values /\
    (reveal observation).qk_values == reveal qk_values)

fn kplay_run
  (weight_count:U64.t{U64.v weight_count <= 2147483647})
  (input0:bf16_array (U64.v weight_count){is_global_array input0})
  (input1:bf16_array hidden_elements{is_global_array input1})
  (input2:bf16_array hidden_elements{is_global_array input2})
  (input3:bf16_array rotary_elements{is_global_array input3})
  (input4 input5 input6 input7 input8 input9:I64.t)
  (input10:I64.t{
    valid_weight_offsets (U64.v weight_count)
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)})
  (input11:bool)
  (output0:bf16_array hidden_elements{is_global_array output0})
  (output1:bf16_array hidden_elements{is_global_array output1})
  (#weights:erased (bf16_values (U64.v weight_count)))
  (#residual #delta:erased (bf16_values hidden_elements))
  (#rotary:erased (bf16_values rotary_elements))
  (#old_output0 #old_output1:erased (bf16_values hidden_elements))
  (#f0 #f1 #f2 #f3:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac f0 weights) **
    on gpu_loc (input1 |-> Frac f1 residual) **
    on gpu_loc (input2 |-> Frac f2 delta) **
    on gpu_loc (input3 |-> Frac f3 rotary)
  requires
    on gpu_loc (output0 |-> old_output0 ** output1 |-> old_output1) **
    pure (input_representation weights residual delta rotary
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)
      input11 logical)
  returns observation:erased cache_observation
  ensures
    exists* (result0:bf16_values hidden_elements) (result1:bf16_values hidden_elements).
      on gpu_loc (output0 |-> result0 ** output1 |-> result1) **
      pure (semantic_postcondition (input_mapping logical input11 layer before)
        result0 result1 (reveal observation))

(* This corollary drops only the algebraic result, not a numerical hypothesis:
   neither this contract nor kplay_run requires native finite arithmetic. *)
inline_for_extraction noextract
fn kplay_run_safety
  (weight_count:U64.t{U64.v weight_count <= 2147483647})
  (input0:bf16_array (U64.v weight_count){is_global_array input0})
  (input1:bf16_array hidden_elements{is_global_array input1})
  (input2:bf16_array hidden_elements{is_global_array input2})
  (input3:bf16_array rotary_elements{is_global_array input3})
  (input4 input5 input6 input7 input8 input9:I64.t)
  (input10:I64.t{
    valid_weight_offsets (U64.v weight_count)
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)})
  (input11:bool)
  (output0:bf16_array hidden_elements{is_global_array output0})
  (output1:bf16_array hidden_elements{is_global_array output1})
  (#weights:erased (bf16_values (U64.v weight_count)))
  (#residual #delta:erased (bf16_values hidden_elements))
  (#rotary:erased (bf16_values rotary_elements))
  (#old_output0 #old_output1:erased (bf16_values hidden_elements))
  (#f0 #f1 #f2 #f3:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac f0 weights) **
    on gpu_loc (input1 |-> Frac f1 residual) **
    on gpu_loc (input2 |-> Frac f2 delta) **
    on gpu_loc (input3 |-> Frac f3 rotary)
  requires
    on gpu_loc (output0 |-> old_output0 ** output1 |-> old_output1) **
    pure (input_representation weights residual delta rotary
      (offsets_of_scalars input4 input5 input6 input7 input8 input9 input10)
      input11 logical)
  ensures
    exists* (result0:bf16_values hidden_elements) (result1:bf16_values hidden_elements).
      on gpu_loc (output0 |-> result0 ** output1 |-> result1)
