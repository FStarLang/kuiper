module KPlay.Functional

#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module BF16 = Kuiper.BFloat16
module Shared = KPlay.SharedSpec

#set-options "--fuel 4 --ifuel 2"

(* These extents come from the frozen configuration/call, not observations. *)
inline_for_extraction noextract
let embedding_count : pos = Shared.vocab_size * Shared.hidden_size

inline_for_extraction noextract
let hidden_count : pos = Shared.prompt_length * Shared.hidden_size

inline_for_extraction noextract
let rotary_plane_count : pos = Shared.prompt_length * Shared.head_dim

inline_for_extraction noextract
let rotary_count : pos = 2 * rotary_plane_count

inline_for_extraction noextract
let embedding_values = lseq BF16.t embedding_count

inline_for_extraction noextract
let hidden_values = lseq BF16.t hidden_count

inline_for_extraction noextract
let rotary_values = lseq BF16.t rotary_count

inline_for_extraction noextract
let valid_tokens (ts:lseq I64.t Shared.prompt_length) : prop =
  forall (p:Shared.prompt_index).
    0 <= I64.v (Seq.index ts p) /\
    I64.v (Seq.index ts p) < Shared.vocab_size

inline_for_extraction noextract
let token_values =
  ts:lseq I64.t Shared.prompt_length{valid_tokens ts}

(* The scalar is retained in the ABI; only the shared prefill call is in scope. *)
inline_for_extraction noextract
let prefill_position = p:I64.t{I64.v p == 0}

inline_for_extraction noextract
let rotary_view = Shared.prompt_index -> Shared.rotary_index -> real

inline_for_extraction noextract
let embedding_offset (v:Shared.token_id) (h:Shared.hidden_index)
  : i:nat{i < embedding_count} =
  v * Shared.hidden_size + h

inline_for_extraction noextract
let hidden_offset (p:Shared.prompt_index) (h:Shared.hidden_index)
  : i:nat{i < hidden_count} =
  p * Shared.hidden_size + h

inline_for_extraction noextract
let rotary_offset (component:nat{component < 2})
  (p:Shared.prompt_index) (d:Shared.rotary_index)
  : i:nat{i < rotary_count} =
  component * rotary_plane_count + p * Shared.head_dim + d

inline_for_extraction noextract
let token_at (ts:token_values) (p:Shared.prompt_index) : Shared.token_id =
  I64.v (Seq.index ts p)

inline_for_extraction noextract
let input_mapping (weights:embedding_values) (ts:token_values)
  : Shared.inputs BF16.t = {
  embedding_weight = (fun v h -> Seq.index weights (embedding_offset v h));
  tokens = token_at ts;
  decode_steps = 1
}

(* No cache/state buffer exists at this ABI. Unit denotes fresh logical state,
   not a forgotten previous cache or a claim about physical initialization. *)
inline_for_extraction noextract
let state_mapping (_:unit) : Shared.decoder_entry_state BF16.t =
  Shared.initial_decoder_state ()

inline_for_extraction noextract
let state_mapping_satisfies_shared ()
  : Lemma (Shared.state_postcondition (state_mapping ())) =
  ()

(* Positions, bias and fresh state are implicit. The two real views are
   witnesses related to physical BF16 storage below, not supplied tables. *)
inline_for_extraction noextract
let output_mapping (first_position:prefill_position) (hidden:hidden_values)
  (cosines sines:rotary_view) : Shared.outputs BF16.t = {
  hidden_rows = (fun p h -> Seq.index hidden (hidden_offset p h));
  position_ids = (fun p -> I64.v first_position + p);
  cache_positions = (fun p -> p);
  causal_bias = Shared.full_attention_bias;
  rope_cos = cosines;
  rope_sin = sines;
  decoder_state = state_mapping ()
}

inline_for_extraction noextract
let rotary_representation (rotary:rotary_values) (cosines sines:rotary_view)
  : prop =
  forall (p:Shared.prompt_index) (d:Shared.rotary_index).
    v_approximates (Seq.index rotary (rotary_offset 0 p d)) (cosines p d) /\
    v_approximates (Seq.index rotary (rotary_offset 1 p d)) (sines p d)

inline_for_extraction noextract
let semantic_postcondition
  (weights:embedding_values) (ts:token_values)
  (first_position:prefill_position) (hidden:hidden_values) (rotary:rotary_values)
  : prop =
  exists (cosines:rotary_view) (sines:rotary_view).
    rotary_representation rotary cosines sines /\
    Shared.postcondition (input_mapping weights ts)
      (output_mapping first_position hidden cosines sines)

inline_for_extraction noextract
let pointwise_postcondition
  (weights:embedding_values) (ts:token_values)
  (hidden:hidden_values) (rotary:rotary_values) : prop =
  (forall (p:Shared.prompt_index) (h:Shared.hidden_index).
    Seq.index hidden (hidden_offset p h) ==
    Seq.index weights (embedding_offset (token_at ts p) h)) /\
  rotary_representation rotary Shared.cosine Shared.sine

inline_for_extraction noextract
let establish_semantic_postcondition
  (weights:embedding_values) (ts:token_values)
  (first_position:prefill_position) (hidden:hidden_values) (rotary:rotary_values)
  : Lemma
      (requires pointwise_postcondition weights ts hidden rotary)
      (ensures semantic_postcondition weights ts first_position hidden rotary) =
  state_mapping_satisfies_shared ();
  assert (Shared.postcondition (input_mapping weights ts)
    (output_mapping first_position hidden Shared.cosine Shared.sine));
  assert (rotary_representation rotary Shared.cosine Shared.sine)

inline_for_extraction noextract
let semantic_postcondition_implies_pointwise
  (weights:embedding_values) (ts:token_values)
  (first_position:prefill_position) (hidden:hidden_values) (rotary:rotary_values)
  : Lemma
      (requires semantic_postcondition weights ts first_position hidden rotary)
      (ensures pointwise_postcondition weights ts hidden rotary) =
  let from_views (cosines sines:rotary_view)
    : Lemma
      (requires
        rotary_representation rotary cosines sines /\
        Shared.postcondition (input_mapping weights ts)
          (output_mapping first_position hidden cosines sines))
      (ensures pointwise_postcondition weights ts hidden rotary) =
    let logical_input = input_mapping weights ts in
    let logical_output = output_mapping first_position hidden cosines sines in
    let hidden_at (p:Shared.prompt_index) (h:Shared.hidden_index)
      : Lemma (
          Seq.index hidden (hidden_offset p h) ==
          Seq.index weights (embedding_offset (token_at ts p) h)) =
      assert (logical_output.hidden_rows p h ==
        logical_input.embedding_weight (logical_input.tokens p) h)
    in
    FStar.Classical.forall_intro_2 hidden_at;
    let rotary_at (p:Shared.prompt_index) (d:Shared.rotary_index)
      : Lemma (
          v_approximates (Seq.index rotary (rotary_offset 0 p d))
            (Shared.cosine p d) /\
          v_approximates (Seq.index rotary (rotary_offset 1 p d))
            (Shared.sine p d)) =
      assert (logical_output.rope_cos p d == Shared.cosine p d /\
              logical_output.rope_sin p d == Shared.sine p d);
      assert (cosines p d == Shared.cosine p d /\
              sines p d == Shared.sine p d);
      assert (v_approximates (Seq.index rotary (rotary_offset 0 p d))
                (cosines p d) /\
              v_approximates (Seq.index rotary (rotary_offset 1 p d))
                (sines p d))
    in
    FStar.Classical.forall_intro_2 rotary_at
  in
  let implication (cosines sines:rotary_view)
    : Lemma (
        rotary_representation rotary cosines sines /\
        Shared.postcondition (input_mapping weights ts)
          (output_mapping first_position hidden cosines sines) ==>
        pointwise_postcondition weights ts hidden rotary) =
    FStar.Classical.move_requires_2 from_views cosines sines
  in
  FStar.Classical.forall_intro_2 implication

(* Ownership is returned on gpu_loc, not pledged to a pending stream. *)
fn kplay_run
  (weights:larray BF16.t embedding_count{is_global_array weights})
  (tokens:larray I64.t Shared.prompt_length{is_global_array tokens})
  (first_position:prefill_position)
  (hidden:larray BF16.t hidden_count{is_global_array hidden})
  (rotary:larray BF16.t rotary_count{is_global_array rotary})
  (#weights_before:erased embedding_values)
  (#tokens_before:erased (lseq I64.t Shared.prompt_length){
    valid_tokens tokens_before})
  preserves cpu **
    on gpu_loc (weights |-> weights_before ** tokens |-> tokens_before)
  requires exists* (hidden_before:hidden_values) (rotary_before:rotary_values).
    on gpu_loc (hidden |-> hidden_before ** rotary |-> rotary_before)
  ensures exists* (hidden_after:hidden_values) (rotary_after:rotary_values).
    on gpu_loc (hidden |-> hidden_after ** rotary |-> rotary_after) **
    pure (semantic_postcondition weights_before tokens_before first_position
      hidden_after rotary_after)
