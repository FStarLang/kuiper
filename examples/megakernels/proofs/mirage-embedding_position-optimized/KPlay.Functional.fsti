module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module Shared = KPlay.SharedSpec
module R = FStar.Real
module Trig = Kuiper.Real.Trigonometry
module C = FStar.Classical

#set-options "--fuel 4 --ifuel 2 --z3rlimit 80"

inline_for_extraction noextract
let embedding_count : pos = Shared.vocab_size * Shared.hidden_size

(* Physical extents of this producer, not new logical input restrictions. *)
inline_for_extraction noextract
let padded_rows : pos = 8

inline_for_extraction noextract
let table_positions : pos = 4096

inline_for_extraction noextract
let hidden_count : pos = padded_rows * Shared.hidden_size

inline_for_extraction noextract
let table_plane_count : pos = table_positions * Shared.head_dim

inline_for_extraction noextract
let table_count : pos = 2 * table_plane_count

inline_for_extraction noextract
let embedding_values = lseq bf16 embedding_count

inline_for_extraction noextract
let hidden_values = lseq bf16 hidden_count

inline_for_extraction noextract
let table_values = lseq bf16 table_count

inline_for_extraction noextract
let physical_row = p:nat{p < padded_rows}

inline_for_extraction noextract
let table_position = p:nat{p < table_positions}

inline_for_extraction noextract
let valid_tokens (ts:lseq I64.t Shared.prompt_length) : prop =
  forall (p:Shared.prompt_index).
    0 <= I64.v (S.index ts p) /\
    I64.v (S.index ts p) < Shared.vocab_size

inline_for_extraction noextract
let token_values =
  ts:lseq I64.t Shared.prompt_length{valid_tokens ts}

inline_for_extraction noextract
let token_at (ts:token_values) (p:Shared.prompt_index) : Shared.token_id =
  I64.v (S.index ts p)

inline_for_extraction noextract
let physical_token (ts:token_values) (p:physical_row) : Shared.token_id =
  if p < Shared.prompt_length then token_at ts p else 0

inline_for_extraction noextract
let embedding_offset (v:Shared.token_id) (h:Shared.hidden_index)
  : i:nat{i < embedding_count} =
  v * Shared.hidden_size + h

inline_for_extraction noextract
let hidden_offset (p:physical_row) (h:Shared.hidden_index)
  : i:nat{i < hidden_count} =
  p * Shared.hidden_size + h

inline_for_extraction noextract
let table_offset (component:nat{component < 2})
  (p:table_position) (d:Shared.rotary_index)
  : i:nat{i < table_count} =
  (component * table_positions + p) * Shared.head_dim + d

inline_for_extraction noextract
let input_mapping (weights:embedding_values) (ts:token_values)
  : Shared.inputs bf16 = {
  embedding_weight = (fun v h -> S.index weights (embedding_offset v h));
  tokens = token_at ts;
  decode_steps = 1
}

(* Unit represents absence of physical state at this fresh-prefill boundary.
   It does not erase any previous history or initialize a decoder's storage. *)
inline_for_extraction noextract
let state_mapping (_:unit) : Shared.decoder_entry_state bf16 =
  Shared.initial_decoder_state ()

inline_for_extraction noextract
let state_mapping_satisfies_shared ()
  : Lemma (Shared.state_postcondition (state_mapping ())) =
  ()

inline_for_extraction noextract
let rotary_view = Shared.prompt_index -> Shared.rotary_index -> real

inline_for_extraction noextract
let output_mapping (hidden:hidden_values) (cosines sines:rotary_view)
  : Shared.outputs bf16 = {
  hidden_rows = (fun p h -> S.index hidden (hidden_offset p h));
  position_ids = (fun p -> p);
  cache_positions = (fun p -> p);
  causal_bias = Shared.full_attention_bias;
  rope_cos = cosines;
  rope_sin = sines;
  decoder_state = state_mapping ()
}

inline_for_extraction noextract
let rotary_representation (table:table_values) (cosines sines:rotary_view)
  : prop =
  forall (p:Shared.prompt_index) (d:Shared.rotary_index).
    v_approximates (S.index table (table_offset 0 p d)) (cosines p d) /\
    v_approximates (S.index table (table_offset 1 p d)) (sines p d)

(* Extend the same ordinary mathematics to the producer's extra table rows.
   This is not an extension or alteration of the shared prefill domain. *)
inline_for_extraction noextract
let table_angle (p:table_position) (d:Shared.rotary_index) : real =
  Shared.inverse_frequency (Shared.frequency_of_coordinate d) *. R.of_int p

inline_for_extraction noextract
let table_cosine (p:table_position) (d:Shared.rotary_index) : real =
  Trig.cos (table_angle p d)

inline_for_extraction noextract
let table_sine (p:table_position) (d:Shared.rotary_index) : real =
  Trig.sin (table_angle p d)

inline_for_extraction noextract
let table_restriction (p:Shared.prompt_index) (d:Shared.rotary_index)
  : Lemma (
      table_cosine p d == Shared.cosine p d /\
      table_sine p d == Shared.sine p d) =
  ()

inline_for_extraction noextract
let physical_postcondition
  (weights:embedding_values) (ts:token_values)
  (hidden:hidden_values) (table:table_values) : prop =
  (forall (p:physical_row) (h:Shared.hidden_index).
    S.index hidden (hidden_offset p h) ==
      S.index weights (embedding_offset (physical_token ts p) h)) /\
  (forall (p:table_position) (d:Shared.rotary_index).
    v_approximates (S.index table (table_offset 0 p d)) (table_cosine p d) /\
    v_approximates (S.index table (table_offset 1 p d)) (table_sine p d)) /\
  (forall (p:table_position) (r:Shared.frequency_index).
    S.index table (table_offset 0 p r) ==
      S.index table (table_offset 0 p (r + Shared.half_dim)) /\
    S.index table (table_offset 1 p r) ==
      S.index table (table_offset 1 p (r + Shared.half_dim)))

inline_for_extraction noextract
let semantic_postcondition
  (weights:embedding_values) (ts:token_values)
  (hidden:hidden_values) (table:table_values) : prop =
  physical_postcondition weights ts hidden table /\
  (exists (cosines:rotary_view) (sines:rotary_view).
    rotary_representation table cosines sines /\
    Shared.postcondition (input_mapping weights ts)
      (output_mapping hidden cosines sines))

inline_for_extraction noextract
let establish_semantic_postcondition
  (weights:embedding_values) (ts:token_values)
  (hidden:hidden_values) (table:table_values)
  : Lemma
      (requires physical_postcondition weights ts hidden table)
      (ensures semantic_postcondition weights ts hidden table) =
  let rotary_at (p:Shared.prompt_index) (d:Shared.rotary_index)
    : Lemma (
        v_approximates (S.index table (table_offset 0 p d)) (Shared.cosine p d) /\
        v_approximates (S.index table (table_offset 1 p d)) (Shared.sine p d)) =
    table_restriction p d
  in
  C.forall_intro_2 rotary_at;
  state_mapping_satisfies_shared ();
  assert (rotary_representation table Shared.cosine Shared.sine);
  assert (Shared.postcondition (input_mapping weights ts)
    (output_mapping hidden Shared.cosine Shared.sine))

fn kplay_run
  (weights:larray bf16 embedding_count{is_global_array weights})
  (prompt:larray I64.t Shared.prompt_length{is_global_array prompt})
  (embedding:larray bf16 hidden_count{is_global_array embedding})
  (table:larray bf16 table_count{is_global_array table})
  (#weights_perm #prompt_perm:perm)
  (#w:erased embedding_values)
  (#t:erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  preserves cpu **
    on gpu_loc (
      weights |-> Frac weights_perm (reveal w) **
      prompt |-> Frac prompt_perm (reveal t))
  requires exists* (embedding_before:hidden_values) (table_before:table_values).
    on gpu_loc (embedding |-> embedding_before ** table |-> table_before)
  ensures exists* (embedding_after:hidden_values) (table_after:table_values).
    on gpu_loc (embedding |-> embedding_after ** table |-> table_after) **
    pure (semantic_postcondition w t embedding_after table_after)
