module KPlay.Functional

#lang-pulse

open Kuiper
open FStar.Ghost

module Shared = KPlay.SharedSpec
module Seq = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64
module Cast = Kuiper.Float.Casts.Base

(* These extents come from the frozen model, not the replay samples. *)
inline_for_extraction noextract
let hidden_elements : pos = Shared.prompt_length * Shared.hidden_size

inline_for_extraction noextract
let embedding_elements : pos = Shared.vocab_size * Shared.hidden_size

inline_for_extraction noextract
let logit_elements : pos = Shared.prompt_length * Shared.vocab_size

inline_for_extraction noextract
type packed_extent = n:U64.t{
  embedding_elements <= U64.v n /\ U64.v n <= 2147483647}

inline_for_extraction noextract
type weight_offset (n span:nat) = o:I64.t{
  0 <= I64.v o /\ I64.v o + span <= n}

inline_for_extraction noextract
let hidden_index (p:Shared.prompt_index) (d:Shared.hidden_index)
  : i:nat{i < hidden_elements}
  = p * Shared.hidden_size + d

inline_for_extraction noextract
let logit_index (p:Shared.prompt_index) (v:Shared.vocab_index)
  : i:nat{i < logit_elements}
  = p * Shared.vocab_size + v

inline_for_extraction noextract
noeq type input_snapshot (n:nat) = {
  packed_values: lseq bf16 n;
  residual_values: lseq bf16 hidden_elements;
  delta_values: lseq bf16 hidden_elements
}

inline_for_extraction noextract
noeq type input_denotation (n:nat) = {
  packed: i:nat{i < n} -> GTot real;
  residual: Shared.hidden_rows;
  delta: Shared.hidden_rows
}

inline_for_extraction noextract
let input_representation (#n:nat)
  (physical:input_snapshot n) (logical:input_denotation n) : GTot prop =
  (forall (i:nat{i < n}).
    Seq.index physical.packed_values i %~ logical.packed i) /\
  (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
    Seq.index physical.residual_values (hidden_index p d) %~
      logical.residual p d) /\
  (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
    Seq.index physical.delta_values (hidden_index p d) %~
      logical.delta p d)

(* A canonical representative exists for every BF16 snapshot. Callers may
   instead supply any related denotation, allowing upstream composition. *)
inline_for_extraction noextract
let canonical_input_denotation (#n:nat) (physical:input_snapshot n)
  : GTot (input_denotation n) =
  {
    packed = (fun i -> to_real (Seq.index physical.packed_values i));
    residual = (fun p d ->
      to_real (Seq.index physical.residual_values (hidden_index p d)));
    delta = (fun p d ->
      to_real (Seq.index physical.delta_values (hidden_index p d)))
  }

inline_for_extraction noextract
let canonical_input_is_represented (#n:nat) (physical:input_snapshot n)
  : Lemma (input_representation physical (canonical_input_denotation physical))
  = ()

inline_for_extraction noextract
let map_inputs (#n:nat)
  (norm_offset:weight_offset n Shared.hidden_size)
  (projection_offset:weight_offset n embedding_elements)
  (logical:input_denotation n) : GTot Shared.inputs =
  {
    hidden_in = (fun p d -> logical.residual p d +. logical.delta p d);
    norm_weight = (fun d -> logical.packed (I64.v norm_offset + d));
    embedding = (fun v d ->
      logical.packed
        (I64.v projection_offset + v * Shared.hidden_size + d))
  }

(* The pending residual's BF16 rounding boundary refines the real sum;
   neither this boundary nor an instruction schedule defines the function. *)
inline_for_extraction noextract
let residual_add_refines (r d:bf16) (rr dr:real)
  : Lemma
      (requires r %~ rr /\ d %~ dr)
      (ensures
        Cast.cast_f32_to_bf16
          (add (Cast.cast_bf16_to_f32 r) (Cast.cast_bf16_to_f32 d)) %~
        (rr +. dr))
  =
  Cast.cast_bf16_to_f32_ok r rr;
  Cast.cast_bf16_to_f32_ok d dr;
  a_add (Cast.cast_bf16_to_f32 r) (Cast.cast_bf16_to_f32 d) rr dr;
  Cast.cast_f32_to_bf16_ok
    (add (Cast.cast_bf16_to_f32 r) (Cast.cast_bf16_to_f32 d)) (rr +. dr)

inline_for_extraction noextract
type all_logits = Shared.prompt_index -> GTot Shared.vocabulary_logits

inline_for_extraction noextract
noeq type output_denotation = {
  hidden: Shared.hidden_rows;
  scores: all_logits
}

inline_for_extraction noextract
let output_representation
  (hidden:lseq bf16 hidden_elements)
  (logits:lseq bf16 logit_elements)
  (logical:output_denotation) : GTot prop =
  (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
    Seq.index hidden (hidden_index p d) %~ logical.hidden p d) /\
  (forall (p:Shared.prompt_index) (v:Shared.vocab_index).
    Seq.index logits (logit_index p v) %~ logical.scores p v)

inline_for_extraction noextract
let map_outputs (logical:output_denotation) : GTot Shared.outputs =
  {
    normalized_hidden = logical.hidden;
    final_logits = logical.scores (Shared.prompt_length - 1)
  }

(* State is kept in its original BF16 representation, including unused
   packed-weight cells. Other ambient resources are preserved by framing. *)
inline_for_extraction noextract
let map_state (#n:nat) (physical:input_snapshot n)
  : GTot (seq bf16 & seq bf16 & seq bf16)
  =
  (physical.packed_values, physical.residual_values, physical.delta_values)

inline_for_extraction noextract
let all_rows_projection (x:Shared.inputs) (observed:output_denotation)
  : GTot prop =
  forall (p:Shared.prompt_index) (v:Shared.vocab_index).
    observed.scores p v ==
      Shared.tied_projection x.embedding (observed.hidden p) v

(* The existential is a represented logical observation, not an inverse of
   approximation. No injectivity or comparison preservation is used. *)
inline_for_extraction noextract
let postcondition (#n:nat)
  (norm_offset:weight_offset n Shared.hidden_size)
  (projection_offset:weight_offset n embedding_elements)
  (logical:input_denotation n)
  (before after:input_snapshot n)
  (hidden:lseq bf16 hidden_elements)
  (logits:lseq bf16 logit_elements) : GTot prop =
  let x = map_inputs norm_offset projection_offset logical in
  input_representation before logical /\
  (exists (observed:output_denotation).
    output_representation hidden logits observed /\
    Shared.postcondition x (map_state before)
      (map_outputs observed) (map_state after) /\
    all_rows_projection x observed)

inline_for_extraction noextract
let expected_outputs (x:Shared.inputs) : GTot output_denotation =
  let hidden = Shared.final_rms_norm x in
  {
    hidden = hidden;
    scores = (fun p -> Shared.tied_projection x.embedding (hidden p))
  }

inline_for_extraction noextract
let expected_outputs_refine_shared (#ambient:Type0)
  (x:Shared.inputs) (state:ambient)
  : Lemma
      (Shared.postcondition x state (map_outputs (expected_outputs x)) state /\
       all_rows_projection x (expected_outputs x))
  = ()

inline_for_extraction noextract
let represented_outputs_establish_postcondition (#n:nat)
  (norm_offset:weight_offset n Shared.hidden_size)
  (projection_offset:weight_offset n embedding_elements)
  (logical:input_denotation n)
  (before:input_snapshot n)
  (hidden:lseq bf16 hidden_elements)
  (logits:lseq bf16 logit_elements)
  : Lemma
      (requires
        input_representation before logical /\
        output_representation hidden logits
          (expected_outputs (map_inputs norm_offset projection_offset logical)))
      (ensures
        postcondition norm_offset projection_offset logical
          before before hidden logits)
  =
  expected_outputs_refine_shared
    (map_inputs norm_offset projection_offset logical) (map_state before)

fn kplay_run
  (weights_numel:packed_extent)
  (weights:larray bf16 (U64.v weights_numel){is_global_array weights})
  (residual:larray bf16 hidden_elements{is_global_array residual})
  (delta:larray bf16 hidden_elements{is_global_array delta})
  (norm_offset:weight_offset (U64.v weights_numel) Shared.hidden_size)
  (projection_offset:weight_offset (U64.v weights_numel) embedding_elements)
  (#before:erased (input_snapshot (U64.v weights_numel)))
  (#logical:erased (input_denotation (U64.v weights_numel)))
  (#weights_perm #residual_perm #delta_perm:perm)
  (output_hidden:larray bf16 hidden_elements{is_global_array output_hidden})
  (output_logits:larray bf16 logit_elements{is_global_array output_logits})
  preserves cpu **
    on gpu_loc (weights |-> Frac weights_perm (reveal before).packed_values) **
    on gpu_loc (residual |-> Frac residual_perm (reveal before).residual_values) **
    on gpu_loc (delta |-> Frac delta_perm (reveal before).delta_values)
  requires pure (input_representation (reveal before) (reveal logical))
  requires on gpu_loc (
    exists* (old_hidden:lseq bf16 hidden_elements).
      output_hidden |-> old_hidden)
  requires on gpu_loc (
    exists* (old_logits:lseq bf16 logit_elements).
      output_logits |-> old_logits)
  ensures exists*
    (hidden:lseq bf16 hidden_elements)
    (logits:lseq bf16 logit_elements).
    on gpu_loc (output_hidden |-> hidden ** output_logits |-> logits) **
    pure (postcondition norm_offset projection_offset (reveal logical)
      (reveal before) (reveal before) hidden logits)
