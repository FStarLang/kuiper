module KPlay.Functional

#lang-pulse

open Kuiper
module A = Pulse.Lib.Array
module Seq = FStar.Seq
module I64 = FStar.Int64
module BF16 = Kuiper.BFloat16
module Approx = Kuiper.Approximates.Base
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let hidden_size : pos = Shared.reference_hidden_size

inline_for_extraction noextract
let vocab_size : pos = Shared.reference_vocab_size

inline_for_extraction noextract
let embedding_elements : nat = vocab_size * hidden_size

inline_for_extraction noextract
let packed_elements : nat = embedding_elements + hidden_size

inline_for_extraction noextract
let position_t = p:I64.t{0 <= I64.v p /\ I64.v p < 2048}

inline_for_extraction noextract
let prompt_length_t = n:I64.t{1 <= I64.v n /\ I64.v n <= 2048}

inline_for_extraction noextract
let clamped_difference (left right:int) (offset:nat) : nat =
  if left - right + offset > 0 then left - right + offset else 0

inline_for_extraction noextract
let previous_rows (p:position_t) (n:prompt_length_t) : nat =
  clamped_difference (I64.v p) (I64.v n) 1

inline_for_extraction noextract
let emitted_rows (p:position_t) (n:prompt_length_t) : nat =
  clamped_difference (I64.v p) (I64.v n) 2

inline_for_extraction noextract
let emits_observation (p:position_t) (n:prompt_length_t) : bool =
  I64.v p >= I64.v n - 1

inline_for_extraction noextract
let previous_elements (p:position_t) (n:prompt_length_t) : nat =
  previous_rows p n * hidden_size

inline_for_extraction noextract
let emitted_elements (p:position_t) (n:prompt_length_t) : nat =
  emitted_rows p n * hidden_size

inline_for_extraction noextract
let extent_refinement (p:position_t) (n:prompt_length_t)
  : Lemma
      (previous_rows p n <= 2047 /\
       emitted_rows p n <= 2048 /\
       (if emits_observation p n then
          emitted_rows p n == previous_rows p n + 1 /\
          emitted_elements p n == previous_elements p n + hidden_size
        else
          previous_rows p n == 0 /\ emitted_rows p n == 0 /\
          previous_elements p n == 0 /\ emitted_elements p n == 0)) =
  ()

inline_for_extraction noextract
let vector_view (#size:nat) (values:lseq BF16.t size)
  : Shared.vector BF16.t size =
  fun i -> Seq.index values i

inline_for_extraction noextract
let norm_weight_view (packed:lseq BF16.t packed_elements)
  : Shared.vector BF16.t hidden_size =
  fun i -> Seq.index packed (embedding_elements + i)

inline_for_extraction noextract
let embedding_view (packed:lseq BF16.t packed_elements)
  : Shared.matrix BF16.t vocab_size hidden_size =
  fun token i -> Seq.index packed (token * hidden_size + i)

inline_for_extraction noextract
let state_view (#state:Type0) (p:position_t) (rest:state)
  : I64.t & state =
  (p, rest)

inline_for_extraction noextract
let emitted_history
  (p:position_t) (n:prompt_length_t)
  (previous:lseq BF16.t (previous_elements p n))
  (hidden:lseq BF16.t hidden_size)
  : lseq BF16.t (emitted_elements p n) =
  extent_refinement p n;
  if emits_observation p n then Seq.append previous hidden else previous

(* A prefill call retains the current observation in the read-only hidden
   input; emitting calls expose it as the new history tail. *)
inline_for_extraction noextract
let observation_view
  (p:position_t) (n:prompt_length_t)
  (hidden:lseq BF16.t hidden_size)
  (history:lseq BF16.t (emitted_elements p n))
  : Shared.vector BF16.t hidden_size =
  extent_refinement p n;
  if emits_observation p n then
    fun i -> Seq.index history (previous_elements p n + i)
  else
    vector_view hidden

inline_for_extraction noextract
let emitted_history_observation
  (p:position_t) (n:prompt_length_t)
  (previous:lseq BF16.t (previous_elements p n))
  (hidden:lseq BF16.t hidden_size)
  (history:lseq BF16.t (emitted_elements p n))
  : Lemma
      (requires history == emitted_history p n previous hidden)
      (ensures
        forall (i:Shared.index hidden_size).
          observation_view p n hidden history i == Seq.index hidden i) =
  extent_refinement p n;
  if emits_observation p n then () else ()

inline_for_extraction noextract
let semantic_postcondition
  (#state:Type0)
  (p:position_t) (n:prompt_length_t)
  (packed:lseq BF16.t packed_elements)
  (hidden:lseq BF16.t hidden_size)
  (previous:lseq BF16.t (previous_elements p n))
  (before after:state)
  (scores:lseq BF16.t vocab_size)
  (history:lseq BF16.t (emitted_elements p n))
  : prop =
  Shared.postcondition
    Shared.reference_epsilon
    (vector_view hidden) (norm_weight_view packed) (embedding_view packed)
    (state_view p before)
    (vector_view scores) (observation_view p n hidden history)
    (state_view p after) /\
  history == emitted_history p n previous hidden

inline_for_extraction noextract
let establish_semantic_postcondition
  (#state:Type0)
  (p:position_t) (n:prompt_length_t)
  (packed:lseq BF16.t packed_elements)
  (hidden:lseq BF16.t hidden_size)
  (previous:lseq BF16.t (previous_elements p n))
  (before after:state)
  (scores:lseq BF16.t vocab_size)
  (history:lseq BF16.t (emitted_elements p n))
  : Lemma
      (requires
        (forall (token:Shared.index vocab_size).
          Approx.v_approximates (Seq.index scores token)
            (Shared.logical_logits Shared.reference_epsilon
              (vector_view hidden) (norm_weight_view packed)
              (embedding_view packed) token)) /\
        history == emitted_history p n previous hidden /\
        after == before)
      (ensures
        semantic_postcondition p n packed hidden previous before after
          scores history) =
  emitted_history_observation p n previous hidden history

inline_for_extraction noextract
let refines_shared
  (#state:Type0)
  (p:position_t) (n:prompt_length_t)
  (packed:lseq BF16.t packed_elements)
  (hidden:lseq BF16.t hidden_size)
  (previous:lseq BF16.t (previous_elements p n))
  (before after:state)
  (scores:lseq BF16.t vocab_size)
  (history:lseq BF16.t (emitted_elements p n))
  : Lemma
      (requires
        semantic_postcondition p n packed hidden previous before after
          scores history)
      (ensures
        Shared.postcondition
          Shared.reference_epsilon
          (vector_view hidden) (norm_weight_view packed) (embedding_view packed)
          (state_view p before)
          (vector_view scores) (observation_view p n hidden history)
          (state_view p after) /\
        after == before) =
  ()

(* Empty tensor pointers may be null. Pulse array ownership implies non-null,
   so no array resource or location/length fact is required for an empty view. *)
inline_for_extraction noextract
let bf16_buffer (size:nat) =
  ptr:A.array BF16.t{
    size > 0 ==> A.length ptr == size /\ is_global_array ptr}

inline_for_extraction noextract
let buffer_owns
  (#size:nat)
  (ptr:bf16_buffer size) (fraction:perm) (values:lseq BF16.t size)
  : slprop =
  if size = 0 then emp else on gpu_loc (A.pts_to ptr #fraction values)

fn kplay_run
  (#state:Type0)
  (#state_ownership:state -> slprop)
  (#before:erased state)
  (#packed_fraction #hidden_fraction #previous_fraction:perm)
  (position:position_t)
  (prompt_length:prompt_length_t)
  (#packed_values:erased (lseq BF16.t packed_elements))
  (#hidden_values:erased (lseq BF16.t hidden_size))
  (#previous_values:erased
    (lseq BF16.t (previous_elements position prompt_length)))
  (packed:bf16_buffer packed_elements)
  (hidden:bf16_buffer hidden_size)
  (previous:bf16_buffer (previous_elements position prompt_length))
  (scores:bf16_buffer vocab_size)
  (history:bf16_buffer (emitted_elements position prompt_length))
  preserves
    cpu **
    buffer_owns packed packed_fraction packed_values **
    buffer_owns hidden hidden_fraction hidden_values **
    buffer_owns previous previous_fraction previous_values
  requires
    state_ownership before **
    (exists* (initial_scores:lseq BF16.t vocab_size).
      buffer_owns scores 1.0R initial_scores) **
    (exists* (initial_history:lseq BF16.t
        (emitted_elements position prompt_length)).
      buffer_owns history 1.0R initial_history)
  ensures
    exists* (score_values:lseq BF16.t vocab_size)
      (history_values:lseq BF16.t (emitted_elements position prompt_length))
      (after:state).
      buffer_owns scores 1.0R score_values **
      buffer_owns history 1.0R history_values **
      state_ownership after **
      pure (semantic_postcondition position prompt_length
        packed_values hidden_values previous_values before after
        score_values history_values)
