module KPlay.Proof.Optimized.FamilyProjection

#lang-pulse

open Kuiper
module S = FStar.Seq
module R = FStar.Real
module Sqrt = FStar.Math.Sqrt
module I64 = FStar.Int64
module Shared = KPlay.Proof.Shared.Projection

inline_for_extraction noextract
let hidden_width : pos = Shared.reference_hidden_size
inline_for_extraction noextract
let vocabulary_size : pos = Shared.reference_vocab_size
inline_for_extraction noextract
let embedding_elements : pos = vocabulary_size * hidden_width
inline_for_extraction noextract
let weight_elements : pos = embedding_elements + hidden_width
inline_for_extraction noextract
let epsilon : Sqrt.rpos = Shared.reference_epsilon

inline_for_extraction noextract
let valid_position (position prompt_length : int) : prop =
  0 <= position /\ position < 2048 /\
  1 <= prompt_length /\ prompt_length <= 2048

inline_for_extraction noextract
let previous_rows (position prompt_length : int) : nat =
  if position < prompt_length then 0
  else position - prompt_length + 1

inline_for_extraction noextract
let result_rows (position prompt_length : int) : nat =
  if position < prompt_length - 1 then 0
  else position - prompt_length + 2

(* Intermediate algebraic invariants of the reused witness; the exported
   entrypoint below requires the imported shared postcondition. *)
inline_for_extraction noextract
let rec sum_squares
  (hidden : lseq real hidden_width)
  (count : nat{count <= hidden_width})
  : GTot Sqrt.rnonneg (decreases count)
  = if count = 0 then 0.0R
    else
      let x = S.index hidden (count - 1) in
      sum_squares hidden (count - 1) +. x *. x

inline_for_extraction noextract
let rms_scale (hidden : lseq real hidden_width) : GTot Sqrt.rpos =
  Sqrt.rsqrt
    (sum_squares hidden hidden_width /. R.of_int hidden_width +. epsilon)

inline_for_extraction noextract
let normalized_cell
  (weights : lseq real weight_elements)
  (hidden : lseq real hidden_width)
  (column : nat{column < hidden_width})
  : GTot real
  = (S.index hidden column *. rms_scale hidden) *.
      (1.0R +. S.index weights (embedding_elements + column))

inline_for_extraction noextract
let projected_cell
  (weights : lseq real weight_elements)
  (hidden : lseq real hidden_width)
  (row : nat{row < vocabulary_size})
  : GTot real
  = rsum (S.init_ghost hidden_width (fun column ->
      S.index weights (row * hidden_width + column) *.
        normalized_cell weights hidden column))

inline_for_extraction noextract
let projection_post
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (scores : lseq f32 vocabulary_size)
  : prop
  = forall (real_weights : lseq real weight_elements)
           (real_hidden : lseq real hidden_width).
      weights %~ real_weights /\ hidden %~ real_hidden ==>
      (forall (row : nat{row < vocabulary_size}).
        S.index scores row %~ projected_cell real_weights real_hidden row)

inline_for_extraction noextract
let history_model
  (position prompt_length : int)
  (previous : lseq bf16 (previous_rows position prompt_length * hidden_width))
  (hidden : lseq bf16 hidden_width)
  : Tot (lseq bf16 (result_rows position prompt_length * hidden_width))
  = if position < prompt_length - 1 then previous
    else S.append previous hidden

inline_for_extraction noextract
let final_norm_projection_post
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (position prompt_length : int)
  (previous : lseq bf16 (previous_rows position prompt_length * hidden_width))
  (scores : lseq f32 vocabulary_size)
  (history : lseq bf16 (result_rows position prompt_length * hidden_width))
  : prop
  = projection_post weights hidden scores /\
    history == history_model position prompt_length previous hidden

(* Empty histories are inactive pointers, not owned zero-length allocations. *)
inline_for_extraction noextract
let history_ownership
  (buffer : array bf16)
  (rows : nat)
  (fraction : perm)
  (contents : lseq bf16 (rows * hidden_width))
  : slprop
  = if rows = 0 then emp
    else pure (is_global_array buffer) ** buffer |-> Frac fraction contents

inline_for_extraction noextract
let history_extents (position prompt_length : int)
  : Lemma
      (requires valid_position position prompt_length)
      (ensures
        previous_rows position prompt_length <= 2047 /\
        result_rows position prompt_length <= 2048 /\
        (if position < prompt_length - 1 then
          previous_rows position prompt_length == 0 /\
          result_rows position prompt_length == 0
         else
          result_rows position prompt_length ==
            previous_rows position prompt_length + 1)) =
  ()

inline_for_extraction noextract
let vector_view (#a : Type0) (#n : nat) (values : lseq a n)
  : Shared.vector a n =
  fun i -> S.index values i

inline_for_extraction noextract
let norm_weight_view (weights : lseq bf16 weight_elements)
  : Shared.vector bf16 hidden_width =
  fun i -> S.index weights (embedding_elements + i)

inline_for_extraction noextract
let embedding_view (weights : lseq bf16 weight_elements)
  : Shared.matrix bf16 vocabulary_size hidden_width =
  fun token i -> S.index weights (token * hidden_width + i)

inline_for_extraction noextract
let packed_partition ()
  : Lemma (
      embedding_elements + hidden_width == weight_elements /\
      (forall (token : Shared.index vocabulary_size)
              (i : Shared.index hidden_width).
        token * hidden_width + i < embedding_elements) /\
      (forall (i : Shared.index hidden_width).
        embedding_elements <= embedding_elements + i /\
        embedding_elements + i < weight_elements)) =
  ()

inline_for_extraction noextract
let state_view (#state : Type0) (position : i64) (rest : state)
  : i64 & state =
  (position, rest)

(* As in the base representation, early prefill exposes the observation in
   returned read-only hidden ownership, not in the empty history allocation. *)
inline_for_extraction noextract
let observation_view
  (position prompt_length : int)
  (returned_hidden : lseq bf16 hidden_width)
  (history : lseq bf16 (result_rows position prompt_length * hidden_width))
  : Shared.vector bf16 hidden_width =
  if position < prompt_length - 1 then vector_view returned_hidden
  else fun i ->
    S.index history (previous_rows position prompt_length * hidden_width + i)

inline_for_extraction noextract
let history_observation
  (position prompt_length : int)
  (previous : lseq bf16 (previous_rows position prompt_length * hidden_width))
  (hidden : lseq bf16 hidden_width)
  (history : lseq bf16 (result_rows position prompt_length * hidden_width))
  : Lemma
      (requires history == history_model position prompt_length previous hidden)
      (ensures forall (i : Shared.index hidden_width).
        observation_view position prompt_length hidden history i ==
          S.index hidden i) =
  if position < prompt_length - 1 then () else ()

inline_for_extraction noextract
let semantic_postcondition
  (#state : Type0)
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (position prompt_length : i64)
  (previous : lseq bf16
    (previous_rows (I64.v position) (I64.v prompt_length) * hidden_width))
  (before after : state)
  (scores : lseq f32 vocabulary_size)
  (history : lseq bf16
    (result_rows (I64.v position) (I64.v prompt_length) * hidden_width))
  : prop =
  Shared.postcondition
    Shared.reference_epsilon
    (vector_view hidden) (norm_weight_view weights) (embedding_view weights)
    (state_view position before)
    (vector_view scores)
    (observation_view (I64.v position) (I64.v prompt_length) hidden history)
    (state_view position after) /\
  history == history_model (I64.v position) (I64.v prompt_length) previous hidden

inline_for_extraction noextract
let establish_semantic_postcondition
  (#state : Type0)
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (position prompt_length : i64)
  (previous : lseq bf16
    (previous_rows (I64.v position) (I64.v prompt_length) * hidden_width))
  (before after : state)
  (scores : lseq f32 vocabulary_size)
  (history : lseq bf16
    (result_rows (I64.v position) (I64.v prompt_length) * hidden_width))
  : Lemma
      (requires
        (forall (token : Shared.index vocabulary_size).
          S.index scores token %~
            Shared.logical_logits Shared.reference_epsilon
              (vector_view hidden) (norm_weight_view weights)
              (embedding_view weights) token) /\
        history == history_model (I64.v position) (I64.v prompt_length)
          previous hidden /\
        after == before)
      (ensures semantic_postcondition weights hidden position prompt_length
        previous before after scores history) =
  history_observation (I64.v position) (I64.v prompt_length)
    previous hidden history

inline_for_extraction noextract
let refines_shared
  (#state : Type0)
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (position prompt_length : i64)
  (previous : lseq bf16
    (previous_rows (I64.v position) (I64.v prompt_length) * hidden_width))
  (before after : state)
  (scores : lseq f32 vocabulary_size)
  (history : lseq bf16
    (result_rows (I64.v position) (I64.v prompt_length) * hidden_width))
  : Lemma
      (requires semantic_postcondition weights hidden position prompt_length
        previous before after scores history)
      (ensures
        Shared.postcondition
          Shared.reference_epsilon
          (vector_view hidden) (norm_weight_view weights) (embedding_view weights)
          (state_view position before)
          (vector_view scores)
          (observation_view (I64.v position) (I64.v prompt_length) hidden history)
          (state_view position after) /\
        after == before) =
  ()
