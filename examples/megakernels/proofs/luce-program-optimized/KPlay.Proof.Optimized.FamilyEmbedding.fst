module KPlay.Proof.Optimized.FamilyEmbedding

#lang-pulse

open Kuiper
module S = FStar.Seq.Base
module I64 = FStar.Int64
module R = FStar.Real
module A = Kuiper.Approximates.Base
module PA = Pulse.Lib.Array
module Shared = KPlay.Proof.Shared.Embedding

inline_for_extraction noextract
let embedding_width : pos = Shared.hidden_size

inline_for_extraction noextract
let vocabulary_size : pos = Shared.vocab_size

inline_for_extraction noextract
let embedding_elements : pos = vocabulary_size * embedding_width

inline_for_extraction noextract
let position_limit : pos = 2048

inline_for_extraction noextract
let flat_weights = lseq bf16 embedding_elements

inline_for_extraction noextract
let narrow_token (value:i64) : GTot int =
  let bits = I64.v value % 4294967296 in
  if bits < 2147483648 then bits else bits - 4294967296

inline_for_extraction noextract
let selected_token
  (prompt_length:pos)
  (position:nat)
  (prompt:lseq i64 prompt_length)
  (previous_token:lseq i64 1)
  : GTot int
  = let carried = narrow_token (S.index previous_token 0) in
    if position < prompt_length
    then narrow_token (S.index prompt position)
    else carried

(* The shared ID is the exact selected int64, not its low 32 bits. *)
inline_for_extraction noextract
let logical_token_value
  (prompt_length:pos)
  (position:nat)
  (prompt:lseq i64 prompt_length)
  (previous_token:lseq i64 1)
  : GTot int
  = if position < prompt_length
    then I64.v (S.index prompt position)
    else I64.v (S.index previous_token 0)

inline_for_extraction noextract
let selection_represents
  (prompt_length:pos)
  (position:nat)
  (prompt:lseq i64 prompt_length)
  (previous_token:lseq i64 1)
  (t:Shared.token_id)
  : prop
  = logical_token_value prompt_length position prompt previous_token == (t <: int)

inline_for_extraction noextract
let narrow_token_exact
  (value:i64{0 <= I64.v value /\ I64.v value < vocabulary_size})
  : Lemma (narrow_token value == I64.v value)
  = ()

inline_for_extraction noextract
let selection_refines
  (prompt_length:pos)
  (position:nat)
  (prompt:lseq i64 prompt_length)
  (previous_token:lseq i64 1)
  (t:Shared.token_id)
  : Lemma
      (requires selection_represents prompt_length position prompt previous_token t)
      (ensures selected_token prompt_length position prompt previous_token == (t <: int))
  = if position < prompt_length
    then narrow_token_exact (S.index prompt position)
    else narrow_token_exact (S.index previous_token 0)

inline_for_extraction noextract
let encode_token (t:Shared.token_id) : Tot (x:i64{I64.v x == (t <: int)})
  = assert_norm (pow2 63 == 9223372036854775808);
    I64.int_to_t t

inline_for_extraction noextract
let every_token_at_every_position
  (t:Shared.token_id) (n:pos{n <= position_limit})
  (p:nat{p < position_limit})
  : Lemma (
      selection_represents n p (S.create n (encode_token t))
        (S.create 1 (encode_token t)) t /\
      selected_token n p (S.create n (encode_token t))
        (S.create 1 (encode_token t)) == (t <: int))
  = selection_refines n p (S.create n (encode_token t))
      (S.create 1 (encode_token t)) t

inline_for_extraction noextract
let logical_row (w:flat_weights) (t:Shared.token_id)
  : Tot (Shared.hidden_vector bf16)
  = S.slice w (t * embedding_width) ((t + 1) * embedding_width)

inline_for_extraction noextract
let logical_weights (w:flat_weights)
  : Tot (Shared.embedding_weights bf16)
  = S.init vocabulary_size (logical_row w)

inline_for_extraction noextract
let logical_hidden (h:Shared.hidden_vector bf16)
  : Tot (Shared.hidden_vector bf16)
  = h

inline_for_extraction noextract
let logical_state : unit = ()

inline_for_extraction noextract
let logical_weights_cell
  (w:flat_weights) (t:Shared.token_id) (d:Shared.hidden_index)
  : Lemma (
      S.index (S.index (logical_weights w) t) d ==
      S.index w (t * embedding_width + d))
  = S.init_index_ vocabulary_size (logical_row w) t;
    S.lemma_index_slice w (t * embedding_width) ((t + 1) * embedding_width) d

inline_for_extraction noextract
let flatten_cell
  (e:Shared.embedding_weights bf16)
  (i:nat{i < embedding_elements})
  : Tot bf16
  = S.index (S.index e (i / embedding_width)) (i % embedding_width)

inline_for_extraction noextract
let flatten_weights (e:Shared.embedding_weights bf16) : Tot flat_weights
  = S.init embedding_elements (flatten_cell e)

inline_for_extraction noextract
let flatten_preserves_cell
  (e:Shared.embedding_weights bf16) (t:Shared.token_id) (d:Shared.hidden_index)
  : Lemma (
      S.index (S.index (logical_weights (flatten_weights e)) t) d ==
      S.index (S.index e t) d)
  = logical_weights_cell (flatten_weights e) t d;
    S.init_index_ embedding_elements (flatten_cell e) (t * embedding_width + d)

inline_for_extraction noextract
let every_table_is_representable (e:Shared.embedding_weights bf16)
  : Lemma (logical_weights (flatten_weights e) == e)
  = let at_row (t:Shared.token_id)
      : Lemma (
          S.index (logical_weights (flatten_weights e)) t == S.index e t)
      = let at_cell (d:Shared.hidden_index)
          : Lemma (
              S.index (S.index (logical_weights (flatten_weights e)) t) d ==
              S.index (S.index e t) d)
          = flatten_preserves_cell e t d
        in FStar.Classical.forall_intro at_cell;
        S.lemma_eq_intro (S.index (logical_weights (flatten_weights e)) t) (S.index e t);
        S.lemma_eq_elim (S.index (logical_weights (flatten_weights e)) t) (S.index e t)
    in FStar.Classical.forall_intro at_row;
    S.lemma_eq_intro (logical_weights (flatten_weights e)) e;
    S.lemma_eq_elim (logical_weights (flatten_weights e)) e

inline_for_extraction noextract
let flat_lookup_post
  (w:flat_weights) (t:Shared.token_id) (h:Shared.hidden_vector bf16)
  : prop
  = forall (d:Shared.hidden_index).
      S.index h d == S.index w (t * embedding_width + d)

inline_for_extraction noextract
let flat_lookup_refines_shared
  (w:flat_weights) (t:Shared.token_id) (h:Shared.hidden_vector bf16)
  : Lemma
      (requires flat_lookup_post w t h)
      (ensures Shared.embedding_post (logical_weights w) t (logical_hidden h))
  = let at_index (d:Shared.hidden_index)
      : Lemma (
          S.index h d == S.index (S.index (logical_weights w) t) d)
      = logical_weights_cell w t d
    in FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let weights_model
  (w:flat_weights) (r:Shared.embedding_weights R.real)
  : prop
  = forall (t:Shared.token_id) (d:Shared.hidden_index).
      A.v_approximates (S.index (S.index (logical_weights w) t) d)
                      (S.index (S.index r t) d)

inline_for_extraction noextract
let canonical_row_model (w:flat_weights) (t:Shared.token_id)
  : GTot (Shared.hidden_vector R.real)
  = S.init_ghost embedding_width
      (fun d -> A.to_real (S.index (S.index (logical_weights w) t) d))

inline_for_extraction noextract
let canonical_weights_model (w:flat_weights)
  : GTot (Shared.embedding_weights R.real)
  = S.init_ghost vocabulary_size (canonical_row_model w)

inline_for_extraction noextract
let canonical_model_is_total (w:flat_weights)
  : Lemma (weights_model w (canonical_weights_model w))
  = let at_row (t:Shared.token_id)
      : Lemma (
          forall (d:Shared.hidden_index).
            A.v_approximates (S.index (S.index (logical_weights w) t) d)
              (S.index (S.index (canonical_weights_model w) t) d))
      = S.init_ghost_index_ vocabulary_size (canonical_row_model w) t;
        let at_cell (d:Shared.hidden_index)
          : Lemma (
              A.v_approximates (S.index (S.index (logical_weights w) t) d)
                (S.index (S.index (canonical_weights_model w) t) d))
          = S.init_ghost_index_ embedding_width
              (fun j -> A.to_real (S.index (S.index (logical_weights w) t) j)) d;
            A.to_real_ok (S.index (S.index (logical_weights w) t) d)
        in FStar.Classical.forall_intro at_cell
    in FStar.Classical.forall_intro at_row

inline_for_extraction noextract
let hidden_model
  (h:Shared.hidden_vector bf16) (r:Shared.hidden_vector R.real)
  : prop
  = forall (d:Shared.hidden_index).
      A.v_approximates (S.index h d) (S.index r d)

inline_for_extraction noextract
let semantic_post
  (w:flat_weights) (t:Shared.token_id)
  (model:Shared.embedding_weights R.real) (h:Shared.hidden_vector bf16)
  : prop
  = flat_lookup_post w t h /\
    Shared.embedding_post (logical_weights w) t (logical_hidden h) /\
    Shared.algebraic_post model t (Shared.algebraic_embedding model t) /\
    hidden_model h (Shared.algebraic_embedding model t)

inline_for_extraction noextract
let lookup_transports_model
  (w:flat_weights) (t:Shared.token_id)
  (model:Shared.embedding_weights R.real) (h:Shared.hidden_vector bf16)
  : Lemma
      (requires flat_lookup_post w t h /\ weights_model w model)
      (ensures semantic_post w t model h)
  = flat_lookup_refines_shared w t h;
    Shared.embedding_post_unique (logical_weights w) t h;
    Shared.embedding_transports_relation
      (fun (x:bf16) (y:R.real) -> A.v_approximates x y)
      (logical_weights w) model t;
    Shared.embedding_correct model t
