module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Float.Casts

module S = FStar.Seq.Base
module I64 = FStar.Int64
module U64 = FStar.UInt64
module R = FStar.Real
module A = Kuiper.Approximates.Base
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let vocab_size : pos = Shared.vocab_size

inline_for_extraction noextract
let hidden_size : pos = Shared.hidden_size

inline_for_extraction noextract
let weight_cells : pos = vocab_size * hidden_size

inline_for_extraction noextract
let position_limit : pos = 2048

inline_for_extraction noextract
let rotary_half : pos = 32

inline_for_extraction noextract
let rotary_cells : pos = 2 * rotary_half

inline_for_extraction noextract
let state_layers : pos = 18

inline_for_extraction noextract
let recurrent_layer_cells : pos = 262144

inline_for_extraction noextract
let convolution_layer_cells : pos = 24576

inline_for_extraction noextract
let position = p:i64{0 <= I64.v p /\ I64.v p < position_limit}

inline_for_extraction noextract
let initial_recurrent_cells (p:position) : nat =
  if I64.v p = 0 then state_layers * recurrent_layer_cells else 0

inline_for_extraction noextract
let initial_convolution_cells (p:position) : nat =
  if I64.v p = 0 then state_layers * convolution_layer_cells else 0

inline_for_extraction noextract
let flat_weights = w:S.seq bf16{S.length w == weight_cells}

inline_for_extraction noextract
let token_value_valid (x:i64) : prop =
  0 <= I64.v x /\ I64.v x < vocab_size

inline_for_extraction noextract
let prompt_values =
  q:S.seq i64{
    0 < S.length q /\ S.length q <= position_limit /\
    (forall (i:nat{i < S.length q}). token_value_valid (S.index q i))}

inline_for_extraction noextract
let logical_token_value (q:prompt_values) (previous:i64) (p:position) : int =
  if I64.v p < S.length q
  then I64.v (S.index q (I64.v p))
  else I64.v previous

inline_for_extraction noextract
let position_zero_selects_prompt (q:prompt_values) (previous:i64)
  : Lemma (
      logical_token_value q previous 0L == I64.v (S.index q 0))
  = ()

inline_for_extraction noextract
let logical_token_is_valid (q:prompt_values) (previous:i64) (p:position)
  : Lemma
      (requires I64.v p < S.length q \/ token_value_valid previous)
      (ensures
        0 <= logical_token_value q previous p /\
        logical_token_value q previous p < vocab_size)
  = if I64.v p < S.length q then () else ()

inline_for_extraction noextract
let logical_row (w:flat_weights) (t:Shared.token_id)
  : Tot (Shared.hidden_vector bf16)
  = S.slice w (t * hidden_size) ((t + 1) * hidden_size)

inline_for_extraction noextract
let logical_weights (w:flat_weights)
  : Tot (Shared.embedding_weights bf16)
  = S.init vocab_size (logical_row w)

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
      S.index w (t * hidden_size + d))
  = S.init_index_ vocab_size (logical_row w) t;
    S.lemma_index_slice w (t * hidden_size) ((t + 1) * hidden_size) d

inline_for_extraction noextract
let flat_lookup_post
  (w:flat_weights) (t:Shared.token_id) (h:Shared.hidden_vector bf16)
  : prop
  = forall (d:Shared.hidden_index).
      S.index h d == S.index w (t * hidden_size + d)

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
  = S.init_ghost hidden_size
      (fun d -> A.to_real (S.index (S.index (logical_weights w) t) d))

inline_for_extraction noextract
let canonical_weights_model (w:flat_weights)
  : GTot (Shared.embedding_weights R.real)
  = S.init_ghost vocab_size (canonical_row_model w)

inline_for_extraction noextract
let canonical_model_is_total (w:flat_weights)
  : Lemma (weights_model w (canonical_weights_model w))
  = let at_row (t:Shared.token_id)
      : Lemma (
          forall (d:Shared.hidden_index).
            A.v_approximates (S.index (S.index (logical_weights w) t) d)
              (S.index (S.index (canonical_weights_model w) t) d))
      = S.init_ghost_index_ vocab_size (canonical_row_model w) t;
        let at_cell (d:Shared.hidden_index)
          : Lemma (
              A.v_approximates (S.index (S.index (logical_weights w) t) d)
                (S.index (S.index (canonical_weights_model w) t) d))
          = S.init_ghost_index_ hidden_size
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
let lookup_transports_model
  (w:flat_weights)
  (t:Shared.token_id)
  (r:Shared.embedding_weights R.real)
  (h:Shared.hidden_vector bf16)
  : Lemma
      (requires flat_lookup_post w t h /\ weights_model w r)
      (ensures
        Shared.embedding_post (logical_weights w) t (logical_hidden h) /\
        Shared.algebraic_post r t (Shared.algebraic_embedding r t) /\
        hidden_model h (Shared.algebraic_embedding r t))
  = flat_lookup_refines_shared w t h;
    Shared.embedding_post_unique (logical_weights w) t h;
    Shared.embedding_transports_relation
      (fun (x:bf16) (y:R.real) -> A.v_approximates x y)
      (logical_weights w) r t;
    Shared.embedding_correct r t

inline_for_extraction noextract
let inverse_frequency (i:nat{i < rotary_half}) : f32 =
  match i with
  | 0  -> of_literal "1.0"
  | 1  -> of_literal "0.604296386241912841796875"
  | 2  -> of_literal "0.3651741445064544677734375"
  | 3  -> of_literal "0.22067339718341827392578125"
  | 4  -> of_literal "0.13335214555263519287109375"
  | 5  -> of_literal "0.08058421313762664794921875"
  | 6  -> of_literal "0.0486967526376247406005859375"
  | 7  -> of_literal "0.02942727319896221160888671875"
  | 8  -> of_literal "0.01778279431164264678955078125"
  | 9  -> of_literal "0.01074607856571674346923828125"
  | 10 -> of_literal "0.0064938166178762912750244140625"
  | 11 -> of_literal "0.0039241896010935306549072265625"
  | 12 -> of_literal "0.00237137381918728351593017578125"
  | 13 -> of_literal "0.001433012657798826694488525390625"
  | 14 -> of_literal "0.000865964335389435291290283203125"
  | 15 -> of_literal "0.000523299095220863819122314453125"
  | 16 -> of_literal "0.000316227786242961883544921875"
  | 17 -> of_literal "0.000191095299669541418552398681640625"
  | 18 -> of_literal "0.000115478193038143217563629150390625"
  | 19 -> of_literal "0.000069783054641447961330413818359375"
  | 20 -> of_literal "0.0000421696531702764332294464111328125"
  | 21 -> of_literal "0.000025482966520939953625202178955078125"
  | 22 -> of_literal "0.000015399264157167635858058929443359375"
  | 23 -> of_literal "0.00000930572059587575495243072509765625"
  | 24 -> of_literal "0.00000562341347176698036491870880126953125"
  | 25 -> of_literal "0.00000339820826411596499383449554443359375"
  | 26 -> of_literal "0.00000205352489501819945871829986572265625"
  | 27 -> of_literal "0.00000124093776321387849748134613037109375"
  | 28 -> of_literal "0.00000074989424092564149759709835052490234375"
  | 29 -> of_literal "0.000000453158378377338522113859653472900390625"
  | 30 -> of_literal "0.00000027384197665014653466641902923583984375"
  | _  -> of_literal "0.00000016548170833630138076841831207275390625"

inline_for_extraction noextract
let rotary_angle (p:position) (i:nat{i < rotary_half}) : f32 =
  mul (inverse_frequency i) (of_int p)

inline_for_extraction noextract
let rotary_component (p:position) (d:nat{d < rotary_cells}) : bf16 =
  if d < rotary_half
  then fcast (cos (rotary_angle p d))
  else fcast (sin (rotary_angle p (d - rotary_half)))

inline_for_extraction noextract
let rotary_post (p:position) (r:lseq bf16 rotary_cells) : prop =
  forall (d:nat{d < rotary_cells}).
    S.index r d == rotary_component p d

inline_for_extraction noextract
let recurrent_post
  (p:position) (s:lseq f32 (initial_recurrent_cells p))
  : prop
  = forall (i:nat{i < initial_recurrent_cells p}).
      S.index s i == (zero #f32)

inline_for_extraction noextract
let convolution_post
  (p:position) (c:lseq bf16 (initial_convolution_cells p))
  : prop
  = forall (i:nat{i < initial_convolution_cells p}).
      S.index c i == (zero #bf16)

inline_for_extraction noextract
let recurrent_layer
  (s:lseq f32 (initial_recurrent_cells 0L))
  (layer:nat{layer < state_layers})
  : Tot (lseq f32 recurrent_layer_cells)
  = S.slice s (layer * recurrent_layer_cells)
              ((layer + 1) * recurrent_layer_cells)

inline_for_extraction noextract
let convolution_layer
  (c:lseq bf16 (initial_convolution_cells 0L))
  (layer:nat{layer < state_layers})
  : Tot (lseq bf16 convolution_layer_cells)
  = S.slice c (layer * convolution_layer_cells)
              ((layer + 1) * convolution_layer_cells)

inline_for_extraction noextract
let initial_state_slices_are_zero
  (s:lseq f32 (initial_recurrent_cells 0L))
  (c:lseq bf16 (initial_convolution_cells 0L))
  (layer:nat{layer < state_layers})
  (i:nat{i < recurrent_layer_cells})
  (j:nat{j < convolution_layer_cells})
  : Lemma
      (requires recurrent_post 0L s /\ convolution_post 0L c)
      (ensures
        S.index (recurrent_layer s layer) i == (zero #f32) /\
        S.index (convolution_layer c layer) j == (zero #bf16))
  = S.lemma_index_slice s (layer * recurrent_layer_cells)
                         ((layer + 1) * recurrent_layer_cells) i;
    S.lemma_index_slice c (layer * convolution_layer_cells)
                         ((layer + 1) * convolution_layer_cells) j

inline_for_extraction noextract
let semantic_post
  (w:flat_weights)
  (t:Shared.token_id)
  (model:Shared.embedding_weights R.real)
  (p:position)
  (h:Shared.hidden_vector bf16)
  (r:lseq bf16 rotary_cells)
  (s:lseq f32 (initial_recurrent_cells p))
  (c:lseq bf16 (initial_convolution_cells p))
  : prop
  = flat_lookup_post w t h /\
    Shared.embedding_post (logical_weights w) t (logical_hidden h) /\
    Shared.algebraic_post model t (Shared.algebraic_embedding model t) /\
    hidden_model h (Shared.algebraic_embedding model t) /\
    rotary_post p r /\ recurrent_post p s /\ convolution_post p c

(* At position zero this pointer aliases the first prompt cell in the graph. *)
inline_for_extraction noextract
let previous_permission (p:position) (ptr:array i64) (v:i64) : slprop =
  if I64.v p = 0 then emp
  else Pulse.Lib.Array.pts_to ptr #0.5R (S.create 1 v)

fn kplay_run
  (prompt_length:u64{0 < U64.v prompt_length /\ U64.v prompt_length <= position_limit})
  (p:position)
  (#w:erased flat_weights)
  (#q:erased prompt_values{S.length q == U64.v prompt_length})
  (#previous_value:erased i64)
  (#t:erased Shared.token_id)
  (#model:erased (Shared.embedding_weights R.real))
  (weights:larray bf16 weight_cells{is_global_array weights})
  (prompt:larray i64 (U64.v prompt_length){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 hidden_size{is_global_array hidden})
  (rotary:larray bf16 rotary_cells{is_global_array rotary})
  (initial_state:larray f32 (initial_recurrent_cells p){is_global_array initial_state})
  (initial_conv:larray bf16 (initial_convolution_cells p){is_global_array initial_conv})
  preserves cpu **
    on gpu_loc (
      Pulse.Lib.Array.pts_to weights #0.5R (FStar.Ghost.reveal w) **
      Pulse.Lib.Array.pts_to prompt #0.5R (FStar.Ghost.reveal q) **
      previous_permission p previous previous_value)
  requires pure (
    logical_token_value q previous_value p == (FStar.Ghost.reveal t <: int) /\
    weights_model w model)
  requires exists*
    (h0:lseq bf16 hidden_size)
    (r0:lseq bf16 rotary_cells)
    (s0:lseq f32 (initial_recurrent_cells p))
    (c0:lseq bf16 (initial_convolution_cells p)).
      on gpu_loc (
        Pulse.Lib.Array.pts_to hidden #1.0R h0 **
        Pulse.Lib.Array.pts_to rotary #1.0R r0 **
        Pulse.Lib.Array.pts_to initial_state #1.0R s0 **
        Pulse.Lib.Array.pts_to initial_conv #1.0R c0)
  ensures exists*
    (h:Shared.hidden_vector bf16)
    (r:lseq bf16 rotary_cells)
    (s:lseq f32 (initial_recurrent_cells p))
    (c:lseq bf16 (initial_convolution_cells p)).
      on gpu_loc (
        Pulse.Lib.Array.pts_to hidden #1.0R h **
        Pulse.Lib.Array.pts_to rotary #1.0R r **
        Pulse.Lib.Array.pts_to initial_state #1.0R s **
        Pulse.Lib.Array.pts_to initial_conv #1.0R c) **
      pure (semantic_post w t model p h r s c)
