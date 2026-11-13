module KPlay.Functional

#lang-pulse

open FStar.Real

module Shared = KPlay.SharedSpec
module S = FStar.Seq.Base
module C = FStar.Classical
module N = Kuiper.Approximates.Base
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module NativeNeg = Kuiper.Float32.NativeNeg
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module I64 = FStar.Int64
module Pow = FStar.Math.Pow
module Root = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry
module Sum = Kuiper.Sum
module PC = Pulse.Lib.Core
module PA = Pulse.Lib.Array
module KA = Kuiper.Array.Core
module KL = Kuiper.Locs
module Permission = PulseCore.FractionalPermission

open Pulse.Lib.Core

(* Composition support, not program_refines. The reviewed joint numerical
   adapter supersedes the former rotary gap; native caller obligations remain. *)

inline_for_extraction noextract
type values (a:Type0) (n:nat) = s:S.seq a{S.length s == n}

inline_for_extraction noextract
let embedding_elements : pos = 151936 * 1024

inline_for_extraction noextract
let suffix_weight_elements : pos = embedding_elements + 1024

inline_for_extraction noextract
let layer_weight_elements : pos = 15730944

inline_for_extraction noextract
let weight_elements : pos = 596049920

inline_for_extraction noextract
let layer_start (layer:Shared.index 28) : nat =
  suffix_weight_elements + layer * layer_weight_elements

inline_for_extraction noextract
let source_spans (layer:Shared.index 28)
  : Lemma (
      embedding_elements == 155582464 /\
      suffix_weight_elements == 155583488 /\
      suffix_weight_elements + 28 * layer_weight_elements == weight_elements /\
      weight_elements <= 2147483647 /\
      layer_start layer + layer_weight_elements <= weight_elements /\
      2048 + 4096 * 1024 <= layer_weight_elements /\
      6293504 + 256 <= layer_weight_elements /\
      4196352 + 1024 * 2048 <= layer_weight_elements /\
      6293760 + 6144 * 1024 <= layer_weight_elements /\
      12585216 + 1024 * 3072 == layer_weight_elements) = ()

inline_for_extraction noextract
let valid_prompt (prompt:values I64.t 7) : prop =
  forall (p:Shared.index 7).
    0 <= I64.v (S.index prompt p) /\ I64.v (S.index prompt p) < 151936

inline_for_extraction noextract
noeq type source_inputs = {
  weights: values BF16.t weight_elements;
  prompt: p:values I64.t 7{valid_prompt p};
  steps: n:I64.t{I64.v n == 1}
}

inline_for_extraction noextract
let hidden_view (v:values BF16.t (7 * 1024)) : GTot (Shared.hidden BF16.t) =
  fun p d -> S.index v (p * 1024 + d)

inline_for_extraction noextract
let rotary_view (v:values BF16.t 1792) : GTot (Shared.rotary_tables BF16.t) = {
  cosine = (fun p d -> S.index v (p * 128 + d));
  sine = (fun p d -> S.index v (896 + p * 128 + d))
}

inline_for_extraction noextract
let source_layer_weights (input:source_inputs) (layer:Shared.index 28)
  : GTot (values BF16.t layer_weight_elements) =
  S.slice input.weights (layer_start layer) (layer_start layer + layer_weight_elements)

inline_for_extraction noextract
let source_parameters (input:source_inputs) (layer:Shared.index 28)
  : GTot (Shared.parameters BF16.t) =
  let b = layer_start layer in {
    input_layernorm = (fun d -> S.index input.weights (b + d));
    q_proj = (fun r d -> S.index input.weights (b + 2048 + r * 1024 + d));
    k_proj = (fun r d -> S.index input.weights (b + 2048 + (2048 + r) * 1024 + d));
    v_proj = (fun r d -> S.index input.weights (b + 2048 + (3072 + r) * 1024 + d));
    q_norm = (fun d -> S.index input.weights (b + 6293504 + d));
    k_norm = (fun d -> S.index input.weights (b + 6293504 + 128 + d));
    o_proj = (fun r d -> S.index input.weights (b + 4196352 + r * 2048 + d));
    post_attention_layernorm = (fun d -> S.index input.weights (b + 1024 + d));
    gate_proj = (fun r d -> S.index input.weights (b + 6293760 + r * 1024 + d));
    up_proj = (fun r d -> S.index input.weights (b + 6293760 + (3072 + r) * 1024 + d));
    down_proj = (fun r d -> S.index input.weights (b + 12585216 + r * 3072 + d))
  }

inline_for_extraction noextract
let input_mapping (input:source_inputs) : GTot Shared.inputs = {
  weight = {
    embedding = (fun v d -> S.index input.weights (v * 1024 + d));
    final_norm = (fun d -> S.index input.weights (embedding_elements + d));
    layers = source_parameters input
  };
  prompt = S.init #Shared.token_id 7 (fun p -> I64.v (S.index input.prompt p));
  decode_steps = 1
}

inline_for_extraction noextract
noeq type decoder_snapshot = {
  combined: values BF16.t 7168;
  residual: values BF16.t 7168;
  delta: values BF16.t 7168;
  qkv: values BF16.t 28672;
  qk: values BF16.t 21504
}

inline_for_extraction noextract
noeq type source_snapshots = {
  embedded: values BF16.t 7168;
  rotary: values BF16.t 1792;
  layer: Shared.index 28 -> GTot decoder_snapshot;
  normalized: values BF16.t 7168;
  logits: values BF16.t (7 * 151936);
  returned: values I64.t 8
}

inline_for_extraction noextract
let observed_keys (layer:decoder_snapshot) : GTot (Shared.heads BF16.t 8) =
  fun p h d -> S.index layer.qk (p * 3072 + 2048 + h * 128 + d)

inline_for_extraction noextract
let observed_values (layer:decoder_snapshot) : GTot (Shared.heads BF16.t 8) =
  fun p h d -> S.index layer.qkv (p * 4096 + 3072 + h * 128 + d)

inline_for_extraction noextract
let decoder_input_pair (run:source_snapshots) (layer:Shared.index 28)
  : GTot (values BF16.t 7168 & values BF16.t 7168) =
  if layer = 0 then (run.embedded, run.embedded)
  else ((run.layer (layer - 1)).residual, (run.layer (layer - 1)).delta)

inline_for_extraction noextract
let decoder_first_call_alias (run:source_snapshots)
  : Lemma (decoder_input_pair run 0 == (run.embedded, run.embedded)) = ()

inline_for_extraction noextract
let decoder_successor_feedback (run:source_snapshots) (layer:Shared.index 27)
  : Lemma (decoder_input_pair run (layer + 1) ==
      ((run.layer layer).residual, (run.layer layer).delta)) = ()

inline_for_extraction noextract
noeq type suffix_snapshot = {
  packed_values: values BF16.t suffix_weight_elements;
  residual_values: values BF16.t 7168;
  delta_values: values BF16.t 7168
}

inline_for_extraction noextract
noeq type suffix_denotation = {
  packed: Shared.vector real suffix_weight_elements;
  residual: Shared.hidden real;
  delta: Shared.hidden real
}

inline_for_extraction noextract
noeq type suffix_output = {
  hidden_values: values BF16.t 7168;
  logit_values: values BF16.t (7 * 151936)
}

inline_for_extraction noextract
noeq type suffix_observation = {
  hidden: Shared.hidden real;
  scores: Shared.matrix real 7 151936
}

inline_for_extraction noextract
let source_suffix_input (input:source_inputs) (run:source_snapshots)
  : GTot suffix_snapshot = {
  packed_values = S.slice input.weights 0 suffix_weight_elements;
  residual_values = (run.layer 27).residual;
  delta_values = (run.layer 27).delta
}

inline_for_extraction noextract
let source_suffix_output (run:source_snapshots) : Tot suffix_output = {
  hidden_values = run.normalized;
  logit_values = run.logits
}

inline_for_extraction noextract
let embedding_view (v:values BF16.t suffix_weight_elements)
  : GTot (Shared.matrix BF16.t 151936 1024) =
  fun token d -> S.index v (token * 1024 + d)

inline_for_extraction noextract
let logit_view (v:values BF16.t (7 * 151936))
  : GTot (Shared.matrix BF16.t 7 151936) =
  fun p token -> S.index v (p * 151936 + token)

inline_for_extraction noextract
let actual_scores (physical:suffix_output) : GTot (Shared.vector F32.t 151936) =
  fun token -> Cast.cast_bf16_to_f32 (logit_view physical.logit_values 6 token)

inline_for_extraction noextract
let suffix_input_representation (physical:suffix_snapshot) (logical:suffix_denotation)
  : prop =
  (forall (i:Shared.index suffix_weight_elements).
    N.v_approximates (S.index physical.packed_values i) (logical.packed i)) /\
  (forall (p:Shared.index 7) (d:Shared.index 1024).
    N.v_approximates (S.index physical.residual_values (p * 1024 + d))
      (logical.residual p d)) /\
  (forall (p:Shared.index 7) (d:Shared.index 1024).
    N.v_approximates (S.index physical.delta_values (p * 1024 + d))
      (logical.delta p d))

inline_for_extraction noextract
let canonical_suffix_denotation (physical:suffix_snapshot) : GTot suffix_denotation = {
  packed = (fun i -> N.to_real (S.index physical.packed_values i));
  residual = (fun p d -> N.to_real (S.index physical.residual_values (p * 1024 + d)));
  delta = (fun p d -> N.to_real (S.index physical.delta_values (p * 1024 + d)))
}

inline_for_extraction noextract
let canonical_suffix_represented (physical:suffix_snapshot)
  : Lemma (suffix_input_representation physical (canonical_suffix_denotation physical)) =
  C.forall_intro #(Shared.index suffix_weight_elements)
    #(fun i -> N.v_approximates (S.index physical.packed_values i)
      (N.to_real (S.index physical.packed_values i)))
    (fun i -> N.to_real_ok (S.index physical.packed_values i));
  Shared.real_matrix_related (hidden_view physical.residual_values);
  Shared.real_matrix_related (hidden_view physical.delta_values)

inline_for_extraction noextract
let suffix_embedding (logical:suffix_denotation)
  : GTot (Shared.matrix real 151936 1024) =
  fun token d -> logical.packed (token * 1024 + d)

inline_for_extraction noextract
let suffix_hidden_input (logical:suffix_denotation) : GTot (Shared.hidden real) =
  fun p d -> logical.residual p d +. logical.delta p d

inline_for_extraction noextract
let suffix_norm_weight (logical:suffix_denotation) : GTot (Shared.vector real 1024) =
  fun d -> logical.packed (embedding_elements + d)

inline_for_extraction noextract
let suffix_normalized (logical:suffix_denotation) : GTot (Shared.hidden real) =
  Shared.normalize_hidden (suffix_norm_weight logical) (suffix_hidden_input logical)

inline_for_extraction noextract
let legacy_sum_hidden (f:Shared.index 1024 -> GTot real) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 1024 f

inline_for_extraction noextract
let rec legacy_sum_nonnegative
  (f:Shared.index 1024 -> GTot Root.rnonneg) (lo:nat{lo <= 1024})
  : Lemma
      (Sum.sum #real #Sum.real_add_semigroup lo 1024 f >=. 0.0R)
      (decreases (1024 - lo)) =
  if lo < 1024 then (
    legacy_sum_nonnegative f (lo + 1);
    Sum.sum_pop_left #real #Sum.real_add_semigroup lo 1024 f;
    let tail = Sum.sum #real #Sum.real_add_semigroup (lo + 1) 1024 f in
    assert_norm (Sum.op #real #Sum.real_add_semigroup (f lo) tail == f lo +. tail)
  ) else (
    assert_norm (Sum.id #real #Sum.real_add_semigroup == 0.0R))

inline_for_extraction noextract
let legacy_mean_square (row:Shared.vector real 1024) : GTot Root.rnonneg =
  let squares : Shared.index 1024 -> GTot Root.rnonneg = fun d -> row d *. row d in
  legacy_sum_nonnegative squares 0;
  legacy_sum_hidden squares /. FStar.Real.of_int 1024

inline_for_extraction noextract
let legacy_normalization_scale (row:Shared.vector real 1024) : GTot Root.rpos =
  Root.rsqrt (legacy_mean_square row +. 0.000001R)

inline_for_extraction noextract
let legacy_tied_projection
  (embedding:Shared.matrix real 151936 1024) (row:Shared.vector real 1024)
  : GTot (Shared.vector real 151936) =
  fun token -> legacy_sum_hidden (fun d -> row d *. embedding token d)

inline_for_extraction noextract
let projection_equations_agree
  (embedding:Shared.matrix real 151936 1024) (row:Shared.vector real 1024)
  (token:Shared.token_id)
  : Lemma (legacy_tied_projection embedding row token ==
      Shared.linear #1024 #151936 embedding row token) =
  assert (legacy_tied_projection embedding row token ==
      Shared.linear #1024 #151936 embedding row token) by
    (FStar.Tactics.V2.norm
      [delta_only [`%legacy_tied_projection; `%legacy_sum_hidden;
        `%Shared.linear; `%Shared.dot; `%Shared.sum]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let legacy_normalized (logical:suffix_denotation) : GTot (Shared.hidden real) =
  fun p ->
    let row = suffix_hidden_input logical p in
    let scale = legacy_normalization_scale row in
    fun d -> suffix_norm_weight logical d *. (row d *. scale)

inline_for_extraction noextract
let normalization_equations_agree (logical:suffix_denotation)
  (p:Shared.index 7) (d:Shared.index 1024)
  : Lemma (legacy_normalized logical p d == suffix_normalized logical p d) =
  assert (legacy_normalized logical p d == suffix_normalized logical p d) by
    (FStar.Tactics.V2.norm
      [delta_only [`%legacy_normalized; `%legacy_normalization_scale;
        `%legacy_mean_square; `%legacy_sum_hidden; `%suffix_normalized;
        `%Shared.normalize_hidden; `%Shared.rms_norm; `%Shared.sum]];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let suffix_map_state (physical:suffix_snapshot)
  : Tot (S.seq BF16.t & S.seq BF16.t & S.seq BF16.t) =
  (physical.packed_values, physical.residual_values, physical.delta_values)

inline_for_extraction noextract
let suffix_family_postcondition
  (logical:suffix_denotation) (before after:suffix_snapshot)
  (physical:suffix_output) : prop =
  suffix_input_representation before logical /\
  (exists (observed:suffix_observation).
    (forall (p:Shared.index 7) (d:Shared.index 1024).
      N.v_approximates (S.index physical.hidden_values (p * 1024 + d))
        (observed.hidden p d)) /\
    (forall (p:Shared.index 7) (token:Shared.token_id).
      N.v_approximates (S.index physical.logit_values (p * 151936 + token))
        (observed.scores p token)) /\
    (forall (p:Shared.index 7) (d:Shared.index 1024).
      observed.hidden p d == legacy_normalized logical p d) /\
    (forall (token:Shared.token_id).
      observed.scores 6 token ==
        legacy_tied_projection (suffix_embedding logical) (legacy_normalized logical 6) token) /\
    suffix_map_state after == suffix_map_state before /\
    (forall (p:Shared.index 7) (token:Shared.token_id).
      observed.scores p token ==
        legacy_tied_projection (suffix_embedding logical) (observed.hidden p) token))

(* Specialization of the reviewed, newly proved postcondition, not of the
   historical suffix receipt. Its exact definition binding is in composition.json. *)
inline_for_extraction noextract
let reviewed_universal_projection (before:suffix_snapshot) (physical:suffix_output)
  : prop =
  forall (embedding:Shared.matrix real 151936 1024) (rows:Shared.hidden real).
    (forall (token:Shared.token_id) (d:Shared.index 1024).
      N.v_approximates (S.index before.packed_values (token * 1024 + d))
        (embedding token d)) /\
    (forall (p:Shared.index 7) (d:Shared.index 1024).
      N.v_approximates (S.index physical.hidden_values (p * 1024 + d)) (rows p d)) ==>
    (forall (p:Shared.index 7) (token:Shared.token_id).
      N.v_approximates (S.index physical.logit_values (p * 151936 + token))
        (legacy_tied_projection embedding (rows p) token))

inline_for_extraction noextract
let reviewed_projection_bridge (before:suffix_snapshot) (physical:suffix_output)
  : Lemma
      (requires reviewed_universal_projection before physical)
      (ensures Shared.projection_relation (embedding_view before.packed_values)
        (hidden_view physical.hidden_values) (actual_scores physical)) =
  let related (embedding:Shared.matrix real 151936 1024) (rows:Shared.hidden real)
    : Lemma
        (requires Shared.matrix_approximates (embedding_view before.packed_values) embedding /\
          Shared.matrix_approximates (hidden_view physical.hidden_values) rows)
        (ensures Shared.vector_approximates (actual_scores physical)
          (Shared.linear #1024 #151936 embedding (rows 6))) =
    assert (forall (token:Shared.token_id) (d:Shared.index 1024).
      N.v_approximates (S.index before.packed_values (token * 1024 + d))
        (embedding token d));
    assert (forall (p:Shared.index 7) (d:Shared.index 1024).
      N.v_approximates (S.index physical.hidden_values (p * 1024 + d)) (rows p d));
    let cell (token:Shared.token_id)
      : Lemma (N.v_approximates (actual_scores physical token)
          (Shared.linear #1024 #151936 embedding (rows 6) token)) =
      projection_equations_agree embedding (rows 6) token;
      Cast.cast_bf16_to_f32_ok (S.index physical.logit_values (6 * 151936 + token))
        (legacy_tied_projection embedding (rows 6) token)
    in C.forall_intro cell
  in C.forall_intro_2 (fun embedding -> C.move_requires (related embedding))

inline_for_extraction noextract
let suffix_completed_residual (before:suffix_snapshot)
  : GTot (values BF16.t 7168) =
  S.init_ghost 7168 (fun i ->
    Cast.cast_f32_to_bf16
      (F.add (Cast.cast_bf16_to_f32 (S.index before.residual_values i))
        (Cast.cast_bf16_to_f32 (S.index before.delta_values i))))

inline_for_extraction noextract
let suffix_native_norm_weight (before:suffix_snapshot)
  : GTot (Shared.vector BF16.t 1024) =
  fun d -> S.index before.packed_values (embedding_elements + d)

inline_for_extraction noextract
let reviewed_suffix_postcondition
  (logical:suffix_denotation) (before after:suffix_snapshot)
  (physical:suffix_output) : prop =
  suffix_family_postcondition logical before after physical /\
  reviewed_universal_projection before physical /\
  Shared.projection_relation (embedding_view before.packed_values)
    (hidden_view physical.hidden_values) (actual_scores physical) /\
  Shared.final_norm_relation (suffix_native_norm_weight before)
    (hidden_view (suffix_completed_residual before))
    (hidden_view physical.hidden_values)

inline_for_extraction noextract
let source_embedding_adapter (input:source_inputs) (run:source_snapshots)
  (embedding:Shared.matrix real 151936 1024)
  : Lemma (
      Shared.matrix_approximates (input_mapping input).weight.embedding embedding <==>
      Shared.matrix_approximates
        (embedding_view (source_suffix_input input run).packed_values) embedding) = ()

inline_for_extraction noextract
let source_suffix_caller (input:source_inputs) (run:source_snapshots)
  : Lemma (
      suffix_input_representation (source_suffix_input input run)
        (canonical_suffix_denotation (source_suffix_input input run)) /\
      0 + embedding_elements <= suffix_weight_elements /\
      embedding_elements + 1024 <= suffix_weight_elements /\
      suffix_weight_elements <= 2147483647) =
  canonical_suffix_represented (source_suffix_input input run)

inline_for_extraction noextract
let source_suffix_projection (input:source_inputs) (run:source_snapshots)
  : Lemma
      (requires reviewed_suffix_postcondition
        (canonical_suffix_denotation (source_suffix_input input run))
        (source_suffix_input input run) (source_suffix_input input run)
        (source_suffix_output run))
      (ensures Shared.projection_relation (input_mapping input).weight.embedding
        (hidden_view run.normalized) (actual_scores (source_suffix_output run))) =
  source_suffix_caller input run;
  let related (embedding:Shared.matrix real 151936 1024) (rows:Shared.hidden real)
    : Lemma
        (requires Shared.matrix_approximates (input_mapping input).weight.embedding embedding /\
          Shared.matrix_approximates (hidden_view run.normalized) rows)
        (ensures Shared.vector_approximates (actual_scores (source_suffix_output run))
          (Shared.linear #1024 #151936 embedding (rows 6))) =
    source_embedding_adapter input run embedding
  in C.forall_intro_2 (fun embedding -> C.move_requires (related embedding))

inline_for_extraction noextract
noeq type suffix_addresses = {
  weights: a:PA.larray BF16.t suffix_weight_elements{KA.is_global_array a};
  residual: a:PA.larray BF16.t 7168{KA.is_global_array a};
  delta: a:PA.larray BF16.t 7168{KA.is_global_array a};
  normalized: a:PA.larray BF16.t 7168{KA.is_global_array a};
  logits: a:PA.larray BF16.t (7 * 151936){KA.is_global_array a};
  weights_permission: Permission.perm;
  residual_permission: Permission.perm;
  delta_permission: Permission.perm
}

inline_for_extraction noextract
let suffix_read_frame (addresses:suffix_addresses) (before:suffix_snapshot) : PC.slprop =
  KL.cpu **
  PC.on KL.gpu_loc
    (PA.pts_to addresses.weights #addresses.weights_permission before.packed_values) **
  PC.on KL.gpu_loc
    (PA.pts_to addresses.residual #addresses.residual_permission before.residual_values) **
  PC.on KL.gpu_loc
    (PA.pts_to addresses.delta #addresses.delta_permission before.delta_values)

inline_for_extraction noextract
let source_suffix_pre (input:source_inputs) (run:source_snapshots)
  (addresses:suffix_addresses) (old:suffix_output) : PC.slprop =
  suffix_read_frame addresses (source_suffix_input input run) **
  PC.on KL.gpu_loc (PA.pts_to addresses.normalized old.hidden_values) **
  PC.on KL.gpu_loc (PA.pts_to addresses.logits old.logit_values) **
  PC.pure (suffix_input_representation (source_suffix_input input run)
    (canonical_suffix_denotation (source_suffix_input input run)))

inline_for_extraction noextract
let source_suffix_post (input:source_inputs) (run:source_snapshots)
  (addresses:suffix_addresses) : PC.slprop =
  suffix_read_frame addresses (source_suffix_input input run) **
  PC.on KL.gpu_loc (PA.pts_to addresses.normalized run.normalized) **
  PC.on KL.gpu_loc (PA.pts_to addresses.logits run.logits) **
  PC.pure (reviewed_suffix_postcondition
    (canonical_suffix_denotation (source_suffix_input input run))
    (source_suffix_input input run) (source_suffix_input input run)
    (source_suffix_output run))

(* Exact accepted decoder attention-data definitions, renamed locally. *)
inline_for_extraction noextract
noeq type decoder_attention_data = {
  position_ids: Shared.index 7 -> int;
  cache_positions: Shared.index 7 -> int;
  cosine: Shared.matrix real 7 128;
  sine: Shared.matrix real 7 128;
  additive_mask: Shared.matrix real 7 7
}

inline_for_extraction noextract
let decoder_inverse_frequency (lane:Shared.index 128) : real =
  1.0R /. Pow.pow 1000000.0R (of_int (2 * (lane % 64)) /. 128.0R)

inline_for_extraction noextract
let decoder_rope_angle (position:int) (lane:Shared.index 128) : real =
  of_int position *. decoder_inverse_frequency lane

inline_for_extraction noextract
let decoder_prefill_cos (token:Shared.index 7) (lane:Shared.index 128) : real =
  Trig.cos (decoder_rope_angle token lane)

inline_for_extraction noextract
let decoder_prefill_sin (token:Shared.index 7) (lane:Shared.index 128) : real =
  Trig.sin (decoder_rope_angle token lane)

inline_for_extraction noextract
let decoder_bf16_minimum () : real =
  of_int (-338953138925153547590470800371487866880)

inline_for_extraction noextract
let decoder_prefill_mask (query key:Shared.index 7) : real =
  if key <= query then 0.0R else decoder_bf16_minimum ()

inline_for_extraction noextract
let decoder_valid_attention_data (data:decoder_attention_data) : prop =
  (forall (token:Shared.index 7).
    data.position_ids token == token /\ data.cache_positions token == token) /\
  (forall (token:Shared.index 7) (lane:Shared.index 128).
    data.cosine token lane == decoder_prefill_cos token lane /\
    data.sine token lane == decoder_prefill_sin token lane) /\
  (forall (query key:Shared.index 7).
    data.additive_mask query key == decoder_prefill_mask query key)

inline_for_extraction noextract
let decoder_attention_view (tables:Shared.rotary_tables real)
  : GTot decoder_attention_data = {
  position_ids = (fun token -> token);
  cache_positions = (fun token -> token);
  cosine = tables.cosine;
  sine = tables.sine;
  additive_mask = decoder_prefill_mask
}

inline_for_extraction noextract
let decoder_rotary_representation (storage:values BF16.t 1792)
  (logical:decoder_attention_data) : prop =
  forall (token:Shared.index 7) (lane:Shared.index 128).
    N.v_approximates (S.index storage (token * 128 + lane)) (logical.cosine token lane) /\
    N.v_approximates (S.index storage (896 + token * 128 + lane)) (logical.sine token lane)

inline_for_extraction noextract
let decoder_rotary_aliases (token:Shared.index 7) (lane:Shared.index 128)
  : Lemma (
      decoder_prefill_cos token lane == (Shared.canonical_rotary ()).cosine token lane /\
      decoder_prefill_sin token lane == (Shared.canonical_rotary ()).sine token lane) = ()

inline_for_extraction noextract
let decoder_rotary_adapter (storage:values BF16.t 1792)
  (tables:Shared.rotary_tables real)
  : Lemma (
      decoder_rotary_representation storage (decoder_attention_view tables) <==>
      Shared.rotary_approximates (rotary_view storage) tables) = ()

inline_for_extraction noextract
let decoder_caller_domain (tables:Shared.rotary_tables real)
  : Lemma (
      decoder_valid_attention_data (decoder_attention_view tables) <==>
      (Shared.same_matrix tables.cosine (Shared.canonical_rotary ()).cosine /\
       Shared.same_matrix tables.sine (Shared.canonical_rotary ()).sine)) =
  C.forall_intro_2 decoder_rotary_aliases

inline_for_extraction noextract
let decoder_canonical_caller (storage:values BF16.t 1792)
  : Lemma
      (requires Shared.rotary_approximates (rotary_view storage) (Shared.canonical_rotary ()))
      (ensures
        decoder_valid_attention_data (decoder_attention_view (Shared.canonical_rotary ())) /\
        decoder_rotary_representation storage (decoder_attention_view (Shared.canonical_rotary ()))) =
  decoder_caller_domain (Shared.canonical_rotary ());
  decoder_rotary_adapter storage (Shared.canonical_rotary ())

(* The interior score model does not use data. This checks the alternative
   canonical-data adapter without claiming a universal completed decoder. *)
inline_for_extraction noextract
let decoder_score_model (qk:Shared.heads real 24) (_data:decoder_attention_data)
  (row:Shared.index 7) (head:Shared.index 16)
  : GTot (Shared.vector real (row + 1)) =
  fun key -> Shared.dot (qk row head) (qk key (16 + head / 2)) *.
    FStar.Math.Sqrt.rsqrt 128.0R

inline_for_extraction noextract
let decoder_score_data_irrelevant (qk:Shared.heads real 24)
  (left right:decoder_attention_data) (row:Shared.index 7) (head:Shared.index 16)
  (key:Shared.index (row + 1))
  : Lemma (decoder_score_model qk left row head key ==
      decoder_score_model qk right row head key) = ()

inline_for_extraction noextract
let selector_row_scores (logits:values BF16.t (7 * 151936))
  : Tot (values F32.t 151936) =
  S.init 151936 (fun token -> Cast.cast_bf16_to_f32 (S.index logits (6 * 151936 + token)))

inline_for_extraction noextract
let selector_first_winner (scores:values F32.t 151936) (winner:Shared.token_id) : prop =
  if F.NaN? (F.kind (S.index scores winner))
  then forall (j:nat{j < winner}). not (F.NaN? (F.kind (S.index scores j)))
  else
    (forall (j:Shared.token_id).
      not (F.NaN? (F.kind (S.index scores j))) /\
      not (F.lt (S.index scores winner) (S.index scores j))) /\
    (forall (j:nat{j < winner}). F.lt (S.index scores j) (S.index scores winner))

inline_for_extraction noextract
let selector_same_scores (run:source_snapshots) (token:Shared.token_id)
  : Lemma (S.index (selector_row_scores run.logits) token ==
      actual_scores (source_suffix_output run) token) = ()

inline_for_extraction noextract
let selector_maximal_actual (run:source_snapshots) (winner:Shared.token_id)
  : Lemma
      (requires Shared.selector_domain (actual_scores (source_suffix_output run)) /\
        selector_first_winner (selector_row_scores run.logits) winner)
      (ensures Shared.maximal_actual_index (actual_scores (source_suffix_output run)) winner) =
  let point (token:Shared.token_id)
    : Lemma (F.lte (actual_scores (source_suffix_output run) token)
        (actual_scores (source_suffix_output run) winner)) =
    selector_same_scores run token;
    selector_same_scores run winner;
    F.negate_lt_is_lte (S.index (selector_row_scores run.logits) winner)
      (S.index (selector_row_scores run.logits) token)
  in C.forall_intro point

inline_for_extraction noextract
let selected_token_feedback (input:source_inputs) (run:source_snapshots)
  (winner:Shared.token_id)
  : Lemma
      (requires
        (forall (p:Shared.index 7). S.index run.returned p == S.index input.prompt p) /\
        I64.v (S.index run.returned 7) == winner)
      (ensures forall (p:Shared.index 8).
        I64.v (S.index run.returned p) ==
          S.index (Shared.append_token (input_mapping input).prompt winner) p) = ()

(* Exact pure conjuncts of the reviewed decoder result, specialized to the
   original layer slice and its seven relative offsets. *)
inline_for_extraction noextract
let decoder_source_combination (fused:bool)
  (left right combined:values BF16.t 7168) : prop =
  if fused then
    forall (i:Shared.index 7168) (lr rr:real).
      N.v_approximates (S.index left i) lr /\
      N.v_approximates (S.index right i) rr ==>
      N.v_approximates (S.index combined i) (lr +. rr)
  else combined == left

inline_for_extraction noextract
let decoder_slice_parameters (storage:values BF16.t layer_weight_elements)
  : GTot (Shared.parameters BF16.t) = {
  input_layernorm = (fun d -> S.index storage d);
  q_proj = (fun r d -> S.index storage (2048 + r * 1024 + d));
  k_proj = (fun r d -> S.index storage (2048 + (2048 + r) * 1024 + d));
  v_proj = (fun r d -> S.index storage (2048 + (3072 + r) * 1024 + d));
  q_norm = (fun d -> S.index storage (6293504 + d));
  k_norm = (fun d -> S.index storage (6293504 + 128 + d));
  o_proj = (fun r d -> S.index storage (4196352 + r * 2048 + d));
  post_attention_layernorm = (fun d -> S.index storage (1024 + d));
  gate_proj = (fun r d -> S.index storage (6293760 + r * 1024 + d));
  up_proj = (fun r d -> S.index storage (6293760 + (3072 + r) * 1024 + d));
  down_proj = (fun r d -> S.index storage (12585216 + r * 3072 + d))
}

inline_for_extraction noextract
let decoder_slice_adapter (input:source_inputs) (layer:Shared.index 28)
  (wr:Shared.parameters real)
  : Lemma (
      Shared.parameters_approximates
        (decoder_slice_parameters (source_layer_weights input layer)) wr <==>
      Shared.parameters_approximates (source_parameters input layer) wr) = ()

inline_for_extraction noextract
let decoder_native_layer (combined:values BF16.t 7168)
  (observation:decoder_snapshot) : GTot (Shared.layer_values BF16.t) = {
  hidden_after = hidden_view combined;
  keys = observed_keys observation;
  values = observed_values observation
}

inline_for_extraction noextract
let reviewed_pending_universal
  (weights:values BF16.t layer_weight_elements)
  (rotary:values BF16.t 1792) (observation:decoder_snapshot) : prop =
  forall (combined:values BF16.t 7168).
    decoder_source_combination true observation.residual observation.delta combined ==>
    Shared.decoder_relation (decoder_slice_parameters weights)
      (hidden_view observation.combined) (rotary_view rotary)
      (decoder_native_layer combined observation)

inline_for_extraction noextract
let reviewed_decoder_source_post (input:source_inputs) (run:source_snapshots)
  (layer:Shared.index 28) : prop =
  let pair = decoder_input_pair run layer in
  decoder_source_combination (layer <> 0) pair._1 pair._2 (run.layer layer).combined /\
  reviewed_pending_universal (source_layer_weights input layer) run.rotary (run.layer layer)

inline_for_extraction noextract
let source_completed_hidden (input:source_inputs) (run:source_snapshots)
  (layer:Shared.index 28) : GTot (values BF16.t 7168) =
  if layer < 27 then (run.layer (layer + 1)).combined
  else suffix_completed_residual (source_suffix_input input run)

inline_for_extraction noextract
let source_layer_observation (input:source_inputs) (run:source_snapshots)
  (layer:Shared.index 28) : GTot (Shared.layer_values BF16.t) =
  decoder_native_layer (source_completed_hidden input run layer) (run.layer layer)

inline_for_extraction noextract
let suffix_source_combination (before:suffix_snapshot)
  : Lemma (decoder_source_combination true before.residual_values before.delta_values
      (suffix_completed_residual before)) =
  let cell (i:Shared.index 7168) (lr rr:real)
    : Lemma
        (requires N.v_approximates (S.index before.residual_values i) lr /\
          N.v_approximates (S.index before.delta_values i) rr)
        (ensures N.v_approximates (S.index (suffix_completed_residual before) i)
          (lr +. rr)) =
    Cast.cast_bf16_to_f32_ok (S.index before.residual_values i) lr;
    Cast.cast_bf16_to_f32_ok (S.index before.delta_values i) rr;
    N.a_add (Cast.cast_bf16_to_f32 (S.index before.residual_values i))
      (Cast.cast_bf16_to_f32 (S.index before.delta_values i)) lr rr;
    Cast.cast_f32_to_bf16_ok
      (F.add (Cast.cast_bf16_to_f32 (S.index before.residual_values i))
        (Cast.cast_bf16_to_f32 (S.index before.delta_values i))) (lr +. rr)
  in C.forall_intro_3 (fun i lr -> C.move_requires (cell i lr))

inline_for_extraction noextract
let source_decoder_input (input:source_inputs) (run:source_snapshots)
  (layer:Shared.index 28) : GTot (Shared.hidden BF16.t) =
  if layer = 0 then hidden_view run.embedded
  else (source_layer_observation input run (layer - 1)).hidden_after

#push-options "--fuel 4 --ifuel 2"
inline_for_extraction noextract
let source_decoder_at (input:source_inputs) (run:source_snapshots)
  (layer:Shared.index 28)
  : Lemma
      (requires forall (l:Shared.index 28). reviewed_decoder_source_post input run l)
      (ensures Shared.decoder_relation (source_parameters input layer)
        (source_decoder_input input run layer) (rotary_view run.rotary)
        (source_layer_observation input run layer)) =
  if layer < 27 then (
    assert (reviewed_decoder_source_post input run (layer + 1));
    assert (decoder_input_pair run (layer + 1) ==
      ((run.layer layer).residual, (run.layer layer).delta))
  ) else suffix_source_combination (source_suffix_input input run);
  assert (decoder_source_combination true (run.layer layer).residual
    (run.layer layer).delta (source_completed_hidden input run layer));
  assert (Shared.decoder_relation
    (decoder_slice_parameters (source_layer_weights input layer))
    (hidden_view (run.layer layer).combined) (rotary_view run.rotary)
    (source_layer_observation input run layer));
  assert (Shared.same_matrix (hidden_view (run.layer layer).combined)
    (source_decoder_input input run layer));
  let related (wr:Shared.parameters real) (xr:Shared.hidden real)
    (dr:Shared.rotary_tables real)
    : Lemma
        (requires Shared.parameters_approximates (source_parameters input layer) wr /\
          Shared.matrix_approximates (source_decoder_input input run layer) xr /\
          Shared.rotary_approximates (rotary_view run.rotary) dr)
        (ensures Shared.layer_approximates (source_layer_observation input run layer)
          (Shared.decoder_layer wr dr xr)) =
    decoder_slice_adapter input layer wr
  in C.forall_intro_3 (fun wr xr -> C.move_requires (related wr xr))

inline_for_extraction noextract
let source_decoder_chain (input:source_inputs) (run:source_snapshots)
  : Lemma
      (requires forall (layer:Shared.index 28).
        reviewed_decoder_source_post input run layer)
      (ensures
        Shared.decoder_relation ((input_mapping input).weight.layers 0)
          (hidden_view run.embedded) (rotary_view run.rotary)
          (source_layer_observation input run 0) /\
        (forall (layer:Shared.index 28{0 < layer}).
          Shared.decoder_relation ((input_mapping input).weight.layers layer)
            (source_layer_observation input run (layer - 1)).hidden_after
            (rotary_view run.rotary) (source_layer_observation input run layer))) =
  source_decoder_at input run 0;
  C.forall_intro #(layer:Shared.index 28{0 < layer})
    #(fun layer ->
      Shared.decoder_relation ((input_mapping input).weight.layers layer)
        (source_layer_observation input run (layer - 1)).hidden_after
        (rotary_view run.rotary) (source_layer_observation input run layer))
    (fun layer ->
    source_decoder_at input run layer)

inline_for_extraction noextract
let source_final_normalization (input:source_inputs) (run:source_snapshots)
  : Lemma
      (requires reviewed_suffix_postcondition
        (canonical_suffix_denotation (source_suffix_input input run))
        (source_suffix_input input run) (source_suffix_input input run)
        (source_suffix_output run))
      (ensures Shared.final_norm_relation (input_mapping input).weight.final_norm
        (source_layer_observation input run 27).hidden_after (hidden_view run.normalized)) =
  let related (wr:Shared.vector real 1024) (xr:Shared.hidden real)
    : Lemma
        (requires Shared.vector_approximates (input_mapping input).weight.final_norm wr /\
          Shared.matrix_approximates
            (source_layer_observation input run 27).hidden_after xr)
        (ensures Shared.matrix_approximates (hidden_view run.normalized)
          (Shared.normalize_hidden wr xr)) =
    assert (Shared.vector_approximates
      (suffix_native_norm_weight (source_suffix_input input run)) wr)
  in C.forall_intro_2 (fun wr -> C.move_requires (related wr))
#pop-options

inline_for_extraction noextract
let source_inverse_frequency (r:Shared.index 64) : F32.t =
  let theta = F.of_int #F32.t 1000000L in
  let numerator = F.of_int #F32.t (I64.int_to_t r) in
  let denominator = F.of_int #F32.t 64L in
  F.div (F.one #F32.t) (F.pow theta (F.div numerator denominator))

inline_for_extraction noextract
let source_inverse_frequency_related (r:Shared.index 64)
  : Lemma (N.v_approximates (source_inverse_frequency r) (Shared.inverse_frequency r)) =
  let theta = F.of_int #F32.t 1000000L in
  let numerator = F.of_int #F32.t (I64.int_to_t r) in
  let denominator = F.of_int #F32.t 64L in
  let exponent = F.div numerator denominator in
  let exponent_real = of_int r /. 64.0R in
  N.of_int_approx #F32.t 1000000L;
  N.of_int_approx #F32.t (I64.int_to_t r);
  N.of_int_approx #F32.t 64L;
  N.div_approx numerator denominator (of_int r) 64.0R;
  F32.pow_approx theta exponent 1000000.0R exponent_real;
  N.div_approx (F.one #F32.t) (F.pow theta exponent) 1.0R
    (Pow.pow 1000000.0R exponent_real);
  assert (exponent_real == of_int (2 * r) /. 128.0R)

inline_for_extraction noextract
let source_embedding_post (input:source_inputs) (run:source_snapshots) : prop =
  (forall (p:Shared.index 7) (d:Shared.index 1024).
    S.index run.embedded (p * 1024 + d) ==
      S.index input.weights (I64.v (S.index input.prompt p) * 1024 + d)) /\
  Shared.rotary_approximates (rotary_view run.rotary) (Shared.canonical_rotary ())

inline_for_extraction noextract
let source_embedding_caller (input:source_inputs)
  : Lemma (
    embedding_elements <= weight_elements /\
    (forall (p:Shared.index 7) (d:Shared.index 1024).
      I64.v (S.index input.prompt p) * 1024 + d < embedding_elements /\
      S.index (S.slice input.weights 0 embedding_elements)
        (I64.v (S.index input.prompt p) * 1024 + d) ==
      S.index input.weights (I64.v (S.index input.prompt p) * 1024 + d))) = ()

inline_for_extraction noextract
let source_decoder_rotary_caller (input:source_inputs) (run:source_snapshots)
  : Lemma
      (requires source_embedding_post input run)
      (ensures
        decoder_valid_attention_data (decoder_attention_view (Shared.canonical_rotary ())) /\
        decoder_rotary_representation run.rotary
          (decoder_attention_view (Shared.canonical_rotary ()))) =
  decoder_canonical_caller run.rotary

inline_for_extraction noextract
let source_cache_prefix (run:source_snapshots) (count:nat{count <= 28})
  : GTot (Shared.cache_state BF16.t) =
  fun layer ->
    if layer < count then
      Shared.Prefilled {
        cached_keys = observed_keys (run.layer layer);
        cached_values = observed_values (run.layer layer)
      }
    else Shared.Empty

inline_for_extraction noextract
let source_cache_update (input:source_inputs) (run:source_snapshots)
  (layer:Shared.index 28)
  : Lemma (
    Shared.same_cache (source_cache_prefix run (layer + 1))
      (Shared.update_cache (source_cache_prefix run layer) layer
        (source_layer_observation input run layer)) /\
    (forall (other:Shared.index 28{other <> layer}).
      source_cache_prefix run (layer + 1) other == source_cache_prefix run layer other) /\
    Shared.cache_length (source_cache_prefix run (layer + 1) layer) == 7) = ()

inline_for_extraction noextract
let source_cache_complete (run:source_snapshots)
  : Lemma (
    (forall (layer:Shared.index 28).
      Shared.same_cache_slot (source_cache_prefix run 0 layer) Shared.Empty) /\
    (forall (layer:Shared.index 28).
      Shared.cache_length (source_cache_prefix run 28 layer) == 7 /\
      Shared.same_cache_slot (source_cache_prefix run 28 layer)
        (Shared.Prefilled {
          cached_keys = observed_keys (run.layer layer);
          cached_values = observed_values (run.layer layer)
        }))) = ()

inline_for_extraction noextract
let embedding_frequency_coordinate (d:Shared.index 128) : Shared.index 64 =
  if d < 64 then d else d - 64

inline_for_extraction noextract
let embedding_angle (p:Shared.index 7) (d:Shared.index 128) : real =
  (1.0R /. Pow.pow 1000000.0R
    (of_int (2 * embedding_frequency_coordinate d) /. 128.0R)) *. of_int p

inline_for_extraction noextract
let embedding_cosine (p:Shared.index 7) (d:Shared.index 128) : real =
  Trig.cos (embedding_angle p d) *. 1.0R

inline_for_extraction noextract
let embedding_sine (p:Shared.index 7) (d:Shared.index 128) : real =
  Trig.sin (embedding_angle p d) *. 1.0R

inline_for_extraction noextract
let embedding_math_adapter (p:Shared.index 7) (d:Shared.index 128)
  : Lemma (
    embedding_cosine p d == (Shared.canonical_rotary ()).cosine p d /\
    embedding_sine p d == (Shared.canonical_rotary ()).sine p d) =
  assert (embedding_frequency_coordinate d == d % 64)

inline_for_extraction noextract
let accepted_embedding_pointwise
  (weights:values BF16.t embedding_elements)
  (prompt:values I64.t 7{valid_prompt prompt})
  (hidden:values BF16.t 7168) (rotary:values BF16.t 1792) : prop =
  (forall (p:Shared.index 7) (d:Shared.index 1024).
    S.index hidden (p * 1024 + d) ==
      S.index weights (I64.v (S.index prompt p) * 1024 + d)) /\
  (forall (p:Shared.index 7) (d:Shared.index 128).
    N.v_approximates (S.index rotary (p * 128 + d)) (embedding_cosine p d) /\
    N.v_approximates (S.index rotary (896 + p * 128 + d)) (embedding_sine p d))

inline_for_extraction noextract
let source_embedding_call_adapter (input:source_inputs) (run:source_snapshots)
  : Lemma
      (requires accepted_embedding_pointwise
        (S.slice input.weights 0 embedding_elements) input.prompt run.embedded run.rotary)
      (ensures source_embedding_post input run) =
  source_embedding_caller input;
  C.forall_intro_2 embedding_math_adapter

inline_for_extraction noextract
let source_selector_physical_post (input:source_inputs) (run:source_snapshots) : prop =
  (forall (p:Shared.index 7). S.index run.returned p == S.index input.prompt p) /\
  (exists (winner:Shared.token_id).
    I64.v (S.index run.returned 7) == winner /\
    selector_first_winner (selector_row_scores run.logits) winner)

inline_for_extraction noextract
let source_selector_result (input:source_inputs) (run:source_snapshots)
  : Lemma
      (requires source_selector_physical_post input run /\
        Shared.selector_domain (actual_scores (source_suffix_output run)))
      (ensures exists (winner:Shared.token_id).
        I64.v (S.index run.returned 7) == winner /\
        Shared.maximal_actual_index (actual_scores (source_suffix_output run)) winner /\
        (forall (p:Shared.index 8).
          I64.v (S.index run.returned p) ==
            S.index (Shared.append_token (input_mapping input).prompt winner) p)) =
  let winner:Shared.token_id = I64.v (S.index run.returned 7) in
  selector_maximal_actual run winner;
  selected_token_feedback input run winner;
  C.exists_intro
    (fun (chosen:Shared.token_id) ->
      I64.v (S.index run.returned 7) == chosen /\
      Shared.maximal_actual_index (actual_scores (source_suffix_output run)) chosen /\
      (forall (p:Shared.index 8).
        I64.v (S.index run.returned p) ==
          S.index (Shared.append_token (input_mapping input).prompt chosen) p))
    winner

inline_for_extraction noextract
type packed_weights_address =
  a:PA.larray BF16.t weight_elements{KA.is_global_array a}

inline_for_extraction noextract
noeq type source_input_addresses = {
  weights_address: packed_weights_address;
  prompt_address: a:PA.larray I64.t 7{KA.is_global_array a}
}

inline_for_extraction noextract
let source_caller_pre (input:source_inputs) (addresses:source_input_addresses)
  (ambient:PC.slprop) : PC.slprop =
  KL.cpu ** ambient **
  PC.on KL.gpu_loc
    (PA.pts_to addresses.weights_address input.weights **
     PA.pts_to addresses.prompt_address input.prompt)

inline_for_extraction noextract
let source_weight_window (weights:packed_weights_address)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  : GTot (PA.larray BF16.t (hi - lo)) =
  PA.gsub weights lo hi

inline_for_extraction noextract
let source_weight_window_shape (weights:packed_weights_address)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  : Lemma (
      PA.length (source_weight_window weights lo hi) == hi - lo /\
      PA.base_of (source_weight_window weights lo hi) == PA.base_of weights /\
      PA.offset_of (source_weight_window weights lo hi) == PA.offset_of weights + lo) =
  PA.length_gsub weights lo hi;
  PA.base_of_gsub weights lo hi;
  PA.offset_of_gsub weights lo hi

inline_for_extraction noextract
let source_embedding_window (weights:packed_weights_address)
  : GTot (PA.larray BF16.t embedding_elements) =
  source_weight_window weights 0 embedding_elements

inline_for_extraction noextract
let source_embedding_window_is_not_parent (weights:packed_weights_address)
  : Lemma (source_embedding_window weights =!= weights) =
  source_weight_window_shape weights 0 embedding_elements

inline_for_extraction noextract
let source_weight_window_resources (weights:packed_weights_address)
  (contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (fraction:Permission.perm) : PC.slprop =
  PA.pts_to_range weights 0 lo #fraction (S.slice contents 0 lo) **
  PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi) **
  PA.pts_to_range weights hi weight_elements #fraction
    (S.slice contents hi weight_elements)

inline_for_extraction noextract
ghost
fn take_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires PC.on KL.gpu_loc (PA.pts_to weights #fraction contents)
  ensures PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)

inline_for_extraction noextract
ghost
fn return_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)
  ensures PC.on KL.gpu_loc (PA.pts_to weights #fraction contents)

inline_for_extraction noextract
ghost
fn split_first_decoder_alias
  (hidden:PA.larray BF16.t 7168)
  (#contents:values BF16.t 7168)
  requires PC.on KL.gpu_loc (PA.pts_to hidden contents)
  ensures PC.on KL.gpu_loc
    (PA.pts_to hidden #0.5R contents ** PA.pts_to hidden #0.5R contents)

inline_for_extraction noextract
ghost
fn reassemble_first_decoder_alias
  (hidden:PA.larray BF16.t 7168)
  (#left #right:values BF16.t 7168)
  requires PC.on KL.gpu_loc
    (PA.pts_to hidden #0.5R left ** PA.pts_to hidden #0.5R right)
  ensures PC.on KL.gpu_loc (PA.pts_to hidden left) ** PC.pure (left == right)

inline_for_extraction noextract
let source_weight_window_residency (weights:packed_weights_address)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  : Lemma (KA.is_global_array (source_weight_window weights lo hi)) =
  PA.loc_id_of_array_gsub weights lo hi;
  KA.visibility_of_gsub weights lo hi

inline_for_extraction noextract
ghost
fn source_embedding_call_residency
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (ambient:PC.slprop)
  preserves KL.cpu ** ambient ** PC.on KL.gpu_loc (PA.pts_to weights contents)
  ensures PC.pure (KA.is_global_array (source_embedding_window weights))

inline_for_extraction noextract
let source_weight_window_frame (weights:packed_weights_address)
  (contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (fraction:Permission.perm) : PC.slprop =
  PC.on KL.gpu_loc
    (PA.pts_to_range weights 0 lo #fraction (S.slice contents 0 lo)) **
  PC.on KL.gpu_loc
    (PA.pts_to_range weights hi weight_elements #fraction
      (S.slice contents hi weight_elements))

inline_for_extraction noextract
ghost
fn expose_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)
  ensures source_weight_window_frame weights contents lo hi fraction **
    PC.on KL.gpu_loc
      (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi))

inline_for_extraction noextract
ghost
fn conceal_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires source_weight_window_frame weights contents lo hi fraction **
    PC.on KL.gpu_loc
      (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi))
  ensures PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)

module Ghost = FStar.Ghost
module HostSize = FStar.SizeT

inline_for_extraction noextract
fn materialize_source_weight_window
  (weights:packed_weights_address)
  (lo:HostSize.t)
  (hi:Ghost.erased nat{HostSize.v lo <= hi /\ hi <= weight_elements})
  (#contents:Ghost.erased (values BF16.t weight_elements))
  (ambient:PC.slprop)
  preserves KL.cpu ** ambient ** PC.on KL.gpu_loc (PA.pts_to weights contents)
  returns window:PA.array BF16.t
  ensures rewrites_to window (source_weight_window weights (HostSize.v lo) hi)
  ensures pure (PA.length window == hi - HostSize.v lo /\ KA.is_global_array window)

module SourceCasts = Kuiper.Float.Casts
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Kernel = Kuiper.Kernel
module Pledge = Pulse.Lib.Pledge

inline_for_extraction noextract
type source_embedding_values = values BF16.t embedding_elements

inline_for_extraction noextract
type source_hidden_values = values BF16.t 7168

inline_for_extraction noextract
type source_rotary_values = values BF16.t 1792

inline_for_extraction noextract
let source_loaded_approximates (value:BF16.t) (r:real) : prop =
  N.v_approximates value r \/ N.v_approximates (SourceCasts.fcast #BF16.t #F32.t value) r

inline_for_extraction noextract
let source_loaded_widen (value:BF16.t) (r:real)
  : Lemma
      (requires source_loaded_approximates value r)
      (ensures N.v_approximates (SourceCasts.fcast #BF16.t #F32.t value) r) =
  if t2b (N.v_approximates value r) then SourceCasts.fcast_approx #BF16.t #F32.t value r

inline_for_extraction noextract
let source_loaded_hidden (actual:source_hidden_values) (logical:Shared.hidden real) : prop =
  forall (row:Shared.index 7) (column:Shared.index 1024).
    source_loaded_approximates (S.index actual (row * 1024 + column)) (logical row column)

inline_for_extraction noextract
let source_loaded_rotary (actual:source_rotary_values) (logical:Shared.rotary_tables real) : prop =
  (forall (row:Shared.index 7) (column:Shared.index 128).
    source_loaded_approximates (S.index actual (row * 128 + column)) (logical.cosine row column) /\
    source_loaded_approximates (S.index actual (896 + row * 128 + column)) (logical.sine row column))

inline_for_extraction noextract
let source_hidden32 (actual:source_hidden_values) : GTot (Shared.hidden F32.t) =
  fun row column -> SourceCasts.fcast #BF16.t #F32.t (S.index actual (row * 1024 + column))

inline_for_extraction noextract
type source_token_values = ts:values I64.t 7{valid_prompt ts}

inline_for_extraction noextract
type source_prefill_position = p:I64.t{I64.v p == 0}

inline_for_extraction noextract
let source_position32 (p:Shared.index 7) : F32.t =
  F.of_int #F32.t (I64.int_to_t p)

inline_for_extraction noextract
let source_angle32 (p:Shared.index 7) (d:Shared.index 128) : F32.t =
  F.mul (source_position32 p) (source_inverse_frequency (d % 64))

inline_for_extraction noextract
let source_cosine32 (p:Shared.index 7) (d:Shared.index 128) : F32.t =
  F.cos (source_angle32 p d)

inline_for_extraction noextract
let source_sine32 (p:Shared.index 7) (d:Shared.index 128) : F32.t =
  F.sin (source_angle32 p d)

inline_for_extraction noextract
let source_rotary_cell (plane:Shared.index 2) (p:Shared.index 7) (d:Shared.index 128)
  : BF16.t =
  SourceCasts.fcast #F32.t #BF16.t
    (if plane = 0 then source_cosine32 p d else source_sine32 p d)

inline_for_extraction noextract
let source_frequency_definition_binding (r:Shared.index 64)
  : Lemma (
    source_inverse_frequency r ==
      (let base = F.of_int #F32.t 1000000L in
       let exponent = F.div (F.of_int #F32.t (I64.int_to_t r)) (F.of_int #F32.t 64L) in
       F.div (F.one #F32.t) (F.pow base exponent))) = ()

inline_for_extraction noextract
let source_rotary_definition_binding
  (plane:Shared.index 2) (p:Shared.index 7) (d:Shared.index 128)
  : Lemma (
    source_position32 p == F.of_int #F32.t (I64.int_to_t p) /\
    source_angle32 p d ==
      F.mul (source_position32 p) (source_inverse_frequency (d % 64)) /\
    source_rotary_cell plane p d ==
      SourceCasts.fcast #F32.t #BF16.t
        (if plane = 0 then F.cos (source_angle32 p d) else F.sin (source_angle32 p d))) = ()

inline_for_extraction noextract
let source_exact_rotary (rotary:source_rotary_values) (count:nat{count <= 896})
  : prop =
  forall (p:Shared.index 7) (d:Shared.index 128).
    p * 128 + d < count ==>
    S.index rotary (p * 128 + d) == source_rotary_cell 0 p d /\
    S.index rotary (896 + p * 128 + d) == source_rotary_cell 1 p d

inline_for_extraction noextract
let source_rotary_domain () : prop =
  forall (p:Shared.index 7) (d:Shared.index 128).
    not (F.NaN? (F.kind (source_cosine32 p d))) /\
    not (F.NaN? (F.kind (source_sine32 p d)))

inline_for_extraction noextract
let source_observed_rotary (rotary:source_rotary_values)
  : GTot (Shared.rotary_tables F32.t) = {
  cosine = (fun p d -> SourceCasts.fcast #BF16.t #F32.t (S.index rotary (p * 128 + d)));
  sine = (fun p d -> SourceCasts.fcast #BF16.t #F32.t (S.index rotary (896 + p * 128 + d)))
}

inline_for_extraction noextract
let source_observed_rotary_state () : GTot (Shared.rotary_state F32.t) = {
  inv_freq = source_inverse_frequency;
  original_inv_freq = source_inverse_frequency;
  attention_scaling = F.one #F32.t;
  max_seq_len_cached = 40960;
  original_max_seq_len = 40960
}

inline_for_extraction noextract
let source_angle_related (p:Shared.index 7) (d:Shared.index 128) (frequency:real)
  : Lemma
      (requires N.v_approximates (source_inverse_frequency (d % 64)) frequency)
      (ensures N.v_approximates (source_angle32 p d) (of_int p *. frequency)) =
  N.of_int_approx #F32.t (I64.int_to_t p);
  N.a_mul (source_position32 p) (source_inverse_frequency (d % 64)) (of_int p) frequency

inline_for_extraction noextract
let source_cell_related (plane:Shared.index 2) (p:Shared.index 7) (d:Shared.index 128)
  (frequency factor:real)
  : Lemma
      (requires
        N.v_approximates (source_inverse_frequency (d % 64)) frequency /\
        N.v_approximates (F.one #F32.t) factor /\
        not (F.NaN? (F.kind (if plane = 0 then source_cosine32 p d else source_sine32 p d))))
      (ensures N.v_approximates (source_rotary_cell plane p d)
        ((if plane = 0 then Trig.cos (of_int p *. frequency)
          else Trig.sin (of_int p *. frequency)) *. factor)) =
  source_angle_related p d frequency;
  let value = if plane = 0 then source_cosine32 p d else source_sine32 p d in
  let target = if plane = 0 then Trig.cos (of_int p *. frequency)
    else Trig.sin (of_int p *. frequency) in
  if plane = 0 then F32.cos_approx (source_angle32 p d) (of_int p *. frequency)
  else F32.sin_approx (source_angle32 p d) (of_int p *. frequency);
  F.mul_one value;
  N.a_mul value (F.one #F32.t) target factor;
  SourceCasts.fcast_approx #F32.t #BF16.t value (target *. factor)

inline_for_extraction noextract
let source_exact_rotary_generation (rotary:source_rotary_values)
  : Lemma
      (requires source_exact_rotary rotary 896)
      (ensures
        Shared.rotary_initialized (source_observed_rotary_state ()) /\
        (source_rotary_domain () ==>
          Shared.rotary_generation (source_observed_rotary_state ()) (source_observed_rotary rotary))) =
  C.forall_intro source_inverse_frequency_related;
  if t2b (source_rotary_domain ()) then (
    let related (frequency:Shared.vector real 64) (factor:real)
      : Lemma
          (requires Shared.vector_approximates source_inverse_frequency frequency /\
            N.v_approximates (F.one #F32.t) factor)
          (ensures Shared.rotary_approximates (source_observed_rotary rotary)
            (Shared.tables_from_frequency frequency factor)) =
      let cell (p:Shared.index 7) (d:Shared.index 128)
        : Lemma (
            N.v_approximates ((source_observed_rotary rotary).cosine p d)
              ((Shared.tables_from_frequency frequency factor).cosine p d) /\
            N.v_approximates ((source_observed_rotary rotary).sine p d)
              ((Shared.tables_from_frequency frequency factor).sine p d)) =
        source_cell_related 0 p d (frequency (d % 64)) factor;
        source_cell_related 1 p d (frequency (d % 64)) factor;
        SourceCasts.fcast_approx #BF16.t #F32.t (source_rotary_cell 0 p d)
          (Trig.cos (of_int p *. frequency (d % 64)) *. factor);
        SourceCasts.fcast_approx #BF16.t #F32.t (source_rotary_cell 1 p d)
          (Trig.sin (of_int p *. frequency (d % 64)) *. factor)
      in C.forall_intro_2 cell
    in C.forall_intro_2 (fun frequency -> C.move_requires (related frequency)))

inline_for_extraction noextract
noeq type source_embedding_addresses = {
  embedded_address: a:PA.larray BF16.t 7168{KA.is_global_array a};
  rotary_address: a:PA.larray BF16.t 1792{KA.is_global_array a}
}

inline_for_extraction noextract
let source_embedding_kernel_pre
  (addresses:source_input_addresses) (outputs:source_embedding_addresses)
  (input:source_inputs) : PC.slprop =
  PA.pts_to addresses.weights_address input.weights **
  PA.pts_to addresses.prompt_address input.prompt **
  (exists* (hidden:source_hidden_values) (rotary:source_rotary_values).
    PA.pts_to outputs.embedded_address hidden ** PA.pts_to outputs.rotary_address rotary)

inline_for_extraction noextract
let source_embedding_result (input:source_inputs)
  (hidden:source_hidden_values) (rotary:source_rotary_values) : prop =
  accepted_embedding_pointwise (S.slice input.weights 0 embedding_elements)
    input.prompt hidden rotary /\
  source_exact_rotary rotary 896 /\
  Shared.rotary_initialized (source_observed_rotary_state ()) /\
  (source_rotary_domain () ==>
    Shared.rotary_generation (source_observed_rotary_state ()) (source_observed_rotary rotary))

inline_for_extraction noextract
let source_embedding_kernel_post
  (addresses:source_input_addresses) (outputs:source_embedding_addresses)
  (input:source_inputs) : PC.slprop =
  PA.pts_to addresses.weights_address input.weights **
  PA.pts_to addresses.prompt_address input.prompt **
  (exists* (hidden:source_hidden_values) (rotary:source_rotary_values).
    PA.pts_to outputs.embedded_address hidden ** PA.pts_to outputs.rotary_address rotary **
    PC.pure (source_embedding_result input hidden rotary))

inline_for_extraction noextract
val source_embedding_descriptor
  (addresses:source_input_addresses)
  (outputs:source_embedding_addresses)
  (window:PA.larray BF16.t embedding_elements{
    window == source_embedding_window addresses.weights_address})
  (#input:Ghost.erased source_inputs)
  : Kernel.kernel_desc
      (source_embedding_kernel_pre addresses outputs input)
      (source_embedding_kernel_post addresses outputs input)

inline_for_extraction noextract
fn source_allocate_embedding_outputs (ambient:PC.slprop)
  preserves KL.cpu ** ambient
  returns outputs:source_embedding_addresses
  ensures exists* (hidden:source_hidden_values) (rotary:source_rotary_values).
    PC.on KL.gpu_loc
      (PA.pts_to outputs.embedded_address hidden ** PA.pts_to outputs.rotary_address rotary)

inline_for_extraction noextract
fn source_enqueue_embedding
  (addresses:source_input_addresses)
  (outputs:source_embedding_addresses)
  (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t)
  (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** ambient
  requires Epoch.epoch_live stream epoch **
    PC.on KL.gpu_loc (source_embedding_kernel_pre addresses outputs input)
  ensures Epoch.epoch_live stream (Epoch.epoch_next epoch) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
      (PC.on KL.gpu_loc (source_embedding_kernel_post addresses outputs input))

module RT = Kuiper.Tensor.Runtime
module RuntimeHelpers = Kuiper.Tensor.Runtime.Helpers
module List = FStar.List.Tot
module ListProps = FStar.List.Tot.Properties

open Pulse.Lib.Send

inline_for_extraction noextract
type source_allocation_shape = shape:list pos{RT.numel shape <= 2147483647}

inline_for_extraction noextract
let source_layer_allocation_shapes : list source_allocation_shape =
  [[7; 1024]; [7; 4096]; [7; 2048]; [7; 1024]; [7; 1024];
   [7; 1024]; [7; 6144]; [7; 3072]; [7; 1024]; [7; 1024]]

inline_for_extraction noextract
let rec source_layer_allocations (count:nat{count <= 28})
  : Tot (list source_allocation_shape) (decreases count) =
  if count = 0 then []
  else List.append source_layer_allocation_shapes (source_layer_allocations (count - 1))

inline_for_extraction noextract
let source_bf16_allocation_shapes : list source_allocation_shape =
  List.append [[7; 1024]; [2; 7; 128]]
    (List.append (source_layer_allocations 28) [[7; 1024]; [7; 151936]])

inline_for_extraction noextract
let rec source_layer_allocation_count (count:nat{count <= 28})
  : Lemma (List.length (source_layer_allocations count) == 10 * count)
      (decreases count) =
  if count > 0 then (
    source_layer_allocation_count (count - 1);
    ListProps.append_length source_layer_allocation_shapes (source_layer_allocations (count - 1));
    assert_norm (List.length source_layer_allocation_shapes == 10))

inline_for_extraction noextract
let source_allocation_count ()
  : Lemma (List.length source_bf16_allocation_shapes == 284 /\
      List.length source_bf16_allocation_shapes + 1 == 285) =
  source_layer_allocation_count 28;
  ListProps.append_length (source_layer_allocations 28) [[7; 1024]; [7; 151936]];
  ListProps.append_length [[7; 1024]; [2; 7; 128]]
    (List.append (source_layer_allocations 28) [[7; 1024]; [7; 151936]])

inline_for_extraction noextract
let rec source_allocation_shapes_match
  (shapes:list source_allocation_shape) (buffers:list (RT.tensor BF16.t))
  : GTot prop (decreases shapes) =
  match shapes, buffers with
  | [], [] -> True
  | shape::rest, value::tail ->
    RT.metadata value shape /\ source_allocation_shapes_match rest tail
  | _ -> False

inline_for_extraction noextract
let rec source_matching_allocation_count
  (shapes:list source_allocation_shape) (buffers:list (RT.tensor BF16.t))
  : Lemma
      (requires source_allocation_shapes_match shapes buffers)
      (ensures List.length shapes == List.length buffers)
      (decreases shapes) =
  match shapes, buffers with
  | _::rest, _::tail -> source_matching_allocation_count rest tail
  | _ -> ()

inline_for_extraction noextract
let rec source_matching_allocation_at
  (shapes:list source_allocation_shape) (buffers:list (RT.tensor BF16.t))
  (index:nat{index < List.length shapes /\ index < List.length buffers})
  : Lemma
      (requires source_allocation_shapes_match shapes buffers)
      (ensures RT.metadata (List.index buffers index) (List.index shapes index))
      (decreases index) =
  if index > 0 then
    source_matching_allocation_at (List.tail shapes) (List.tail buffers) (index - 1)

inline_for_extraction noextract
noeq type source_allocated_buffers = {
  bf16_buffers: list (RT.tensor BF16.t);
  token_buffer: RT.tensor I64.t
}

inline_for_extraction noextract
let source_allocation_metadata (buffers:source_allocated_buffers) : prop =
  source_allocation_shapes_match source_bf16_allocation_shapes buffers.bf16_buffers /\
  RT.metadata buffers.token_buffer [1; 8]

inline_for_extraction noextract
let source_allocated_owners (stream:Stream.stream_t) (buffers:source_allocated_buffers)
  : PC.slprop =
  RuntimeHelpers.all_owners stream buffers.bf16_buffers ** RT.owner stream buffers.token_buffer

inline_for_extraction noextract
let source_allocated_storage (buffers:source_allocated_buffers) (frame:PC.slprop)
  : PC.slprop =
  RT.contents buffers.token_buffer **
  RuntimeHelpers.all_contents buffers.bf16_buffers ** frame

inline_for_extraction noextract
noeq type source_allocation_outcome =
  | SourceAllocationFailed : error:RT.allocation_error -> source_allocation_outcome
  | SourceAllocated : buffers:source_allocated_buffers -> source_allocation_outcome

(* All declarations precede every launch. Failure unwinds only the successful
   preceding allocations, in reverse order, retaining the native error. *)
inline_for_extraction noextract
fn source_prepare_allocations
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  (frame:PC.slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream start
  requires Pledge.pledge0 (Epoch.epoch_done stream start) frame
  returns outcome:source_allocation_outcome
  ensures (
    match outcome with
    | SourceAllocationFailed _ -> Pledge.pledge0 (Epoch.epoch_done stream start) frame
    | SourceAllocated buffers ->
      PC.pure (source_allocation_metadata buffers) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_allocated_storage buffers frame))

inline_for_extraction noextract
let source_allocated_embedding
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : GTot source_embedding_addresses =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  source_matching_allocation_at source_bf16_allocation_shapes buffers.bf16_buffers 0;
  source_matching_allocation_at source_bf16_allocation_shapes buffers.bf16_buffers 1;
  {
    embedded_address = RT.data (List.index buffers.bf16_buffers 0);
    rotary_address = RT.data (List.index buffers.bf16_buffers 1)
  }

inline_for_extraction noextract
let source_remaining_bf16
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Tot (list (RT.tensor BF16.t)) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  List.tail (List.tail buffers.bf16_buffers)

inline_for_extraction noextract
let source_allocations_head_tail
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Lemma (
      List.length buffers.bf16_buffers == 284 /\
      buffers.bf16_buffers ==
        List.index buffers.bf16_buffers 0 ::
        List.index buffers.bf16_buffers 1 :: source_remaining_bf16 buffers) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers

inline_for_extraction noextract
let source_borrowed_inputs (addresses:source_input_addresses) (input:source_inputs)
  : PC.slprop =
  PC.on KL.gpu_loc (PA.pts_to addresses.weights_address input.weights) **
  PC.on KL.gpu_loc (PA.pts_to addresses.prompt_address input.prompt)

inline_for_extraction noextract
let source_remaining_storage
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) : PC.slprop =
  RT.contents buffers.token_buffer ** RuntimeHelpers.all_contents (source_remaining_bf16 buffers)

inline_for_extraction noextract
let source_after_embedding (addresses:source_input_addresses) (input:source_inputs)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) : PC.slprop =
  PC.on KL.gpu_loc
    (source_embedding_kernel_post addresses (source_allocated_embedding buffers) input) **
  source_remaining_storage buffers

inline_for_extraction noextract
noeq type source_prefix_outcome =
  | SourcePrefixFailed : error:RT.allocation_error -> source_prefix_outcome
  | SourcePrefixEnqueued :
      buffers:source_allocated_buffers{source_allocation_metadata buffers} -> source_prefix_outcome

(* This is the complete allocation prefix and launch 1, not program_refines.
   Every scratch and output owner remains live; there is no early retirement. *)
inline_for_extraction noextract
fn source_prepare_and_enqueue
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_prefix_outcome
  ensures (
    match outcome with
    | SourcePrefixFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourcePrefixEnqueued buffers ->
      Epoch.epoch_live stream (Epoch.epoch_next start) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next start))
        (source_after_embedding addresses input buffers))

module TensorLayout = Kuiper.Tensor.Layout
module Strided = Kuiper.Array2.Strided
module DeviceSize = Kuiper.SizeT

(* Source linear supplies the eighth MMA column in registers, not memory.
   This rules out only the direct strided-memory adapter for that operand. *)
inline_for_extraction noextract
let source_mma_padding_is_not_a_native_view
  (width:pos) (layout:TensorLayout.layout2 16 8)
  (strides:Strided.strided_col_major layout)
  : Lemma
      (requires layout.ulen == 7 * width /\ DeviceSize.v strides.stride == width)
      (ensures False) =
  let _ = strides.pf 0 7 in
  assert (Strided.cell_of_pos layout 0 7 ==
    DeviceSize.v strides.offset + 7 * width);
  assert (Strided.cell_of_pos layout 0 7 < layout.ulen)

open Kuiper.Divides { (/?+) }

(* Implemented by the complete accepted SourceMMA body retained below, not
   an external native declaration or a premise about a linear result. *)
inline_for_extraction noextract
val source_mma_arrays_array_result
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width})
  (#n:nat) (weights:values BF16.t n)
  (input:values BF16.t (width * 7)) (old:values BF16.t (columns * 7))
  (offset:nat{offset + columns * width <= n})
  (steps:pos) (splits:pos{steps * splits == width / 16})
  : GTot (values BF16.t (columns * 7))

inline_for_extraction noextract
let source_linear_values
  (columns width:pos{16 /?+ columns /\ 16 /?+ width})
  (weights:values BF16.t weight_elements) (input:values BF16.t (width * 7))
  (offset:nat{offset + columns * width <= weight_elements})
  (steps:pos) (splits:pos{steps * splits == width / 16})
  : GTot (values BF16.t (columns * 7)) =
  source_mma_arrays_array_result #columns #width #weight_elements
    weights input (S.create (columns * 7) (F.zero #BF16.t)) offset steps splits

inline_for_extraction noextract
let source_rms_level (count:nat) : prop =
  count == 1 \/ count == 2 \/ count == 4 \/ count == 8 \/
  count == 16 \/ count == 32 \/ count == 64 \/ count == 128

inline_for_extraction noextract
let rec source_rms_component
  (input:source_hidden_values) (row:Shared.index 7)
  (lane:Shared.index 128) (component:Shared.index 4)
  (count:nat{count <= 2}) : GTot F32.t (decreases count) =
  if count = 0 then F.zero #F32.t
  else
    let value = SourceCasts.fcast #BF16.t #F32.t
      (S.index input (row * 1024 + lane * 4 + component + (count - 1) * 512)) in
    F.add (source_rms_component input row lane component (count - 1))
      (F.mul value value)

inline_for_extraction noextract
let source_rms_lane (input:source_hidden_values)
  (row:Shared.index 7) (lane:Shared.index 128) : GTot F32.t =
  F.add
    (F.add
      (F.add (source_rms_component input row lane 0 2)
        (source_rms_component input row lane 1 2))
      (source_rms_component input row lane 2 2))
    (source_rms_component input row lane 3 2)

inline_for_extraction noextract
let rec source_rms_tree (input:source_hidden_values) (row:Shared.index 7)
  (count:pos{source_rms_level count}) (lane:nat{lane < count})
  : GTot F32.t (decreases (128 - count)) =
  if count = 128 then source_rms_lane input row lane
  else F.add (source_rms_tree input row (2 * count) lane)
    (source_rms_tree input row (2 * count) (lane + count))

inline_for_extraction noextract
let source_rms_scale (input:source_hidden_values) (row:Shared.index 7)
  : GTot F32.t =
  F.rsqrt
    (F.add
      (F.mul (source_rms_tree input row 1 0)
        (F.div (F.one #F32.t) (F.of_int #F32.t 1024L)))
      (F.of_literal #F32.t "0.000001"))

inline_for_extraction noextract
let source_rms_cell (weights:values BF16.t weight_elements)
  (input:source_hidden_values) (offset:nat{offset + 1024 <= weight_elements})
  (row:Shared.index 7) (column:Shared.index 1024) : GTot BF16.t =
  let normalized = SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t (S.index input (row * 1024 + column)))
      (source_rms_scale input row)) in
  SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t normalized)
      (SourceCasts.fcast #BF16.t #F32.t (S.index weights (offset + column))))

inline_for_extraction noextract
let source_rms_result (weights:values BF16.t weight_elements)
  (input output:source_hidden_values) (offset:nat{offset + 1024 <= weight_elements})
  : prop =
  forall (row:Shared.index 7) (column:Shared.index 1024).
    S.index output (row * 1024 + column) == source_rms_cell weights input offset row column

inline_for_extraction noextract
let source_rms_kernel_pre (weights:packed_weights_address)
  (input output:PA.larray BF16.t 7168)
  (wv:values BF16.t weight_elements) (xv:source_hidden_values) : PC.slprop =
  PA.pts_to weights wv ** PA.pts_to input xv **
  (exists* (old:source_hidden_values). PA.pts_to output old)

inline_for_extraction noextract
let source_rms_kernel_post (weights:packed_weights_address)
  (input output:PA.larray BF16.t 7168)
  (wv:values BF16.t weight_elements) (xv:source_hidden_values)
  (offset:nat{offset + 1024 <= weight_elements}) : PC.slprop =
  PA.pts_to weights wv ** PA.pts_to input xv **
  (exists* (result:source_hidden_values).
    PA.pts_to output result ** PC.pure (source_rms_result wv xv result offset))

(* reduced.cu:24-50, with private uninitialized partial[128] and no helper
   launches. This is a source-operation result, not a decoder postcondition. *)
inline_for_extraction noextract
fn source_rmsnorm_body
  (weights:packed_weights_address)
  (input output:PA.larray BF16.t 7168)
  (offset:DeviceSize.t{DeviceSize.v offset + 1024 <= weight_elements})
  (#wv:Ghost.erased (values BF16.t weight_elements))
  (#xv:Ghost.erased source_hidden_values)
  preserves KL.gpu
  requires source_rms_kernel_pre weights input output wv xv
  ensures source_rms_kernel_post weights input output wv xv (DeviceSize.v offset)

inline_for_extraction noextract
val source_rmsnorm_descriptor
  (weights:packed_weights_address)
  (input:PA.larray BF16.t 7168{KA.is_global_array input})
  (output:PA.larray BF16.t 7168{KA.is_global_array output})
  (offset:DeviceSize.t{DeviceSize.v offset + 1024 <= weight_elements})
  (#wv:Ghost.erased (values BF16.t weight_elements))
  (#xv:Ghost.erased source_hidden_values)
  : Kernel.kernel_desc
      (source_rms_kernel_pre weights input output wv xv)
      (source_rms_kernel_post weights input output wv xv (DeviceSize.v offset))

inline_for_extraction noextract
let source_allocated_norm0
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : GTot (a:PA.larray BF16.t 7168{KA.is_global_array a}) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  source_matching_allocation_at source_bf16_allocation_shapes buffers.bf16_buffers 2;
  assert_norm (List.index source_bf16_allocation_shapes 2 == [7; 1024]);
  RT.data (List.index buffers.bf16_buffers 2)

inline_for_extraction noextract
let source_remaining_after_rms
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Tot (list (RT.tensor BF16.t)) =
  source_allocations_head_tail buffers;
  List.tail (source_remaining_bf16 buffers)

inline_for_extraction noextract
let source_allocations_first_three
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Lemma (
      source_remaining_bf16 buffers ==
        List.index buffers.bf16_buffers 2 :: source_remaining_after_rms buffers /\
      buffers.bf16_buffers ==
        List.index buffers.bf16_buffers 0 :: List.index buffers.bf16_buffers 1 ::
        List.index buffers.bf16_buffers 2 :: source_remaining_after_rms buffers /\
      List.length (source_remaining_after_rms buffers) == 281) =
  source_allocations_head_tail buffers

inline_for_extraction noextract
let source_first_rms_pre (addresses:source_input_addresses)
  (outputs:source_embedding_addresses) (norm:PA.larray BF16.t 7168)
  (input:source_inputs) : PC.slprop =
  source_embedding_kernel_post addresses outputs input **
  (exists* (old:source_hidden_values). PA.pts_to norm old)

inline_for_extraction noextract
let source_first_rms_post (addresses:source_input_addresses)
  (outputs:source_embedding_addresses) (norm:PA.larray BF16.t 7168)
  (input:source_inputs) : PC.slprop =
  PA.pts_to addresses.weights_address input.weights **
  PA.pts_to addresses.prompt_address input.prompt **
  (exists* (hidden:source_hidden_values) (rotary:source_rotary_values)
    (normalized:source_hidden_values).
    PA.pts_to outputs.embedded_address hidden **
    PA.pts_to outputs.rotary_address rotary **
    PA.pts_to norm normalized **
    PC.pure (source_embedding_result input hidden rotary /\
      source_rms_result input.weights hidden normalized 155583488))

inline_for_extraction noextract
let source_remaining_rms_storage
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) : PC.slprop =
  RT.contents buffers.token_buffer **
  RuntimeHelpers.all_contents (source_remaining_after_rms buffers)

inline_for_extraction noextract
let source_after_rmsnorm (addresses:source_input_addresses) (input:source_inputs)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) : PC.slprop =
  PC.on KL.gpu_loc
    (source_first_rms_post addresses (source_allocated_embedding buffers)
      (source_allocated_norm0 buffers) input) **
  source_remaining_rms_storage buffers

(* Only original launch 2 is appended to the retained allocation/embedding
   prefix. The other 227 source launches and both returned results remain owed. *)
inline_for_extraction noextract
fn source_prepare_and_enqueue_rms
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_prefix_outcome
  ensures (
    match outcome with
    | SourcePrefixFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourcePrefixEnqueued buffers ->
      Epoch.epoch_live stream (start + 2) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream (start + 2))
        (source_after_rmsnorm addresses input buffers))

inline_for_extraction noextract
let source_added_values (left right:source_hidden_values) : GTot source_hidden_values =
  S.init_ghost 7168 (fun i ->
    SourceCasts.fcast #F32.t #BF16.t
      (F.add (SourceCasts.fcast #BF16.t #F32.t (S.index left i))
        (SourceCasts.fcast #BF16.t #F32.t (S.index right i))))

inline_for_extraction noextract
let source_add_norm_result (weights:values BF16.t weight_elements)
  (left right combined normalized:source_hidden_values)
  (offset:nat{offset + 1024 <= weight_elements}) : prop =
  combined == source_added_values left right /\
  source_rms_result weights combined normalized offset

inline_for_extraction noextract
let source_silu_cell (input:values BF16.t 43008)
  (row:Shared.index 7) (column:Shared.index 3072) : BF16.t =
  let gate = SourceCasts.fcast #BF16.t #F32.t (S.index input (row * 6144 + column)) in
  let activated = SourceCasts.fcast #F32.t #BF16.t
    (F.div gate (F.add (F.one #F32.t) (F.fexp (NativeNeg.neg gate)))) in
  SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t activated)
      (SourceCasts.fcast #BF16.t #F32.t (S.index input (row * 6144 + 3072 + column))))

inline_for_extraction noextract
let source_silu_result (input:values BF16.t 43008) (output:values BF16.t 21504) : prop =
  forall (row:Shared.index 7) (column:Shared.index 3072).
    S.index output (row * 3072 + column) == source_silu_cell input row column

inline_for_extraction noextract
val source_qk_values (weights:values BF16.t weight_elements)
  (qkv:values BF16.t 28672) (rotary:source_rotary_values)
  (offset:nat{offset + 256 <= weight_elements}) : GTot (values BF16.t 21504)

inline_for_extraction noextract
val source_attention_result
  (weights:values BF16.t weight_elements) (qkv:values BF16.t 28672)
  (rotary:source_rotary_values) (qk:values BF16.t 21504)
  (output:values BF16.t 14336) (offset:nat{offset + 256 <= weight_elements}) : GTot prop

inline_for_extraction noextract
noeq type source_block_values = {
  hidden0: source_hidden_values;
  rotary0: source_rotary_values;
  norm0: source_hidden_values;
  qkv0: values BF16.t 28672;
  attn0: values BF16.t 14336;
  projected0: source_hidden_values;
  residual0: source_hidden_values;
  postnorm0: source_hidden_values;
  gateup0: values BF16.t 43008;
  activated0: values BF16.t 21504;
  down0: source_hidden_values;
  hidden1: source_hidden_values;
  norm1: source_hidden_values
}

inline_for_extraction noextract
noeq type source_block_addresses = {
  b_hidden0: a:PA.larray BF16.t 7168{KA.is_global_array a};
  b_rotary0: a:PA.larray BF16.t 1792{KA.is_global_array a};
  b_norm0: a:PA.larray BF16.t 7168{KA.is_global_array a};
  b_qkv0: a:PA.larray BF16.t 28672{KA.is_global_array a};
  b_attn0: a:PA.larray BF16.t 14336{KA.is_global_array a};
  b_projected0: a:PA.larray BF16.t 7168{KA.is_global_array a};
  b_residual0: a:PA.larray BF16.t 7168{KA.is_global_array a};
  b_postnorm0: a:PA.larray BF16.t 7168{KA.is_global_array a};
  b_gateup0: a:PA.larray BF16.t 43008{KA.is_global_array a};
  b_activated0: a:PA.larray BF16.t 21504{KA.is_global_array a};
  b_down0: a:PA.larray BF16.t 7168{KA.is_global_array a};
  b_hidden1: a:PA.larray BF16.t 7168{KA.is_global_array a};
  b_norm1: a:PA.larray BF16.t 7168{KA.is_global_array a}
}

inline_for_extraction noextract
let source_block_storage (a:source_block_addresses) (v:source_block_values)
  (fraction:Permission.perm) : PC.slprop =
  PA.pts_to a.b_hidden0 #fraction v.hidden0 **
  PA.pts_to a.b_rotary0 #fraction v.rotary0 **
  PA.pts_to a.b_norm0 #fraction v.norm0 **
  PA.pts_to a.b_qkv0 #fraction v.qkv0 **
  PA.pts_to a.b_attn0 #fraction v.attn0 **
  PA.pts_to a.b_projected0 #fraction v.projected0 **
  PA.pts_to a.b_residual0 #fraction v.residual0 **
  PA.pts_to a.b_postnorm0 #fraction v.postnorm0 **
  PA.pts_to a.b_gateup0 #fraction v.gateup0 **
  PA.pts_to a.b_activated0 #fraction v.activated0 **
  PA.pts_to a.b_down0 #fraction v.down0 **
  PA.pts_to a.b_hidden1 #fraction v.hidden1 **
  PA.pts_to a.b_norm1 #fraction v.norm1 ** PC.emp

inline_for_extraction noextract
let source_block_arrays (a:source_block_addresses) : list (PA.array BF16.t) =
  [a.b_hidden0; a.b_rotary0; a.b_norm0; a.b_qkv0; a.b_attn0; a.b_projected0;
    a.b_residual0; a.b_postnorm0; a.b_gateup0; a.b_activated0; a.b_down0;
    a.b_hidden1; a.b_norm1]

inline_for_extraction noextract
let source_block_sequences (v:source_block_values) : GTot (list (S.seq BF16.t)) =
  [v.hidden0; v.rotary0; v.norm0; v.qkv0; v.attn0; v.projected0; v.residual0;
    v.postnorm0; v.gateup0; v.activated0; v.down0; v.hidden1; v.norm1]

inline_for_extraction noextract
let source_block_stage (input:source_inputs) (v:source_block_values) (stage:nat{stage <= 8}) : prop =
  source_embedding_result input v.hidden0 v.rotary0 /\
  source_rms_result input.weights v.hidden0 v.norm0 155583488 /\
  (stage >= 1 ==> v.qkv0 == source_linear_values 4096 1024 input.weights v.norm0 155585536 64 1) /\
  (stage >= 2 ==> exists (qk:values BF16.t 21504).
    source_attention_result input.weights v.qkv0 v.rotary0 qk v.attn0 161876992) /\
  (stage >= 3 ==> v.projected0 == source_linear_values 1024 2048 input.weights v.attn0 159779840 128 1) /\
  (stage >= 4 ==> source_add_norm_result input.weights v.hidden0 v.projected0
    v.residual0 v.postnorm0 155584512) /\
  (stage >= 5 ==> v.gateup0 == source_linear_values 6144 1024 input.weights v.postnorm0 161877248 64 1) /\
  (stage >= 6 ==> source_silu_result v.gateup0 v.activated0) /\
  (stage >= 7 ==> v.down0 == source_linear_values 1024 3072 input.weights v.activated0 168168704 96 2) /\
  (stage >= 8 ==> source_add_norm_result input.weights v.residual0 v.down0
    v.hidden1 v.norm1 171314432)

inline_for_extraction noextract
let source_allocated_array_at
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (index:nat{index < 284 /\ index < List.length buffers.bf16_buffers /\
    index < List.length source_bf16_allocation_shapes})
  : GTot (a:PA.array BF16.t{
      a == RT.data (List.index buffers.bf16_buffers index) /\
      PA.length a == RT.numel (List.index source_bf16_allocation_shapes index) /\
      KA.is_global_array a}) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  source_matching_allocation_at source_bf16_allocation_shapes buffers.bf16_buffers index;
  RT.data (List.index buffers.bf16_buffers index)

inline_for_extraction noextract
let source_first_block_shapes ()
  : Lemma
      (requires List.length source_bf16_allocation_shapes == 284)
      (ensures
      List.index source_bf16_allocation_shapes 0 == [7; 1024] /\
      List.index source_bf16_allocation_shapes 1 == [2; 7; 128] /\
      List.index source_bf16_allocation_shapes 2 == [7; 1024] /\
      List.index source_bf16_allocation_shapes 3 == [7; 4096] /\
      List.index source_bf16_allocation_shapes 4 == [7; 2048] /\
      List.index source_bf16_allocation_shapes 5 == [7; 1024] /\
      List.index source_bf16_allocation_shapes 6 == [7; 1024] /\
      List.index source_bf16_allocation_shapes 7 == [7; 1024] /\
      List.index source_bf16_allocation_shapes 8 == [7; 6144] /\
      List.index source_bf16_allocation_shapes 9 == [7; 3072] /\
      List.index source_bf16_allocation_shapes 10 == [7; 1024] /\
      List.index source_bf16_allocation_shapes 11 == [7; 1024] /\
      List.index source_bf16_allocation_shapes 12 == [7; 1024]) =
  assert_norm (List.index source_bf16_allocation_shapes 0 == [7; 1024]);
  assert_norm (List.index source_bf16_allocation_shapes 1 == [2; 7; 128]);
  assert_norm (List.index source_bf16_allocation_shapes 2 == [7; 1024]);
  assert_norm (List.index source_bf16_allocation_shapes 3 == [7; 4096]);
  assert_norm (List.index source_bf16_allocation_shapes 4 == [7; 2048]);
  assert_norm (List.index source_bf16_allocation_shapes 5 == [7; 1024]);
  assert_norm (List.index source_bf16_allocation_shapes 6 == [7; 1024]);
  assert_norm (List.index source_bf16_allocation_shapes 7 == [7; 1024]);
  assert_norm (List.index source_bf16_allocation_shapes 8 == [7; 6144]);
  assert_norm (List.index source_bf16_allocation_shapes 9 == [7; 3072]);
  assert_norm (List.index source_bf16_allocation_shapes 10 == [7; 1024]);
  assert_norm (List.index source_bf16_allocation_shapes 11 == [7; 1024]);
  assert_norm (List.index source_bf16_allocation_shapes 12 == [7; 1024])

inline_for_extraction noextract
let source_allocated_first_block
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : GTot source_block_addresses =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  source_first_block_shapes ();
  {
    b_hidden0 = source_allocated_array_at buffers 0;
    b_rotary0 = source_allocated_array_at buffers 1;
    b_norm0 = source_allocated_array_at buffers 2;
    b_qkv0 = source_allocated_array_at buffers 3;
    b_attn0 = source_allocated_array_at buffers 4;
    b_projected0 = source_allocated_array_at buffers 5;
    b_residual0 = source_allocated_array_at buffers 6;
    b_postnorm0 = source_allocated_array_at buffers 7;
    b_gateup0 = source_allocated_array_at buffers 8;
    b_activated0 = source_allocated_array_at buffers 9;
    b_down0 = source_allocated_array_at buffers 10;
    b_hidden1 = source_allocated_array_at buffers 11;
    b_norm1 = source_allocated_array_at buffers 12
  }

inline_for_extraction noextract
let source_remaining_after_block
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Tot (list (RT.tensor BF16.t)) =
  (List.splitAt 13 buffers.bf16_buffers)._2

inline_for_extraction noextract
let source_remaining_block_storage
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) : PC.slprop =
  RT.contents buffers.token_buffer **
  RuntimeHelpers.all_contents (source_remaining_after_block buffers)

inline_for_extraction noextract
let source_first_block_layer (v:source_block_values) (qk:values BF16.t 21504)
  : GTot (Shared.layer_values F32.t) = {
  hidden_after = (fun p d -> SourceCasts.fcast #BF16.t #F32.t (S.index v.hidden1 (p * 1024 + d)));
  keys = (fun p h d -> SourceCasts.fcast #BF16.t #F32.t (S.index qk (p * 3072 + 2048 + h * 128 + d)));
  values = (fun p h d -> SourceCasts.fcast #BF16.t #F32.t
    (S.index v.qkv0 (p * 4096 + 3072 + h * 128 + d)))
}

inline_for_extraction noextract
let source_first_block_cache (v:source_block_values) (qk:values BF16.t 21504)
  : GTot (Shared.cache_state F32.t) =
  Shared.update_cache (fun _ -> Shared.Empty) 0 (source_first_block_layer v qk)

inline_for_extraction noextract
let source_first_block_cache_frame (v:source_block_values) (qk:values BF16.t 21504)
  : Lemma (
      Shared.cache_length (source_first_block_cache v qk 0) == 7 /\
      (forall (layer:Shared.index 28{layer > 0}).
        source_first_block_cache v qk layer == Shared.Empty)) = ()

inline_for_extraction noextract
let source_first_block_native_layer (v:source_block_values) (qk:values BF16.t 21504)
  : GTot (Shared.layer_values BF16.t) = {
  hidden_after = hidden_view v.hidden1;
  keys = (fun p h d -> S.index qk (p * 3072 + 2048 + h * 128 + d));
  values = (fun p h d -> S.index v.qkv0 (p * 4096 + 3072 + h * 128 + d))
}

inline_for_extraction noextract
let source_first_block_numerical (input:source_inputs) (v:source_block_values) : GTot prop =
  let qk = source_qk_values input.weights v.qkv0 v.rotary0 161876992 in
  Shared.decoder_relation (source_parameters input 0)
    (hidden_view v.hidden0) (rotary_view v.rotary0) (source_first_block_native_layer v qk) /\
  Shared.final_norm_relation (source_parameters input 1).input_layernorm
    (hidden_view v.hidden1) (hidden_view v.norm1)

inline_for_extraction noextract
let source_first_block_observation (input:source_inputs) (v:source_block_values) : GTot prop =
  let qk = source_qk_values input.weights v.qkv0 v.rotary0 161876992 in
  source_block_stage input v 8 /\
  source_first_block_numerical input v /\
  source_attention_result input.weights v.qkv0 v.rotary0 qk v.attn0 161876992 /\
  Shared.cache_length (source_first_block_cache v qk 0) == 7 /\
  (forall (layer:Shared.index 28{layer > 0}).
    source_first_block_cache v qk layer == Shared.Empty)

inline_for_extraction noextract
let source_after_first_block (addresses:source_input_addresses) (input:source_inputs)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) : PC.slprop =
  PC.on KL.gpu_loc (
    PA.pts_to addresses.weights_address input.weights **
    PA.pts_to addresses.prompt_address input.prompt **
    (exists* (v:source_block_values).
      source_block_storage (source_allocated_first_block buffers) v 1.0R **
      PC.pure (source_first_block_observation input v))) **
  source_remaining_block_storage buffers

(* This milestone ends at source line 529, not at either returned tensor.
   All 285 owners remain live and every result remains pending on this queue. *)
inline_for_extraction noextract
fn first_block_refines
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_prefix_outcome
  ensures (
    match outcome with
    | SourcePrefixFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourcePrefixEnqueued buffers ->
      Epoch.epoch_live stream (start + 10) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream (start + 10))
        (source_after_first_block addresses input buffers))

inline_for_extraction noextract
let source_layer_stage (input:source_inputs) (layer:Shared.index 28)
  (v:source_block_values) (stage:nat{stage <= 8}) : prop =
  source_rms_result input.weights v.hidden0 v.norm0 (layer_start layer) /\
  (stage >= 1 ==> v.qkv0 == source_linear_values 4096 1024 input.weights v.norm0
    (layer_start layer + 2048) 64 1) /\
  (stage >= 2 ==> source_attention_result input.weights v.qkv0 v.rotary0
    (source_qk_values input.weights v.qkv0 v.rotary0 (layer_start layer + 6293504))
    v.attn0 (layer_start layer + 6293504)) /\
  (stage >= 3 ==> v.projected0 == source_linear_values 1024 2048 input.weights v.attn0
    (layer_start layer + 4196352) 128 1) /\
  (stage >= 4 ==> source_add_norm_result input.weights v.hidden0 v.projected0
    v.residual0 v.postnorm0 (layer_start layer + 1024)) /\
  (stage >= 5 ==> v.gateup0 == source_linear_values 6144 1024 input.weights v.postnorm0
    (layer_start layer + 6293760) 64 1) /\
  (stage >= 6 ==> source_silu_result v.gateup0 v.activated0) /\
  (stage >= 7 ==> v.down0 == source_linear_values 1024 3072 input.weights v.activated0
    (layer_start layer + 12585216) 96 2) /\
  (stage >= 8 ==> v.hidden1 == source_added_values v.residual0 v.down0) /\
  (stage >= 8 /\ layer < 27 ==> source_rms_result input.weights v.hidden1 v.norm1
    (layer_start layer + layer_weight_elements))

inline_for_extraction noextract
fn source_residual_add_body
  (left right output:PA.larray BF16.t 7168)
  (#lv #rv #old:Ghost.erased source_hidden_values)
  preserves KL.gpu ** PA.pts_to left lv ** PA.pts_to right rv
  requires PA.pts_to output old
  ensures PA.pts_to output (source_added_values lv rv)

inline_for_extraction noextract
let source_selection_result
  (logits:values BF16.t (7 * 151936)) (prompt:values I64.t 7)
  (output:values I64.t 8) : prop =
  (forall (p:Shared.index 7). S.index output p == S.index prompt p) /\
  (exists (winner:Shared.token_id).
    I64.v (S.index output 7) == winner /\
    selector_first_winner (selector_row_scores logits) winner)

inline_for_extraction noextract
fn source_select_token_body
  (logits:PA.larray BF16.t (7 * 151936)) (prompt:PA.larray I64.t 7)
  (output:PA.larray I64.t 8)
  (#lv:Ghost.erased (values BF16.t (7 * 151936)))
  (#pv:Ghost.erased (values I64.t 7))
  (#old:Ghost.erased (S.seq I64.t){S.length old == 8})
  preserves KL.gpu ** PA.pts_to logits lv ** PA.pts_to prompt pv
  requires PA.pts_to output old
  ensures exists* (result:values I64.t 8).
    PA.pts_to output result ** PC.pure (source_selection_result lv pv result)

inline_for_extraction noextract
let source_program_size (index:Shared.index 284) : pos =
  if index = 1 then 1792
  else if index = 283 then 7 * 151936
  else if 2 <= index && index < 282 then
    let slot = (index - 2) % 10 in
    if slot = 1 then 28672
    else if slot = 2 then 14336
    else if slot = 6 then 43008
    else if slot = 7 then 21504
    else 7168
  else 7168

inline_for_extraction noextract
let source_program_index (layer:Shared.index 28) (slot:Shared.index 13) : Shared.index 284 =
  if slot = 0 then (if layer = 0 then 0 else 10 * layer + 1)
  else if slot = 1 then 1
  else 10 * layer + slot

inline_for_extraction noextract
type source_program_values =
  s:values (S.seq BF16.t) 284{
    forall (index:Shared.index 284). S.length (S.index s index) == source_program_size index}

inline_for_extraction noextract
let source_program_slot (v:source_program_values) (index:Shared.index 284)
  : GTot (values BF16.t (source_program_size index)) = S.index v index

inline_for_extraction noextract
let source_program_replace (v:source_program_values) (index:Shared.index 284)
  (value:values BF16.t (source_program_size index)) : GTot source_program_values =
  S.upd v index value

inline_for_extraction noextract
let source_program_layer (v:source_program_values) (layer:Shared.index 28)
  : GTot source_block_values = {
  hidden0 = source_program_slot v (source_program_index layer 0);
  rotary0 = source_program_slot v 1;
  norm0 = source_program_slot v (source_program_index layer 2);
  qkv0 = source_program_slot v (source_program_index layer 3);
  attn0 = source_program_slot v (source_program_index layer 4);
  projected0 = source_program_slot v (source_program_index layer 5);
  residual0 = source_program_slot v (source_program_index layer 6);
  postnorm0 = source_program_slot v (source_program_index layer 7);
  gateup0 = source_program_slot v (source_program_index layer 8);
  activated0 = source_program_slot v (source_program_index layer 9);
  down0 = source_program_slot v (source_program_index layer 10);
  hidden1 = source_program_slot v (source_program_index layer 11);
  norm1 = source_program_slot v (source_program_index layer 12)
}

inline_for_extraction noextract
let source_program_stage (input:source_inputs) (v:source_program_values)
  (calls:nat{2 <= calls /\ calls <= 228}) : prop =
  source_embedding_result input (source_program_slot v 0) (source_program_slot v 1) /\
  (forall (layer:Shared.index 28).
    calls >= 2 + 8 * layer ==>
      source_layer_stage input layer (source_program_layer v layer)
        (if calls >= 10 + 8 * layer then 8 else calls - (2 + 8 * layer))) /\
  (calls >= 227 ==> source_rms_result input.weights
    (source_program_slot v 281) (source_program_slot v 282) 155582464) /\
  (calls >= 228 ==> source_program_slot v 283 ==
    source_linear_values 151936 1024 input.weights (source_program_slot v 282) 0 64 1)

inline_for_extraction noextract
let rec source_list_index_append (#a:Type0) (left right:list a)
  (index:nat{index < List.length left + List.length right})
  : Lemma (
      List.length (List.append left right) == List.length left + List.length right /\
      List.index (List.append left right) index ==
        (if index < List.length left then List.index left index
         else List.index right (index - List.length left))) (decreases left) =
  ListProps.append_length left right;
  match left with
  | [] -> ()
  | _::tail -> if index > 0 then source_list_index_append tail right (index - 1)

#push-options "--fuel 20 --ifuel 1"
inline_for_extraction noextract
let rec source_layer_shape_at (count:pos{count <= 28})
  (index:nat{index < 10 * count /\ index < List.length (source_layer_allocations count)})
  : Lemma (List.index (source_layer_allocations count) index ==
      List.index source_layer_allocation_shapes (index % 10)) (decreases count) =
  source_layer_allocation_count (count - 1);
  source_list_index_append source_layer_allocation_shapes (source_layer_allocations (count - 1)) index;
  if index >= 10 then (
    source_layer_shape_at (count - 1) (index - 10);
    assert ((index - 10) % 10 == index % 10))

inline_for_extraction noextract
let source_program_shape_at (index:Shared.index 284)
  : Lemma (
      List.length source_bf16_allocation_shapes == 284 /\
      RT.numel (List.index source_bf16_allocation_shapes index) == source_program_size index) =
  source_allocation_count ();
  source_layer_allocation_count 28;
  source_list_index_append [[7; 1024]; [2; 7; 128]]
    (List.append (source_layer_allocations 28) [[7; 1024]; [7; 151936]]) index;
  if index >= 2 then (
    source_list_index_append (source_layer_allocations 28) [[7; 1024]; [7; 151936]] (index - 2);
    if index < 282 then source_layer_shape_at 28 (index - 2))
#pop-options

inline_for_extraction noextract
let source_program_address
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (index:Shared.index 284)
  : GTot (a:PA.larray BF16.t (source_program_size index){KA.is_global_array a}) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  source_program_shape_at index;
  source_allocated_array_at buffers index

open Kuiper.ForEvery

inline_for_extraction noextract
let source_program_storage
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (v:source_program_values) (fraction:Permission.perm) : PC.slprop =
  forall+ (index:Shared.index 284).
    PA.pts_to (source_program_address buffers index) #fraction (source_program_slot v index)

inline_for_extraction noextract
let source_program_storage_equation
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (v:source_program_values) (fraction:Permission.perm)
  : Lemma (source_program_storage buffers v fraction ==
      (forall+ (i:Shared.index 284).
        PA.pts_to (source_program_address buffers i) #fraction (source_program_slot v i))) =
  assert (source_program_storage buffers v fraction ==
    (forall+ (i:Shared.index 284).
      PA.pts_to (source_program_address buffers i) #fraction (source_program_slot v i))) by
    (FStar.Tactics.V2.norm [delta_only [`%source_program_storage]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let source_program_masked_storage
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (v:source_program_values) (fraction:Permission.perm)
  (mask:Shared.index 284 -> prop) : PC.slprop =
  forall+ (cell_id:Shared.index 284{mask cell_id}).
    PA.pts_to (source_program_address buffers cell_id) #fraction (source_program_slot v cell_id)

inline_for_extraction noextract
let source_program_condition
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228})
  (fraction:Permission.perm) : PC.slprop =
  exists* (v:source_program_values).
    source_program_storage buffers v fraction ** PC.pure (source_program_stage input v calls)

inline_for_extraction noextract
let source_program_core (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228})
  (fraction:Permission.perm) : PC.slprop =
  PA.pts_to addresses.weights_address #fraction input.weights **
  source_program_condition buffers input calls fraction

inline_for_extraction noextract
let source_program_layer32 (input:source_inputs) (v:source_program_values) (layer:Shared.index 28)
  : GTot (Shared.layer_values F32.t) =
  let values = source_program_layer v layer in
  source_first_block_layer values
    (source_qk_values input.weights values.qkv0 values.rotary0 (layer_start layer + 6293504))

inline_for_extraction noextract
let source_program_cache (input:source_inputs) (v:source_program_values) (count:Shared.index 29)
  : GTot (Shared.cache_state F32.t) =
  fun layer -> if layer < count then
    Shared.Prefilled {
      cached_keys = (source_program_layer32 input v layer).keys;
      cached_values = (source_program_layer32 input v layer).values
    }
    else Shared.Empty

inline_for_extraction noextract
let source_program_execution (input:source_inputs) (v:source_program_values) (selected:Shared.token_id)
  : GTot (Shared.execution F32.t F32.t) = {
  embedded = hidden_view (source_program_slot v 0);
  position_ids = (fun p -> p);
  cache_positions = (fun p -> p);
  checked_mask = Shared.prefill_mask;
  rotary_initial = source_observed_rotary_state ();
  rotary_final = source_observed_rotary_state ();
  rope = source_observed_rotary (source_program_slot v 1);
  layer = source_program_layer32 input v;
  cache_at = source_program_cache input v;
  normalized = hidden_view (source_program_slot v 282);
  scores = (fun token -> S.index (selector_row_scores (source_program_slot v 283)) token);
  selected = selected
}

inline_for_extraction noextract
let source_program_tokens (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  : GTot Shared.returned_tokens =
  S.init_ghost #Shared.token_id 8 (fun i -> I64.v (S.index tokens i))

inline_for_extraction noextract
let source_program_outputs (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  : GTot Shared.outputs = {
  tokens = source_program_tokens input v tokens;
  normalized_hidden = hidden_view (source_program_slot v 282)
}

inline_for_extraction noextract
let source_program_after_state (#frame_t:Type0) (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  (ambient:frame_t) : GTot (Shared.post_state F32.t frame_t) = {
  owned_inputs = input_mapping input;
  history = source_program_tokens input v tokens;
  cache = source_program_cache input v 28;
  rotary = source_observed_rotary_state ();
  frame = ambient
}

inline_for_extraction noextract
let source_program_result (#frame_t:Type0) (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  (ambient:frame_t) : GTot prop =
  let selected:Shared.token_id = I64.v (S.index tokens 7) in
  let run = source_program_execution input v selected in
  let output = source_program_outputs input v tokens in
  let after = source_program_after_state input v tokens ambient in
  Shared.structural_postcondition (input_mapping input) ambient run output after /\
  (source_rotary_domain () ==> Shared.numerical_execution (input_mapping input) run) /\
  (Shared.selector_domain run.scores ==> Shared.maximal_actual_index run.scores selected) /\
  (source_rotary_domain () /\ Shared.selector_domain run.scores ==>
    Shared.postcondition (input_mapping input) ambient run output after /\
    Shared.numerical_denotation (input_mapping input) run)

inline_for_extraction noextract
noeq type source_returned_buffers = {
  returned_tokens: RT.tensor I64.t;
  returned_hidden: RT.tensor BF16.t
}

inline_for_extraction noextract
let source_returned_metadata (output:source_returned_buffers) : prop =
  RT.metadata output.returned_tokens [1; 8] /\ RT.metadata output.returned_hidden [7; 1024]

inline_for_extraction noextract
type source_returned_result = output:source_returned_buffers{source_returned_metadata output}

inline_for_extraction noextract
noeq type source_program_outcome =
  | SourceProgramFailed : error:RT.allocation_error -> source_program_outcome
  | SourceProgramReturned :
      output:source_returned_buffers{source_returned_metadata output} -> source_program_outcome

inline_for_extraction noextract
let source_program_return_frame (#frame_t:Type0)
  (addresses:source_input_addresses) (input:source_inputs)
  (output:source_returned_buffers{source_returned_metadata output}) (frame:frame_t) : PC.slprop =
  PC.on KL.gpu_loc (
    PA.pts_to addresses.weights_address input.weights **
    PA.pts_to addresses.prompt_address input.prompt **
    (exists* (v:source_program_values)
      (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens}).
      PA.pts_to (RT.data output.returned_tokens) tokens **
      PA.pts_to (RT.data output.returned_hidden) (source_program_slot v 282) **
      PC.pure (source_program_stage input v 228 /\ source_program_result input v tokens frame)))

inline_for_extraction noextract
fn program_refines (#frame_t:Type0) (#logical_frame:Ghost.erased frame_t)
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_program_outcome
  ensures (
    match outcome with
    | SourceProgramFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourceProgramReturned output ->
      Epoch.epoch_live stream (start + 229) **
      RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
      Pledge.pledge0 (Epoch.epoch_done stream (start + 229))
        (source_program_return_frame addresses input output logical_frame))
