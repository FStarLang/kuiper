module KPlay.Proof.Optimized.FamilyAttention

#lang-pulse

open Kuiper
module S = FStar.Seq
module R = FStar.Real
module Sqrt = FStar.Math.Sqrt
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module Fast = Kuiper.Float32.FastMath
module Approx = Kuiper.Approximates.Base
module Cast = Kuiper.Float.Casts
module C = FStar.Classical
module SC = Kuiper.Seq.Common
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
module Sh = KPlay.Proof.Shared.Attention
module RoPE = Luce.RoPE
module A = Pulse.Lib.Array
module Q = Kuiper.Seq.Common
module Exp = FStar.Math.Exp
module Proof = FStar.Tactics.V2

inline_for_extraction noextract
let hidden_width : pos = 1024
inline_for_extraction noextract
let query_heads : pos = 8
inline_for_extraction noextract
let kv_heads : pos = 2
inline_for_extraction noextract
let head_width : pos = 256
inline_for_extraction noextract
let rotary_half : pos = 32
inline_for_extraction noextract
let cache_positions : pos = 2048
inline_for_extraction noextract
let partitions : pos = 16
inline_for_extraction noextract
let mlp_width : pos = 3584
inline_for_extraction noextract
let activation_width : pos = query_heads * head_width
inline_for_extraction noextract
let query_gate_width : pos = 2 * activation_width
inline_for_extraction noextract
let kv_width : pos = kv_heads * head_width
inline_for_extraction noextract
let cache_head_elements : pos = cache_positions * head_width
inline_for_extraction noextract
let cache_elements : pos = kv_heads * cache_head_elements

inline_for_extraction noextract
let input_norm_offset : nat = 0
inline_for_extraction noextract
let post_norm_offset : nat = hidden_width
inline_for_extraction noextract
let mlp_gate_offset : nat = 2 * hidden_width
inline_for_extraction noextract
let mlp_up_offset : nat = mlp_gate_offset + mlp_width * hidden_width
inline_for_extraction noextract
let mlp_down_offset : nat = mlp_up_offset + mlp_width * hidden_width
inline_for_extraction noextract
let query_gate_offset : nat = mlp_down_offset + hidden_width * mlp_width
inline_for_extraction noextract
let key_offset : nat = query_gate_offset + query_gate_width * hidden_width
inline_for_extraction noextract
let value_offset : nat = key_offset + kv_width * hidden_width
inline_for_extraction noextract
let query_norm_offset : nat = value_offset + kv_width * hidden_width
inline_for_extraction noextract
let key_norm_offset : nat = query_norm_offset + head_width
inline_for_extraction noextract
let output_offset : nat = key_norm_offset + head_width
inline_for_extraction noextract
let weight_elements : pos = output_offset + hidden_width * activation_width

inline_for_extraction noextract
let valid_position (position:int) : prop =
  0 <= position /\ position < cache_positions

inline_for_extraction noextract
let position_index = p:nat{p < cache_positions}

inline_for_extraction noextract
let previous_elements (position:position_index) : nat =
  if position = 0 then 0 else cache_elements

inline_for_extraction noextract
let score_elements (position:position_index) : pos =
  query_heads * (position + 1)

inline_for_extraction noextract
let real_vector (count:nat) = lseq real count

inline_for_extraction noextract
let kv_head_for_query (head:nat{head < query_heads}) : natlt kv_heads =
  head / (query_heads / kv_heads)

inline_for_extraction noextract
let cache_index
  (head:nat{head < kv_heads})
  (time:nat{time < cache_positions})
  (coordinate:nat{coordinate < head_width})
  : natlt cache_elements =
  head * cache_head_elements + time * head_width + coordinate

(* This restricts live cells, not the intentional empty-partition sentinel. *)
inline_for_extraction noextract
let bilateral_live_scores_finite
  (position:position_index) (scores:lseq f32 (score_elements position)) : prop =
  forall (i:nat{i < score_elements position}). Finite? (kind (S.index scores i))

inline_for_extraction noextract
let bfseq (n:nat) = lseq bf16 n

inline_for_extraction noextract
let represents (n:nat) (storage:bfseq n) (logical:Sh.vector n) : prop =
  forall (i:Sh.index n). v_approximates (S.index storage i) (logical i)

(* The weight inverse is reused from the checked base representation. *)
inline_for_extraction noextract
let weights_view (packed:Sh.vector weight_elements) : Sh.layer_weights = {
  input_norm = (fun i -> packed i);
  post_attention_norm = (fun i -> packed (1024 + i));
  query_gate_proj = (fun r c -> packed (11012096 + 1024 * r + c));
  key_proj = (fun r c -> packed (15206400 + 1024 * r + c));
  value_proj = (fun r c -> packed (15730688 + 1024 * r + c));
  query_norm = (fun i -> packed (16254976 + i));
  key_norm = (fun i -> packed (16255232 + i));
  output_proj = (fun r c -> packed (16255488 + 2048 * r + c));
  mlp_gate_proj = (fun r c -> packed (2048 + 1024 * r + c));
  mlp_up_proj = (fun r c -> packed (3672064 + 1024 * r + c));
  mlp_down_proj = (fun r c -> packed (7342080 + 3584 * r + c))
}

inline_for_extraction noextract
let pack_weights (w:Sh.layer_weights) : Sh.vector weight_elements =
  fun i ->
    if i < 1024 then w.input_norm i
    else if i < 2048 then w.post_attention_norm (i - 1024)
    else if i < 3672064 then
      w.mlp_gate_proj ((i - 2048) / 1024) ((i - 2048) % 1024)
    else if i < 7342080 then
      w.mlp_up_proj ((i - 3672064) / 1024) ((i - 3672064) % 1024)
    else if i < 11012096 then
      w.mlp_down_proj ((i - 7342080) / 3584) ((i - 7342080) % 3584)
    else if i < 15206400 then
      w.query_gate_proj ((i - 11012096) / 1024) ((i - 11012096) % 1024)
    else if i < 15730688 then
      w.key_proj ((i - 15206400) / 1024) ((i - 15206400) % 1024)
    else if i < 16254976 then
      w.value_proj ((i - 15730688) / 1024) ((i - 15730688) % 1024)
    else if i < 16255232 then w.query_norm (i - 16254976)
    else if i < 16255488 then w.key_norm (i - 16255232)
    else w.output_proj ((i - 16255488) / 2048) ((i - 16255488) % 2048)

inline_for_extraction noextract
let packed_weights_roundtrip (packed:Sh.vector weight_elements) (i:Sh.index weight_elements)
  : Lemma (pack_weights (weights_view packed) i == packed i) = ()

inline_for_extraction noextract
let logical_weights_roundtrip (w:Sh.layer_weights)
  : Lemma (
      let mapped = weights_view (pack_weights w) in
      (forall (i:Sh.index 1024). mapped.input_norm i == w.input_norm i) /\
      (forall (i:Sh.index 1024). mapped.post_attention_norm i == w.post_attention_norm i) /\
      (forall (r:Sh.index 4096) (c:Sh.index 1024).
        mapped.query_gate_proj r c == w.query_gate_proj r c) /\
      (forall (r:Sh.index 512) (c:Sh.index 1024). mapped.key_proj r c == w.key_proj r c) /\
      (forall (r:Sh.index 512) (c:Sh.index 1024). mapped.value_proj r c == w.value_proj r c) /\
      (forall (i:Sh.index 256). mapped.query_norm i == w.query_norm i) /\
      (forall (i:Sh.index 256). mapped.key_norm i == w.key_norm i) /\
      (forall (r:Sh.index 1024) (c:Sh.index 2048).
        mapped.output_proj r c == w.output_proj r c) /\
      (forall (r:Sh.index 3584) (c:Sh.index 1024).
        mapped.mlp_gate_proj r c == w.mlp_gate_proj r c) /\
      (forall (r:Sh.index 3584) (c:Sh.index 1024).
        mapped.mlp_up_proj r c == w.mlp_up_proj r c) /\
      (forall (r:Sh.index 1024) (c:Sh.index 3584).
        mapped.mlp_down_proj r c == w.mlp_down_proj r c)) = ()

(* Unlike the base ABI, LUCE retains all 2048 positions in head-major order. *)
inline_for_extraction noextract
let kv_view (n:nat{n <= cache_positions}) (flat:Sh.vector cache_elements)
  : Sh.kv_tensor n =
  fun h t d -> flat (cache_index h t d)

inline_for_extraction noextract
let pack_kv
  (n:nat{n <= cache_positions}) (logical:Sh.kv_tensor n)
  (tail:Sh.vector cache_elements) : Sh.vector cache_elements =
  fun i ->
    let h = i / cache_head_elements in
    let t = (i % cache_head_elements) / head_width in
    let d = i % head_width in
    if t < n then logical h t d else tail i

inline_for_extraction noextract
let kv_logical_roundtrip
  (n:nat{n <= cache_positions}) (logical:Sh.kv_tensor n)
  (tail:Sh.vector cache_elements)
  (h:Sh.index 2) (t:Sh.index n) (d:Sh.index 256)
  : Lemma (kv_view n (pack_kv n logical tail) h t d == logical h t d) = ()

inline_for_extraction noextract
let kv_physical_roundtrip
  (n:nat{n <= cache_positions}) (flat:Sh.vector cache_elements)
  (i:Sh.index cache_elements)
  : Lemma (pack_kv n (kv_view n flat) flat i == flat i) = ()

inline_for_extraction noextract
let history_view
  (n:pos{n <= cache_positions}) (keys values:Sh.vector cache_elements)
  : h:Sh.kv_history{h.length == n} = {
  length = n;
  keys = kv_view n keys;
  values = kv_view n values
}

inline_for_extraction noextract
let old_history_view
  (position:position_index)
  (keys values:Sh.vector (previous_elements position))
  : h:option Sh.kv_history{Sh.history_length h == position} =
  if position = 0 then None
  else Some (history_view position keys values)

inline_for_extraction noextract
let cache_view
  (call:Sh.invocation) (frame:Sh.hybrid_cache)
  (keys values:Sh.vector (previous_elements call.position))
  : before:Sh.hybrid_cache{Sh.domain call before} = {
  kv = (fun l ->
    if l = call.layer then old_history_view call.position keys values
    else frame.kv l);
  convolution = frame.convolution;
  recurrent = frame.recurrent
}

inline_for_extraction noextract
let cache_presence
  (call:Sh.invocation) (frame:Sh.hybrid_cache)
  (keys values:Sh.vector (previous_elements call.position))
  : Lemma (
      let before = cache_view call frame keys values in
      (call.position = 0 <==> before.kv call.layer == None) /\
      Sh.history_length (before.kv call.layer) == call.position) = ()

inline_for_extraction noextract
let result_view
  (call:Sh.invocation) (frame:Sh.hybrid_cache)
  (hidden:Sh.vector hidden_width)
  (keys values:Sh.vector cache_elements)
  : Sh.layer_result = {
  hidden = hidden;
  cache = Sh.replace_kv frame call.layer
    (history_view (call.position + 1) keys values)
}

inline_for_extraction noextract
let result_frame
  (call:Sh.invocation) (frame:Sh.hybrid_cache)
  (hidden:Sh.vector hidden_width)
  (keys values:Sh.vector cache_elements)
  : Lemma (
      let result = result_view call frame hidden keys values in
      Sh.history_length (result.cache.kv call.layer) == call.position + 1 /\
      (forall (l:Sh.layer_index). l <> call.layer ==>
        result.cache.kv l == frame.kv l) /\
      result.cache.convolution == frame.convolution /\
      result.cache.recurrent == frame.recurrent) =
  Sh.cache_frame frame call.layer (history_view (call.position + 1) keys values)

noextract noeq
type logical_inputs = {
  call: Sh.invocation;
  packed_weights: Sh.vector weight_elements;
  input_hidden: Sh.vector hidden_width;
  old_keys: Sh.vector (previous_elements call.position);
  old_values: Sh.vector (previous_elements call.position);
  frame: Sh.hybrid_cache
}

inline_for_extraction noextract
let logical_before (input:logical_inputs)
  : before:Sh.hybrid_cache{Sh.domain input.call before} =
  cache_view input.call input.frame input.old_keys input.old_values

inline_for_extraction noextract
let input_representation
  (input:logical_inputs)
  (weights:bfseq weight_elements) (hidden:bfseq hidden_width)
  (keys values:bfseq (previous_elements input.call.position)) : prop =
  represents weight_elements weights input.packed_weights /\
  represents hidden_width hidden input.input_hidden /\
  represents (previous_elements input.call.position) keys input.old_keys /\
  represents (previous_elements input.call.position) values input.old_values

inline_for_extraction noextract
let shared_queries (input:logical_inputs) : Sh.matrix query_heads head_width =
  let w = weights_view input.packed_weights in
  let u = Sh.rms_norm hidden_width w.input_norm input.input_hidden in
  let qg = Sh.linear query_gate_width hidden_width w.query_gate_proj u in
  fun h -> Sh.rotate_partial input.call.position
    (Sh.rms_norm head_width w.query_norm (Sh.split_queries qg h))

inline_for_extraction noextract
let retained_cache_post
  (position:position_index)
  (previous:bfseq (previous_elements position))
  (current:bfseq cache_elements) : prop =
  forall (i:nat{i < cache_elements}).
    (i % cache_head_elements) / head_width <> position ==>
    S.index current i ==
      (if position = 0 then (zero <: bf16) else S.index previous i)

noextract
let state_postcondition
  (input:logical_inputs)
  (old_keys old_values:bfseq (previous_elements input.call.position))
  (keys values:bfseq cache_elements) : prop =
  retained_cache_post input.call.position old_keys keys /\
  retained_cache_post input.call.position old_values values /\
  (forall (h:Sh.vector hidden_width) (k v:Sh.vector cache_elements).
    let result = result_view input.call input.frame h k v in
    Sh.history_length (result.cache.kv input.call.layer) == input.call.position + 1 /\
    not (result.cache.kv input.call.layer == None) /\
    (forall (l:Sh.layer_index). l <> input.call.layer ==>
      result.cache.kv l == (logical_before input).kv l) /\
    result.cache.convolution == (logical_before input).convolution /\
    result.cache.recurrent == (logical_before input).recurrent)

noextract
let semantic_postcondition
  (input:logical_inputs)
  (old_keys old_values:bfseq (previous_elements input.call.position))
  (hidden:bfseq hidden_width) (keys values:bfseq cache_elements) : prop =
  retained_cache_post input.call.position old_keys keys /\
  retained_cache_post input.call.position old_values values /\
  (exists (logical_hidden:Sh.vector hidden_width)
          (logical_keys:Sh.vector cache_elements)
          (logical_values:Sh.vector cache_elements).
    represents hidden_width hidden logical_hidden /\
    represents cache_elements keys logical_keys /\
    represents cache_elements values logical_values /\
    Sh.postcondition input.call (weights_view input.packed_weights)
      input.input_hidden (logical_before input)
      (result_view input.call input.frame logical_hidden logical_keys logical_values))

noextract
let source_score_post
  (input:logical_inputs)
  (old_keys old_values:bfseq (previous_elements input.call.position))
  (hidden:bfseq hidden_width) (keys values:bfseq cache_elements)
  (live_scores:lseq f32 (score_elements input.call.position)) : prop =
  state_postcondition input old_keys old_values keys values /\
  (bilateral_live_scores_finite input.call.position live_scores ==>
    semantic_postcondition input old_keys old_values hidden keys values)
