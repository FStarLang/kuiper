module KPlay.Functional

#lang-pulse

open Kuiper
open FStar.Real
module S = KPlay.SharedSpec
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module GA = Kuiper.Array
module Seq = FStar.Seq

inline_for_extraction noextract
let weight_count : pos = 18352640

inline_for_extraction noextract
let hidden_width : pos = 1024

inline_for_extraction noextract
let rope_width : pos = 64

inline_for_extraction noextract
let kv_width : pos = 512

inline_for_extraction noextract
let prefix_count (position:nat) : nat = position * kv_width

inline_for_extraction noextract
let next_count (position:nat) : pos = (position + 1) * kv_width

inline_for_extraction noextract
let bfseq (n:nat) : Type0 = lseq BF.t n

inline_for_extraction noextract
let represents (n:nat) (storage:bfseq n) (logical:S.vector n) : prop =
  forall (i:S.index n). v_approximates (Seq.index storage i) (logical i)

inline_for_extraction noextract
let weights_view (packed:S.vector weight_count) : S.layer_weights = {
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
let pack_weights (w:S.layer_weights) : S.vector weight_count =
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
let packed_weights_roundtrip (packed:S.vector weight_count) (i:S.index weight_count)
  : Lemma (pack_weights (weights_view packed) i == packed i) = ()

inline_for_extraction noextract
let logical_weights_roundtrip (w:S.layer_weights)
  : Lemma (
      let mapped = weights_view (pack_weights w) in
      (forall (i:S.index 1024). mapped.input_norm i == w.input_norm i) /\
      (forall (i:S.index 1024). mapped.post_attention_norm i == w.post_attention_norm i) /\
      (forall (r:S.index 4096) (c:S.index 1024).
        mapped.query_gate_proj r c == w.query_gate_proj r c) /\
      (forall (r:S.index 512) (c:S.index 1024). mapped.key_proj r c == w.key_proj r c) /\
      (forall (r:S.index 512) (c:S.index 1024). mapped.value_proj r c == w.value_proj r c) /\
      (forall (i:S.index 256). mapped.query_norm i == w.query_norm i) /\
      (forall (i:S.index 256). mapped.key_norm i == w.key_norm i) /\
      (forall (r:S.index 1024) (c:S.index 2048). mapped.output_proj r c == w.output_proj r c) /\
      (forall (r:S.index 3584) (c:S.index 1024).
        mapped.mlp_gate_proj r c == w.mlp_gate_proj r c) /\
      (forall (r:S.index 3584) (c:S.index 1024).
        mapped.mlp_up_proj r c == w.mlp_up_proj r c) /\
      (forall (r:S.index 1024) (c:S.index 3584).
        mapped.mlp_down_proj r c == w.mlp_down_proj r c)) = ()

inline_for_extraction noextract
let kv_view (n:nat) (flat:S.vector (prefix_count n)) : S.kv_tensor n =
  fun h t d -> flat (512 * t + 256 * h + d)

inline_for_extraction noextract
let flatten_kv (n:nat) (logical:S.kv_tensor n) : S.vector (prefix_count n) =
  fun i -> logical ((i % 512) / 256) (i / 512) (i % 256)

inline_for_extraction noextract
let kv_logical_roundtrip
  (n:nat) (logical:S.kv_tensor n)
  (h:S.index 2) (t:S.index n) (d:S.index 256)
  : Lemma (kv_view n (flatten_kv n logical) h t d == logical h t d) = ()

inline_for_extraction noextract
let kv_physical_roundtrip
  (n:nat) (flat:S.vector (prefix_count n)) (i:S.index (prefix_count n))
  : Lemma (flatten_kv n (kv_view n flat) i == flat i) = ()

inline_for_extraction noextract
let history_view
  (n:pos) (keys values:S.vector (prefix_count n))
  : h:S.kv_history{h.length == n} = {
  length = n;
  keys = kv_view n keys;
  values = kv_view n values
}

inline_for_extraction noextract
let old_history_view
  (position:nat) (keys values:S.vector (prefix_count position))
  : h:option S.kv_history{S.history_length h == position} =
  if position = 0 then None
  else Some (history_view position keys values)

inline_for_extraction noextract
let cache_view
  (call:S.invocation) (frame:S.hybrid_cache)
  (keys values:S.vector (prefix_count call.position))
  : before:S.hybrid_cache{S.domain call before} = {
  kv = (fun l ->
    if l = call.layer then old_history_view call.position keys values
    else frame.kv l);
  convolution = frame.convolution;
  recurrent = frame.recurrent
}

inline_for_extraction noextract
let cache_presence
  (call:S.invocation) (frame:S.hybrid_cache)
  (keys values:S.vector (prefix_count call.position))
  : Lemma (
      let before = cache_view call frame keys values in
      (call.position = 0 <==> before.kv call.layer == None) /\
      S.history_length (before.kv call.layer) == call.position) = ()

inline_for_extraction noextract
let result_view
  (call:S.invocation) (frame:S.hybrid_cache)
  (hidden:S.vector hidden_width)
  (keys values:S.vector (next_count call.position))
  : S.layer_result = {
  hidden = hidden;
  cache = S.replace_kv frame call.layer
    (history_view (call.position + 1) keys values)
}

inline_for_extraction noextract
let result_frame
  (call:S.invocation) (frame:S.hybrid_cache)
  (hidden:S.vector hidden_width)
  (keys values:S.vector (next_count call.position))
  : Lemma (
      let result = result_view call frame hidden keys values in
      S.history_length (result.cache.kv call.layer) == call.position + 1 /\
      (forall (l:S.layer_index). l <> call.layer ==>
        result.cache.kv l == frame.kv l) /\
      result.cache.convolution == frame.convolution /\
      result.cache.recurrent == frame.recurrent) =
  S.cache_frame frame call.layer (history_view (call.position + 1) keys values)

noextract noeq
type logical_inputs = {
  call: S.invocation;
  packed_weights: S.vector weight_count;
  input_hidden: S.vector hidden_width;
  old_keys: S.vector (prefix_count call.position);
  old_values: S.vector (prefix_count call.position);
  frame: S.hybrid_cache
}

inline_for_extraction noextract
let logical_before (input:logical_inputs)
  : before:S.hybrid_cache{S.domain input.call before} =
  cache_view input.call input.frame input.old_keys input.old_values

inline_for_extraction noextract
let rope_represents (position:nat) (storage:bfseq rope_width) : prop =
  (forall (j:S.index 32).
    v_approximates (Seq.index storage j) (S.rope_cos position j)) /\
  (forall (j:S.index 32).
    v_approximates (Seq.index storage (32 + j)) (S.rope_sin position j))

inline_for_extraction noextract
let input_representation
  (input:logical_inputs)
  (weights:bfseq weight_count)
  (hidden:bfseq hidden_width)
  (rope:bfseq rope_width)
  (keys values:bfseq (prefix_count input.call.position))
  : prop =
  represents weight_count weights input.packed_weights /\
  represents hidden_width hidden input.input_hidden /\
  rope_represents input.call.position rope /\
  represents (prefix_count input.call.position) keys input.old_keys /\
  represents (prefix_count input.call.position) values input.old_values

(* Native element identity, not IEEE numerical equality, preserves the prefix. *)
inline_for_extraction noextract
let exact_prefix
  (position:nat)
  (old:bfseq (prefix_count position))
  (next:bfseq (next_count position))
  : prop =
  forall (i:S.index (prefix_count position)).
    Seq.index next i == Seq.index old i

(* BASE needs no extra numerical premise under the reviewed qualitative laws.
   The conclusion and heap/frame obligations are unguarded; this does not prove
   native finiteness or satisfaction of the bilateral numerical domain. *)
noextract
let semantic_postcondition
  (input:logical_inputs)
  (old_keys old_values:bfseq (prefix_count input.call.position))
  (hidden:bfseq hidden_width)
  (keys values:bfseq (next_count input.call.position))
  : prop =
  exact_prefix input.call.position old_keys keys /\
  exact_prefix input.call.position old_values values /\
  (exists (logical_hidden:S.vector hidden_width)
          (logical_keys:S.vector (next_count input.call.position))
          (logical_values:S.vector (next_count input.call.position)).
    represents hidden_width hidden logical_hidden /\
    represents (next_count input.call.position) keys logical_keys /\
    represents (next_count input.call.position) values logical_values /\
    S.postcondition input.call
      (weights_view input.packed_weights) input.input_hidden
      (logical_before input)
      (result_view input.call input.frame logical_hidden logical_keys logical_values))

(* Empty source prefixes are inactive; ownership is never fabricated for null. *)
inline_for_extraction noextract
let prefix_ownership
  (position:nat) (pointer:A.array BF.t) (fraction:perm)
  (storage:bfseq (prefix_count position))
  : slprop =
  if position = 0 then pure (pointer == A.null)
  else
    pure (GA.is_global_array pointer /\ A.length pointer == prefix_count position) **
    on gpu_loc (pointer |-> Frac fraction storage)

fn kplay_run
  (#input:erased logical_inputs)
  (#sw:erased (bfseq weight_count))
  (#sx:erased (bfseq hidden_width))
  (#sr:erased (bfseq rope_width))
  (#sk:erased (bfseq (prefix_count input.call.position)))
  (#sv:erased (bfseq (prefix_count input.call.position)))
  (#fw #fx #fr #fk #fv:perm)
  (weights:A.larray BF.t weight_count{GA.is_global_array weights})
  (hidden:A.larray BF.t hidden_width{GA.is_global_array hidden})
  (rope:A.larray BF.t rope_width{GA.is_global_array rope})
  (old_keys:A.array BF.t)
  (old_values:A.array BF.t)
  (position:I64.t{I64.v position == input.call.position})
  (output:A.larray BF.t hidden_width{GA.is_global_array output})
  (new_keys:A.larray BF.t (next_count input.call.position){GA.is_global_array new_keys})
  (new_values:A.larray BF.t (next_count input.call.position){GA.is_global_array new_values})
  preserves cpu **
    on gpu_loc (weights |-> Frac fw sw) **
    on gpu_loc (hidden |-> Frac fx sx) **
    on gpu_loc (rope |-> Frac fr sr) **
    prefix_ownership input.call.position old_keys fk sk **
    prefix_ownership input.call.position old_values fv sv
  requires pure (input_representation input sw sx sr sk sv)
  requires on gpu_loc (
    A.pts_to_uninit output hidden_width **
    A.pts_to_uninit new_keys (next_count input.call.position) **
    A.pts_to_uninit new_values (next_count input.call.position))
  ensures exists* (sy:bfseq hidden_width)
                  (nk:bfseq (next_count input.call.position))
                  (nv:bfseq (next_count input.call.position)).
    on gpu_loc (output |-> sy ** new_keys |-> nk ** new_values |-> nv) **
    pure (semantic_postcondition input sk sv sy nk nv)
