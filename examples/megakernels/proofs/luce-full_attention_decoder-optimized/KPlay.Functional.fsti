module KPlay.Functional

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
module Sh = KPlay.SharedSpec
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

inline_for_extraction noextract
let state_pointer (buffer:array bf16) (position:position_index) : prop =
  if position = 0 then buffer == A.null
  else is_global_array buffer /\ A.length buffer == cache_elements

inline_for_extraction noextract
let state_ownership
  (buffer:array bf16) (position:position_index) (fraction:perm)
  (contents:bfseq (previous_elements position)) : slprop =
  if position = 0 then emp
  else pure (is_global_array buffer) ** buffer |-> Frac fraction contents

instance send_global_array_contents (#a:Type0) (buffer:array a{is_global_array buffer})
  (#fraction:perm) (contents:seq a)
  : is_send_across gpu_of (A.pts_to buffer #fraction contents)
  = Kuiper.Array.Core.is_send_pts_to buffer #fraction contents

ghost
fn move_state_ownership (buffer:array bf16) (position:position_index) (fraction:perm)
  (contents:lseq bf16 (previous_elements position)) (source:loc_id)
  (destination:loc_id{gpu_of source == gpu_of destination})
  requires on source (state_ownership buffer position fraction contents)
  ensures on destination (state_ownership buffer position fraction contents)
{
  if (position = 0) {
    map_loc source #(state_ownership buffer position fraction contents) #emp
      fn () { rewrite state_ownership buffer position fraction contents as emp };
    map_loc destination #emp #(state_ownership buffer position fraction contents)
      fn () { rewrite emp as state_ownership buffer position fraction contents };
  } else {
    map_loc source #(state_ownership buffer position fraction contents)
      #(pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      fn () {
        rewrite state_ownership buffer position fraction contents as
          (pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      };
    is_send_across_elim gpu_of (buffer |-> Frac fraction contents) destination;
    map_loc destination #(pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      #(state_ownership buffer position fraction contents)
      fn () {
        rewrite (pure (is_global_array buffer) ** buffer |-> Frac fraction contents) as
          state_ownership buffer position fraction contents
      };
  }
}

instance send_state_ownership (buffer:array bf16) (position:position_index) (fraction:perm)
  (contents:lseq bf16 (previous_elements position))
  : is_send_across gpu_of (state_ownership buffer position fraction contents)
  = fun source destination -> move_state_ownership buffer position fraction contents source destination

(* Source binding is concrete through fill_scores; shared mathematics remains the result authority. *)
#set-options "--fuel 2 --ifuel 1 --z3rlimit 30"

noextract noeq
type score_observation (position:position_index) = {
  observed_queries: lseq f32 activation_width;
  observed_keys: lseq bf16 cache_elements;
  observed_scores: lseq f32 (score_elements position)
}

(* Internal execution invariants, not the exported semantic contract. *)
inline_for_extraction noextract
let epsilon : Sqrt.rpos = 0.000001R
inline_for_extraction noextract
let rotary_log2_base : real = RoPE.coefficient ()
inline_for_extraction noextract
let score_scale : real = 0.0625R

inline_for_extraction noextract
let vector_slice
  (#n:nat) (values:real_vector n)
  (offset count:nat{offset + count <= n}) : GTot (real_vector count) =
  S.init_ghost count (fun i -> S.index values (offset + i))

inline_for_extraction noextract
let dot (#n:nat) (left right:real_vector n) : GTot real =
  rsum (S.init_ghost n (fun i -> S.index left i *. S.index right i))

inline_for_extraction noextract
let nonnegative_terms (values:seq real) : prop =
  forall (i:nat{i < S.length values}). S.index values i >=. 0.0R

inline_for_extraction noextract
let rec rsum_nonnegative (values:seq real)
  : Lemma (requires nonnegative_terms values)
      (ensures rsum values >=. 0.0R) (decreases S.length values) =
  match Q.view_seq values with
  | Q.SNil -> ()
  | Q.SCons value rest ->
    rsum_nonnegative rest;
    Kuiper.Real.rsum_append (S.create 1 value) rest

inline_for_extraction noextract
let rec rsum_member_lower_bound (values:seq real) (i:nat{i < S.length values})
  : Lemma (requires nonnegative_terms values)
      (ensures rsum values >=. S.index values i) (decreases S.length values) =
  let Q.SCons value rest = Q.view_seq values in
  if i = 0 then rsum_nonnegative rest
  else rsum_member_lower_bound rest (i - 1);
  Kuiper.Real.rsum_append (S.create 1 value) rest

inline_for_extraction noextract
let rec sum_squares
  (#n:nat) (values:real_vector n) (count:nat{count <= n})
  : GTot Sqrt.rnonneg (decreases count) =
  if count = 0 then 0.0R
  else let x = S.index values (count - 1) in
    sum_squares values (count - 1) +. x *. x

inline_for_extraction noextract
let rms_scale (#n:pos) (values:real_vector n) : GTot Sqrt.rpos =
  Sqrt.rsqrt (sum_squares values n /. R.of_int n +. epsilon)

inline_for_extraction noextract
let linear
  (weights:real_vector weight_elements) (offset:nat)
  (rows columns:pos{offset + rows * columns <= weight_elements})
  (values:real_vector columns) : GTot (real_vector rows) =
  S.init_ghost rows (fun row ->
    rsum (S.init_ghost columns (fun column ->
      S.index weights (offset + row * columns + column) *. S.index values column)))

inline_for_extraction noextract
let normalization
  (#n:pos) (weights:real_vector weight_elements)
  (offset:nat{offset + n <= weight_elements}) (values:real_vector n)
  : GTot (real_vector n) =
  let scale = rms_scale values in
  S.init_ghost n (fun i ->
    S.index values i *. scale *. (1.0R +. S.index weights (offset + i)))

inline_for_extraction noextract
let sigmoid (x:real) : GTot real = 1.0R /. (1.0R +. exp (0.0R -. x))
inline_for_extraction noextract
let silu (x:real) : GTot real = x *. sigmoid x

inline_for_extraction noextract
let rotary_phase (position:position_index) (coordinate:nat) : GTot real =
  R.of_int position *.
    Pow.exp2 (0.0R -.
      (R.of_int (coordinate % rotary_half) /. R.of_int rotary_half) *. rotary_log2_base)

inline_for_extraction noextract
let rotary_phase_refines_shared (position:position_index) (coordinate:nat)
  : Lemma (rotary_phase position coordinate ==
      Sh.rope_angle position (coordinate % rotary_half)) =
  RoPE.inverse_frequency_bridge (coordinate % rotary_half);
  FStar.Pervasives.norm_spec
    [delta_only [`%Sh.rope_angle; `%Sh.inverse_frequency;
                 `%Sh.text_axis_position; `%Sh.mrope_axis]; zeta; iota]
    (Sh.rope_angle position (coordinate % rotary_half))

inline_for_extraction noextract
let rotary_lower (value partner phase:real) : GTot real =
  value *. Trig.cos phase -. partner *. Trig.sin phase

inline_for_extraction noextract
let rotary_upper (value partner phase:real) : GTot real =
  value *. Trig.cos phase +. partner *. Trig.sin phase

inline_for_extraction noextract
let key_transform
  (weights:real_vector weight_elements) (position:position_index)
  (raw:real_vector head_width) : GTot (real_vector head_width) =
  let normalized = normalization weights key_norm_offset raw in
  S.init_ghost head_width (fun i ->
    let value = S.index normalized i in
    if i < rotary_half then
      rotary_lower value (S.index normalized (i + rotary_half)) (rotary_phase position i)
    else if i < 2 * rotary_half then
      rotary_upper value (S.index normalized (i - rotary_half)) (rotary_phase position i)
    else value)

inline_for_extraction noextract
let query_transform
  (weights:real_vector weight_elements) (position:position_index)
  (raw:real_vector head_width) : GTot (real_vector head_width) =
  let normalized = normalization weights query_norm_offset raw in
  S.init_ghost head_width (fun i ->
    let value = S.index normalized i in
    if i < rotary_half then
      rotary_lower value (S.index normalized (i + rotary_half)) (rotary_phase position i)
    else if i < 2 * rotary_half then
      rotary_upper value (S.index normalized (i - rotary_half)) (rotary_phase position i)
    else value)

inline_for_extraction noextract
let cache_update
  (position:position_index) (previous:real_vector (previous_elements position))
  (current:real_vector kv_width) : GTot (real_vector cache_elements) =
  S.init_ghost cache_elements (fun i ->
    let head = i / cache_head_elements in
    let time = (i % cache_head_elements) / head_width in
    let coordinate = i % head_width in
    if time = position then S.index current (head * head_width + coordinate)
    else if position = 0 then 0.0R
    else S.index previous i)

inline_for_extraction noextract
let attention_scores
  (position:position_index) (queries:real_vector activation_width)
  (keys:real_vector cache_elements) : GTot (real_vector (score_elements position)) =
  S.init_ghost (score_elements position) (fun i ->
    let head = i / (position + 1) in
    let time = i % (position + 1) in
    score_scale *.
      dot (vector_slice queries (head * head_width) head_width)
        (vector_slice keys (cache_index (kv_head_for_query head) time 0) head_width))

inline_for_extraction noextract
let negative_infinity : f32 = neg (infinity <: f32)

inline_for_extraction noextract
let rec partition_maximum
  (position:position_index) (scores:lseq f32 (score_elements position))
  (head:nat{head < query_heads}) (partition:nat{partition < partitions})
  (count:nat{count <= position + 1}) : GTot f32 (decreases count) =
  if count = 0 then negative_infinity
  else
    let previous = partition_maximum position scores head partition (count - 1) in
    if (count - 1) % partitions = partition then
      fmax previous (S.index scores (head * (position + 1) + count - 1))
    else previous

inline_for_extraction noextract
let partition_included
  (position:position_index) (scores:lseq f32 (score_elements position))
  (head:nat{head < query_heads}) (partition:nat{partition < partitions}) : GTot bool =
  negative_infinity `lt` partition_maximum position scores head partition (position + 1)

inline_for_extraction noextract
let bilateral_sentinel_kind ()
  : Lemma (
      kind negative_infinity == Infinite /\
      not (is_zero negative_infinity)) =
  let _ = kind_infinity #f32 in
  neg_kind (infinity <: f32);
  if is_zero negative_infinity then is_zero_spec negative_infinity

inline_for_extraction noextract
let bilateral_finite_not_sentinel (value:f32)
  : Lemma
      (requires Finite? (kind value))
      (ensures not (eq value negative_infinity)) =
  bilateral_sentinel_kind ();
  eq_spec value negative_infinity

inline_for_extraction noextract
let bilateral_negative_infinity_lt_finite (value:f32)
  : Lemma
      (requires Finite? (kind value))
      (ensures negative_infinity `lt` value) =
  let positive:f32 = infinity in
  let double_negative = neg negative_infinity in
  let negated = neg value in
  bilateral_sentinel_kind ();
  neg_kind positive;
  neg_kind negative_infinity;
  neg_kind value;
  neg_neg positive;
  eq_spec double_negative positive;
  if is_zero positive then is_zero_spec positive;
  assert (double_negative == positive);
  infinity_val_spec negated;
  negate_lt_is_lte positive negated;
  lt_neg_flip value negative_infinity;
  assert (not (value `lt` negative_infinity));
  bilateral_finite_not_sentinel value;
  lte_is_lt_or_eq value negative_infinity;
  negate_lt_is_lte negative_infinity value

inline_for_extraction noextract
let bilateral_finite_or_sentinel (value:f32) : prop =
  Finite? (kind value) \/ value == negative_infinity

inline_for_extraction noextract
let bilateral_fmax_classification (left right:f32)
  : Lemma
      (requires bilateral_finite_or_sentinel left /\
        bilateral_finite_or_sentinel right)
      (ensures bilateral_finite_or_sentinel (fmax left right) /\
        ((Finite? (kind left) \/ Finite? (kind right)) ==>
          Finite? (kind (fmax left right)))) =
  bilateral_sentinel_kind ();
  if Finite? (kind left) then (
    bilateral_negative_infinity_lt_finite left;
    negate_lt_is_lte left negative_infinity;
    lte_is_lt_or_eq negative_infinity left
  );
  if Finite? (kind right) then bilateral_negative_infinity_lt_finite right;
  fmax_spec left right;
  let selected = if left `lt` right then right else left in
  eq_spec (fmax left right) selected;
  if is_zero (fmax left right) then is_zero_spec (fmax left right)

inline_for_extraction noextract
let rec bilateral_partition_maximum_classification
  (position:position_index) (scores:lseq f32 (score_elements position))
  (head:nat{head < query_heads}) (partition:nat{partition < partitions})
  (count:nat{count <= position + 1})
  : Lemma
      (requires bilateral_live_scores_finite position scores)
      (ensures
        bilateral_finite_or_sentinel
          (partition_maximum position scores head partition count) /\
        (partition < count ==>
          Finite? (kind (partition_maximum position scores head partition count))))
      (decreases count) =
  if count > 0 then (
    bilateral_partition_maximum_classification position scores head partition (count - 1);
    if (count - 1) % partitions = partition then (
      assert (Finite? (kind (S.index scores (head * (position + 1) + count - 1))));
      bilateral_fmax_classification
        (partition_maximum position scores head partition (count - 1))
        (S.index scores (head * (position + 1) + count - 1))
    )
  )

inline_for_extraction noextract
let bilateral_live_token_partition_included
  (position:position_index) (scores:lseq f32 (score_elements position))
  (head:nat{head < query_heads}) (time:nat{time <= position})
  : Lemma
      (requires bilateral_live_scores_finite position scores)
      (ensures partition_included position scores head (time % partitions)) =
  bilateral_partition_maximum_classification
    position scores head (time % partitions) (position + 1);
  bilateral_negative_infinity_lt_finite
    (partition_maximum position scores head (time % partitions) (position + 1))

inline_for_extraction noextract
let rec bilateral_empty_partition_maximum
  (position:position_index) (scores:lseq f32 (score_elements position))
  (head:nat{head < query_heads}) (partition:nat{partition < partitions})
  (count:nat{count <= position + 1 /\ count <= partition})
  : Lemma (partition_maximum position scores head partition count == negative_infinity)
      (decreases count) =
  if count > 0 then
    bilateral_empty_partition_maximum position scores head partition (count - 1)

inline_for_extraction noextract
let bilateral_empty_partition_excluded
  (position:position_index) (scores:lseq f32 (score_elements position))
  (head:nat{head < query_heads})
  (partition:nat{partition < partitions /\ position < partition})
  : Lemma (not (partition_included position scores head partition)) =
  bilateral_empty_partition_maximum position scores head partition (position + 1);
  bilateral_sentinel_kind ();
  eq_spec negative_infinity negative_infinity;
  lte_is_lt_or_eq negative_infinity negative_infinity;
  negate_lt_is_lte negative_infinity negative_infinity

inline_for_extraction noextract
let attention_head
  (position:position_index) (controls:lseq f32 (score_elements position))
  (scores:real_vector (score_elements position)) (values:real_vector cache_elements)
  (head:nat{head < query_heads}) : GTot (real_vector head_width) =
  let kv_head = kv_head_for_query head in
  let mass:real_vector (position + 1) = S.init_ghost (position + 1) (fun time ->
    if partition_included position controls head (time % partitions)
    then exp (S.index scores (head * (position + 1) + time))
    else 0.0R) in
  assert (nonnegative_terms mass);
  let denominator = rsum mass in
  S.init_ghost head_width (fun coordinate ->
    rsum (S.init_ghost (position + 1) (fun time ->
      if partition_included position controls head (time % partitions) then
        let _ = rsum_member_lower_bound mass time in
        (S.index mass time /. denominator) *.
          S.index values (cache_index kv_head time coordinate)
      else 0.0R)))

inline_for_extraction noextract
let residual_vector (hidden projected : real_vector hidden_width)
  : GTot (real_vector hidden_width)
  = S.init_ghost hidden_width (fun i -> S.index hidden i +. S.index projected i)

inline_for_extraction noextract
let mlp_vector (gate up : real_vector mlp_width)
  : GTot (real_vector mlp_width)
  = S.init_ghost mlp_width (fun i -> silu (S.index gate i) *. S.index up i)

inline_for_extraction noextract
let mlp_residual
  (weights:real_vector weight_elements) (residual:real_vector hidden_width)
  : GTot (real_vector hidden_width) =
  let normalized = normalization weights post_norm_offset residual in
  let gate = linear weights mlp_gate_offset mlp_width hidden_width normalized in
  let up = linear weights mlp_up_offset mlp_width hidden_width normalized in
  let activation = mlp_vector gate up in
  let down = linear weights mlp_down_offset hidden_width mlp_width activation in
  residual_vector residual down

noeq noextract
type attention_values (position:position_index) = {
  score_result: real_vector (score_elements position);
  hidden_result: real_vector hidden_width;
  key_result: real_vector cache_elements;
  value_result: real_vector cache_elements
}

noeq noextract
type decoder_prefix (position:position_index) = {
  prefix_query_gate: real_vector query_gate_width;
  prefix_queries: real_vector activation_width;
  prefix_keys: real_vector cache_elements;
  prefix_values: real_vector cache_elements;
  prefix_scores: real_vector (score_elements position)
}

noextract
let decoder_prefix_model
  (weights:real_vector weight_elements) (hidden:real_vector hidden_width)
  (position:position_index)
  (previous_keys previous_values:real_vector (previous_elements position))
  : GTot (decoder_prefix position) =
  let normalized = normalization weights input_norm_offset hidden in
  let query_gate = linear weights query_gate_offset query_gate_width hidden_width normalized in
  let key_projection = linear weights key_offset kv_width hidden_width normalized in
  let value_projection = linear weights value_offset kv_width hidden_width normalized in
  let transformed_keys = S.init_ghost kv_heads (fun head ->
    key_transform weights position
      (vector_slice key_projection (head * head_width) head_width)) in
  let current_keys = S.init_ghost kv_width (fun i ->
    S.index (S.index transformed_keys (i / head_width)) (i % head_width)) in
  let keys = cache_update position previous_keys current_keys in
  let values = cache_update position previous_values value_projection in
  let transformed_queries = S.init_ghost query_heads (fun head ->
    query_transform weights position
      (vector_slice query_gate (head * (2 * head_width)) head_width)) in
  let queries = S.init_ghost activation_width (fun i ->
    S.index (S.index transformed_queries (i / head_width)) (i % head_width)) in
  let scores = attention_scores position queries keys in
  {
    prefix_query_gate = query_gate;
    prefix_queries = queries;
    prefix_keys = keys;
    prefix_values = values;
    prefix_scores = scores
  }

inline_for_extraction noextract
let decoder_activation
  (position:position_index) (controls:lseq f32 (score_elements position))
  (query_gate:real_vector query_gate_width) (scores:real_vector (score_elements position))
  (values:real_vector cache_elements) : GTot (real_vector activation_width) =
  let attended = S.init_ghost query_heads (fun head ->
    attention_head position controls scores values head) in
  S.init_ghost activation_width (fun i ->
    let head = i / head_width in
    let coordinate = i % head_width in
    S.index (S.index attended head) coordinate *.
      sigmoid (S.index query_gate (head * (2 * head_width) + head_width + coordinate)))

noextract
let execution_model
  (weights:real_vector weight_elements) (hidden:real_vector hidden_width)
  (position:position_index)
  (previous_keys previous_values:real_vector (previous_elements position))
  (controls:lseq f32 (score_elements position)) : GTot (attention_values position) =
  let first = decoder_prefix_model weights hidden position previous_keys previous_values in
  let activation = decoder_activation position controls first.prefix_query_gate first.prefix_scores first.prefix_values in
  let projected = linear weights output_offset hidden_width activation_width activation in
  let residual = residual_vector hidden projected in
  {
    score_result = first.prefix_scores;
    hidden_result = mlp_residual weights residual;
    key_result = first.prefix_keys;
    value_result = first.prefix_values
  }

noextract
let execution_post
  (weights:lseq bf16 weight_elements) (hidden:lseq bf16 hidden_width)
  (position:position_index)
  (previous_keys previous_values:lseq bf16 (previous_elements position))
  (result:lseq bf16 hidden_width) (keys values:lseq bf16 cache_elements) : prop =
  retained_cache_post position previous_keys keys /\
  retained_cache_post position previous_values values /\
  (exists (controls:lseq f32 (score_elements position)).
    forall (real_weights:real_vector weight_elements) (real_hidden:real_vector hidden_width)
      (real_keys real_values:real_vector (previous_elements position)).
      weights %~ real_weights /\ hidden %~ real_hidden /\
      previous_keys %~ real_keys /\ previous_values %~ real_values ==>
      (let model = execution_model real_weights real_hidden position real_keys real_values controls in
       controls %~ model.score_result /\ result %~ model.hidden_result /\
       keys %~ model.key_result /\ values %~ model.value_result))

inline_for_extraction noextract
let rec prefix (n : nat) (f : nat -> GTot real)
  : GTot real (decreases n)
  = if n = 0 then 0.0R else prefix (n - 1) f +. f (n - 1)

inline_for_extraction noextract
let prefix_step (n : nat) (f : nat -> GTot real)
  : Lemma (prefix (n + 1) f == prefix n f +. f n)
  = ()

inline_for_extraction noextract
let rec prefix_ext (n : nat) (f g : nat -> GTot real)
  : Lemma
      (requires forall (i : nat). i < n ==> f i == g i)
      (ensures prefix n f == prefix n g)
      (decreases n)
  = if n > 0 then prefix_ext (n - 1) f g

inline_for_extraction noextract
let rec prefix_add (n : nat) (f g : nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> f i +. g i) == prefix n f +. prefix n g)
      (decreases n)
  = if n > 0 then prefix_add (n - 1) f g

inline_for_extraction noextract
let rec prefix_zero (n : nat)
  : Lemma (ensures prefix n (fun _ -> 0.0R) == 0.0R) (decreases n)
  = if n > 0 then prefix_zero (n - 1)

inline_for_extraction noextract
let rec prefix_split (n m : nat) (f : nat -> GTot real)
  : Lemma
      (ensures prefix (n + m) f ==
        prefix n f +. prefix m (fun i -> f (n + i)))
      (decreases m)
  = if m > 0 then prefix_split n (m - 1) f

inline_for_extraction noextract
let rec prefix_swap (n m : nat) (f : nat -> nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> prefix m (f i)) ==
        prefix m (fun j -> prefix n (fun i -> f i j)))
      (decreases n)
  = if n = 0 then begin
      prefix_zero m;
      prefix_ext m (fun j -> prefix n (fun i -> f i j)) (fun _ -> 0.0R)
    end
    else begin
      prefix_swap (n - 1) m f;
      prefix_add m (fun j -> prefix (n - 1) (fun i -> f i j)) (f (n - 1));
      prefix_ext m
        (fun j -> prefix n (fun i -> f i j))
        (fun j -> prefix (n - 1) (fun i -> f i j) +. f (n - 1) j)
    end

inline_for_extraction noextract
let rec prefix_blocks (blocks width : nat) (f : nat -> GTot real)
  : Lemma
      (ensures prefix blocks (fun b ->
        prefix width (fun i -> f (b * width + i))) ==
        prefix (blocks * width) f)
      (decreases blocks)
  = if blocks > 0 then begin
      prefix_step (blocks - 1)
        (fun b -> prefix width (fun i -> f (b * width + i)));
      prefix_blocks (blocks - 1) width f;
      prefix_split ((blocks - 1) * width) width f;
      assert (blocks * width == (blocks - 1) * width + width)
    end

inline_for_extraction noextract
let seq_value (s : seq real) (i : nat) : GTot real =
  if i < S.length s then S.index s i else 0.0R

inline_for_extraction noextract
let rec prefix_rsum (s : seq real) (n : nat{n <= S.length s})
  : Lemma
      (ensures prefix n (seq_value s) == rsum (S.slice s 0 n))
      (decreases n)
  = if n = 0 then
      assert (S.equal (S.slice s 0 0) (S.empty #real))
    else begin
      prefix_rsum s (n - 1);
      SC.lemma_seq_fold_left_slice 0.0R (+.) s 0 (n - 1)
    end

inline_for_extraction noextract
let prefix_init (n : nat) (f : nat -> GTot real)
  : Lemma (prefix n f == rsum (S.init_ghost n f))
  = let s = S.init_ghost n f in
    prefix_ext n f (seq_value s);
    prefix_rsum s n;
    assert (S.equal (S.slice s 0 n) s)

(* Sequence/function and sum bridges follow the accepted reference witness. *)
inline_for_extraction noextract
let sequence_of (#n:nat) (values:Sh.vector n) : GTot (real_vector n) =
  S.init_ghost n values

inline_for_extraction noextract
let vector_of (#n:nat) (values:real_vector n) : Sh.vector n =
  fun i -> S.index values i

inline_for_extraction noextract
let sequence_at (#n:nat) (values:Sh.vector n) (i:Sh.index n)
  : Lemma (S.index (sequence_of values) i == values i) =
  S.init_ghost_index_ n values i

inline_for_extraction noextract
let representation_sequence (n:nat) (storage:bfseq n) (logical:Sh.vector n)
  : Lemma (represents n storage logical <==> storage %~ sequence_of logical) = ()

inline_for_extraction noextract
let shared_sum_zero (f:Sh.vector 0) : Lemma (Sh.sum 0 f == 0.0R) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.SharedSpec.sum"]] (Sh.sum 0 f)

inline_for_extraction noextract
let shared_sum_step (n:pos) (f:Sh.vector n)
  : Lemma (Sh.sum n f == Sh.sum (n - 1) (fun i -> f i) +. f (n - 1)) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.SharedSpec.sum"]] (Sh.sum n f)

inline_for_extraction noextract
let rec shared_sum_ext (n:nat) (f g:Sh.vector n)
  : Lemma
      (requires forall (i:Sh.index n). f i == g i)
      (ensures Sh.sum n f == Sh.sum n g)
      (decreases n) =
  if n = 0 then (shared_sum_zero f; shared_sum_zero g)
  else (
    shared_sum_ext (n - 1) (fun i -> f i) (fun i -> g i);
    shared_sum_step n f;
    shared_sum_step n g
  )

inline_for_extraction noextract
let rec shared_squares_ext (n:nat) (f g:Sh.vector n)
  : Lemma
      (requires forall (i:Sh.index n). f i == g i)
      (ensures Sh.sum_squares n f == Sh.sum_squares n g)
      (decreases n) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.SharedSpec.sum_squares"]] (Sh.sum_squares n f);
  FStar.Pervasives.norm_spec [delta_once ["KPlay.SharedSpec.sum_squares"]] (Sh.sum_squares n g);
  if n > 0 then shared_squares_ext (n - 1) (fun i -> f i) (fun i -> g i)

inline_for_extraction noextract
let rec shared_positive_sum_bridge (n:pos) (f:Sh.index n -> Sqrt.rpos)
  : Lemma (Sh.sum n f == Sh.sum_positive n f) (decreases n) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.SharedSpec.sum_positive"]]
    (Sh.sum_positive n f);
  shared_sum_step n f;
  if n = 1 then shared_sum_zero (fun i -> f i)
  else (
    let lower:Sh.index (n - 1) -> Sqrt.rpos = fun i -> f i in
    shared_positive_sum_bridge (n - 1) lower;
    shared_sum_ext (n - 1) lower (fun i -> f i)
  )

inline_for_extraction noextract
let rec sequence_prefix_shared (values:seq real) (count:nat{count <= S.length values})
  : Lemma (prefix count (seq_value values) ==
      Sh.sum count (fun i -> S.index values i)) (decreases count) =
  if count = 0 then shared_sum_zero (fun i -> S.index values i)
  else (
    sequence_prefix_shared values (count - 1);
    shared_sum_step count (fun i -> S.index values i)
  )

inline_for_extraction noextract
let sequence_sum_shared (#n:nat) (values:real_vector n) (terms:Sh.vector n)
  : Lemma
      (requires forall (i:Sh.index n). S.index values i == terms i)
      (ensures rsum values == Sh.sum n terms) =
  prefix_rsum values n;
  assert (S.equal (S.slice values 0 n) values);
  sequence_prefix_shared values n;
  shared_sum_ext n (vector_of values) terms

inline_for_extraction noextract
let rec sequence_squares_shared
  (#n:nat) (values:real_vector n) (logical:Sh.vector n) (count:nat{count <= n})
  : Lemma
      (requires forall (i:Sh.index n). S.index values i == logical i)
      (ensures sum_squares values count == Sh.sum_squares count (fun i -> logical i))
      (decreases count) =
  FStar.Pervasives.norm_spec [delta_once ["KPlay.SharedSpec.sum_squares"]]
    (Sh.sum_squares count (fun i -> logical i));
  if count > 0 then sequence_squares_shared values logical (count - 1)

inline_for_extraction noextract
let normalization_shared
  (#n:pos) (weights:real_vector weight_elements)
  (offset:nat{offset + n <= weight_elements}) (values:real_vector n)
  (weight logical:Sh.vector n)
  : Lemma
      (requires
        (forall (i:Sh.index n). S.index weights (offset + i) == weight i) /\
        (forall (i:Sh.index n). S.index values i == logical i))
      (ensures normalization weights offset values == sequence_of (Sh.rms_norm n weight logical)) =
  sequence_squares_shared values logical n;
  shared_squares_ext n (fun i -> logical i) logical;
  assert (sum_squares values n == Sh.sum_squares n logical);
  FStar.Pervasives.norm_spec
    [delta_only [`%normalization; `%rms_scale; `%epsilon]; zeta; iota]
    (normalization weights offset values);
  FStar.Pervasives.norm_spec [delta_only [`%Sh.rms_norm]; zeta; iota]
    (Sh.rms_norm n weight logical);
  S.lemma_eq_elim (normalization weights offset values)
    (sequence_of (Sh.rms_norm n weight logical))

inline_for_extraction noextract
let linear_shared
  (weights:real_vector weight_elements) (offset:nat)
  (rows columns:pos{offset + rows * columns <= weight_elements})
  (values:real_vector columns) (matrix:Sh.matrix rows columns) (logical:Sh.vector columns)
  : Lemma
      (requires
        (forall (r:Sh.index rows) (c:Sh.index columns).
          S.index weights (offset + r * columns + c) == matrix r c) /\
        (forall (i:Sh.index columns). S.index values i == logical i))
      (ensures linear weights offset rows columns values ==
        sequence_of (Sh.linear rows columns matrix logical)) =
  let point (r:Sh.index rows)
    : Lemma (S.index (linear weights offset rows columns values) r ==
        Sh.linear rows columns matrix logical r) =
    sequence_sum_shared
      (S.init_ghost columns (fun c ->
        S.index weights (offset + r * columns + c) *. S.index values c))
      (fun c -> matrix r c *. logical c)
  in
  C.forall_intro point;
  S.lemma_eq_elim (linear weights offset rows columns values)
    (sequence_of (Sh.linear rows columns matrix logical))

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let transform_shared
  (query:bool) (weights:real_vector weight_elements) (position:position_index)
  (raw:real_vector head_width) (weight logical:Sh.vector head_width)
  : Lemma
      (requires
        (forall (d:Sh.index head_width).
          S.index weights ((if query then query_norm_offset else key_norm_offset) + d) == weight d) /\
        (forall (d:Sh.index head_width). S.index raw d == logical d))
      (ensures
        (if query then query_transform weights position raw else key_transform weights position raw) ==
        sequence_of (Sh.rotate_partial position (Sh.rms_norm head_width weight logical))) =
  let offset = if query then query_norm_offset else key_norm_offset in
  normalization_shared weights offset raw weight logical;
  let transformed =
    if query then query_transform weights position raw else key_transform weights position raw in
  if query then
    FStar.Pervasives.norm_spec [delta_only [`%query_transform]; zeta; iota] transformed
  else
    FStar.Pervasives.norm_spec [delta_only [`%key_transform]; zeta; iota] transformed;
  let point (d:Sh.index head_width)
    : Lemma (S.index transformed d ==
        Sh.rotate_partial position (Sh.rms_norm head_width weight logical) d) =
    if d < 64 then rotary_phase_refines_shared position d;
    FStar.Pervasives.norm_spec
      [delta_only [`%Sh.rotate_partial; `%Sh.rope_cos; `%Sh.rope_sin]; zeta; iota]
      (Sh.rotate_partial position (Sh.rms_norm head_width weight logical) d);
    if d < 32 then assert (d % rotary_half == d)
    else if d < 64 then assert (d % rotary_half == d - 32)
  in
  C.forall_intro point;
  S.lemma_eq_elim transformed
    (sequence_of (Sh.rotate_partial position (Sh.rms_norm head_width weight logical)))
#pop-options

inline_for_extraction noextract
let residual_shared (a b:real_vector hidden_width) (x y:Sh.vector hidden_width)
  : Lemma
      (requires forall (i:Sh.index hidden_width). S.index a i == x i /\ S.index b i == y i)
      (ensures residual_vector a b == sequence_of (Sh.vector_add hidden_width x y)) =
  S.lemma_eq_elim (residual_vector a b) (sequence_of (Sh.vector_add hidden_width x y))

inline_for_extraction noextract
let shared_attention_scores
  (position:nat) (queries:Sh.matrix query_heads head_width)
  (history:Sh.kv_history{history.length == position + 1}) (head:Sh.index query_heads)
  : Sh.vector (position + 1) =
  fun t ->
    Sh.sum head_width (fun d -> queries head d *. history.keys (Sh.kv_head head) t d) *.
      (1.0R /. 16.0R) +. Sh.causal_bias position t

inline_for_extraction noextract
let shared_attention_unfold
  (position:nat) (queries:Sh.matrix query_heads head_width)
  (history:Sh.kv_history{history.length == position + 1})
  (head:Sh.index query_heads) (d:Sh.index head_width)
  : Lemma (
      Sh.attention position queries history head d ==
      Sh.sum (position + 1) (fun t ->
        Sh.softmax (position + 1) (shared_attention_scores position queries history head) t *.
          history.values (Sh.kv_head head) t d)) =
  assert (
      Sh.attention position queries history head d ==
      Sh.sum (position + 1) (fun t ->
        Sh.softmax (position + 1) (shared_attention_scores position queries history head) t *.
          history.values (Sh.kv_head head) t d))
    by (Proof.norm [delta_only [`%Sh.attention; `%shared_attention_scores; `%head_width]];
        Proof.trefl ())

inline_for_extraction noextract
let head_grouping_shared (head:Sh.index query_heads)
  : Lemma (kv_head_for_query head == Sh.kv_head head) =
  assert (kv_head_for_query head == Sh.kv_head head)
    by (Proof.norm [delta_only [`%kv_head_for_query; `%Sh.kv_head; `%query_heads; `%kv_heads]];
        Proof.trefl ())

inline_for_extraction noextract
let scores_cell_shared
  (position:position_index) (queries:real_vector activation_width) (keys:real_vector cache_elements)
  (i:Sh.index (score_elements position))
  : Lemma (
      S.index (attention_scores position queries keys) i ==
      score_scale *. dot
        (vector_slice queries ((i / (position + 1)) * head_width) head_width)
        (vector_slice keys (cache_index (kv_head_for_query (i / (position + 1))) (i % (position + 1)) 0) head_width)) = ()

inline_for_extraction noextract
let dot_shared (#n:nat) (left right:real_vector n) (a b:Sh.vector n)
  : Lemma
      (requires forall (i:Sh.index n). S.index left i == a i /\ S.index right i == b i)
      (ensures dot left right == Sh.sum n (fun i -> a i *. b i)) =
  sequence_sum_shared (S.init_ghost n (fun i -> S.index left i *. S.index right i))
    (fun i -> a i *. b i)

inline_for_extraction noextract
let scores_shared
  (position:position_index) (queries:real_vector activation_width) (keys:real_vector cache_elements)
  (logical:Sh.matrix query_heads head_width)
  (history:Sh.kv_history{history.length == position + 1})
  : Lemma
      (requires
        (forall (h:Sh.index query_heads) (d:Sh.index head_width).
          S.index queries (h * head_width + d) == logical h d) /\
        (forall (h:Sh.index kv_heads) (t:Sh.index (position + 1)) (d:Sh.index head_width).
          S.index keys (cache_index h t d) == history.keys h t d))
      (ensures forall (h:Sh.index query_heads) (t:Sh.index (position + 1)).
        S.index (attention_scores position queries keys) (h * (position + 1) + t) ==
        shared_attention_scores position logical history h t) =
  let point (h:Sh.index query_heads) (t:Sh.index (position + 1))
    : Lemma (
        S.index (attention_scores position queries keys) (h * (position + 1) + t) ==
        shared_attention_scores position logical history h t) =
    FStar.Math.Lemmas.lemma_div_mod_plus t h (position + 1);
    FStar.Math.Lemmas.small_div t (position + 1);
    FStar.Math.Lemmas.small_mod t (position + 1);
    assert ((h * (position + 1) + t) / (position + 1) == h);
    assert ((h * (position + 1) + t) % (position + 1) == t);
    head_grouping_shared h;
    scores_cell_shared position queries keys (h * (position + 1) + t);
    dot_shared (vector_slice queries (h * head_width) head_width)
      (vector_slice keys (cache_index (kv_head_for_query h) t 0) head_width)
      (logical h) (fun d -> history.keys (Sh.kv_head h) t d);
    FStar.Pervasives.norm_spec [delta_only [`%shared_attention_scores; `%Sh.causal_bias]; zeta; iota]
      (shared_attention_scores position logical history h t);
    assert (score_scale == 1.0R /. 16.0R)
  in C.forall_intro_2 point

inline_for_extraction noextract
let online_interpretation (position : position_index) =
  real_vector (score_elements position) & real_vector cache_elements

inline_for_extraction noextract
let model_mass (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads})
  : GTot (real_vector (position + 1)) =
  S.init_ghost (position + 1) (fun time ->
    if partition_included position controls head (time % partitions)
    then exp (S.index (fst model) (head * (position + 1) + time)) else 0.0R)

inline_for_extraction noextract
let attention_terms (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads}) (i : nat{i < head_width})
  : GTot (real_vector (position + 1)) =
  let mass = model_mass position controls model head in
  assert (nonnegative_terms mass);
  let denominator = rsum mass in
  S.init_ghost (position + 1) (fun time ->
    let value_index = cache_index (kv_head_for_query head) time i in
    if partition_included position controls head (time % partitions) then
      let _ = rsum_member_lower_bound mass time in
      (S.index mass time /. denominator) *. S.index (snd model) value_index
    else 0.0R)

inline_for_extraction noextract
let attention_formula (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads})
  : Lemma (attention_head position controls (fst model) (snd model) head ==
      S.init_ghost head_width (fun i -> rsum (attention_terms position controls model head i)))
  = assert (attention_head position controls (fst model) (snd model) head ==
      S.init_ghost head_width (fun i -> rsum (attention_terms position controls model head i)))
    by (FStar.Tactics.V2.norm [delta_only [`%attention_head; `%attention_terms; `%model_mass]; zeta; iota];
      FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let attention_index_terms (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads}) (i : nat{i < head_width})
  : Lemma (S.index (attention_head position controls (fst model) (snd model) head) i ==
      rsum (attention_terms position controls model head i))
  = attention_formula position controls model head;
    S.init_ghost_index_ head_width (fun j -> rsum (attention_terms position controls model head j)) i

inline_for_extraction noextract
let shared_exponentials (n:pos) (scores:Sh.vector n) : Sh.index n -> Sqrt.rpos =
  fun i -> Exp.exp_positive (scores i); Exp.exp (scores i)

inline_for_extraction noextract
let shared_softmax_denominator (n:pos) (scores:Sh.vector n) (i:Sh.index n)
  : Lemma (
      Sh.softmax n scores i ==
      Exp.exp (scores i) /. Sh.sum_positive n (shared_exponentials n scores)) =
  let original:Sh.index n -> Sqrt.rpos =
    fun j -> Exp.exp_positive (scores j); Exp.exp (scores j) in
  shared_positive_sum_bridge n original;
  shared_sum_ext n original (shared_exponentials n scores);
  shared_positive_sum_bridge n (shared_exponentials n scores);
  FStar.Pervasives.norm_spec [delta_once ["KPlay.SharedSpec.softmax"]] (Sh.softmax n scores i)

#push-options "--fuel 2 --z3rlimit 30"
inline_for_extraction noextract
let attention_head_shared
  (position:position_index) (controls:lseq f32 (score_elements position))
  (scores:real_vector (score_elements position)) (values:real_vector cache_elements)
  (queries:Sh.matrix query_heads head_width)
  (history:Sh.kv_history{history.length == position + 1}) (head:Sh.index query_heads)
  : Lemma
      (requires bilateral_live_scores_finite position controls /\
        (forall (t:Sh.index (position + 1)).
          S.index scores (head * (position + 1) + t) ==
          shared_attention_scores position queries history head t) /\
        (forall (h:Sh.index kv_heads) (t:Sh.index (position + 1)) (d:Sh.index head_width).
          S.index values (cache_index h t d) == history.values h t d))
      (ensures attention_head position controls scores values head ==
        sequence_of (Sh.attention position queries history head)) =
  head_grouping_shared head;
  let included (t:Sh.index (position + 1))
    : Lemma (partition_included position controls head (t % partitions)) =
    bilateral_live_token_partition_included position controls head t
  in
  C.forall_intro included;
  let logical_scores = shared_attention_scores position queries history head in
  let exponentials = shared_exponentials (position + 1) logical_scores in
  let mass = model_mass position controls (scores, values) head in
  sequence_sum_shared mass exponentials;
  shared_positive_sum_bridge (position + 1) exponentials;
  let probability (t:Sh.index (position + 1))
    : Lemma (S.index mass t /. rsum mass == Sh.softmax (position + 1) logical_scores t) =
    shared_softmax_denominator (position + 1) logical_scores t
  in
  C.forall_intro probability;
  let point (d:Sh.index head_width)
    : Lemma (S.index (attention_head position controls scores values head) d ==
        Sh.attention position queries history head d) =
    shared_attention_unfold position queries history head d;
    let numerator = attention_terms position controls (scores, values) head d in
    let terms:Sh.vector (position + 1) =
      fun t -> Sh.softmax (position + 1) logical_scores t *. history.values (Sh.kv_head head) t d in
    let term (t:Sh.index (position + 1))
      : Lemma (S.index numerator t == terms t) =
      included t;
      probability t;
      assert (S.index values (cache_index (kv_head_for_query head) t d) ==
        history.values (Sh.kv_head head) t d)
    in
    C.forall_intro term;
    sequence_sum_shared numerator terms;
    shared_sum_ext (position + 1) terms
      (fun t -> Sh.softmax (position + 1)
        (shared_attention_scores position queries history head) t *. history.values (Sh.kv_head head) t d);
    attention_index_terms position controls (scores, values) head d
  in
  C.forall_intro point;
  S.lemma_eq_elim (attention_head position controls scores values head)
    (sequence_of (Sh.attention position queries history head))
#pop-options

inline_for_extraction noextract
let shared_normalized (input:logical_inputs) : Sh.vector hidden_width =
  Sh.rms_norm hidden_width (weights_view input.packed_weights).input_norm input.input_hidden

inline_for_extraction noextract
let shared_query_gate (input:logical_inputs) : Sh.vector query_gate_width =
  Sh.linear query_gate_width hidden_width (weights_view input.packed_weights).query_gate_proj
    (shared_normalized input)

inline_for_extraction noextract
let shared_current_keys (input:logical_inputs) : Sh.matrix kv_heads head_width =
  let w = weights_view input.packed_weights in
  let keys = Sh.reshape_kv (Sh.linear kv_width hidden_width w.key_proj (shared_normalized input)) in
  fun h -> Sh.rotate_partial input.call.position (Sh.rms_norm head_width w.key_norm (keys h))

inline_for_extraction noextract
let shared_current_values (input:logical_inputs) : Sh.matrix kv_heads head_width =
  Sh.reshape_kv (Sh.linear kv_width hidden_width (weights_view input.packed_weights).value_proj
    (shared_normalized input))

inline_for_extraction noextract
let shared_history (input:logical_inputs)
  : h:Sh.kv_history{h.length == input.call.position + 1} =
  Sh.append_kv input.call.position ((logical_before input).kv input.call.layer)
    (shared_current_keys input) (shared_current_values input)

inline_for_extraction noextract
let shared_gated_context (input:logical_inputs) : Sh.vector activation_width =
  let context = Sh.attention input.call.position (shared_queries input) (shared_history input) in
  let gates = Sh.split_gates (shared_query_gate input) in
  fun i -> context (i / head_width) (i % head_width) *.
    Sh.sigmoid (gates (i / head_width) (i % head_width))

inline_for_extraction noextract
let gate_offset_shared (h:Sh.index query_heads) (d:Sh.index head_width)
  : Lemma (h * (2 * head_width) + head_width + d == 512 * h + 256 + d) = ()

inline_for_extraction noextract
let shared_gated_cell (input:logical_inputs) (i:Sh.index activation_width)
  : Lemma (
      shared_gated_context input i ==
      Sh.attention input.call.position (shared_queries input) (shared_history input)
        (i / head_width) (i % head_width) *.
      Sh.sigmoid (shared_query_gate input
        ((i / head_width) * (2 * head_width) + head_width + i % head_width))) =
  gate_offset_shared (i / head_width) (i % head_width);
  FStar.Pervasives.norm_spec [delta_only [`%shared_gated_context; `%Sh.split_gates]; zeta; iota]
    (shared_gated_context input i)

inline_for_extraction noextract
let shared_residual (input:logical_inputs) : Sh.vector hidden_width =
  Sh.vector_add hidden_width input.input_hidden
    (Sh.linear hidden_width activation_width (weights_view input.packed_weights).output_proj
      (shared_gated_context input))

inline_for_extraction noextract
let shared_mlp_product (gate up:Sh.vector mlp_width) : Sh.vector mlp_width =
  fun i -> Sh.silu (gate i) *. up i

inline_for_extraction noextract
let shared_mlp (w:Sh.layer_weights) (residual:Sh.vector hidden_width) : Sh.vector hidden_width =
  let normalized = Sh.rms_norm hidden_width w.post_attention_norm residual in
  let gate = Sh.linear mlp_width hidden_width w.mlp_gate_proj normalized in
  let up = Sh.linear mlp_width hidden_width w.mlp_up_proj normalized in
  let product = shared_mlp_product gate up in
  Sh.vector_add hidden_width residual (Sh.linear hidden_width mlp_width w.mlp_down_proj product)

noextract
let shared_decoder_decomposition (input:logical_inputs)
  : Lemma (
      let expected = Sh.full_attention_decoder input.call (weights_view input.packed_weights)
        input.input_hidden (logical_before input) in
      expected.hidden == shared_mlp (weights_view input.packed_weights) (shared_residual input) /\
      expected.cache == Sh.replace_kv (logical_before input) input.call.layer (shared_history input)) =
  assert (
      let expected = Sh.full_attention_decoder input.call (weights_view input.packed_weights)
        input.input_hidden (logical_before input) in
      expected.hidden == shared_mlp (weights_view input.packed_weights) (shared_residual input) /\
      expected.cache == Sh.replace_kv (logical_before input) input.call.layer (shared_history input))
    by (Proof.norm [nbe; unmeta; unascribe; delta_only [
          `%Sh.full_attention_decoder; `%shared_mlp; `%shared_mlp_product; `%shared_residual;
          `%shared_gated_context; `%shared_history; `%shared_current_keys; `%shared_current_values;
          `%shared_queries; `%shared_query_gate; `%shared_normalized;
          `%hidden_width; `%head_width; `%query_gate_width; `%activation_width;
          `%query_heads; `%kv_heads; `%kv_width; `%mlp_width]];
        Proof.split ();
        Proof.trefl ();
        Proof.trefl ())

inline_for_extraction noextract
let cache_coordinates
  (h:Sh.index kv_heads) (t:nat{t < cache_positions}) (d:Sh.index head_width)
  : Lemma (
      cache_index h t d / cache_head_elements == h /\
      (cache_index h t d % cache_head_elements) / head_width == t /\
      cache_index h t d % head_width == d) = ()

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let cache_append_shared
  (position:position_index)
  (previous_keys previous_values:real_vector (previous_elements position))
  (current_keys current_values:real_vector kv_width)
  (old_keys old_values:Sh.vector (previous_elements position))
  (new_keys new_values:Sh.matrix kv_heads head_width)
  : Lemma
      (requires
        (forall (i:Sh.index (previous_elements position)).
          S.index previous_keys i == old_keys i /\ S.index previous_values i == old_values i) /\
        (forall (h:Sh.index kv_heads) (d:Sh.index head_width).
          S.index current_keys (h * head_width + d) == new_keys h d /\
          S.index current_values (h * head_width + d) == new_values h d))
      (ensures (
        let next = Sh.append_kv position (old_history_view position old_keys old_values) new_keys new_values in
        forall (h:Sh.index kv_heads) (t:Sh.index (position + 1)) (d:Sh.index head_width).
          S.index (cache_update position previous_keys current_keys) (cache_index h t d) == next.keys h t d /\
          S.index (cache_update position previous_values current_values) (cache_index h t d) == next.values h t d)) =
  let next = Sh.append_kv position (old_history_view position old_keys old_values) new_keys new_values in
  let point (h:Sh.index kv_heads) (t:Sh.index (position + 1)) (d:Sh.index head_width)
    : Lemma (
        S.index (cache_update position previous_keys current_keys) (cache_index h t d) == next.keys h t d /\
        S.index (cache_update position previous_values current_values) (cache_index h t d) == next.values h t d) =
    cache_coordinates h t d;
    if position = 0 then
      Sh.append_includes_current position (old_history_view position old_keys old_values) new_keys new_values h d
    else if t < position then
      Sh.append_preserves_prefix (history_view position old_keys old_values) new_keys new_values h t d
    else
      Sh.append_includes_current position (old_history_view position old_keys old_values) new_keys new_values h d
  in C.forall_intro_3 point
#pop-options

inline_for_extraction noextract
let flatten_heads_shared
  (heads:pos) (vectors:lseq (real_vector head_width) heads) (logical:Sh.matrix heads head_width)
  : Lemma
      (requires forall (h:Sh.index heads). S.index vectors h == sequence_of (logical h))
      (ensures forall (h:Sh.index heads) (d:Sh.index head_width).
        S.index (S.init_ghost (heads * head_width) (fun i ->
          S.index (S.index vectors (i / head_width)) (i % head_width))) (h * head_width + d) ==
        logical h d) =
  let point (h:Sh.index heads) (d:Sh.index head_width)
    : Lemma (
        S.index (S.init_ghost (heads * head_width) (fun i ->
          S.index (S.index vectors (i / head_width)) (i % head_width))) (h * head_width + d) ==
        logical h d) =
    FStar.Math.Lemmas.lemma_div_mod_plus d h head_width;
    FStar.Math.Lemmas.small_div d head_width;
    FStar.Math.Lemmas.small_mod d head_width
  in C.forall_intro_2 point

#push-options "--fuel 4 --z3rlimit 60"
noextract
let decoder_prefix_shared (input:logical_inputs)
  : Lemma (
      let first = decoder_prefix_model (sequence_of input.packed_weights) (sequence_of input.input_hidden)
        input.call.position (sequence_of input.old_keys) (sequence_of input.old_values) in
      first.prefix_query_gate == sequence_of (shared_query_gate input) /\
      (forall (h:Sh.index query_heads) (d:Sh.index head_width).
        S.index first.prefix_queries (h * head_width + d) == shared_queries input h d) /\
      (forall (h:Sh.index kv_heads) (t:Sh.index (input.call.position + 1)) (d:Sh.index head_width).
        S.index first.prefix_keys (cache_index h t d) == (shared_history input).keys h t d /\
        S.index first.prefix_values (cache_index h t d) == (shared_history input).values h t d) /\
      (forall (h:Sh.index query_heads) (t:Sh.index (input.call.position + 1)).
        S.index first.prefix_scores (h * (input.call.position + 1) + t) ==
        shared_attention_scores input.call.position (shared_queries input) (shared_history input) h t)) =
  let weights = sequence_of input.packed_weights in
  let hidden = sequence_of input.input_hidden in
  let w = weights_view input.packed_weights in
  let position = input.call.position in
  let normalized = normalization weights input_norm_offset hidden in
  normalization_shared weights input_norm_offset hidden w.input_norm input.input_hidden;
  let qg = linear weights query_gate_offset query_gate_width hidden_width normalized in
  let kp = linear weights key_offset kv_width hidden_width normalized in
  let vp = linear weights value_offset kv_width hidden_width normalized in
  linear_shared weights query_gate_offset query_gate_width hidden_width normalized
    w.query_gate_proj (shared_normalized input);
  linear_shared weights key_offset kv_width hidden_width normalized w.key_proj (shared_normalized input);
  linear_shared weights value_offset kv_width hidden_width normalized w.value_proj (shared_normalized input);
  let key_heads = S.init_ghost kv_heads (fun h ->
    key_transform weights position (vector_slice kp (h * head_width) head_width)) in
  let current_keys = S.init_ghost kv_width (fun i ->
    S.index (S.index key_heads (i / head_width)) (i % head_width)) in
  let key_head (h:Sh.index kv_heads)
    : Lemma (S.index key_heads h == sequence_of (shared_current_keys input h)) =
    transform_shared false weights position (vector_slice kp (h * head_width) head_width)
      w.key_norm (Sh.reshape_kv (Sh.linear kv_width hidden_width w.key_proj (shared_normalized input)) h)
  in
  C.forall_intro key_head;
  flatten_heads_shared kv_heads key_heads (shared_current_keys input);
  let query_heads_values = S.init_ghost query_heads (fun h ->
    query_transform weights position (vector_slice qg (h * (2 * head_width)) head_width)) in
  let queries = S.init_ghost activation_width (fun i ->
    S.index (S.index query_heads_values (i / head_width)) (i % head_width)) in
  let query_head (h:Sh.index query_heads)
    : Lemma (S.index query_heads_values h == sequence_of (shared_queries input h)) =
    transform_shared true weights position (vector_slice qg (h * (2 * head_width)) head_width)
      w.query_norm (Sh.split_queries (shared_query_gate input) h)
  in
  C.forall_intro query_head;
  flatten_heads_shared query_heads query_heads_values (shared_queries input);
  cache_append_shared position (sequence_of input.old_keys) (sequence_of input.old_values)
    current_keys vp input.old_keys input.old_values (shared_current_keys input) (shared_current_values input);
  let keys = cache_update position (sequence_of input.old_keys) current_keys in
  scores_shared position queries keys (shared_queries input) (shared_history input);
  let first = decoder_prefix_model weights hidden position
    (sequence_of input.old_keys) (sequence_of input.old_values) in
  FStar.Pervasives.norm_spec [delta_only [`%decoder_prefix_model]; zeta; iota]
    (decoder_prefix_model weights hidden position (sequence_of input.old_keys) (sequence_of input.old_values));
  S.lemma_eq_elim first.prefix_query_gate qg;
  S.lemma_eq_elim first.prefix_queries queries;
  S.lemma_eq_elim first.prefix_keys keys;
  S.lemma_eq_elim first.prefix_values (cache_update position (sequence_of input.old_values) vp);
  S.lemma_eq_elim first.prefix_scores (attention_scores position queries keys)
#pop-options

inline_for_extraction noextract
let sigmoid_shared (x:real) : Lemma (sigmoid x == Sh.sigmoid x /\ silu x == Sh.silu x) = ()

#push-options "--fuel 4 --z3rlimit 10"
inline_for_extraction noextract
let mlp_shared (packed:Sh.vector weight_elements) (residual:real_vector hidden_width)
  (logical:Sh.vector hidden_width)
  : Lemma
      (requires forall (i:Sh.index hidden_width). S.index residual i == logical i)
      (ensures mlp_residual (sequence_of packed) residual ==
        sequence_of (shared_mlp (weights_view packed) logical)) =
  let weights = sequence_of packed in
  let w = weights_view packed in
  let normalized = normalization weights post_norm_offset residual in
  let logical_normalized = Sh.rms_norm hidden_width w.post_attention_norm logical in
  normalization_shared weights post_norm_offset residual w.post_attention_norm logical;
  let gate = linear weights mlp_gate_offset mlp_width hidden_width normalized in
  let up = linear weights mlp_up_offset mlp_width hidden_width normalized in
  let logical_gate = Sh.linear mlp_width hidden_width w.mlp_gate_proj logical_normalized in
  let logical_up = Sh.linear mlp_width hidden_width w.mlp_up_proj logical_normalized in
  linear_shared weights mlp_gate_offset mlp_width hidden_width normalized w.mlp_gate_proj logical_normalized;
  linear_shared weights mlp_up_offset mlp_width hidden_width normalized w.mlp_up_proj logical_normalized;
  let product = mlp_vector gate up in
  let logical_product = shared_mlp_product logical_gate logical_up in
  let point (i:Sh.index mlp_width)
    : Lemma (S.index product i == logical_product i) =
    sigmoid_shared (logical_gate i)
  in
  C.forall_intro point;
  linear_shared weights mlp_down_offset hidden_width mlp_width product w.mlp_down_proj logical_product;
  residual_shared residual (linear weights mlp_down_offset hidden_width mlp_width product)
    logical (Sh.linear hidden_width mlp_width w.mlp_down_proj logical_product);
  FStar.Pervasives.norm_spec [delta_only [`%mlp_residual]; zeta; iota] (mlp_residual weights residual);
  FStar.Pervasives.norm_spec [delta_only [`%shared_mlp]; zeta; iota] (shared_mlp w logical);
  S.lemma_eq_elim (mlp_residual weights residual) (sequence_of (shared_mlp w logical))
#pop-options

inline_for_extraction noextract
let decoder_activation_cell
  (position:position_index) (controls:lseq f32 (score_elements position))
  (query_gate:real_vector query_gate_width) (scores:real_vector (score_elements position))
  (values:real_vector cache_elements) (i:Sh.index activation_width)
  : Lemma (
      S.index (decoder_activation position controls query_gate scores values) i ==
      S.index (attention_head position controls scores values (i / head_width)) (i % head_width) *.
        sigmoid (S.index query_gate ((i / head_width) * (2 * head_width) + head_width + i % head_width))) = ()

#push-options "--fuel 4 --z3rlimit 10"
noextract
let execution_model_shared
  (input:logical_inputs) (controls:lseq f32 (score_elements input.call.position))
  : Lemma
      (requires bilateral_live_scores_finite input.call.position controls)
      (ensures (
        let full = execution_model (sequence_of input.packed_weights) (sequence_of input.input_hidden)
          input.call.position (sequence_of input.old_keys) (sequence_of input.old_values) controls in
        let expected = Sh.full_attention_decoder input.call (weights_view input.packed_weights)
          input.input_hidden (logical_before input) in
        full.hidden_result == sequence_of expected.hidden /\
        (forall (h:Sh.index kv_heads) (t:Sh.index (input.call.position + 1)) (d:Sh.index head_width).
          S.index full.key_result (cache_index h t d) == (shared_history input).keys h t d /\
          S.index full.value_result (cache_index h t d) == (shared_history input).values h t d))) =
  let position = input.call.position in
  let weights = sequence_of input.packed_weights in
  let hidden = sequence_of input.input_hidden in
  let w = weights_view input.packed_weights in
  let first = decoder_prefix_model weights hidden position (sequence_of input.old_keys) (sequence_of input.old_values) in
  decoder_prefix_shared input;
  let head (h:Sh.index query_heads)
    : Lemma (attention_head position controls first.prefix_scores first.prefix_values h ==
        sequence_of (Sh.attention position (shared_queries input) (shared_history input) h)) =
    attention_head_shared position controls first.prefix_scores first.prefix_values
      (shared_queries input) (shared_history input) h
  in
  C.forall_intro head;
  let activation = decoder_activation position controls
    first.prefix_query_gate first.prefix_scores first.prefix_values in
  let gated (i:Sh.index activation_width)
    : Lemma (S.index activation i == shared_gated_context input i) =
    decoder_activation_cell position controls first.prefix_query_gate first.prefix_scores first.prefix_values i;
    head (i / head_width);
    sequence_at (Sh.attention position (shared_queries input) (shared_history input) (i / head_width))
      (i % head_width);
    sequence_at (shared_query_gate input)
      ((i / head_width) * (2 * head_width) + head_width + i % head_width);
    shared_gated_cell input i;
    sigmoid_shared (shared_query_gate input
      ((i / head_width) * (2 * head_width) + head_width + i % head_width))
  in
  C.forall_intro gated;
  linear_shared weights output_offset hidden_width activation_width activation w.output_proj
    (shared_gated_context input);
  let projected = linear weights output_offset hidden_width activation_width activation in
  let residual = residual_vector hidden projected in
  residual_shared hidden projected input.input_hidden
    (Sh.linear hidden_width activation_width w.output_proj (shared_gated_context input));
  mlp_shared input.packed_weights residual (shared_residual input);
  shared_decoder_decomposition input;
  FStar.Pervasives.norm_spec [delta_only [`%execution_model]; zeta; iota]
    (execution_model weights hidden position (sequence_of input.old_keys) (sequence_of input.old_values) controls)
#pop-options

inline_for_extraction noextract
let packed_cache_cell
  (n:pos{n <= cache_positions}) (logical:Sh.kv_tensor n) (tail:Sh.vector cache_elements)
  (h:Sh.index kv_heads) (t:Sh.index n) (d:Sh.index head_width)
  : Lemma (pack_kv n logical tail (cache_index h t d) == logical h t d) =
  kv_logical_roundtrip n logical tail h t d

inline_for_extraction noextract
let initial_history_pack_equality
  (old_keys old_values:Sh.vector 0) (new_keys new_values:Sh.matrix kv_heads head_width)
  (key_tail value_tail:Sh.vector cache_elements)
  : Lemma (
      let next = Sh.append_kv 0 (old_history_view 0 old_keys old_values) new_keys new_values in
      history_view 1 (pack_kv 1 next.keys key_tail) (pack_kv 1 next.values value_tail) == next) =
  assert (
      let next = Sh.append_kv 0 (old_history_view 0 old_keys old_values) new_keys new_values in
      history_view 1 (pack_kv 1 next.keys key_tail) (pack_kv 1 next.values value_tail) == next)
    by (Proof.norm [delta_only [`%Sh.append_kv; `%old_history_view; `%history_view; `%kv_view]];
        Proof.l_to_r [`packed_cache_cell];
        Proof.trefl ())

inline_for_extraction noextract
let positive_old_history (position:pos{position < cache_positions})
  (old_keys old_values:Sh.vector (previous_elements position))
  : Lemma (old_history_view position old_keys old_values ==
      Some (history_view position old_keys old_values)) = ()

inline_for_extraction noextract
let positive_history_pack_equality
  (position:pos{position < cache_positions})
  (old_keys old_values:Sh.vector (previous_elements position))
  (new_keys new_values:Sh.matrix kv_heads head_width) (key_tail value_tail:Sh.vector cache_elements)
  : Lemma (
      let next = Sh.append_kv position (old_history_view position old_keys old_values) new_keys new_values in
      history_view (position + 1)
        (pack_kv (position + 1) next.keys key_tail) (pack_kv (position + 1) next.values value_tail) == next) =
  assert (
      let next = Sh.append_kv position (old_history_view position old_keys old_values) new_keys new_values in
      history_view (position + 1)
        (pack_kv (position + 1) next.keys key_tail) (pack_kv (position + 1) next.values value_tail) == next)
    by (Proof.l_to_r [`positive_old_history];
        Proof.norm [delta_only [`%Sh.append_kv; `%history_view; `%kv_view]];
        Proof.l_to_r [`packed_cache_cell];
        Proof.trefl ())

inline_for_extraction noextract
let history_pack_equality
  (position:position_index)
  (old_keys old_values:Sh.vector (previous_elements position))
  (new_keys new_values:Sh.matrix kv_heads head_width) (key_tail value_tail:Sh.vector cache_elements)
  : Lemma (
      let next = Sh.append_kv position (old_history_view position old_keys old_values) new_keys new_values in
      history_view (position + 1)
        (pack_kv (position + 1) next.keys key_tail) (pack_kv (position + 1) next.values value_tail) == next) =
  if position = 0 then initial_history_pack_equality old_keys old_values new_keys new_values key_tail value_tail
  else positive_history_pack_equality position old_keys old_values new_keys new_values key_tail value_tail

inline_for_extraction noextract
let choose_kv (selected:bool) (next frame:option Sh.kv_history) : option Sh.kv_history =
  if selected then next else frame

inline_for_extraction noextract
let choose_equal_conditions
  (selected also_selected:bool{selected == also_selected})
  (next old frame:option Sh.kv_history)
  : Lemma (
      choose_kv selected next (choose_kv also_selected old frame) ==
      choose_kv selected next frame) = ()

inline_for_extraction noextract
let cache_slot
  (call:Sh.invocation) (frame:Sh.hybrid_cache)
  (old_keys old_values:Sh.vector (previous_elements call.position)) (l:Sh.layer_index)
  : Lemma (
      (cache_view call frame old_keys old_values).kv l ==
      (if l = call.layer then old_history_view call.position old_keys old_values else frame.kv l)) = ()

inline_for_extraction noextract
let composed_cache
  (call:Sh.invocation) (frame:Sh.hybrid_cache)
  (old_keys old_values:Sh.vector (previous_elements call.position))
  (next:Sh.kv_history) : Sh.hybrid_cache = {
  kv = (fun l -> choose_kv (Prims.op_Equals #Sh.layer_index l call.layer) (Some next)
    (choose_kv (l = call.layer) (old_history_view call.position old_keys old_values) (frame.kv l)));
  convolution = frame.convolution;
  recurrent = frame.recurrent
}

inline_for_extraction noextract
let single_cache (call:Sh.invocation) (frame:Sh.hybrid_cache) (next:Sh.kv_history) : Sh.hybrid_cache = {
  kv = (fun l -> choose_kv (Prims.op_Equals #Sh.layer_index l call.layer) (Some next) (frame.kv l));
  convolution = frame.convolution;
  recurrent = frame.recurrent
}

inline_for_extraction noextract
let replace_cache_frame
  (call:Sh.invocation) (frame:Sh.hybrid_cache)
  (old_keys old_values:Sh.vector (previous_elements call.position)) (next:Sh.kv_history)
  : Lemma (
      Sh.replace_kv (cache_view call frame old_keys old_values) call.layer next ==
      Sh.replace_kv frame call.layer next) =
  assert (
      Sh.replace_kv (cache_view call frame old_keys old_values) call.layer next ==
      composed_cache call frame old_keys old_values next)
    by (Proof.norm [delta_only [`%Sh.replace_kv; `%cache_view; `%composed_cache; `%choose_kv]];
        Proof.l_to_r [`cache_slot];
        Proof.trefl ());
  assert (composed_cache call frame old_keys old_values next == single_cache call frame next)
    by (Proof.norm [delta_only [`%composed_cache; `%single_cache]];
        Proof.l_to_r [`choose_equal_conditions];
        Proof.trefl ());
  assert (single_cache call frame next == Sh.replace_kv frame call.layer next)
    by (Proof.norm [delta_only [`%Sh.replace_kv; `%single_cache; `%choose_kv]];
        Proof.trefl ())

inline_for_extraction noextract
let packed_cache_representation
  (n:pos{n <= cache_positions}) (storage:bfseq cache_elements) (model:real_vector cache_elements)
  (logical:Sh.kv_tensor n)
  : Lemma
      (requires storage %~ model /\
        (forall (h:Sh.index kv_heads) (t:Sh.index n) (d:Sh.index head_width).
          S.index model (cache_index h t d) == logical h t d))
      (ensures represents cache_elements storage (pack_kv n logical (vector_of model))) =
  let point (i:Sh.index cache_elements)
    : Lemma (v_approximates (S.index storage i) (pack_kv n logical (vector_of model) i)) =
    let h = i / cache_head_elements in
    let t = (i % cache_head_elements) / head_width in
    let d = i % head_width in
    assert (cache_index h t d == i);
    if t < n then assert (S.index model i == logical h t d)
  in C.forall_intro point

inline_for_extraction noextract
let unconditional_state_post
  (input:logical_inputs) (previous_keys previous_values:bfseq (previous_elements input.call.position))
  (keys values:bfseq cache_elements)
  : Lemma
      (requires retained_cache_post input.call.position previous_keys keys /\
        retained_cache_post input.call.position previous_values values)
      (ensures state_postcondition input previous_keys previous_values keys values) =
  let frame (h:Sh.vector hidden_width) (k v:Sh.vector cache_elements)
    : Lemma (
        let result = result_view input.call input.frame h k v in
        Sh.history_length (result.cache.kv input.call.layer) == input.call.position + 1 /\
        not (result.cache.kv input.call.layer == None) /\
        (forall (l:Sh.layer_index). l <> input.call.layer ==>
          result.cache.kv l == (logical_before input).kv l) /\
        result.cache.convolution == (logical_before input).convolution /\
        result.cache.recurrent == (logical_before input).recurrent) =
    result_frame input.call input.frame h k v
  in C.forall_intro_3 frame

noextract
let semantic_from_model
  (input:logical_inputs) (previous_keys previous_values:bfseq (previous_elements input.call.position))
  (result:bfseq hidden_width) (keys values:bfseq cache_elements)
  (controls:lseq f32 (score_elements input.call.position))
  : Lemma
      (requires bilateral_live_scores_finite input.call.position controls /\
        retained_cache_post input.call.position previous_keys keys /\
        retained_cache_post input.call.position previous_values values /\
        (let full = execution_model (sequence_of input.packed_weights) (sequence_of input.input_hidden)
          input.call.position (sequence_of input.old_keys) (sequence_of input.old_values) controls in
         result %~ full.hidden_result /\ keys %~ full.key_result /\ values %~ full.value_result))
      (ensures semantic_postcondition input previous_keys previous_values result keys values) =
  let full = execution_model (sequence_of input.packed_weights) (sequence_of input.input_hidden)
    input.call.position (sequence_of input.old_keys) (sequence_of input.old_values) controls in
  let expected = Sh.full_attention_decoder input.call (weights_view input.packed_weights)
    input.input_hidden (logical_before input) in
  let next = shared_history input in
  let logical_keys = pack_kv (input.call.position + 1) next.keys (vector_of full.key_result) in
  let logical_values = pack_kv (input.call.position + 1) next.values (vector_of full.value_result) in
  execution_model_shared input controls;
  packed_cache_representation (input.call.position + 1) keys full.key_result next.keys;
  packed_cache_representation (input.call.position + 1) values full.value_result next.values;
  representation_sequence hidden_width result expected.hidden;
  history_pack_equality input.call.position input.old_keys input.old_values
    (shared_current_keys input) (shared_current_values input) (vector_of full.key_result) (vector_of full.value_result);
  replace_cache_frame input.call input.frame input.old_keys input.old_values next;
  shared_decoder_decomposition input;
  assert (Sh.postcondition input.call (weights_view input.packed_weights) input.input_hidden
    (logical_before input) (result_view input.call input.frame expected.hidden logical_keys logical_values));
  assert (exists (logical_hidden:Sh.vector hidden_width)
                 (logical_keys:Sh.vector cache_elements)
                 (logical_values:Sh.vector cache_elements).
    represents hidden_width result logical_hidden /\
    represents cache_elements keys logical_keys /\
    represents cache_elements values logical_values /\
    Sh.postcondition input.call (weights_view input.packed_weights) input.input_hidden (logical_before input)
      (result_view input.call input.frame logical_hidden logical_keys logical_values))

inline_for_extraction noextract
let tree_width (n : nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32

inline_for_extraction noextract
let rec tree_value
  (f : nat -> GTot real)
  (width : nat{tree_width width})
  (lane : nat)
  : GTot real (decreases 32 - width)
  = if width = 32 then f lane
    else tree_value f (2 * width) lane +.
      tree_value f (2 * width) (lane + width)

inline_for_extraction noextract
let rec tree_total
  (f : nat -> GTot real)
  (width : nat{tree_width width})
  : Lemma
      (ensures prefix width (tree_value f width) == prefix 32 f)
      (decreases 32 - width)
  = if width = 32 then prefix_ext 32 (tree_value f 32) f
    else begin
      tree_total f (2 * width);
      prefix_ext width (tree_value f width)
        (fun i -> tree_value f (2 * width) i +.
          tree_value f (2 * width) (i + width));
      prefix_add width (tree_value f (2 * width))
        (fun i -> tree_value f (2 * width) (i + width));
      prefix_split width width (tree_value f (2 * width));
      prefix_ext width
        (fun i -> tree_value f (2 * width) (i + width))
        (fun i -> tree_value f (2 * width) (width + i))
    end

inline_for_extraction noextract
let tree_root (f : nat -> GTot real)
  : Lemma (tree_value f 1 0 == prefix 32 f)
  = tree_total f 1

inline_for_extraction noextract
let lift_sum32
  (#parameter : Type)
  (premise : parameter -> prop)
  (model : parameter -> nat -> GTot real)
  (values : lseq f32 32)
  (result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==>
          (forall (i : nat{i < 32}). S.index values i %~ model p i)) /\
        (forall (r : lseq real 32).
          values %~ r ==> result %~ prefix 32 (seq_value r)))
      (ensures forall (p : parameter).
        premise p ==> result %~ prefix 32 (model p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ prefix 32 (model p))
      = let r = S.init_ghost 32 (model p) in
        assert (values %~ r);
        prefix_ext 32 (seq_value r) (model p)
    in
    C.forall_intro aux

inline_for_extraction noextract
let lift_fma
  (#parameter : Type)
  (premise : parameter -> prop)
  (xm ym zm : parameter -> GTot real)
  (x y z result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==>
          x %~ xm p /\ y %~ ym p /\ z %~ zm p) /\
        (forall (xr yr zr : real).
          x %~ xr /\ y %~ yr /\ z %~ zr ==>
          result %~ (xr *. yr +. zr)))
      (ensures forall (p : parameter). premise p ==>
        result %~ (xm p *. ym p +. zm p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ (xm p *. ym p +. zm p))
      = ()
    in
    C.forall_intro aux

inline_for_extraction noextract
let lift_add
  (#parameter : Type)
  (premise : parameter -> prop)
  (xm ym : parameter -> GTot real)
  (x y result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==> x %~ xm p /\ y %~ ym p) /\
        (forall (xr yr : real).
          x %~ xr /\ y %~ yr ==> result %~ (xr +. yr)))
      (ensures forall (p : parameter). premise p ==> result %~ (xm p +. ym p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ (xm p +. ym p))
      = ()
    in
    C.forall_intro aux

inline_for_extraction noextract
let lift_mul
  (#parameter : Type)
  (premise : parameter -> prop)
  (xm ym : parameter -> GTot real)
  (x y result : f32)
  : Lemma
      (requires
        (forall (p : parameter). premise p ==> x %~ xm p /\ y %~ ym p) /\
        (forall (xr yr : real).
          x %~ xr /\ y %~ yr ==> result %~ (xr *. yr)))
      (ensures forall (p : parameter). premise p ==> result %~ (xm p *. ym p))
  = let aux (p : parameter{premise p})
      : Lemma (result %~ (xm p *. ym p))
      = ()
    in
    C.forall_intro aux

inline_for_extraction noextract
let inverse_width () : Pure f32
  (requires True)
  (ensures fun result -> result %~ (1.0R /. 1024.0R))
  = let numerator : f32 = of_int 1L in
    let denominator : f32 = of_int 1024L in
    let _ = of_int_approx #f32 1L in
    let _ = of_int_approx #f32 1024L in
    div_approx numerator denominator 1.0R 1024.0R;
    div numerator denominator

(* Constant division is folded to the accepted binary32 literal by NVCC. *)
inline_for_extraction noextract
let epsilon32 () : Pure f32
  (requires True)
  (ensures fun result -> result %~ 0.000001R)
  = let numerator : f32 = of_int 1L in
    let denominator : f32 = of_int 1000000L in
    let _ = of_int_approx #f32 1L in
    let _ = of_int_approx #f32 1000000L in
    div_approx numerator denominator 1.0R 1000000.0R;
    div numerator denominator

inline_for_extraction noextract
let tree_invariant
  (original current : lseq f32 32)
  (width : nat{tree_width width})
  : prop
  = forall (model : lseq real 32).
      original %~ model ==>
      (forall (i : nat{i < width}).
        S.index current i %~ tree_value (seq_value model) width i)

inline_for_extraction noextract
let tree_progress
  (original current : lseq f32 32)
  (distance : nat{tree_width distance /\ distance < 32})
  (done : nat{done <= distance})
  : prop
  = forall (model : lseq real 32).
      original %~ model ==>
      (forall (i : nat{i < 2 * distance}).
        S.index current i %~
          (if i < done then tree_value (seq_value model) distance i
           else tree_value (seq_value model) (2 * distance) i))

inline_for_extraction noextract
let tree_initial (values : lseq f32 32)
  : Lemma (tree_invariant values values 32)
  = ()

inline_for_extraction noextract
let tree_step
  (original current : lseq f32 32)
  (distance : nat{tree_width distance /\ distance < 32})
  (i : nat{i < distance})
  (value : f32)
  : Lemma
      (requires tree_progress original current distance i /\
        (forall (xr yr : real).
          S.index current i %~ xr /\
          S.index current (i + distance) %~ yr ==>
          value %~ (xr +. yr)))
      (ensures tree_progress original (S.upd current i value) distance (i + 1))
  = ()

inline_for_extraction noextract
let tree_finished (original current : lseq f32 32)
  : Lemma
      (requires tree_invariant original current 1)
      (ensures forall (model : lseq real 32).
        original %~ model ==>
        S.index current 0 %~ prefix 32 (seq_value model))
  = let aux (model : lseq real 32)
      : Lemma (tree_value (seq_value model) 1 0 ==
        prefix 32 (seq_value model))
      = tree_root (seq_value model)
    in
    C.forall_intro aux

inline_for_extraction noextract
fn reduce_level
  (partial : larray f32 32)
  (distance : sz{tree_width (SZ.v distance) /\ distance < 32})
  (#original #before : erased (lseq f32 32))
  preserves gpu
  requires partial |-> before **
    pure (tree_invariant original before (2 * SZ.v distance))
  ensures exists* (after : lseq f32 32).
    partial |-> after ** pure (tree_invariant original after (SZ.v distance))
{
  let mut lane = 0sz;
  while (!lane <^ distance)
    invariant exists* (i : sz{i <= distance}) (values : lseq f32 32).
      lane |-> i ** partial |-> values **
      pure (tree_progress original values (SZ.v distance) (SZ.v i))
    decreases (distance - !lane)
  {
    let i = !lane;
    with values. assert partial |-> values;
    let a = partial.(i);
    let b = partial.(i +^ distance);
    let value = F32.add_rn_ftz a b;
    tree_step original values (SZ.v distance) (SZ.v i) value;
    partial.(i) <- value;
    lane := i +^ 1sz;
  }
}

inline_for_extraction noextract
fn sum32
  (partial : larray f32 32)
  (#before : erased (lseq f32 32))
  preserves gpu
  requires partial |-> before
  returns result : f32
  ensures exists* (after : lseq f32 32).
    partial |-> after **
    pure (forall (model : lseq real 32).
      before %~ model ==> result %~ prefix 32 (seq_value model))
{
  tree_initial before;
  let mut width = 32sz;
  while (1sz <^ !width)
    invariant exists* (w : sz{tree_width (SZ.v w)}) (values : lseq f32 32).
      width |-> w ** partial |-> values **
      pure (tree_invariant before values (SZ.v w))
    decreases (SZ.v (!width))
  {
    let w = !width;
    let distance = w /^ 2sz;
    reduce_level partial distance #before;
    width := distance;
  };
  with values. assert partial |-> values;
  tree_finished before values;
  partial.(0sz)
}

inline_for_extraction noextract
let square_cell (hidden : lseq real hidden_width) (i : nat) : GTot real =
  if i < hidden_width then
    let x = S.index hidden i in x *. x
  else 0.0R

inline_for_extraction noextract
let square_pair (hidden : lseq real hidden_width) (warp lane : nat) : GTot real =
  square_cell hidden (warp * 32 + lane) +.
  square_cell hidden (512 + (warp * 32 + lane))

inline_for_extraction noextract
let warp_square_sum (hidden : lseq real hidden_width) (warp : nat) : GTot real =
  if warp < 16 then prefix 32 (square_pair hidden warp) else 0.0R

inline_for_extraction noextract
let rec squares_prefix
  (hidden : lseq real hidden_width)
  (count : nat{count <= hidden_width})
  : Lemma
      (ensures sum_squares hidden count == prefix count (square_cell hidden))
      (decreases count)
  = if count > 0 then begin
      squares_prefix hidden (count - 1);
      prefix_step (count - 1) (square_cell hidden)
    end

inline_for_extraction noextract
let warp_squares_split
  (hidden : lseq real hidden_width)
  (warp : nat{warp < 16})
  : Lemma (warp_square_sum hidden warp ==
      prefix 32 (fun lane -> square_cell hidden (warp * 32 + lane)) +.
      prefix 32 (fun lane -> square_cell hidden (512 + (warp * 32 + lane))))
  = prefix_ext 32 (square_pair hidden warp)
      (fun lane -> square_cell hidden (warp * 32 + lane) +.
        square_cell hidden (512 + (warp * 32 + lane)));
    prefix_add 32
      (fun lane -> square_cell hidden (warp * 32 + lane))
      (fun lane -> square_cell hidden (512 + (warp * 32 + lane)))

inline_for_extraction noextract
let all_squares
  (hidden : lseq real hidden_width)
  : Lemma (prefix 32 (warp_square_sum hidden) == sum_squares hidden hidden_width)
  = let first = (fun (warp : nat) -> prefix 32
      (fun lane -> square_cell hidden (warp * 32 + lane))) in
    let second = (fun (warp : nat) -> prefix 32
      (fun lane -> square_cell hidden (512 + (warp * 32 + lane)))) in
    C.forall_intro (warp_squares_split hidden);
    prefix_ext 16 (warp_square_sum hidden) (fun warp -> first warp +. second warp);
    prefix_add 16 first second;
    prefix_blocks 16 32 (square_cell hidden);
    prefix_blocks 16 32 (fun i -> square_cell hidden (512 + i));
    prefix_split 512 512 (square_cell hidden);
    squares_prefix hidden 1024;
    prefix_split 16 16 (warp_square_sum hidden);
    prefix_ext 16 (fun i -> warp_square_sum hidden (16 + i)) (fun _ -> 0.0R);
    prefix_zero 16

inline_for_extraction noextract
let lanes_squared
  (hidden : lseq bf16 hidden_width)
  (values : lseq f32 32)
  (warp : nat{warp < 16})
  (done : nat{done <= 32})
  : prop
  = forall (model : lseq real hidden_width).
      hidden %~ model ==>
      (forall (i : nat{i < done}).
        S.index values i %~ square_pair model warp i)

inline_for_extraction noextract
let warps_squared
  (hidden : lseq bf16 hidden_width)
  (values : lseq f32 32)
  (done : nat{done <= 16})
  : prop
  = forall (model : lseq real hidden_width).
      hidden %~ model ==>
      (forall (i : nat{i < 32}).
        if i < done then S.index values i %~ warp_square_sum model i
        else S.index values i == zero #f32)

inline_for_extraction noextract
let rms_finish
  (hidden : lseq bf16 hidden_width)
  (argument result : f32)
  : Lemma
      (requires
        (forall (model : lseq real hidden_width).
          hidden %~ model ==> argument %~
            (sum_squares model hidden_width /. 1024.0R +. 0.000001R <: real)) /\
        (forall (x : Sqrt.rpos).
          argument %~ (x <: real) ==>
          result %~ (1.0R /. Sqrt.sqrt x)))
      (ensures forall (model : lseq real hidden_width).
        hidden %~ model ==> result %~ (rms_scale model <: real))
  = let aux (model : lseq real hidden_width{hidden %~ model})
      : Lemma (result %~ (rms_scale model <: real))
      = ()
    in
    C.forall_intro aux

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
fn hidden_rstd
  (hidden : larray bf16 hidden_width)
  (#before : erased (lseq bf16 hidden_width))
  (#fraction : perm)
  preserves gpu ** hidden |-> Frac fraction before
  returns result : f32
  ensures pure (forall (model : lseq real hidden_width).
    before %~ model ==> result %~ (rms_scale model <: real))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut lanes = [| zero #f32 #_; 32sz |];
  let mut warps = [| zero #f32 #_; 32sz |];
  let mut warp = 0sz;
  while (!warp <^ 16sz)
    invariant exists* (w : sz{w <= 16}) (values : lseq f32 32).
      warp |-> w ** warps |-> values ** pure (warps_squared before values (SZ.v w))
    invariant exists* (values : lseq f32 32). lanes |-> values
    decreases (16 - !warp)
  {
    let w = !warp;
    let mut lane = 0sz;
    while (!lane <^ 32sz)
      invariant exists* (i : sz{i <= 32}) (values : lseq f32 32).
        lane |-> i ** lanes |-> values **
        pure (lanes_squared before values (SZ.v w) (SZ.v i))
      decreases (32 - !lane)
    {
      let i = !lane;
      let a : f32 = Cast.fcast (hidden.(w *^ 32sz +^ i));
      let b : f32 = Cast.fcast (hidden.(w *^ 32sz +^ i +^ 512sz));
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==>
        a %~ S.index model (SZ.v w * 32 + SZ.v i) /\
        b %~ S.index model (SZ.v w * 32 + SZ.v i + 512));
      let a_squared = F32.fma_rn_ftz a a (zero #f32);
      lift_fma
        (fun (model : lseq real hidden_width) -> before %~ model)
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i))
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i))
        (fun _ -> 0.0R)
        a a (zero #f32) a_squared;
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==>
        a_squared %~ square_cell model (SZ.v w * 32 + SZ.v i));
      let pair = F32.fma_rn_ftz b b a_squared;
      lift_fma
        (fun (model : lseq real hidden_width) -> before %~ model)
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i + 512))
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i + 512))
        (fun model -> square_cell model (SZ.v w * 32 + SZ.v i))
        b b a_squared pair;
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==> pair %~
          (S.index model (SZ.v w * 32 + SZ.v i + 512) *.
           S.index model (SZ.v w * 32 + SZ.v i + 512) +.
           square_cell model (SZ.v w * 32 + SZ.v i) <: real));
      assert pure (forall (model : lseq real hidden_width).
        before %~ model ==> pair %~ square_pair model (SZ.v w) (SZ.v i));
      lanes.(i) <- pair;
      lane := i +^ 1sz;
    };
    with lane_values. assert lanes |-> lane_values;
    let warp_total = sum32 lanes;
    lift_sum32
      (fun (model : lseq real hidden_width) -> before %~ model)
      (fun model -> square_pair model (SZ.v w))
      lane_values warp_total;
    warps.(w) <- warp_total;
    warp := w +^ 1sz;
  };
  with warp_values. assert warps |-> warp_values;
  let squared_total = sum32 warps;
  lift_sum32
    (fun (model : lseq real hidden_width) -> before %~ model)
    warp_square_sum warp_values squared_total;
  C.forall_intro all_squares;
  assert pure (forall (model : lseq real hidden_width).
    before %~ model ==> squared_total %~ (sum_squares model hidden_width <: real));
  let factor = inverse_width ();
  let epsilon = epsilon32 ();
  assert pure (factor %~ (1.0R /. 1024.0R));
  assert pure (epsilon %~ 0.000001R);
  let argument = F32.fma_rn_ftz squared_total factor epsilon;
  lift_fma
    (fun (model : lseq real hidden_width) -> before %~ model)
    (fun model -> (sum_squares model hidden_width <: real))
    (fun _ -> 1.0R /. 1024.0R)
    (fun _ -> 0.000001R)
    squared_total factor epsilon argument;
  let result = F32.rsqrt_approx_ftz argument;
  rms_finish before argument result;
  result
}
#pop-options

inline_for_extraction noextract
let flush_exponential (x exponential result : f32)
  : Lemma
      (requires
        (forall (r : real).
          Approx.v_approximates x r ==>
          Approx.v_approximates exponential (Kuiper.Real.exp r)) /\
        (forall (a b : real).
          Approx.v_approximates exponential a /\
          Approx.v_approximates (one #f32) b ==>
          Approx.v_approximates result (a *. b)))
      (ensures forall (r : real).
        Approx.v_approximates x r ==>
        Approx.v_approximates result (Kuiper.Real.exp r))
  = let _ = Approx.a1 #f32 in
    let related (r : real{Approx.v_approximates x r})
      : Lemma (Approx.v_approximates result (Kuiper.Real.exp r))
      = assert (Approx.v_approximates exponential (Kuiper.Real.exp r));
        assert (Approx.v_approximates (one #f32) 1.0R);
        let product (a b : real) =
          Approx.v_approximates exponential a /\
          Approx.v_approximates (one #f32) b ==>
          Approx.v_approximates result (a *. b)
        in
        let at_exp = FStar.Classical.Sugar.forall_elim
          #real #(fun a -> forall (b : real). product a b)
          (Kuiper.Real.exp r) ()
        in
        let _ = FStar.Classical.Sugar.forall_elim
          #real #(product (Kuiper.Real.exp r)) 1.0R at_exp
        in ()
    in FStar.Classical.forall_intro related

[@@CPrologue "__device__"]
fn exponential_ftz (x : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (r : real). x %~ r ==> result %~ Kuiper.Real.exp r)
{
  let exponential = Fast.exp x;
  let result = F32.mul_rn_ftz exponential (one #f32);
  flush_exponential x exponential result;
  result
}

inline_for_extraction noextract
let negate32 (x : f32) : Pure f32
  (requires True)
  (ensures fun result -> forall (r : real). x %~ r ==> result %~ (0.0R -. r))
  = let minus_one : f32 = of_int (-1L) in
    let _ = of_int_approx #f32 (-1L) in
    mul minus_one x

[@@CPrologue "__device__"]
fn negative (x : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (r : real). x %~ r ==> result %~ (0.0R -. r))
{
  negate32 x
}

inline_for_extraction noextract
let sigmoid_argument (x exponential argument : f32)
  : Lemma
      (requires
        (forall (r : real). x %~ r ==>
          exponential %~ Kuiper.Real.exp (0.0R -. r)) /\
        (forall (a b : real).
          (one #f32) %~ a /\ exponential %~ b ==> argument %~ (a +. b)))
      (ensures forall (r : real). x %~ r ==>
        argument %~ (1.0R +. Kuiper.Real.exp (0.0R -. r)) /\
        1.0R +. Kuiper.Real.exp (0.0R -. r) >. 0.0R)
  = let related (r : real{x %~ r})
      : Lemma (
          argument %~ (1.0R +. Kuiper.Real.exp (0.0R -. r)) /\
          1.0R +. Kuiper.Real.exp (0.0R -. r) >. 0.0R)
      = exp_positive (0.0R -. r)
    in FStar.Classical.forall_intro related

[@@CPrologue "__device__"]
fn sigmoid32 (x : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (r : real). x %~ r ==> result %~ sigmoid r)
{
  let negated = negate32 x;
  let exponential = exponential_ftz negated;
  assert pure (forall (r : real). x %~ r ==>
    exponential %~ Kuiper.Real.exp (0.0R -. r));
  let argument = F32.add_rn_ftz (one #f32) exponential;
  sigmoid_argument x exponential argument;
  F32.rcp_approx_ftz argument
}

[@@CPrologue "__device__"]
fn silu32 (x : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (r : real). x %~ r ==> result %~ silu r)
{
  let sigmoid = sigmoid32 x;
  F32.mul_rn_ftz x sigmoid
}

inline_for_extraction noextract
let matrix_interpretation (n : nat) =
  real_vector weight_elements & real_vector n

inline_for_extraction noextract
let matrix_inputs
  (#n : nat)
  (weights : lseq bf16 weight_elements)
  (values : lseq f32 n)
  (model : matrix_interpretation n)
  : prop = weights %~ fst model /\ values %~ snd model

inline_for_extraction noextract
let matrix_term
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (column : nat)
  : GTot real
  = if column < n then
      S.index (fst model) (offset + column) *. S.index (snd model) column
    else 0.0R

inline_for_extraction noextract
let matrix_block
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (lane group : nat)
  : GTot real
  = prefix 8 (fun j -> matrix_term model offset (group * 256 + (lane * 8 + j)))

inline_for_extraction noextract
let matrix_lane
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (lane : nat)
  : GTot real
  = prefix (n / 256) (matrix_block model offset lane)

inline_for_extraction noextract
let matrix_partial
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (lane group count : nat)
  (chunks : bool)
  : GTot real
  = (if chunks then 0.0R else prefix group (matrix_block model offset lane)) +.
      prefix count (fun j -> matrix_term model offset (group * 256 + (lane * 8 + j)))

inline_for_extraction noextract
let matrix_partial_step
  (#n : nat)
  (offset : nat{offset + n <= weight_elements})
  (lane group count : nat)
  (chunks : bool)
  (model : matrix_interpretation n)
  : Lemma (matrix_partial model offset lane group (count + 1) chunks ==
      matrix_partial model offset lane group count chunks +.
      matrix_term model offset (group * 256 + (lane * 8 + count)))
  = prefix_step count (fun j -> matrix_term model offset (group * 256 + (lane * 8 + j)))

inline_for_extraction noextract
let matrix_group_end
  (#n : nat)
  (offset : nat{offset + n <= weight_elements})
  (lane group : nat)
  (model : matrix_interpretation n)
  : Lemma (
      matrix_partial model offset lane group 8 false ==
        prefix (group + 1) (matrix_block model offset lane) /\
      prefix group (matrix_block model offset lane) +.
        matrix_partial model offset lane group 8 true ==
        prefix (group + 1) (matrix_block model offset lane))
  = prefix_step group (matrix_block model offset lane)

inline_for_extraction noextract
let matrix_group_sum
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (group : nat)
  : Lemma (
      prefix 32 (fun lane -> matrix_block model offset lane group) ==
      prefix 256 (fun column -> matrix_term model offset (group * 256 + column)))
  = prefix_blocks 32 8 (fun column -> matrix_term model offset (group * 256 + column));
    prefix_ext 32 (fun lane -> matrix_block model offset lane group)
      (fun lane -> prefix 8
        (fun j -> matrix_term model offset (group * 256 + (lane * 8 + j))))

inline_for_extraction noextract
let matrix_all_columns
  (#n : pos{n % 256 = 0})
  (offset : nat{offset + n <= weight_elements})
  (model : matrix_interpretation n)
  : Lemma (prefix 32 (matrix_lane model offset) ==
      rsum (S.init_ghost n (matrix_term model offset)))
  = let expand_lane (lane : nat)
      : Lemma (matrix_lane model offset lane ==
          prefix (n / 256) (fun group -> matrix_block model offset lane group))
      = prefix_ext (n / 256) (matrix_block model offset lane)
          (fun group -> matrix_block model offset lane group)
    in
    C.forall_intro expand_lane;
    prefix_ext 32 (matrix_lane model offset)
      (fun lane -> prefix (n / 256) (fun group -> matrix_block model offset lane group));
    prefix_swap 32 (n / 256) (fun lane group -> matrix_block model offset lane group);
    C.forall_intro (matrix_group_sum model offset);
    prefix_ext (n / 256)
      (fun group -> prefix 32 (fun lane -> matrix_block model offset lane group))
      (fun group -> prefix 256 (fun column -> matrix_term model offset (group * 256 + column)));
    prefix_blocks (n / 256) 256 (matrix_term model offset);
    prefix_init n (matrix_term model offset)

inline_for_extraction noextract
fn matrix_lane_run
  (#n : erased pos{n % 256 = 0 /\ n <= 8192})
  (weights : larray bf16 weight_elements)
  (values : larray f32 n)
  (count : sz{SZ.v count = n})
  (offset : sz{SZ.v offset + n <= weight_elements})
  (lane : sz{lane < 32})
  (chunks : bool)
  (#wv : erased (lseq bf16 weight_elements))
  (#xv : erased (lseq f32 n))
  (#fw #fx : perm)
  preserves gpu ** weights |-> Frac fw wv ** values |-> Frac fx xv
  returns result : f32
  ensures pure (forall (model : matrix_interpretation n).
    matrix_inputs wv xv model ==> result %~ matrix_lane model (SZ.v offset) (SZ.v lane))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut accumulator : f32 = zero;
  let mut group = 0sz;
  while (!group <^ count /^ 256sz)
    invariant exists* (g : sz{g <= n / 256}) (value : f32).
      group |-> g ** accumulator |-> value **
      pure (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
        value %~ prefix (SZ.v g) (matrix_block model (SZ.v offset) (SZ.v lane)))
    decreases (n / 256 - !group)
  {
    let g = !group;
    let previous = !accumulator;
    let mut chunk : f32 = (if chunks then (zero #f32 #_) else previous);
    let mut element = 0sz;
    while (!element <^ 8sz)
      invariant exists* (j : sz{j <= 8}) (value : f32).
        element |-> j ** chunk |-> value **
        pure (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
          value %~ matrix_partial model (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) chunks)
      decreases (8 - !element)
    {
      let j = !element;
      let column = g *^ 256sz +^ lane *^ 8sz +^ j;
      let w : f32 = Cast.fcast (weights.(offset +^ column));
      let x = values.(column);
      let prior = !chunk;
      assert pure (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
        w %~ S.index (fst model) (SZ.v offset + SZ.v column) /\
        x %~ S.index (snd model) (SZ.v column));
      let result = F32.fma_rn_ftz w x prior;
      lift_fma #(matrix_interpretation n) (matrix_inputs #n wv xv)
        (fun model -> S.index (fst model) (SZ.v offset + SZ.v column))
        (fun model -> S.index (snd model) (SZ.v column))
        (fun model -> matrix_partial #n model (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) chunks)
        w x prior result;
      C.forall_intro (matrix_partial_step #n (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) chunks);
      chunk := result;
      element := j +^ 1sz;
    };
    let last = !chunk;
    C.forall_intro (matrix_group_end #n (SZ.v offset) (SZ.v lane) (SZ.v g));
    if chunks {
      let result = F32.add_rn_ftz previous last;
      lift_add #(matrix_interpretation n) (matrix_inputs #n wv xv)
        (fun model -> prefix (SZ.v g) (matrix_block #n model (SZ.v offset) (SZ.v lane)))
        (fun model -> matrix_partial #n model (SZ.v offset) (SZ.v lane) (SZ.v g) 8 true)
        previous last result;
      accumulator := result;
      group := g +^ 1sz;
    } else {
      accumulator := last;
      group := g +^ 1sz;
    }
  };
  !accumulator
}

inline_for_extraction noextract
fn matrix_dot
  (#n : erased pos{n % 256 = 0 /\ n <= 8192})
  (weights : larray bf16 weight_elements)
  (values : larray f32 n)
  (count : sz{SZ.v count = n})
  (offset : sz{SZ.v offset + n <= weight_elements})
  (chunks : bool)
  (#wv : erased (lseq bf16 weight_elements))
  (#xv : erased (lseq f32 n))
  (#fw #fx : perm)
  preserves gpu ** weights |-> Frac fw wv ** values |-> Frac fx xv
  returns result : f32
  ensures pure (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
    result %~ rsum (S.init_ghost n (matrix_term model (SZ.v offset))))
{
  let mut partial = [| zero #f32 #_; 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i : sz{i <= 32}) (current : lseq f32 32).
      lane |-> i ** partial |-> current **
      pure (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index current j %~ matrix_lane model (SZ.v offset) j))
    decreases (32 - !lane)
  {
    let i = !lane;
    let value = matrix_lane_run weights values count offset i chunks;
    partial.(i) <- value;
    lane := i +^ 1sz;
  };
  with before. assert partial |-> before;
  let result = sum32 partial;
  lift_sum32 #(matrix_interpretation n) (matrix_inputs #n wv xv)
    (fun model -> matrix_lane #n model (SZ.v offset)) before result;
  C.forall_intro (matrix_all_columns #n (SZ.v offset));
  result
}

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let matrix_row_model
  (#n : pos) (#m : pos)
  (offset : nat{offset + m * n <= weight_elements})
  (row : nat{row < m})
  (model : matrix_interpretation n)
  : Lemma (
    rsum (S.init_ghost n (matrix_term model (offset + row * n))) ==
      S.index (linear (fst model) offset m n (snd model)) row)
  = FStar.Pervasives.norm_spec [delta_only [`%linear]; zeta; iota]
      (linear (fst model) offset m n (snd model));
    S.init_ghost_index_ m (fun r ->
      rsum (S.init_ghost n (fun column ->
        S.index (fst model) (offset + r * n + column) *. S.index (snd model) column))) row;
    S.lemma_eq_elim
      (S.init_ghost n (matrix_term model (offset + row * n)))
      (S.init_ghost n (fun column ->
        S.index (fst model) (offset + row * n + column) *. S.index (snd model) column))
#pop-options

[@@CPrologue "__device__"]
fn matrix_project
  (#n : erased pos{n % 256 = 0 /\ n <= 8192})
  (#m : erased pos{m <= 8192})
  (weights : larray bf16 weight_elements)
  (values : larray f32 n)
  (output : larray f32 m)
  (count : sz{SZ.v count = n})
  (rows : sz{SZ.v rows = m})
  (offset : sz{SZ.v offset + m * n <= weight_elements})
  (chunks : bool)
  (#wv : erased (lseq bf16 weight_elements))
  (#xv : erased (lseq f32 n))
  (#ov : erased (lseq f32 m))
  (#fw #fx : perm)
  preserves gpu ** weights |-> Frac fw wv ** values |-> Frac fx xv
  requires output |-> ov
  ensures exists* (after : lseq f32 m). output |-> after **
    pure (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
      after %~ linear (fst model) (SZ.v offset) m n (snd model))
{
  let mut row = 0sz;
  while (!row <^ rows)
    invariant exists* (i : sz{i <= m}) (current : lseq f32 m).
      row |-> i ** output |-> current **
      pure (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index current j %~ S.index (linear (fst model) (SZ.v offset) m n (snd model)) j))
    decreases (m - !row)
  {
    let i = !row;
    let result = matrix_dot weights values count (offset +^ i *^ count) chunks;
    C.forall_intro (matrix_row_model #n #m (SZ.v offset) (SZ.v i));
    output.(i) <- result;
    row := i +^ 1sz;
  }
}

inline_for_extraction noextract
let normalization_inputs
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (model : matrix_interpretation hidden_width)
  : prop = weights %~ fst model /\ hidden %~ snd model

[@@CPrologue "__device__"]
fn normalize_hidden
  (weights : larray bf16 weight_elements)
  (hidden : larray bf16 hidden_width)
  (output : larray f32 hidden_width)
  (offset : sz{SZ.v offset + hidden_width <= weight_elements})
  (#wv : erased (lseq bf16 weight_elements))
  (#hv : erased (lseq bf16 hidden_width))
  (#ov : erased (lseq f32 hidden_width))
  (#fw #fh : perm)
  preserves gpu ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires output |-> ov
  ensures exists* (after : lseq f32 hidden_width). output |-> after **
    pure (forall (model : matrix_interpretation hidden_width).
      normalization_inputs wv hv model ==>
      after %~ normalization (fst model) (SZ.v offset) (snd model))
{
  let scale = hidden_rstd hidden;
  assert pure ((one #f32) %~ 1.0R);
  let mut column = 0sz;
  while (!column <^ 1024sz)
    invariant exists* (i : sz{i <= 1024}) (current : lseq f32 hidden_width).
      column |-> i ** output |-> current **
      pure (forall (model : matrix_interpretation hidden_width).
        normalization_inputs wv hv model ==>
        (forall (j : nat{j < SZ.v i}). S.index current j %~
          S.index (normalization (fst model) (SZ.v offset) (snd model)) j))
    decreases (1024 - !column)
  {
    let i = !column;
    let h : f32 = Cast.fcast (hidden.(i));
    let w : f32 = Cast.fcast (weights.(offset +^ i));
    assert pure (forall (model : matrix_interpretation hidden_width).
      normalization_inputs wv hv model ==>
      h %~ S.index (snd model) (SZ.v i) /\
      w %~ S.index (fst model) (SZ.v offset + SZ.v i) /\
      scale %~ (rms_scale #hidden_width (snd model) <: real));
    let scaled = F32.mul_rn_ftz h scale;
    lift_mul #(matrix_interpretation hidden_width) (normalization_inputs wv hv)
      (fun model -> S.index (snd model) (SZ.v i))
      (fun model -> (rms_scale #hidden_width (snd model) <: real)) h scale scaled;
    let adjusted = F32.add_rn_ftz (one #f32) w;
    lift_add #(matrix_interpretation hidden_width) (normalization_inputs wv hv) (fun _ -> 1.0R)
      (fun model -> S.index (fst model) (SZ.v offset + SZ.v i))
      (one #f32) w adjusted;
    let value = F32.mul_rn_ftz scaled adjusted;
    lift_mul #(matrix_interpretation hidden_width) (normalization_inputs wv hv)
      (fun model -> S.index (snd model) (SZ.v i) *. rms_scale #hidden_width (snd model))
      (fun model -> 1.0R +. S.index (fst model) (SZ.v offset + SZ.v i))
      scaled adjusted value;
    let rounded : bf16 = Cast.fcast value;
    let stored : f32 = Cast.fcast rounded;
    output.(i) <- stored;
    column := i +^ 1sz;
  }
}

inline_for_extraction noextract
let rec prefix_scale (n : nat) (a : real) (f : nat -> GTot real)
  : Lemma (ensures prefix n (fun i -> a *. f i) == a *. prefix n f)
      (decreases n)
  = if n > 0 then prefix_scale (n - 1) a f

inline_for_extraction noextract
let head_square (values : real_vector head_width) (i : nat) : GTot real =
  if i < head_width then S.index values i *. S.index values i else 0.0R

inline_for_extraction noextract
let head_square_lane (values : real_vector head_width) (lane group : nat) : GTot real =
  head_square values (group * 32 + lane)

inline_for_extraction noextract
let head_square_lane_step (lane group : nat) (model : real_vector head_width)
  : Lemma (prefix (group + 1) (head_square_lane model lane) ==
      prefix group (head_square_lane model lane) +. head_square model (group * 32 + lane))
  = prefix_step group (head_square_lane model lane)

inline_for_extraction noextract
let rec head_squares_prefix (values : real_vector head_width)
  (count : nat{count <= head_width})
  : Lemma (ensures prefix count (head_square values) == sum_squares values count)
      (decreases count)
  = if count > 0 then head_squares_prefix values (count - 1)

inline_for_extraction noextract
let square_strided_model (values : real_vector head_width)
  : Lemma (prefix 32 (fun lane ->
      prefix 8 (head_square_lane values lane)) ==
      sum_squares values head_width)
  = let expand_lane (lane : nat)
      : Lemma (prefix 8 (head_square_lane values lane) ==
          prefix 8 (fun group -> head_square values (group * 32 + lane)))
      = prefix_ext 8 (head_square_lane values lane)
          (fun group -> head_square values (group * 32 + lane))
    in
    C.forall_intro expand_lane;
    prefix_ext 32 (fun lane -> prefix 8 (head_square_lane values lane))
      (fun lane -> prefix 8 (fun group -> head_square values (group * 32 + lane)));
    prefix_swap 32 8 (fun lane group -> head_square values (group * 32 + lane));
    prefix_blocks 8 32 (head_square values);
    head_squares_prefix values head_width

inline_for_extraction noextract
fn square_strided
  (input : larray f32 head_width)
  (#before : erased (lseq f32 head_width))
  (#fraction : perm)
  preserves gpu ** input |-> Frac fraction before
  returns result : f32
  ensures pure (forall (model : real_vector head_width).
    before %~ model ==> result %~ (sum_squares model head_width <: real))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut partial = [| zero #f32 #_; 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i : sz{i <= 32}) (values : lseq f32 32).
      lane |-> i ** partial |-> values **
      pure (forall (model : real_vector head_width). before %~ model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~ prefix 8 (head_square_lane model j)))
    decreases (32 - !lane)
  {
    let i = !lane;
    let mut group = 0sz;
    let mut accumulator : f32 = zero;
    while (!group <^ 8sz)
      invariant exists* (g : sz{g <= 8}) (value : f32).
        group |-> g ** accumulator |-> value **
        pure (forall (model : real_vector head_width). before %~ model ==>
          value %~ prefix (SZ.v g) (head_square_lane model (SZ.v i)))
      decreases (8 - !group)
    {
      let g = !group;
      let j = g *^ 32sz +^ i;
      let x = input.(j);
      let previous = !accumulator;
      let value = F32.fma_rn_ftz x x previous;
      lift_fma (fun (model : real_vector head_width) -> before %~ model)
        (fun model -> S.index model (SZ.v j))
        (fun model -> S.index model (SZ.v j))
        (fun model -> prefix (SZ.v g) (head_square_lane model (SZ.v i)))
        x x previous value;
      C.forall_intro (head_square_lane_step (SZ.v i) (SZ.v g));
      accumulator := value;
      group := g +^ 1sz;
    };
    partial.(i) <- !accumulator;
    lane := i +^ 1sz;
  };
  with values. assert partial |-> values;
  let result = sum32 partial;
  lift_sum32 (fun (model : real_vector head_width) -> before %~ model)
    (fun model lane -> prefix 8 (head_square_lane model lane)) values result;
  C.forall_intro square_strided_model;
  result
}

inline_for_extraction noextract
let replace_slice
  (#a : Type) (#n #m : nat)
  (before : lseq a n) (offset : nat{offset + m <= n}) (patch : lseq a m)
  : GTot (lseq a n)
  = S.init_ghost n (fun i ->
      if offset <= i /\ i < offset + m then S.index patch (i - offset) else S.index before i)

inline_for_extraction noextract
let replace_prefix_relation
  (#n #m : nat)
  (before : lseq f32 n) (model : real_vector n)
  (offset : nat{offset + m <= n}) (patch : lseq f32 m)
  : Lemma
      (requires
        (forall (i : nat{i < offset}). S.index before i %~ S.index model i) /\
        patch %~ vector_slice model offset m)
      (ensures forall (i : nat{i < offset + m}).
        S.index (replace_slice before offset patch) i %~ S.index model i)
  = let related (i : nat{i < offset + m})
      : Lemma (S.index (replace_slice before offset patch) i %~ S.index model i)
      = if i < offset then () else ()
    in C.forall_intro related

inline_for_extraction noextract
let lift_replace_prefix
  (#p : Type) (#n #m : nat)
  (premise : p -> prop) (target : p -> GTot (real_vector n))
  (before : lseq f32 n) (offset : nat{offset + m <= n}) (patch : lseq f32 m)
  : Lemma
      (requires forall (model : p). premise model ==>
        (forall (i : nat{i < offset}). S.index before i %~ S.index (target model) i) /\
        patch %~ vector_slice (target model) offset m)
      (ensures forall (model : p). premise model ==>
        (forall (i : nat{i < offset + m}).
          S.index (replace_slice before offset patch) i %~ S.index (target model) i))
  = let related (model : p{premise model})
      : Lemma (forall (i : nat{i < offset + m}).
          S.index (replace_slice before offset patch) i %~ S.index (target model) i)
      = replace_prefix_relation #n #m before (target model) offset patch;
        ()
    in C.forall_intro related

inline_for_extraction noextract
let f32_slice_relation
  (#n : nat) (before : lseq f32 n) (model : real_vector n)
  (offset count : nat{offset + count <= n})
  : Lemma
      (requires before %~ model)
      (ensures S.slice before offset (offset + count) %~ vector_slice model offset count)
  = ()

inline_for_extraction noextract
fn copy_f32_slice
  (#n #m : pos)
  (source : larray f32 n) (output : larray f32 m)
  (offset : sz{SZ.v offset + m <= n}) (count : sz{SZ.v count = m})
  (#before : erased (lseq f32 n)) (#old : erased (lseq f32 m))
  (#fraction : perm)
  preserves gpu ** source |-> Frac fraction before
  requires output |-> old
  ensures output |-> S.slice before (SZ.v offset) (SZ.v offset + m)
{
  let mut index = 0sz;
  while (!index <^ count)
    invariant exists* (i : sz{i <= m}) (values : lseq f32 m).
      index |-> i ** output |-> values **
      pure (forall (j : nat{j < SZ.v i}). S.index values j == S.index before (SZ.v offset + j))
    decreases (m - !index)
  {
    let i = !index;
    output.(i) <- source.(offset +^ i);
    index := i +^ 1sz;
  };
  with values. assert output |-> values;
  assert pure (S.equal values (S.slice before (SZ.v offset) (SZ.v offset + m)));
}

inline_for_extraction noextract
fn write_f32_slice
  (#n #m : pos)
  (output : larray f32 n) (patch : larray f32 m)
  (offset : sz{SZ.v offset + m <= n}) (count : sz{SZ.v count = m})
  (#before : erased (lseq f32 n)) (#values : erased (lseq f32 m))
  (#fraction : perm)
  preserves gpu ** patch |-> Frac fraction values
  requires output |-> before
  ensures output |-> replace_slice before (SZ.v offset) values
{
  let mut index = 0sz;
  while (!index <^ count)
    invariant exists* (i : sz{i <= m}) (current : lseq f32 n).
      index |-> i ** output |-> current **
      pure (forall (j : nat{j < n}).
        S.index current j ==
          (if SZ.v offset <= j /\ j < SZ.v offset + SZ.v i
           then S.index values (j - SZ.v offset) else S.index before j))
    decreases (m - !index)
  {
    let i = !index;
    output.(offset +^ i) <- patch.(i);
    index := i +^ 1sz;
  };
  with current. assert output |-> current;
  assert pure (S.equal current (replace_slice before (SZ.v offset) values));
}

inline_for_extraction noextract
let compose_relations
  (#p : Type) (before middle after : p -> prop)
  : Lemma
      (requires (forall (model : p). before model ==> middle model) /\
        (forall (model : p). middle model ==> after model))
      (ensures forall (model : p). before model ==> after model)
  = let related (model : p{before model}) : Lemma (after model) = ()
    in C.forall_intro related

inline_for_extraction noextract
let map_relations
  (#p #q : Type)
  (before : p -> prop) (middle after : q -> prop) (map_model : p -> GTot q)
  : Lemma
      (requires (forall (model : p). before model ==> middle (map_model model)) /\
        (forall (model : q). middle model ==> after model))
      (ensures forall (model : p). before model ==> after (map_model model))
  = let related (model : p{before model}) : Lemma (after (map_model model))
      = assert (middle (map_model model))
    in C.forall_intro related

inline_for_extraction noextract
fn copy_hidden
  (input output : larray bf16 hidden_width)
  (#before : erased (lseq bf16 hidden_width))
  (#fraction : perm)
  preserves gpu ** input |-> Frac fraction before
  requires A.pts_to_uninit output hidden_width
  ensures output |-> before
{
  unfold A.pts_to_uninit output hidden_width;
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i : sz{i <= 1024}) (values : seq (option bf16)).
      index |-> i ** A.pts_to_mask output values (fun _ -> True) **
      pure (S.length values == hidden_width /\
        (forall (j : nat{j < SZ.v i}). S.index values j == Some (S.index before j)))
    decreases (1024 - !index)
  {
    let i = !index;
    let value = input.(i);
    A.mask_write output i value;
    index := i +^ 1sz;
  };
  A.from_mask output;
  with values. assert output |-> values;
  assert pure (S.equal values before);
}

inline_for_extraction noextract
let residual_inputs
  (hidden : lseq bf16 hidden_width) (projected : lseq f32 hidden_width)
  (model : real_vector hidden_width & real_vector hidden_width)
  : prop = hidden %~ fst model /\ projected %~ snd model

inline_for_extraction noextract
fn add_residual
  (hidden : larray bf16 hidden_width) (projected : larray f32 hidden_width)
  (#before : erased (lseq bf16 hidden_width))
  (#pv : erased (lseq f32 hidden_width))
  (#fraction : perm)
  preserves gpu ** projected |-> Frac fraction pv
  requires hidden |-> before
  ensures exists* (after : lseq bf16 hidden_width). hidden |-> after **
    pure (forall (model : real_vector hidden_width & real_vector hidden_width).
      residual_inputs before pv model ==> after %~ residual_vector (fst model) (snd model))
{
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i : sz{i <= 1024}) (values : lseq bf16 hidden_width).
      index |-> i ** hidden |-> values **
      pure (forall (j : nat{SZ.v i <= j /\ j < hidden_width}).
        S.index values j == S.index before j) **
      pure (forall (model : real_vector hidden_width & real_vector hidden_width).
        residual_inputs before pv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~ S.index (residual_vector (fst model) (snd model)) j))
    decreases (1024 - !index)
  {
    let i = !index;
    let h : f32 = Cast.fcast (hidden.(i));
    let p = projected.(i);
    let sum = F32.add_rn_ftz p h;
    lift_add #(real_vector hidden_width & real_vector hidden_width) (residual_inputs before pv)
      (fun model -> S.index (snd model) (SZ.v i))
      (fun model -> S.index (fst model) (SZ.v i)) p h sum;
    let result : bf16 = Cast.fcast sum;
    hidden.(i) <- result;
    index := i +^ 1sz;
  }
}

inline_for_extraction noextract
let mlp_inputs
  (gate up : lseq f32 mlp_width) (model : real_vector mlp_width & real_vector mlp_width)
  : prop = gate %~ fst model /\ up %~ snd model

inline_for_extraction noextract
fn activate_mlp
  (gate up : larray f32 mlp_width)
  (#before #uv : erased (lseq f32 mlp_width))
  (#fraction : perm)
  preserves gpu ** up |-> Frac fraction uv
  requires gate |-> before
  ensures exists* (after : lseq f32 mlp_width). gate |-> after **
    pure (forall (model : real_vector mlp_width & real_vector mlp_width).
      mlp_inputs before uv model ==> after %~ mlp_vector (fst model) (snd model))
{
  let mut index = 0sz;
  while (!index <^ 3584sz)
    invariant exists* (i : sz{i <= mlp_width}) (values : lseq f32 mlp_width).
      index |-> i ** gate |-> values **
      pure (forall (j : nat{SZ.v i <= j /\ j < mlp_width}).
        S.index values j == S.index before j) **
      pure (forall (model : real_vector mlp_width & real_vector mlp_width).
        mlp_inputs before uv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~ S.index (mlp_vector (fst model) (snd model)) j))
    decreases (mlp_width - !index)
  {
    let i = !index;
    let g = gate.(i);
    let u = up.(i);
    let activated = silu32 g;
    assert pure (forall (model : real_vector mlp_width & real_vector mlp_width).
      mlp_inputs before uv model ==> activated %~ silu (S.index (fst model) (SZ.v i)));
    let result = F32.mul_rn_ftz activated u;
    lift_mul #(real_vector mlp_width & real_vector mlp_width) (mlp_inputs before uv)
      (fun model -> silu (S.index (fst model) (SZ.v i)))
      (fun model -> S.index (snd model) (SZ.v i)) activated u result;
    gate.(i) <- result;
    index := i +^ 1sz;
  }
}

inline_for_extraction noextract
let lift_matrix
  (#p : Type) (#n #m : pos)
  (premise : p -> prop)
  (wm : p -> GTot (real_vector weight_elements)) (xm : p -> GTot (real_vector n))
  (offset : nat{offset + m * n <= weight_elements})
  (wv : lseq bf16 weight_elements) (xv : lseq f32 n) (out : lseq f32 m)
  : Lemma
      (requires
        (forall (v : p). premise v ==> wv %~ wm v /\ xv %~ xm v) /\
        (forall (model : matrix_interpretation n). matrix_inputs wv xv model ==>
          out %~ linear (fst model) offset m n (snd model)))
      (ensures forall (v : p). premise v ==> out %~ linear (wm v) offset m n (xm v))
  = let related (v : p{premise v})
      : Lemma (out %~ linear (wm v) offset m n (xm v))
      = assert (matrix_inputs #n wv xv (wm v, xm v))
    in C.forall_intro related

inline_for_extraction noextract
let lift_normalization
  (#p : Type) (premise : p -> prop)
  (wm : p -> GTot (real_vector weight_elements)) (hm : p -> GTot (real_vector hidden_width))
  (offset : nat{offset + hidden_width <= weight_elements})
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width) (out : lseq f32 hidden_width)
  : Lemma
      (requires
        (forall (v : p). premise v ==> wv %~ wm v /\ hv %~ hm v) /\
        (forall (model : matrix_interpretation hidden_width). normalization_inputs wv hv model ==>
          out %~ normalization (fst model) offset (snd model)))
      (ensures forall (v : p). premise v ==> out %~ normalization (wm v) offset (hm v))
  = let related (v : p{premise v})
      : Lemma (out %~ normalization (wm v) offset (hm v))
      = assert (normalization_inputs wv hv (wm v, hm v))
    in C.forall_intro related

inline_for_extraction noextract
let lift_residual
  (#p : Type) (premise : p -> prop)
  (hm pm : p -> GTot (real_vector hidden_width))
  (hv : lseq bf16 hidden_width) (pv : lseq f32 hidden_width) (out : lseq bf16 hidden_width)
  : Lemma
      (requires
        (forall (v : p). premise v ==> hv %~ hm v /\ pv %~ pm v) /\
        (forall (model : real_vector hidden_width & real_vector hidden_width).
          residual_inputs hv pv model ==> out %~ residual_vector (fst model) (snd model)))
      (ensures forall (v : p). premise v ==> out %~ residual_vector (hm v) (pm v))
  = let related (v : p{premise v})
      : Lemma (out %~ residual_vector (hm v) (pm v))
      = assert (residual_inputs hv pv (hm v, pm v))
    in C.forall_intro related

inline_for_extraction noextract
let lift_mlp
  (#p : Type) (premise : p -> prop)
  (gm um : p -> GTot (real_vector mlp_width))
  (gv uv out : lseq f32 mlp_width)
  : Lemma
      (requires
        (forall (v : p). premise v ==> gv %~ gm v /\ uv %~ um v) /\
        (forall (model : real_vector mlp_width & real_vector mlp_width).
          mlp_inputs gv uv model ==> out %~ mlp_vector (fst model) (snd model)))
      (ensures forall (v : p). premise v ==> out %~ mlp_vector (gm v) (um v))
  = let related (v : p{premise v})
      : Lemma (out %~ mlp_vector (gm v) (um v))
      = assert (mlp_inputs gv uv (gm v, um v))
    in C.forall_intro related

inline_for_extraction noextract
let ratio32 (numerator : i64) (denominator : i64{I64.v denominator <> 0}) : Pure f32
  (requires True)
  (ensures fun result -> result %~ (R.of_int (I64.v numerator) /. R.of_int (I64.v denominator)))
  = let n : f32 = of_int numerator in
    let d : f32 = of_int denominator in
    let _ = of_int_approx #f32 numerator in
    let _ = of_int_approx #f32 denominator in
    div_approx n d (R.of_int (I64.v numerator)) (R.of_int (I64.v denominator));
    div n d

inline_for_extraction noextract
fn normalize32 (x scale bias : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (xr sr wr : real).
    x %~ xr /\ scale %~ sr /\ bias %~ wr ==>
    result %~ (xr *. sr *. (1.0R +. wr)))
{
  let scaled = F32.mul_rn_ftz x scale;
  let gain = F32.add_rn_ftz (one #f32) bias;
  F32.mul_rn_ftz scaled gain
}

inline_for_extraction noextract
let rope_finish (value partner cosine term result : f32) (phase : real) (lower : bool)
  : Lemma
      (requires cosine %~ Trig.cos phase /\
        (forall (yr : real). partner %~ yr ==>
          term %~ (if lower then 0.0R -. yr *. Trig.sin phase else yr *. Trig.sin phase)) /\
        (forall (xr yr zr : real).
          value %~ xr /\ cosine %~ yr /\ term %~ zr ==> result %~ (xr *. yr +. zr)))
      (ensures forall (xr yr : real). value %~ xr /\ partner %~ yr ==>
        result %~ (if lower then rotary_lower xr yr phase else rotary_upper xr yr phase))
  = let related (xr : real{value %~ xr}) (yr : real{partner %~ yr})
      : Lemma (result %~ (if lower then rotary_lower xr yr phase else rotary_upper xr yr phase))
      = assert (term %~ (if lower then 0.0R -. yr *. Trig.sin phase else yr *. Trig.sin phase));
        assert (result %~ (xr *. Trig.cos phase +.
          (if lower then 0.0R -. yr *. Trig.sin phase else yr *. Trig.sin phase)))
    in C.forall_intro_2 related

inline_for_extraction noextract
let rope_upper_finish (value partner cosine sine direct result:f32) (phase:real)
  : Lemma
      (requires cosine %~ Trig.cos phase /\ sine %~ Trig.sin phase /\
        (forall (xr:real). value %~ xr ==> direct %~ (xr *. Trig.cos phase)) /\
        (forall (yr sr zr:real).
          partner %~ yr /\ sine %~ sr /\ direct %~ zr ==> result %~ (yr *. sr +. zr)))
      (ensures forall (xr yr:real). value %~ xr /\ partner %~ yr ==>
        result %~ rotary_upper xr yr phase) =
  let related (xr:real{value %~ xr}) (yr:real{partner %~ yr})
    : Lemma (result %~ rotary_upper xr yr phase) =
    assert (direct %~ (xr *. Trig.cos phase));
    assert (result %~ (yr *. Trig.sin phase +. xr *. Trig.cos phase))
  in C.forall_intro_2 related

(* The authorized native trigonometry is used in the source's fused schedule. *)
[@@CPrologue "__device__"]
fn rope32 (value partner : f32) (index : sz{index < 64})
  (position : i64{valid_position (I64.v position)})
  preserves gpu
  returns result : f32
  ensures pure (forall (xr yr : real).
    value %~ xr /\ partner %~ yr ==>
    result %~ (if SZ.v index < 32
      then rotary_lower xr yr (rotary_phase (I64.v position) (SZ.v index))
      else rotary_upper xr yr (rotary_phase (I64.v position) (SZ.v index))))
{
  let remainder = index %^ 32sz;
  let integer = IC.uint32_to_int64 (SZ.sizet_to_u32 remainder);
  let coordinate : f32 = of_int integer;
  let _ = of_int_approx #f32 integer;
  let factor = ratio32 1L 32L;
  let exponent = mul coordinate factor;
  assert pure (exponent %~ (R.of_int (SZ.v index % 32) /. 32.0R));
  let base:f32 = of_literal "23.2534961700439453125";
  RoPE.coefficient_bounds ();
  RoPE.literal_cell_check ();
  F32.literal_interval_approx "23.2534961700439453125"
    390128936 16777216 390128937 16777216 12191529 (-19) (RoPE.coefficient ());
  assert pure (base %~ rotary_log2_base);
  let product = F32.mul_rn_ftz exponent base;
  assert pure (product %~ ((R.of_int (SZ.v index % 32) /. 32.0R) *. rotary_log2_base));
  let power = negate32 product;
  let inverse = F32.exp2_approx_ftz power;
  let absolute_position : f32 = of_int position;
  let _ = of_int_approx #f32 position;
  let phase = F32.mul_rn_ftz absolute_position inverse;
  assert pure (phase %~ rotary_phase (I64.v position) (SZ.v index));
  let cosine = Fast.cos phase;
  let sine = Fast.sin phase;
  assert pure (cosine %~ Trig.cos (rotary_phase (I64.v position) (SZ.v index)));
  assert pure (sine %~ Trig.sin (rotary_phase (I64.v position) (SZ.v index)));
  if (index <^ 32sz) {
    let cross = F32.mul_rn_ftz partner sine;
    assert pure (forall (yr : real). partner %~ yr ==>
      cross %~ (yr *. Trig.sin (rotary_phase (I64.v position) (SZ.v index))));
    let negative_cross = negative cross;
    assert pure (forall (yr : real). partner %~ yr ==>
      negative_cross %~ (0.0R -. yr *. Trig.sin (rotary_phase (I64.v position) (SZ.v index))));
    let result = F32.fma_rn_ftz value cosine negative_cross;
    rope_finish value partner cosine negative_cross result
      (rotary_phase (I64.v position) (SZ.v index)) true;
    result
  } else {
    let direct = F32.mul_rn_ftz value cosine;
    let result = F32.fma_rn_ftz partner sine direct;
    rope_upper_finish value partner cosine sine direct result
      (rotary_phase (I64.v position) (SZ.v index));
    result
  }
}

inline_for_extraction noextract
let head_rms_finish (before : lseq f32 head_width) (argument result : f32)
  : Lemma
      (requires
        (forall (model : real_vector head_width). before %~ model ==>
          argument %~ (sum_squares model head_width /. 256.0R +. epsilon <: real)) /\
        (forall (r : Sqrt.rpos). argument %~ (r <: real) ==>
          result %~ (1.0R /. Sqrt.sqrt r)))
      (ensures forall (model : real_vector head_width).
        before %~ model ==> result %~ (rms_scale model <: real))
  = let related (model : real_vector head_width{before %~ model})
      : Lemma (result %~ (rms_scale model <: real)) = ()
    in C.forall_intro related

inline_for_extraction noextract
fn head_rstd (input : larray f32 head_width)
  (#before : erased (lseq f32 head_width)) (#fraction : perm)
  preserves gpu ** input |-> Frac fraction before
  returns result : f32
  ensures pure (forall (model : real_vector head_width).
    before %~ model ==> result %~ (rms_scale model <: real))
{
  let squares = square_strided input;
  let factor = ratio32 1L 256L;
  let eps = epsilon32 ();
  let argument = F32.fma_rn_ftz squares factor eps;
  lift_fma (fun (model : real_vector head_width) -> before %~ model)
    (fun model -> (sum_squares model head_width <: real))
    (fun _ -> 1.0R /. 256.0R) (fun _ -> (epsilon <: real))
    squares factor eps argument;
  let result = F32.rsqrt_approx_ftz argument;
  head_rms_finish before argument result;
  result
}

inline_for_extraction noextract
let transformed_head (query : bool) (position : position_index)
  (model : matrix_interpretation head_width) : GTot (real_vector head_width)
  = if query then query_transform (fst model) position (snd model)
    else key_transform (fst model) position (snd model)

inline_for_extraction noextract
let norm_offset (query : bool) : nat =
  if query then query_norm_offset else key_norm_offset

inline_for_extraction noextract
let normalized_cell (query : bool) (model : matrix_interpretation head_width)
  (i : nat{i < head_width}) : GTot real
  = S.index (snd model) i *. rms_scale (snd model) *.
      (1.0R +. S.index (fst model) (norm_offset query + i))

inline_for_extraction noextract
let rotary_partner (i : nat{i < 64}) : natlt head_width =
  if i < 32 then i + 32 else i - 32

inline_for_extraction noextract
let partner_input (query : bool) (position : position_index)
  (i : nat{i < 64}) (model : matrix_interpretation head_width) : GTot real =
  S.index (snd model) (rotary_partner i)

inline_for_extraction noextract
let partner_normalized (query : bool) (position : position_index)
  (i : nat{i < 64}) (model : matrix_interpretation head_width) : GTot real =
  partner_input query position i model *. rms_scale (snd model) *.
    (1.0R +. S.index (fst model) (norm_offset query + rotary_partner i))

#push-options "--fuel 6 --z3rlimit 60"
inline_for_extraction noextract
let transformed_cell (query : bool) (position : position_index)
  (i : nat{i < head_width}) (model : matrix_interpretation head_width)
  : Lemma (S.index (transformed_head query position model) i ==
      (if i < 32 then
         rotary_lower (normalized_cell query model i) (partner_normalized query position i model)
           (rotary_phase position i)
       else if i < 64 then
         rotary_upper (normalized_cell query model i) (partner_normalized query position i model)
           (rotary_phase position i)
       else normalized_cell query model i))
  = ()
#pop-options

inline_for_extraction noextract
fn read_rotary_partner (query : bool)
  (raw output : larray f32 head_width) (i : sz{i < 64})
  (#rv #ov : erased (lseq f32 head_width)) (#fr #fo : perm)
  preserves gpu ** raw |-> Frac fr rv ** output |-> Frac fo ov
  returns result : f32
  ensures pure (result == S.index rv (rotary_partner (SZ.v i)))
{
  let partner = (if (i <^ 32sz) then i +^ 32sz else i -^ 32sz);
  raw.(partner)
}

inline_for_extraction noextract
fn transform_coordinate (query : bool)
  (weights : larray bf16 weight_elements) (raw output : larray f32 head_width)
  (position : i64{valid_position (I64.v position)})
  (i : sz{i < head_width}) (scale : f32)
  (#wv : erased (lseq bf16 weight_elements))
  (#rv #ov : erased (lseq f32 head_width)) (#fw #fr #fo : perm)
  preserves gpu ** weights |-> Frac fw wv ** raw |-> Frac fr rv ** output |-> Frac fo ov
  requires pure (forall (model : matrix_interpretation head_width). matrix_inputs wv rv model ==>
    scale %~ (rms_scale (snd model) <: real) /\
    (forall (j : nat{j < SZ.v i}).
      S.index ov j %~ S.index (transformed_head query (I64.v position) model) j))
  returns result : f32
  ensures pure (forall (model : matrix_interpretation head_width). matrix_inputs wv rv model ==>
    result %~ S.index (transformed_head query (I64.v position) model) (SZ.v i))
{
  let offset = (if query then SZ.uint32_to_sizet 16254976ul else SZ.uint32_to_sizet 16255232ul);
  let x = raw.(i);
  let bias : f32 = Cast.fcast (weights.(offset +^ i));
  let normalized = normalize32 x scale bias;
  assert pure (forall (model : matrix_interpretation head_width).
    matrix_inputs wv rv model ==> normalized %~ normalized_cell query model (SZ.v i));
  C.forall_intro (transformed_cell query (I64.v position) (SZ.v i));
  if (i <^ 64sz) {
    let partner = (if (i <^ 32sz) then i +^ 32sz else i -^ 32sz);
    let reloaded = read_rotary_partner query raw output i;
    assert pure (forall (model : matrix_interpretation head_width).
      matrix_inputs wv rv model ==>
      reloaded %~ partner_input query (I64.v position) (SZ.v i) model);
    let partner_bias : f32 = Cast.fcast (weights.(offset +^ partner));
    let pv = normalize32 reloaded scale partner_bias;
    assert pure (forall (model : matrix_interpretation head_width).
      matrix_inputs wv rv model ==>
      pv %~ partner_normalized query (I64.v position) (SZ.v i) model);
    let result = rope32 normalized pv i position;
    result
  } else {
    normalized
  }
}

inline_for_extraction noextract
fn transform_head (query : bool)
  (weights : larray bf16 weight_elements)
  (raw output : larray f32 head_width)
  (position : i64{valid_position (I64.v position)})
  (#wv : erased (lseq bf16 weight_elements))
  (#rv #old : erased (lseq f32 head_width)) (#fw #fr : perm)
  preserves gpu ** weights |-> Frac fw wv ** raw |-> Frac fr rv
  requires output |-> old
  ensures exists* (after : lseq f32 head_width). output |-> after **
    pure (forall (model : matrix_interpretation head_width).
      matrix_inputs wv rv model ==> after %~ transformed_head query (I64.v position) model)
{
  let scale = head_rstd raw;
  let mut column = 0sz;
  while (!column <^ 256sz)
    invariant exists* (i : sz{i <= 256}) (current : lseq f32 head_width).
      column |-> i ** output |-> current **
      pure (forall (model : matrix_interpretation head_width). matrix_inputs wv rv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index current j %~ S.index (transformed_head query (I64.v position) model) j))
    decreases (256 - !column)
  {
    let i = !column;
    let result = transform_coordinate query weights raw output position i scale;
    output.(i) <- result;
    column := i +^ 1sz;
  }
}

inline_for_extraction noextract
let transform_count (query : bool) : pos = if query then query_heads else kv_heads
inline_for_extraction noextract
let transform_stride (query : bool) : pos = if query then 2 * head_width else head_width
inline_for_extraction noextract
let transform_inputs (query : bool) : pos = transform_count query * transform_stride query
inline_for_extraction noextract
let transform_outputs (query : bool) : pos = transform_count query * head_width

inline_for_extraction noextract
let transformed_heads (query : bool) (position : position_index)
  (model : matrix_interpretation (transform_inputs query))
  : GTot (real_vector (transform_outputs query))
  = S.init_ghost (transform_outputs query) (fun i ->
      S.index (transformed_head query position
        (fst model, vector_slice (snd model) ((i / head_width) * transform_stride query) head_width))
        (i % head_width))

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let transformed_head_slice (query : bool) (position : position_index)
  (head : nat{head < transform_count query})
  (model : matrix_interpretation (transform_inputs query))
  : Lemma (vector_slice (transformed_heads query position model) (head * head_width) head_width ==
      transformed_head query position
        (fst model, vector_slice (snd model) (head * transform_stride query) head_width))
  = S.lemma_eq_elim
      (vector_slice (transformed_heads query position model) (head * head_width) head_width)
      (transformed_head query position
        (fst model, vector_slice (snd model) (head * transform_stride query) head_width))
#pop-options

inline_for_extraction noextract
let transform_head_lift (query : bool) (position : position_index)
  (head : nat{head < transform_count query})
  (weights : lseq bf16 weight_elements) (raw : lseq f32 (transform_inputs query))
  (local_raw result : lseq f32 head_width)
  : Lemma
      (requires local_raw == S.slice raw (head * transform_stride query)
          (head * transform_stride query + head_width) /\
        (forall (model : matrix_interpretation head_width). matrix_inputs weights local_raw model ==>
          result %~ transformed_head query position model))
      (ensures forall (model : matrix_interpretation (transform_inputs query)).
        matrix_inputs weights raw model ==>
        result %~ vector_slice (transformed_heads query position model) (head * head_width) head_width)
  = let related (model : matrix_interpretation (transform_inputs query){matrix_inputs weights raw model})
      : Lemma (result %~
          vector_slice (transformed_heads query position model) (head * head_width) head_width)
      = f32_slice_relation raw (snd model) (head * transform_stride query) head_width;
        transformed_head_slice query position head model;
        assert (matrix_inputs weights local_raw
          (fst model, vector_slice (snd model) (head * transform_stride query) head_width))
    in C.forall_intro related

[@@CPrologue "__device__"]
fn transform_all_heads (query : bool)
  (weights : larray bf16 weight_elements)
  (raw : larray f32 (transform_inputs query))
  (output : larray f32 (transform_outputs query))
  (position : i64{valid_position (I64.v position)})
  (#wv : erased (lseq bf16 weight_elements))
  (#rv : erased (lseq f32 (transform_inputs query)))
  (#old : erased (lseq f32 (transform_outputs query))) (#fw #fr : perm)
  preserves gpu ** weights |-> Frac fw wv ** raw |-> Frac fr rv
  requires output |-> old
  ensures exists* (after : lseq f32 (transform_outputs query)). output |-> after **
    pure (forall (model : matrix_interpretation (transform_inputs query)).
      matrix_inputs wv rv model ==> after %~ transformed_heads query (I64.v position) model)
{
  let mut local_raw = [| zero #f32 #_; 256sz |];
  let mut transformed = [| zero #f32 #_; 256sz |];
  let limit = (if query then 8sz else 2sz);
  let stride = (if query then 512sz else 256sz);
  let mut head = 0sz;
  while (!head <^ limit)
    invariant exists* (h : sz{h <= transform_count query})
      (current : lseq f32 (transform_outputs query)).
      head |-> h ** output |-> current **
      pure (forall (model : matrix_interpretation (transform_inputs query)). matrix_inputs wv rv model ==>
        (forall (i : nat{i < SZ.v h * head_width}).
          S.index current i %~ S.index (transformed_heads query (I64.v position) model) i))
    invariant exists* (values : lseq f32 head_width). local_raw |-> values
    invariant exists* (values : lseq f32 head_width). transformed |-> values
    decreases (transform_count query - !head)
  {
    let h = !head;
    copy_f32_slice raw local_raw (h *^ stride) 256sz;
    with lr. assert local_raw |-> lr;
    transform_head query weights local_raw transformed position;
    with tr. assert transformed |-> tr;
    with current. assert output |-> current;
    transform_head_lift query (I64.v position) (SZ.v h) wv rv lr tr;
    lift_replace_prefix
      (matrix_inputs #(transform_inputs query) wv rv)
      (transformed_heads query (I64.v position))
      current (SZ.v h * head_width) tr;
    write_f32_slice output transformed (h *^ 256sz) 256sz;
    head := h +^ 1sz;
  }
}

inline_for_extraction noextract
let physical_cache (position : position_index)
  (previous : lseq bf16 (previous_elements position)) (current : lseq f32 kv_width)
  : GTot (lseq bf16 cache_elements)
  = S.init_ghost cache_elements (fun i ->
      if (i % cache_head_elements) / head_width = position then
        Cast.fcast (S.index current ((i / cache_head_elements) * head_width + i % head_width))
      else if position = 0 then (zero <: bf16) else S.index previous i)

inline_for_extraction noextract
let cache_write_index (position : position_index) (coordinate : nat{coordinate < kv_width})
  : GTot (natlt cache_elements) =
  (coordinate / head_width) * cache_head_elements + position * head_width + coordinate % head_width

inline_for_extraction noextract
let physical_cache_prefix (position : position_index)
  (previous : lseq bf16 (previous_elements position)) (current : lseq f32 kv_width)
  (count : nat{count <= kv_width}) : GTot (lseq bf16 cache_elements) =
  S.init_ghost cache_elements (fun i ->
    let coordinate = (i / cache_head_elements) * head_width + i % head_width in
    if (i % cache_head_elements) / head_width = position /\ coordinate < count
    then Cast.fcast (S.index current coordinate)
    else if position = 0 then (zero <: bf16) else S.index previous i)

inline_for_extraction noextract
let cache_write_index_coordinates (position : position_index) (coordinate : nat{coordinate < kv_width})
  : Lemma (
      let i = cache_write_index position coordinate in
      (i % cache_head_elements) / head_width == position /\
      (i / cache_head_elements) * head_width + i % head_width == coordinate)
  = ()

inline_for_extraction noextract
let cache_current_index (position : position_index)
  (i : nat{i < cache_elements /\ (i % cache_head_elements) / head_width = position})
  : Lemma (cache_write_index position
      ((i / cache_head_elements) * head_width + i % head_width) == i)
  = ()

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let physical_cache_prefix_step (position : position_index)
  (previous : lseq bf16 (previous_elements position)) (current : lseq f32 kv_width)
  (count : nat{count < kv_width})
  : Lemma (
      S.upd (physical_cache_prefix position previous current count)
        (cache_write_index position count) (Cast.fcast (S.index current count)) ==
      physical_cache_prefix position previous current (count + 1))
  = cache_write_index_coordinates position count;
    let at (i : nat{i < cache_elements})
      : Lemma (
          S.index (S.upd (physical_cache_prefix position previous current count)
            (cache_write_index position count) (Cast.fcast (S.index current count))) i ==
          S.index (physical_cache_prefix position previous current (count + 1)) i)
      = if i = cache_write_index position count then ()
        else if (i % cache_head_elements) / head_width = position then
          cache_current_index position i
    in
    C.forall_intro at;
    S.lemma_eq_elim
      (S.upd (physical_cache_prefix position previous current count)
        (cache_write_index position count) (Cast.fcast (S.index current count)))
      (physical_cache_prefix position previous current (count + 1))

inline_for_extraction noextract
let physical_cache_prefix_full (position : position_index)
  (previous : lseq bf16 (previous_elements position)) (current : lseq f32 kv_width)
  : Lemma (physical_cache_prefix position previous current kv_width ==
      physical_cache position previous current)
  = S.lemma_eq_elim (physical_cache_prefix position previous current kv_width)
      (physical_cache position previous current)
#pop-options

inline_for_extraction noextract
let initial_position (position : i64) : b:bool{b <==> I64.v position = 0} =
  I64.eq position 0L

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let cache_write_offset
  (position:i64{valid_position (I64.v position)})
  (coordinate:sz{coordinate < kv_width})
  : Pure sz
      (requires True)
      (ensures fun result ->
        SZ.v result < cache_elements /\
        SZ.v result == cache_write_index (I64.v position) (SZ.v coordinate)) =
  let head = coordinate /^ 256sz in
  let time = SZ.uint32_to_sizet (IC.int64_to_uint32 position) in
  assert (SZ.v head < 2 /\ SZ.v time == I64.v position /\ SZ.v time < 2048);
  head *^ SZ.uint32_to_sizet 524288ul +^ time *^ 256sz +^ coordinate %^ 256sz
#pop-options

inline_for_extraction noextract
fn initial_cache_cell (input : array bf16)
  (position : i64{valid_position (I64.v position)}) (index : sz{index < cache_elements})
  (#previous : erased (lseq bf16 (previous_elements (I64.v position)))) (#fraction : perm)
  preserves gpu ** state_ownership input (I64.v position) fraction previous
  returns result : bf16
  ensures pure (result == (if I64.v position = 0 then (zero <: bf16) else S.index previous (SZ.v index)))
{
  if (initial_position position) {
    zero #bf16
  } else {
    rewrite state_ownership input (I64.v position) fraction previous as
      (pure (is_global_array input) ** input |-> Frac fraction previous);
    gpu_pts_to_ref input;
    let result = input.(index);
    rewrite (pure (is_global_array input) ** input |-> Frac fraction previous) as
      state_ownership input (I64.v position) fraction previous;
    result
  }
}

#push-options "--fuel 4 --z3rlimit 120"
[@@CPrologue "__device__"]
fn store_cache (input : array bf16) (output : larray bf16 cache_elements)
  (current : larray f32 kv_width) (position : i64{valid_position (I64.v position)})
  (#previous : erased (lseq bf16 (previous_elements (I64.v position))))
  (#cv : erased (lseq f32 kv_width))
  (#fraction #fc : perm)
  preserves gpu ** state_ownership input (I64.v position) fraction previous **
    current |-> Frac fc cv
  requires A.pts_to_uninit output cache_elements
  ensures output |-> physical_cache (I64.v position) previous cv
{
  unfold A.pts_to_uninit output cache_elements;
  let mut index = 0sz;
  while (!index <^ SZ.uint32_to_sizet 1048576ul)
    invariant exists* (i : sz{i <= cache_elements}) (values : seq (option bf16)).
      index |-> i ** A.pts_to_mask output values (fun _ -> True) **
      pure (S.length values == cache_elements /\
        (forall (j : nat{j < SZ.v i}).
          S.index values j == Some (S.index (physical_cache_prefix (I64.v position) previous cv 0) j)))
    decreases (cache_elements - !index)
  {
    let i = !index;
    let result = initial_cache_cell input position i;
    A.mask_write output i result;
    index := i +^ 1sz;
  };
  A.from_mask output;
  with values. assert output |-> values;
  assert pure (S.equal values (physical_cache_prefix (I64.v position) previous cv 0));
  let mut coordinate = 0sz;
  while (!coordinate <^ 512sz)
    invariant exists* (i : sz{i <= kv_width}).
      coordinate |-> i ** output |-> physical_cache_prefix (I64.v position) previous cv (SZ.v i)
    decreases (kv_width - !coordinate)
  {
    let i = !coordinate;
    let offset = cache_write_offset position i;
    assert pure (SZ.v offset < cache_elements /\
      SZ.v offset == cache_write_index (I64.v position) (SZ.v i));
    A.pts_to_len output;
    A.pts_to_len current;
    let value = current.(i);
    let result : bf16 = Cast.fcast value;
    physical_cache_prefix_step (I64.v position) previous cv (SZ.v i);
    output.(offset) <- result;
    with after. assert output |-> after;
    assert pure (after == physical_cache_prefix (I64.v position) previous cv (SZ.v i + 1));
    rewrite (output |-> after) as
      (output |-> physical_cache_prefix (I64.v position) previous cv (SZ.v i + 1));
    coordinate := i +^ 1sz;
  };
  physical_cache_prefix_full (I64.v position) previous cv;
  rewrite (output |-> physical_cache_prefix (I64.v position) previous cv kv_width) as
    (output |-> physical_cache (I64.v position) previous cv);
}
#pop-options

inline_for_extraction noextract
let physical_cache_relation (position : position_index)
  (previous : lseq bf16 (previous_elements position)) (current : lseq f32 kv_width)
  : Lemma (
      retained_cache_post position previous (physical_cache position previous current) /\
      (forall (pr : real_vector (previous_elements position)) (cr : real_vector kv_width).
        previous %~ pr /\ current %~ cr ==>
        physical_cache position previous current %~ cache_update position pr cr))
  = let _ = Approx.a0 #bf16 in ()

inline_for_extraction noextract
let score_interpretation = real_vector activation_width & real_vector cache_elements

inline_for_extraction noextract
let score_inputs (queries : lseq f32 activation_width) (keys : lseq bf16 cache_elements)
  (model : score_interpretation) : prop = queries %~ fst model /\ keys %~ snd model

inline_for_extraction noextract
let score_term (model : score_interpretation)
  (head : nat{head < query_heads}) (time : nat{time < cache_positions}) (i : nat)
  : GTot real =
  if i < head_width then
    S.index (fst model) (head * head_width + i) *.
      S.index (snd model) (cache_index (kv_head_for_query head) time i)
  else 0.0R

inline_for_extraction noextract
let score_lane_value (model : score_interpretation)
  (head : nat{head < query_heads}) (time : nat{time < cache_positions}) (lane : nat)
  : GTot real = prefix 8 (fun e -> score_term model head time (lane * 8 + e))

inline_for_extraction noextract
let score_dot_value (model : score_interpretation)
  (head : nat{head < query_heads}) (time : nat{time < cache_positions})
  : GTot real = dot (vector_slice (fst model) (head * head_width) head_width)
    (vector_slice (snd model) (cache_index (kv_head_for_query head) time 0) head_width)

inline_for_extraction noextract
let score_lane_unfold (head : nat{head < query_heads}) (time : nat{time < cache_positions})
  (lane : nat) (model : score_interpretation)
  : Lemma (score_lane_value model head time lane ==
      prefix 8 (fun j -> score_term model head time (lane * 8 + j)))
  = ()

inline_for_extraction noextract
let score_reduction_model (head : nat{head < query_heads}) (time : nat{time < cache_positions})
  (model : score_interpretation)
  : Lemma (prefix 32 (score_lane_value model head time) == score_dot_value model head time)
  = prefix_ext 32 (score_lane_value model head time)
      (fun lane -> prefix 8 (fun e -> score_term model head time (lane * 8 + e)));
    prefix_blocks 32 8 (score_term model head time);
    prefix_init 256 (score_term model head time);
    FStar.Pervasives.norm_spec [delta_only [`%score_dot_value; `%dot]; zeta; iota]
      (score_dot_value model head time);
    S.lemma_eq_elim (S.init_ghost head_width (score_term model head time))
      (S.init_ghost head_width (fun i ->
        S.index (vector_slice (fst model) (head * head_width) head_width) i *.
        S.index (vector_slice (snd model) (cache_index (kv_head_for_query head) time 0) head_width) i))

inline_for_extraction noextract
fn attention_score_lane
  (queries : larray f32 activation_width) (keys : larray bf16 cache_elements)
  (head : sz{head < query_heads}) (time : sz{time < cache_positions}) (lane : sz{lane < 32})
  (#qv : erased (lseq f32 activation_width)) (#kv : erased (lseq bf16 cache_elements))
  (#fq #fk : perm)
  preserves gpu ** queries |-> Frac fq qv ** keys |-> Frac fk kv
  returns result : f32
  ensures pure (forall (model : score_interpretation). score_inputs qv kv model ==>
    result %~ score_lane_value model (SZ.v head) (SZ.v time) (SZ.v lane))
{
  let offset = (head /^ 4sz) *^ SZ.uint32_to_sizet 524288ul +^ time *^ 256sz;
  let mut element = 0sz;
  let mut accumulator : f32 = zero;
  while (!element <^ 8sz)
    invariant exists* (e : sz{e <= 8}) (acc : f32).
      element |-> e ** accumulator |-> acc **
      pure (forall (model : score_interpretation). score_inputs qv kv model ==>
        acc %~ prefix (SZ.v e) (fun j -> score_term model (SZ.v head) (SZ.v time) (SZ.v lane * 8 + j)))
    decreases (8 - !element)
  {
    let e = !element;
    let coordinate = lane *^ 8sz +^ e;
    let q = queries.(head *^ 256sz +^ coordinate);
    let k : f32 = Cast.fcast (keys.(offset +^ coordinate));
    let prior = !accumulator;
    let result = F32.fma_rn_ftz q k prior;
    lift_fma #score_interpretation (score_inputs qv kv)
      (fun model -> S.index (fst model) (SZ.v head * head_width + SZ.v coordinate))
      (fun model -> S.index (snd model) (cache_index (kv_head_for_query (SZ.v head))
        (SZ.v time) (SZ.v coordinate)))
      (fun model -> prefix (SZ.v e)
        (fun j -> score_term model (SZ.v head) (SZ.v time) (SZ.v lane * 8 + j)))
      q k prior result;
    accumulator := result;
    element := e +^ 1sz;
  };
  C.forall_intro (score_lane_unfold (SZ.v head) (SZ.v time) (SZ.v lane));
  !accumulator
}

inline_for_extraction noextract
fn attention_score
  (queries : larray f32 activation_width) (keys : larray bf16 cache_elements)
  (head : sz{head < query_heads}) (time : sz{time < cache_positions})
  (#qv : erased (lseq f32 activation_width)) (#kv : erased (lseq bf16 cache_elements))
  (#fq #fk : perm)
  preserves gpu ** queries |-> Frac fq qv ** keys |-> Frac fk kv
  returns result : f32
  ensures pure (forall (model : score_interpretation). score_inputs qv kv model ==>
    result %~ (score_scale *. score_dot_value model (SZ.v head) (SZ.v time)))
{
  let mut lanes = [| zero #f32 #_; 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i : sz{i <= 32}) (values : lseq f32 32).
      lane |-> i ** lanes |-> values **
      pure (forall (model : score_interpretation). score_inputs qv kv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~ score_lane_value model (SZ.v head) (SZ.v time) j))
    decreases (32 - !lane)
  {
    let i = !lane;
    let result = attention_score_lane queries keys head time i;
    lanes.(i) <- result;
    lane := i +^ 1sz;
  };
  with values. assert lanes |-> values;
  let summed = sum32 lanes;
  lift_sum32 #score_interpretation (score_inputs qv kv)
    (fun model -> score_lane_value model (SZ.v head) (SZ.v time)) values summed;
  C.forall_intro (score_reduction_model (SZ.v head) (SZ.v time));
  let factor = ratio32 1L 16L;
  let result = F32.mul_rn_ftz summed factor;
  lift_mul #score_interpretation (score_inputs qv kv)
    (fun model -> score_dot_value model (SZ.v head) (SZ.v time))
    (fun _ -> score_scale) summed factor result;
  result
}

inline_for_extraction noextract
let attention_score_cell (position : position_index) (i : nat{i < score_elements position})
  (model : score_interpretation)
  : Lemma (S.index (attention_scores position (fst model) (snd model)) i ==
      score_scale *. score_dot_value model (i / (position + 1)) (i % (position + 1)))
  = ()

[@@CPrologue "__device__"]
fn fill_scores
  (position : i64{valid_position (I64.v position)})
  (queries : larray f32 activation_width) (keys : larray bf16 cache_elements)
  (scores : larray f32 (score_elements (I64.v position)))
  (#qv : erased (lseq f32 activation_width)) (#kv : erased (lseq bf16 cache_elements))
  (#old : erased (lseq f32 (score_elements (I64.v position)))) (#fq #fk : perm)
  preserves gpu ** queries |-> Frac fq qv ** keys |-> Frac fk kv
  requires scores |-> old
  ensures exists* (after : lseq f32 (score_elements (I64.v position))).
    scores |-> after **
    pure (forall (model : score_interpretation).
      score_inputs qv kv model ==> after %~ attention_scores (I64.v position) (fst model) (snd model))
{
  let count = SZ.uint32_to_sizet (IC.int64_to_uint32 position) +^ 1sz;
  let mut index = 0sz;
  while (!index <^ 8sz *^ count)
    invariant exists* (i : sz{i <= score_elements (I64.v position)})
      (values : lseq f32 (score_elements (I64.v position))).
      index |-> i ** scores |-> values **
      pure (forall (model : score_interpretation). score_inputs qv kv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~ S.index (attention_scores (I64.v position) (fst model) (snd model)) j))
    decreases (score_elements (I64.v position) - !index)
  {
    let i = !index;
    let result = attention_score queries keys (i /^ count) (i %^ count);
    C.forall_intro (attention_score_cell (I64.v position) (SZ.v i));
    scores.(i) <- result;
    index := i +^ 1sz;
  }
}

inline_for_extraction noextract
let online_inputs (#position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (model : online_interpretation position) : prop = scores %~ fst model /\ values %~ snd model

inline_for_extraction noextract
let mass_term (position : position_index) (model : online_interpretation position)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions}) (time : nat)
  : GTot real =
  if time <= position /\ time % partitions = partition
  then exp (S.index (fst model) (head * (position + 1) + time)) else 0.0R

inline_for_extraction noextract
let numerator_term (position : position_index) (model : online_interpretation position)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (coordinate : nat{coordinate < head_width}) (time : nat)
  : GTot real =
  if time <= position then mass_term position model head partition time *.
    S.index (snd model) (cache_index (kv_head_for_query head) time coordinate)
  else 0.0R

inline_for_extraction noextract
let partition_mass (position : position_index) (model : online_interpretation position)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions}) (count : nat)
  : GTot real = prefix count (mass_term position model head partition)

inline_for_extraction noextract
let partition_numerator (position : position_index) (model : online_interpretation position)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (count : nat) (coordinate : nat{coordinate < head_width})
  : GTot real = prefix count (numerator_term position model head partition coordinate)

inline_for_extraction noextract
let shifted_partition (position : position_index) (model : online_interpretation position)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (count : nat) (maximum : f32) : GTot (real_vector head_width) =
  S.init_ghost head_width (fun i ->
    partition_numerator position model head partition count i /. exp (to_real maximum))

inline_for_extraction noextract
let cancel_ratio (a : real) (b : real{b =!= 0.0R}) (c : real{c =!= 0.0R})
  : Lemma ((a /. b) *. (b /. c) == a /. c) = ()

inline_for_extraction noextract
let divide_product (a b : real) (c : real{c =!= 0.0R})
  : Lemma ((a /. c) *. b == (a *. b) /. c) = ()

inline_for_extraction noextract
let divide_sum (a b : real) (c : real{c =!= 0.0R})
  : Lemma ((a +. b) /. c == a /. c +. b /. c) = ()

inline_for_extraction noextract
let multiply_inverse (value : real) (denominator : real{denominator =!= 0.0R})
  : Lemma ((1.0R /. denominator) *. value == value /. denominator) = ()

inline_for_extraction noextract
let online_rebase (a x value old_shift new_shift : real)
  : Lemma (
      (a /. exp old_shift) *. exp (old_shift -. new_shift) +.
        exp (x -. new_shift) *. value ==
      (a +. exp x *. value) /. exp new_shift)
  = exp_sub old_shift new_shift;
    exp_sub x new_shift;
    cancel_ratio a (exp old_shift) (exp new_shift);
    divide_product (exp x) value (exp new_shift);
    divide_sum a (exp x *. value) (exp new_shift)

inline_for_extraction noextract
let merge_rebase (a b old_shift new_shift : real)
  : Lemma (
      (a /. exp old_shift) *. exp (old_shift -. new_shift) +. b /. exp new_shift ==
      (b +. a) /. exp new_shift)
  = exp_sub old_shift new_shift;
    cancel_ratio a (exp old_shift) (exp new_shift);
    divide_sum b a (exp new_shift)

inline_for_extraction noextract
let exponential_difference_finish (left right negative_right difference result : f32)
  : Lemma
      (requires
        (forall (r : real). right %~ r ==> negative_right %~ (0.0R -. r)) /\
        (forall (x y : real). left %~ x /\ negative_right %~ y ==> difference %~ (x +. y)) /\
        (forall (r : real). difference %~ r ==> result %~ exp r))
      (ensures forall (lr rr : real). left %~ lr /\ right %~ rr ==> result %~ exp (lr -. rr))
  = let related (lr : real{left %~ lr}) (rr : real{right %~ rr})
      : Lemma (result %~ exp (lr -. rr))
      = assert (negative_right %~ (0.0R -. rr));
        assert (difference %~ (lr +. (0.0R -. rr)));
        assert (result %~ exp (lr +. (0.0R -. rr)))
    in C.forall_intro_2 related

inline_for_extraction noextract
fn exponential_difference (left right : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (lr rr : real). left %~ lr /\ right %~ rr ==>
    result %~ exp (lr -. rr))
{
  let negative_right = negative right;
  let difference = F32.add_rn_ftz left negative_right;
  let result = exponential_ftz difference;
  exponential_difference_finish left right negative_right difference result;
  result
}

inline_for_extraction noextract
fn zero_vector (#n : pos) (output : larray f32 n) (count : sz{SZ.v count = n})
  (#old : erased (lseq f32 n))
  preserves gpu
  requires output |-> old
  ensures output |-> S.create n (zero #f32)
{
  let mut index = 0sz;
  while (!index <^ count)
    invariant exists* (i : sz{i <= n}) (values : lseq f32 n).
      index |-> i ** output |-> values **
      pure (forall (j : nat{j < SZ.v i}). S.index values j == zero #f32)
    decreases (n - !index)
  {
    let i = !index;
    output.(i) <- zero #f32;
    index := i +^ 1sz;
  };
  with values. assert output |-> values;
  assert pure (S.equal values (S.create n (zero #f32)));
}

inline_for_extraction noextract
let partition_extend (position : position_index)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (time : nat{time <= position /\ time % partitions = partition})
  (model : online_interpretation position)
  : Lemma (
      partition_mass position model head partition (time + 1) ==
        partition_mass position model head partition time +.
          exp (S.index (fst model) (head * (position + 1) + time)) /\
      (forall (i : nat{i < head_width}).
        partition_numerator position model head partition (time + 1) i ==
        partition_numerator position model head partition time i +.
          exp (S.index (fst model) (head * (position + 1) + time)) *.
            S.index (snd model) (cache_index (kv_head_for_query head) time i)))
  = prefix_step time (mass_term position model head partition);
    let at (i : nat{i < head_width})
      : Lemma (
          partition_numerator position model head partition (time + 1) i ==
          partition_numerator position model head partition time i +.
            exp (S.index (fst model) (head * (position + 1) + time)) *.
              S.index (snd model) (cache_index (kv_head_for_query head) time i))
      = prefix_step time (numerator_term position model head partition i)
    in C.forall_intro at

inline_for_extraction noextract
let partition_value_update (position : position_index)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (time : nat{time <= position /\ time % partitions = partition})
  (coordinate : nat{coordinate < head_width}) (old_maximum new_maximum : f32)
  (model : online_interpretation position)
  : Lemma (
      S.index (shifted_partition position model head partition (time + 1) new_maximum) coordinate ==
      S.index (shifted_partition position model head partition time old_maximum) coordinate *.
        exp (to_real old_maximum -. to_real new_maximum) +.
      exp (S.index (fst model) (head * (position + 1) + time) -. to_real new_maximum) *.
        S.index (snd model) (cache_index (kv_head_for_query head) time coordinate))
  = partition_extend position head partition time model;
    online_rebase (partition_numerator position model head partition time coordinate)
      (S.index (fst model) (head * (position + 1) + time))
      (S.index (snd model) (cache_index (kv_head_for_query head) time coordinate))
      (to_real old_maximum) (to_real new_maximum)

inline_for_extraction noextract
let partition_mass_update (position : position_index)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (time : nat{time <= position /\ time % partitions = partition})
  (old_maximum new_maximum : f32) (model : online_interpretation position)
  : Lemma (
      partition_mass position model head partition (time + 1) /. exp (to_real new_maximum) ==
      (partition_mass position model head partition time /. exp (to_real old_maximum)) *.
        exp (to_real old_maximum -. to_real new_maximum) +.
      exp (S.index (fst model) (head * (position + 1) + time) -. to_real new_maximum))
  = partition_extend position head partition time model;
    online_rebase (partition_mass position model head partition time)
      (S.index (fst model) (head * (position + 1) + time)) 1.0R
      (to_real old_maximum) (to_real new_maximum)

inline_for_extraction noextract
let partition_invariant (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (count : nat{count <= position + 1}) (state : lseq f32 2) (output : lseq f32 head_width)
  : prop =
  S.index state 0 == partition_maximum position scores head partition count /\
  (forall (model : online_interpretation position). online_inputs scores values model ==>
    S.index state 1 %~
      (partition_mass position model head partition count /. exp (to_real (S.index state 0))) /\
    output %~ shifted_partition position model head partition count (S.index state 0))

inline_for_extraction noextract
let partition_skip (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (count : nat{count <= position /\ count % partitions <> partition})
  (state : lseq f32 2) (output : lseq f32 head_width)
  : Lemma (
      partition_invariant position scores values head partition (count + 1) state output ==
      partition_invariant position scores values head partition count state output)
  = let unchanged (model : online_interpretation position)
      : Lemma (
          partition_mass position model head partition (count + 1) ==
            partition_mass position model head partition count /\
          shifted_partition position model head partition (count + 1) (S.index state 0) ==
            shifted_partition position model head partition count (S.index state 0))
      = prefix_step count (mass_term position model head partition);
        let at (i : nat{i < head_width})
          : Lemma (partition_numerator position model head partition (count + 1) i ==
              partition_numerator position model head partition count i)
          = prefix_step count (numerator_term position model head partition i)
        in
        C.forall_intro at;
        S.lemma_eq_elim
          (shifted_partition position model head partition (count + 1) (S.index state 0))
          (shifted_partition position model head partition count (S.index state 0))
    in C.forall_intro unchanged

inline_for_extraction noextract
fn update_partition_vector
  (position : i64{valid_position (I64.v position)})
  (values : larray bf16 cache_elements) (output : larray f32 head_width)
  (head : sz{head < query_heads}) (partition : sz{partition < partitions})
  (time : sz{time <= I64.v position /\ SZ.v time % partitions = SZ.v partition})
  (old_maximum new_maximum correction weight : f32)
  (#scores : erased (lseq f32 (score_elements (I64.v position))))
  (#vv : erased (lseq bf16 cache_elements)) (#before : erased (lseq f32 head_width)) (#fv : perm)
  preserves gpu ** values |-> Frac fv vv
  requires output |-> before **
    pure (correction %~ exp (to_real old_maximum -. to_real new_maximum)) **
    pure (forall (model : online_interpretation (I64.v position)). online_inputs scores vv model ==>
      weight %~ exp (S.index (fst model) (SZ.v head * (I64.v position + 1) + SZ.v time) -. to_real new_maximum) /\
      before %~ shifted_partition (I64.v position) model (SZ.v head) (SZ.v partition) (SZ.v time) old_maximum)
  ensures exists* (after : lseq f32 head_width). output |-> after **
    pure (forall (model : online_interpretation (I64.v position)). online_inputs scores vv model ==>
      after %~ shifted_partition (I64.v position) model (SZ.v head) (SZ.v partition) (SZ.v time + 1) new_maximum)
{
  let offset = (head /^ 4sz) *^ SZ.uint32_to_sizet 524288ul +^ time *^ 256sz;
  let mut coordinate = 0sz;
  while (!coordinate <^ 256sz)
    invariant exists* (i : sz{i <= head_width}) (current : lseq f32 head_width).
      coordinate |-> i ** output |-> current **
      pure (forall (j : nat{SZ.v i <= j /\ j < head_width}). S.index current j == S.index before j) **
      pure (forall (model : online_interpretation (I64.v position)). online_inputs scores vv model ==>
        (forall (j : nat{j < SZ.v i}). S.index current j %~
          S.index (shifted_partition (I64.v position) model (SZ.v head) (SZ.v partition)
            (SZ.v time + 1) new_maximum) j))
    decreases (head_width - !coordinate)
  {
    let i = !coordinate;
    let v : f32 = Cast.fcast (values.(offset +^ i));
    let prior = output.(i);
    let product = F32.mul_rn_ftz weight v;
    lift_mul #(online_interpretation (I64.v position)) (online_inputs #(I64.v position) scores vv)
      (fun model -> exp (S.index (fst model) (SZ.v head * (I64.v position + 1) + SZ.v time) -. to_real new_maximum))
      (fun model -> S.index (snd model) (cache_index (kv_head_for_query (SZ.v head)) (SZ.v time) (SZ.v i)))
      weight v product;
    let result = F32.fma_rn_ftz prior correction product;
    lift_fma #(online_interpretation (I64.v position)) (online_inputs #(I64.v position) scores vv)
      (fun model -> S.index (shifted_partition (I64.v position) model (SZ.v head) (SZ.v partition)
        (SZ.v time) old_maximum) (SZ.v i))
      (fun _ -> exp (to_real old_maximum -. to_real new_maximum))
      (fun model -> exp (S.index (fst model) (SZ.v head * (I64.v position + 1) + SZ.v time) -. to_real new_maximum) *.
        S.index (snd model) (cache_index (kv_head_for_query (SZ.v head)) (SZ.v time) (SZ.v i)))
      prior correction product result;
    C.forall_intro (partition_value_update (I64.v position) (SZ.v head) (SZ.v partition)
      (SZ.v time) (SZ.v i) old_maximum new_maximum);
    output.(i) <- result;
    coordinate := i +^ 1sz;
  }
}

inline_for_extraction noextract
let equal_size (x y : sz) : b:bool{b <==> SZ.v x == SZ.v y} =
  FStar.UInt32.eq (SZ.sizet_to_u32 x) (SZ.sizet_to_u32 y)

ghost
fn apply_partition_skip (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads}) (partition : nat{partition < partitions})
  (count : nat{count <= position /\ count % partitions <> partition})
  (state : lseq f32 2) (output : lseq f32 head_width)
  requires pure (partition_invariant position scores values head partition count state output)
  ensures pure (partition_invariant position scores values head partition (count + 1) state output)
{
  partition_skip position scores values head partition count state output;
  rewrite pure (partition_invariant position scores values head partition count state output) as
    pure (partition_invariant position scores values head partition (count + 1) state output);
}

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn partition_step
  (position : i64{valid_position (I64.v position)})
  (scores : larray f32 (score_elements (I64.v position))) (values : larray bf16 cache_elements)
  (state : larray f32 2) (output : larray f32 head_width)
  (head : sz{head < query_heads}) (partition : sz{partition < partitions})
  (time : sz{time <= I64.v position})
  (#sv : erased (lseq f32 (score_elements (I64.v position))))
  (#vv : erased (lseq bf16 cache_elements)) (#before : erased (lseq f32 2))
  (#old : erased (lseq f32 head_width)) (#fs #fv : perm)
  preserves gpu ** scores |-> Frac fs sv ** values |-> Frac fv vv
  requires state |-> before ** output |-> old **
    pure (partition_invariant (I64.v position) sv vv (SZ.v head) (SZ.v partition) (SZ.v time) before old)
  ensures exists* (after : lseq f32 2) (out : lseq f32 head_width).
    state |-> after ** output |-> out **
    pure (partition_invariant (I64.v position) sv vv (SZ.v head) (SZ.v partition) (SZ.v time + 1) after out)
{
  let member = equal_size (time %^ 16sz) partition;
  if member {
    assert pure (SZ.v time % partitions == SZ.v partition);
    let count = SZ.uint32_to_sizet (IC.int64_to_uint32 position) +^ 1sz;
    let score = scores.(head *^ count +^ time);
    let previous = state.(0sz);
    let denominator = state.(1sz);
    let maximum = fmax previous score;
    Approx.to_real_ok previous;
    Approx.to_real_ok maximum;
    let correction = exponential_difference previous maximum;
    assert pure (correction %~ exp (to_real previous -. to_real maximum));
    let weight = exponential_difference score maximum;
    assert pure (forall (model : online_interpretation (I64.v position)). online_inputs sv vv model ==>
      weight %~ exp (S.index (fst model) (SZ.v head * (I64.v position + 1) + SZ.v time) -. to_real maximum));
    let updated = F32.fma_rn_ftz denominator correction weight;
    lift_fma #(online_interpretation (I64.v position)) (online_inputs #(I64.v position) sv vv)
      (fun model -> partition_mass (I64.v position) model (SZ.v head) (SZ.v partition) (SZ.v time) /. exp (to_real previous))
      (fun _ -> exp (to_real previous -. to_real maximum))
      (fun model -> exp (S.index (fst model) (SZ.v head * (I64.v position + 1) + SZ.v time) -. to_real maximum))
      denominator correction weight updated;
    C.forall_intro (partition_mass_update (I64.v position) (SZ.v head) (SZ.v partition)
      (SZ.v time) previous maximum);
    update_partition_vector position values output head partition time previous maximum correction weight #sv;
    state.(0sz) <- maximum;
    state.(1sz) <- updated;
  } else {
    assert pure (SZ.v time % partitions <> SZ.v partition);
    apply_partition_skip (I64.v position) sv vv (SZ.v head) (SZ.v partition) (SZ.v time) before old;
  }
}
#pop-options

inline_for_extraction noextract
fn evaluate_partition
  (position : i64{valid_position (I64.v position)})
  (scores : larray f32 (score_elements (I64.v position))) (values : larray bf16 cache_elements)
  (state : larray f32 2) (output : larray f32 head_width)
  (head : sz{head < query_heads}) (partition : sz{partition < partitions})
  (#sv : erased (lseq f32 (score_elements (I64.v position))))
  (#vv : erased (lseq bf16 cache_elements)) (#before : erased (lseq f32 2))
  (#old : erased (lseq f32 head_width)) (#fs #fv : perm)
  preserves gpu ** scores |-> Frac fs sv ** values |-> Frac fv vv
  requires state |-> before ** output |-> old
  ensures exists* (after : lseq f32 2) (out : lseq f32 head_width).
    state |-> after ** output |-> out **
    pure (partition_invariant (I64.v position) sv vv (SZ.v head) (SZ.v partition)
      (I64.v position + 1) after out)
{
  state.(0sz) <- negative_infinity;
  state.(1sz) <- zero #f32;
  zero_vector output 256sz;
  let count = SZ.uint32_to_sizet (IC.int64_to_uint32 position) +^ 1sz;
  let mut time = 0sz;
  while (!time <^ count)
    invariant exists* (t : sz{t <= I64.v position + 1}) (st : lseq f32 2) (out : lseq f32 head_width).
      time |-> t ** state |-> st ** output |-> out **
      pure (partition_invariant (I64.v position) sv vv (SZ.v head) (SZ.v partition) (SZ.v t) st out)
    decreases (I64.v position + 1 - !time)
  {
    let t = !time;
    partition_step position scores values state output head partition t;
    time := t +^ 1sz;
  }
}

inline_for_extraction noextract
let collected_partitions (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads}) (done : nat{done <= partitions})
  (maxima denominators : lseq f32 partitions) (partial : lseq f32 (partitions * head_width))
  : prop =
  (forall (p : nat{p < done}).
    S.index maxima p == partition_maximum position scores head p (position + 1)) /\
  (forall (model : online_interpretation position). online_inputs scores values model ==>
    (forall (p : nat{p < done}).
      S.index denominators p %~
        (partition_mass position model head p (position + 1) /. exp (to_real (S.index maxima p))) /\
      (forall (i : nat{i < head_width}).
        S.index partial (p * head_width + i) %~
          S.index (shifted_partition position model head p (position + 1) (S.index maxima p)) i)))

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let collect_partition_step (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads}) (p : nat{p < partitions})
  (maxima denominators : lseq f32 partitions) (partial : lseq f32 (partitions * head_width))
  (state : lseq f32 2) (output : lseq f32 head_width)
  : Lemma
      (requires collected_partitions position scores values head p maxima denominators partial /\
        partition_invariant position scores values head p (position + 1) state output)
      (ensures collected_partitions position scores values head (p + 1)
        (S.upd maxima p (S.index state 0)) (S.upd denominators p (S.index state 1))
        (replace_slice partial (p * head_width) output))
  = ()
#pop-options

inline_for_extraction noextract
fn collect_partitions
  (position : i64{valid_position (I64.v position)})
  (scores : larray f32 (score_elements (I64.v position))) (values : larray bf16 cache_elements)
  (head : sz{head < query_heads})
  (maxima denominators : larray f32 partitions) (partial : larray f32 (partitions * head_width))
  (#sv : erased (lseq f32 (score_elements (I64.v position))))
  (#vv : erased (lseq bf16 cache_elements)) (#old_m #old_d : erased (lseq f32 partitions))
  (#old_p : erased (lseq f32 (partitions * head_width))) (#fs #fv : perm)
  preserves gpu ** scores |-> Frac fs sv ** values |-> Frac fv vv
  requires maxima |-> old_m ** denominators |-> old_d ** partial |-> old_p
  ensures exists* (mv dv : lseq f32 partitions) (pv : lseq f32 (partitions * head_width)).
    maxima |-> mv ** denominators |-> dv ** partial |-> pv **
    pure (collected_partitions (I64.v position) sv vv (SZ.v head) partitions mv dv pv)
{
  let mut state = [| zero #f32 #_; 2sz |];
  let mut output = [| zero #f32 #_; 256sz |];
  let mut partition = 0sz;
  while (!partition <^ 16sz)
    invariant exists* (p : sz{p <= partitions}) (mv dv : lseq f32 partitions)
      (pv : lseq f32 (partitions * head_width)).
      partition |-> p ** maxima |-> mv ** denominators |-> dv ** partial |-> pv **
      pure (collected_partitions (I64.v position) sv vv (SZ.v head) (SZ.v p) mv dv pv)
    invariant exists* (st : lseq f32 2). state |-> st
    invariant exists* (out : lseq f32 head_width). output |-> out
    decreases (partitions - !partition)
  {
    let p = !partition;
    evaluate_partition position scores values state output head p;
    with st. assert state |-> st;
    with out. assert output |-> out;
    with mv. assert maxima |-> mv;
    with dv. assert denominators |-> dv;
    with pv. assert partial |-> pv;
    collect_partition_step (I64.v position) sv vv (SZ.v head) (SZ.v p) mv dv pv st out;
    maxima.(p) <- state.(0sz);
    denominators.(p) <- state.(1sz);
    write_f32_slice partial output (p *^ 256sz) 256sz;
    partition := p +^ 1sz;
  }
}

inline_for_extraction noextract
let rec merged_maximum (maxima : lseq f32 partitions) (count : nat{count <= partitions})
  : GTot f32 (decreases count) =
  if count = 0 then negative_infinity else
    let previous = merged_maximum maxima (count - 1) in
    let next = S.index maxima (count - 1) in
    if negative_infinity `lt` next then fmax previous next else previous

inline_for_extraction noextract
fn select_maximum (maxima : larray f32 partitions)
  (#mv : erased (lseq f32 partitions)) (#fraction : perm)
  preserves gpu ** maxima |-> Frac fraction mv
  returns result : f32
  ensures pure (result == merged_maximum mv partitions)
{
  let mut partition = 0sz;
  let mut maximum = negative_infinity;
  while (!partition <^ 16sz)
    invariant exists* (p : sz{p <= partitions}) (m : f32).
      partition |-> p ** maximum |-> m ** pure (m == merged_maximum mv (SZ.v p))
    decreases (partitions - !partition)
  {
    let p = !partition;
    let current = maxima.(p);
    let previous = !maximum;
    maximum := (if negative_infinity `lt` current then fmax previous current else previous);
    partition := p +^ 1sz;
  };
  !maximum
}

inline_for_extraction noextract
let selected_mass_term (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads})
  (done : nat{done <= partitions}) (time : nat) : GTot real =
  if time <= position /\ time % partitions < done /\
      partition_included position controls head (time % partitions)
  then exp (S.index (fst model) (head * (position + 1) + time)) else 0.0R

inline_for_extraction noextract
let selected_numerator_term (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads})
  (done : nat{done <= partitions}) (coordinate : nat{coordinate < head_width}) (time : nat)
  : GTot real =
  if time <= position then selected_mass_term position controls model head done time *.
    S.index (snd model) (cache_index (kv_head_for_query head) time coordinate) else 0.0R

inline_for_extraction noextract
let selected_mass (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads})
  (done : nat{done <= partitions}) : GTot real =
  prefix (position + 1) (selected_mass_term position controls model head done)

inline_for_extraction noextract
let selected_numerator (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads})
  (done : nat{done <= partitions}) (coordinate : nat{coordinate < head_width}) : GTot real =
  prefix (position + 1) (selected_numerator_term position controls model head done coordinate)

inline_for_extraction noextract
let shifted_selection (position : position_index) (controls : lseq f32 (score_elements position))
  (model : online_interpretation position) (head : nat{head < query_heads})
  (done : nat{done <= partitions}) (maximum : f32) : GTot (real_vector head_width) =
  S.init_ghost head_width (fun i ->
    selected_numerator position controls model head done i /. exp (to_real maximum))

inline_for_extraction noextract
let selected_mass_step (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (p : nat{p < partitions}) (model : online_interpretation position)
  : Lemma (
      selected_mass position controls model head (p + 1) ==
      selected_mass position controls model head p +.
        (if partition_included position controls head p then
          partition_mass position model head p (position + 1) else 0.0R))
  = let old = selected_mass_term position controls model head p in
    let add = (fun t -> if partition_included position controls head p
      then mass_term position model head p t else 0.0R) in
    prefix_ext (position + 1) (selected_mass_term position controls model head (p + 1))
      (fun t -> old t +. add t);
    prefix_add (position + 1) old add;
    if partition_included position controls head p then
      prefix_ext (position + 1) add (mass_term position model head p)
    else begin
      prefix_ext (position + 1) add (fun _ -> 0.0R);
      prefix_zero (position + 1)
    end

inline_for_extraction noextract
let selected_numerator_step (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (p : nat{p < partitions}) (i : nat{i < head_width})
  (model : online_interpretation position)
  : Lemma (
      selected_numerator position controls model head (p + 1) i ==
      selected_numerator position controls model head p i +.
        (if partition_included position controls head p then
          partition_numerator position model head p (position + 1) i else 0.0R))
  = let old = selected_numerator_term position controls model head p i in
    let add = (fun t -> if partition_included position controls head p
      then numerator_term position model head p i t else 0.0R) in
    prefix_ext (position + 1) (selected_numerator_term position controls model head (p + 1) i)
      (fun t -> old t +. add t);
    prefix_add (position + 1) old add;
    if partition_included position controls head p then
      prefix_ext (position + 1) add (numerator_term position model head p i)
    else begin
      prefix_ext (position + 1) add (fun _ -> 0.0R);
      prefix_zero (position + 1)
    end

inline_for_extraction noextract
let selected_empty (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (maximum : f32) (model : online_interpretation position)
  : Lemma (
      selected_mass position controls model head 0 == 0.0R /\
      shifted_selection position controls model head 0 maximum == S.create head_width 0.0R)
  = prefix_ext (position + 1) (selected_mass_term position controls model head 0) (fun _ -> 0.0R);
    prefix_zero (position + 1);
    let at (i : nat{i < head_width})
      : Lemma (selected_numerator position controls model head 0 i == 0.0R)
      = prefix_ext (position + 1) (selected_numerator_term position controls model head 0 i) (fun _ -> 0.0R);
        prefix_zero (position + 1)
    in
    C.forall_intro at;
    S.lemma_eq_elim (shifted_selection position controls model head 0 maximum) (S.create head_width 0.0R)

inline_for_extraction noextract
let merge_invariant (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads}) (done : nat{done <= partitions})
  (maximum total_mass32 : f32) (output : lseq f32 head_width) : prop =
  forall (model : online_interpretation position). online_inputs scores values model ==>
    total_mass32 %~ (selected_mass position scores model head done /. exp (to_real maximum)) /\
    output %~ shifted_selection position scores model head done maximum

inline_for_extraction noextract
let merge_mass_equation (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (p : nat{p < partitions})
  (local_maximum maximum : f32) (model : online_interpretation position)
  : Lemma
      (requires partition_included position controls head p)
      (ensures
        selected_mass position controls model head (p + 1) /. exp (to_real maximum) ==
        (partition_mass position model head p (position + 1) /. exp (to_real local_maximum)) *.
          exp (to_real local_maximum -. to_real maximum) +.
        selected_mass position controls model head p /. exp (to_real maximum))
  = selected_mass_step position controls head p model;
    merge_rebase (partition_mass position model head p (position + 1))
      (selected_mass position controls model head p) (to_real local_maximum) (to_real maximum)

inline_for_extraction noextract
let merge_value_equation (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (p : nat{p < partitions}) (i : nat{i < head_width})
  (local_maximum maximum : f32) (model : online_interpretation position)
  : Lemma
      (requires partition_included position controls head p)
      (ensures
        S.index (shifted_selection position controls model head (p + 1) maximum) i ==
        S.index (shifted_partition position model head p (position + 1) local_maximum) i *.
          exp (to_real local_maximum -. to_real maximum) +.
        S.index (shifted_selection position controls model head p maximum) i)
  = selected_numerator_step position controls head p i model;
    merge_rebase (partition_numerator position model head p (position + 1) i)
      (selected_numerator position controls model head p i) (to_real local_maximum) (to_real maximum)

inline_for_extraction noextract
let merge_skip (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads})
  (p : nat{p < partitions /\ ~(partition_included position scores head p)})
  (maximum total_mass32 : f32) (output : lseq f32 head_width)
  : Lemma (
      merge_invariant position scores values head (p + 1) maximum total_mass32 output ==
      merge_invariant position scores values head p maximum total_mass32 output)
  = let unchanged (model : online_interpretation position)
      : Lemma (
          selected_mass position scores model head (p + 1) == selected_mass position scores model head p /\
          shifted_selection position scores model head (p + 1) maximum ==
            shifted_selection position scores model head p maximum)
      = selected_mass_step position scores head p model;
        let at (i : nat{i < head_width})
          : Lemma (selected_numerator position scores model head (p + 1) i ==
              selected_numerator position scores model head p i)
          = selected_numerator_step position scores head p i model
        in
        C.forall_intro at;
        S.lemma_eq_elim (shifted_selection position scores model head (p + 1) maximum)
          (shifted_selection position scores model head p maximum)
    in C.forall_intro unchanged

ghost
fn apply_merge_skip (position : position_index)
  (scores : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads})
  (p : nat{p < partitions /\ ~(partition_included position scores head p)})
  (maximum total_mass32 : f32) (output : lseq f32 head_width)
  requires pure (merge_invariant position scores values head p maximum total_mass32 output)
  ensures pure (merge_invariant position scores values head (p + 1) maximum total_mass32 output)
{
  merge_skip position scores values head p maximum total_mass32 output;
  rewrite pure (merge_invariant position scores values head p maximum total_mass32 output) as
    pure (merge_invariant position scores values head (p + 1) maximum total_mass32 output);
}

inline_for_extraction noextract
fn merge_vector
  (position : i64{valid_position (I64.v position)})
  (partial : larray f32 (partitions * head_width)) (output : larray f32 head_width)
  (head : sz{head < query_heads}) (p : sz{p < partitions})
  (local_maximum maximum scale : f32)
  (#scores : erased (lseq f32 (score_elements (I64.v position))))
  (#values : erased (lseq bf16 cache_elements))
  (#pv : erased (lseq f32 (partitions * head_width))) (#before : erased (lseq f32 head_width)) (#fp : perm)
  preserves gpu ** partial |-> Frac fp pv
  requires output |-> before **
    pure (partition_included (I64.v position) scores (SZ.v head) (SZ.v p)) **
    pure (scale %~ exp (to_real local_maximum -. to_real maximum)) **
    pure (forall (model : online_interpretation (I64.v position)). online_inputs scores values model ==>
      before %~ shifted_selection (I64.v position) scores model (SZ.v head) (SZ.v p) maximum /\
      (forall (i : nat{i < head_width}). S.index pv (SZ.v p * head_width + i) %~
        S.index (shifted_partition (I64.v position) model (SZ.v head) (SZ.v p)
          (I64.v position + 1) local_maximum) i))
  ensures exists* (after : lseq f32 head_width). output |-> after **
    pure (forall (model : online_interpretation (I64.v position)). online_inputs scores values model ==>
      after %~ shifted_selection (I64.v position) scores model (SZ.v head) (SZ.v p + 1) maximum)
{
  let mut coordinate = 0sz;
  while (!coordinate <^ 256sz)
    invariant exists* (i : sz{i <= head_width}) (current : lseq f32 head_width).
      coordinate |-> i ** output |-> current **
      pure (forall (j : nat{SZ.v i <= j /\ j < head_width}). S.index current j == S.index before j) **
      pure (forall (model : online_interpretation (I64.v position)). online_inputs scores values model ==>
        (forall (j : nat{j < SZ.v i}). S.index current j %~
          S.index (shifted_selection (I64.v position) scores model (SZ.v head) (SZ.v p + 1) maximum) j))
    decreases (head_width - !coordinate)
  {
    let i = !coordinate;
    let part = partial.(p *^ 256sz +^ i);
    let prior = output.(i);
    let result = F32.fma_rn_ftz part scale prior;
    lift_fma #(online_interpretation (I64.v position)) (online_inputs #(I64.v position) scores values)
      (fun model -> S.index (shifted_partition (I64.v position) model (SZ.v head) (SZ.v p)
        (I64.v position + 1) local_maximum) (SZ.v i))
      (fun _ -> exp (to_real local_maximum -. to_real maximum))
      (fun model -> S.index (shifted_selection (I64.v position) scores model (SZ.v head) (SZ.v p) maximum) (SZ.v i))
      part scale prior result;
    C.forall_intro (merge_value_equation (I64.v position) scores (SZ.v head) (SZ.v p) (SZ.v i)
      local_maximum maximum);
    output.(i) <- result;
    coordinate := i +^ 1sz;
  }
}

inline_for_extraction noextract
fn merge_partition_step
  (position : i64{valid_position (I64.v position)})
  (head : sz{head < query_heads}) (p : sz{p < partitions}) (maximum : f32)
  (maxima denominators : larray f32 partitions) (partial : larray f32 (partitions * head_width))
  (state : larray f32 1) (output : larray f32 head_width)
  (#scores : erased (lseq f32 (score_elements (I64.v position))))
  (#values : erased (lseq bf16 cache_elements)) (#mv #dv : erased (lseq f32 partitions))
  (#pv : erased (lseq f32 (partitions * head_width)))
  (#before : erased (lseq f32 1)) (#old : erased (lseq f32 head_width)) (#fm #fd #fp : perm)
  preserves gpu ** maxima |-> Frac fm mv ** denominators |-> Frac fd dv ** partial |-> Frac fp pv
  requires state |-> before ** output |-> old **
    pure (collected_partitions (I64.v position) scores values (SZ.v head) partitions mv dv pv) **
    pure (merge_invariant (I64.v position) scores values (SZ.v head) (SZ.v p) maximum (S.index before 0) old)
  ensures exists* (after : lseq f32 1) (out : lseq f32 head_width).
    state |-> after ** output |-> out **
    pure (merge_invariant (I64.v position) scores values (SZ.v head) (SZ.v p + 1) maximum (S.index after 0) out)
{
  let local_maximum = maxima.(p);
  if (negative_infinity `lt` local_maximum) {
    assert pure (partition_included (I64.v position) scores (SZ.v head) (SZ.v p));
    Approx.to_real_ok local_maximum;
    Approx.to_real_ok maximum;
    let scale = exponential_difference local_maximum maximum;
    assert pure (scale %~ exp (to_real local_maximum -. to_real maximum));
    let d = denominators.(p);
    let total_mass32 = state.(0sz);
    let result = F32.fma_rn_ftz d scale total_mass32;
    lift_fma #(online_interpretation (I64.v position)) (online_inputs #(I64.v position) scores values)
      (fun model -> partition_mass (I64.v position) model (SZ.v head) (SZ.v p) (I64.v position + 1) /.
        exp (to_real local_maximum))
      (fun _ -> exp (to_real local_maximum -. to_real maximum))
      (fun model -> selected_mass (I64.v position) scores model (SZ.v head) (SZ.v p) /. exp (to_real maximum))
      d scale total_mass32 result;
    C.forall_intro (merge_mass_equation (I64.v position) scores (SZ.v head) (SZ.v p) local_maximum maximum);
    merge_vector position partial output head p local_maximum maximum scale #scores #values;
    state.(0sz) <- result;
  } else {
    assert pure (~(partition_included (I64.v position) scores (SZ.v head) (SZ.v p)));
    apply_merge_skip (I64.v position) scores values (SZ.v head) (SZ.v p) maximum (S.index before 0) old;
  }
}

inline_for_extraction noextract
fn merge_partitions
  (position : i64{valid_position (I64.v position)})
  (head : sz{head < query_heads}) (maximum : f32)
  (maxima denominators : larray f32 partitions) (partial : larray f32 (partitions * head_width))
  (state : larray f32 1) (output : larray f32 head_width)
  (#scores : erased (lseq f32 (score_elements (I64.v position))))
  (#values : erased (lseq bf16 cache_elements)) (#mv #dv : erased (lseq f32 partitions))
  (#pv : erased (lseq f32 (partitions * head_width)))
  (#before : erased (lseq f32 1)) (#old : erased (lseq f32 head_width)) (#fm #fd #fp : perm)
  preserves gpu ** maxima |-> Frac fm mv ** denominators |-> Frac fd dv ** partial |-> Frac fp pv
  requires state |-> before ** output |-> old **
    pure (collected_partitions (I64.v position) scores values (SZ.v head) partitions mv dv pv)
  ensures exists* (after : lseq f32 1) (out : lseq f32 head_width).
    state |-> after ** output |-> out **
    pure (merge_invariant (I64.v position) scores values (SZ.v head) partitions maximum (S.index after 0) out)
{
  state.(0sz) <- zero #f32;
  zero_vector output 256sz;
  C.forall_intro (selected_empty (I64.v position) scores (SZ.v head) maximum);
  let mut partition = 0sz;
  while (!partition <^ 16sz)
    invariant exists* (p : sz{p <= partitions}) (st : lseq f32 1) (out : lseq f32 head_width).
      partition |-> p ** state |-> st ** output |-> out **
      pure (merge_invariant (I64.v position) scores values (SZ.v head) (SZ.v p) maximum (S.index st 0) out)
    decreases (partitions - !partition)
  {
    let p = !partition;
    merge_partition_step position head p maximum maxima denominators partial state output #scores #values;
    partition := p +^ 1sz;
  }
}

inline_for_extraction noextract
let selected_mass_model (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (model : online_interpretation position)
  : Lemma (
      selected_mass position controls model head partitions ==
        rsum (model_mass position controls model head) /\
      nonnegative_terms (model_mass position controls model head) /\
      selected_mass position controls model head partitions >=. 0.0R)
  = prefix_init (position + 1) (selected_mass_term position controls model head partitions);
    S.lemma_eq_elim
      (S.init_ghost (position + 1) (selected_mass_term position controls model head partitions))
      (model_mass position controls model head);
    rsum_nonnegative (model_mass position controls model head)

inline_for_extraction noextract
let ratio_product (x y : real) (denominator : real{denominator >. 0.0R})
  : Lemma ((1.0R /. denominator) *. (x *. y) == (x /. denominator) *. y)
  = ()

inline_for_extraction noextract
let scaled_term (factor : real) (f : nat -> GTot real) (i : nat) : GTot real =
  factor *. f i

inline_for_extraction noextract
let normalized_sum (n : nat) (f : nat -> GTot real)
  (denominator : real{denominator >. 0.0R}) (terms : real_vector n)
  : Lemma
      (requires S.init_ghost n (scaled_term (1.0R /. denominator) f) == terms)
      (ensures rsum terms == prefix n f /. denominator)
  = prefix_ext n (scaled_term (1.0R /. denominator) f) (fun i -> (1.0R /. denominator) *. f i);
    prefix_scale n (1.0R /. denominator) f;
    prefix_init n (scaled_term (1.0R /. denominator) f);
    multiply_inverse (prefix n f) denominator

inline_for_extraction noextract
let zero_term (_ : nat) : GTot real = 0.0R

inline_for_extraction noextract
let zero_sum (n : nat) (terms : real_vector n)
  : Lemma
      (requires terms == S.init_ghost n zero_term)
      (ensures rsum terms == 0.0R)
  = prefix_init n zero_term;
    prefix_ext n zero_term (fun _ -> 0.0R);
    prefix_zero n

#push-options "--fuel 4 --z3rlimit 30"
inline_for_extraction noextract
let attention_term_scale (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (i : nat{i < head_width}) (model : online_interpretation position)
  (denominator : real{denominator >. 0.0R}) (time : nat{time <= position})
  : Lemma (
      (1.0R /. denominator) *. selected_numerator_term position controls model head partitions i time ==
      (if partition_included position controls head (time % partitions) then
        (S.index (model_mass position controls model head) time /. denominator) *.
          S.index (snd model) (cache_index (kv_head_for_query head) time i)
      else 0.0R))
  = if partition_included position controls head (time % partitions) then
      ratio_product (exp (S.index (fst model) (head * (position + 1) + time)))
        (S.index (snd model) (cache_index (kv_head_for_query head) time i)) denominator
#pop-options

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let attention_cell_positive (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (i : nat{i < head_width}) (model : online_interpretation position)
  : Lemma
      (requires selected_mass position controls model head partitions >. 0.0R)
      (ensures S.index (attention_head position controls (fst model) (snd model) head) i ==
        selected_numerator position controls model head partitions i /.
          selected_mass position controls model head partitions)
  = selected_mass_model position controls head model;
    let mass = model_mass position controls model head in
    let denominator = rsum mass in
    let terms = attention_terms position controls model head i in
    let numer = selected_numerator_term position controls model head partitions i in
    C.forall_intro (attention_term_scale position controls head i model denominator);
    S.lemma_eq_elim
      (S.init_ghost (position + 1) (scaled_term (1.0R /. denominator) numer)) terms;
    normalized_sum (position + 1) numer denominator terms;
    attention_index_terms position controls model head i;
    assert (prefix (position + 1) numer == selected_numerator position controls model head partitions i)
#pop-options

#push-options "--fuel 4 --z3rlimit 60"
inline_for_extraction noextract
let attention_cell_zero (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (i : nat{i < head_width}) (model : online_interpretation position)
  : Lemma
      (requires selected_mass position controls model head partitions == 0.0R)
      (ensures selected_numerator position controls model head partitions i == 0.0R /\
        S.index (attention_head position controls (fst model) (snd model) head) i == 0.0R)
  = selected_mass_model position controls head model;
    let mass = model_mass position controls model head in
    let omitted (time : nat{time <= position})
      : Lemma (~(partition_included position controls head (time % partitions)))
      = if partition_included position controls head (time % partitions) then
          rsum_member_lower_bound mass time
    in
    C.forall_intro omitted;
    let numer = selected_numerator_term position controls model head partitions i in
    prefix_ext (position + 1) numer (fun _ -> 0.0R);
    prefix_zero (position + 1);
    let terms = attention_terms position controls model head i in
    S.lemma_eq_elim terms (S.init_ghost (position + 1) zero_term);
    zero_sum (position + 1) terms;
    attention_index_terms position controls model head i
#pop-options

inline_for_extraction noextract
let unshift_division (numerator : real) (denominator : real{denominator >. 0.0R}) (scale : real{scale >. 0.0R})
  : Lemma ((numerator /. scale) *. (1.0R /. (denominator /. scale)) == numerator /. denominator)
  = ()

(* This is a ghost case distinction, not a numerical zero-total branch. *)
inline_for_extraction noextract
let attention_division_finish (position : position_index)
  (controls : lseq f32 (score_elements position)) (values : lseq bf16 cache_elements)
  (head : nat{head < query_heads}) (i : nat{i < head_width})
  (maximum numerator total_mass32 inverse result : f32)
  : Lemma
      (requires
        (forall (model : online_interpretation position). online_inputs controls values model ==>
          numerator %~ S.index (shifted_selection position controls model head partitions maximum) i /\
          total_mass32 %~ (selected_mass position controls model head partitions /. exp (to_real maximum))) /\
        (forall (r : real{r =!= 0.0R}). total_mass32 %~ (r <: real) ==> inverse %~ (1.0R /. r)) /\
        (forall (x y : real). numerator %~ x /\ inverse %~ y ==> result %~ (x *. y)))
      (ensures forall (model : online_interpretation position). online_inputs controls values model ==>
        result %~ S.index (attention_head position controls (fst model) (snd model) head) i)
  = let related (model : online_interpretation position{online_inputs controls values model})
      : Lemma (result %~ S.index (attention_head position controls (fst model) (snd model) head) i)
      = selected_mass_model position controls head model;
        let mass = selected_mass position controls model head partitions in
        let numer = selected_numerator position controls model head partitions i in
        let scale = exp (to_real maximum) in
        if t2b (mass == 0.0R) then begin
          attention_cell_zero position controls head i model;
          Approx.to_real_ok inverse;
          assert (numerator %~ 0.0R);
          assert (result %~ (0.0R *. to_real inverse))
        end else begin
          attention_cell_positive position controls head i model;
          assert (total_mass32 %~ (mass /. scale));
          assert (inverse %~ (1.0R /. (mass /. scale)));
          assert (numerator %~ (numer /. scale));
          assert (result %~ ((numer /. scale) *. (1.0R /. (mass /. scale))));
          unshift_division numer mass scale
        end
    in C.forall_intro related

inline_for_extraction noextract
fn divide_attention
  (position : i64{valid_position (I64.v position)})
  (head : sz{head < query_heads}) (maximum total_mass32 : f32) (output : larray f32 head_width)
  (#controls : erased (lseq f32 (score_elements (I64.v position))))
  (#values : erased (lseq bf16 cache_elements)) (#before : erased (lseq f32 head_width))
  preserves gpu
  requires output |-> before **
    pure (merge_invariant (I64.v position) controls values (SZ.v head) partitions maximum total_mass32 before)
  ensures exists* (after : lseq f32 head_width). output |-> after **
    pure (forall (model : online_interpretation (I64.v position)). online_inputs controls values model ==>
      after %~ attention_head (I64.v position) controls (fst model) (snd model) (SZ.v head))
{
  let inverse = F32.rcp_approx_ftz total_mass32;
  let mut coordinate = 0sz;
  while (!coordinate <^ 256sz)
    invariant exists* (i : sz{i <= head_width}) (current : lseq f32 head_width).
      coordinate |-> i ** output |-> current **
      pure (forall (j : nat{SZ.v i <= j /\ j < head_width}). S.index current j == S.index before j) **
      pure (forall (model : online_interpretation (I64.v position)). online_inputs controls values model ==>
        (forall (j : nat{j < SZ.v i}). S.index current j %~
          S.index (attention_head (I64.v position) controls (fst model) (snd model) (SZ.v head)) j))
    decreases (head_width - !coordinate)
  {
    let i = !coordinate;
    let numerator = output.(i);
    let result = F32.mul_rn_ftz numerator inverse;
    attention_division_finish (I64.v position) controls values (SZ.v head) (SZ.v i)
      maximum numerator total_mass32 inverse result;
    output.(i) <- result;
    coordinate := i +^ 1sz;
  }
}

inline_for_extraction noextract
fn attend_head
  (position : i64{valid_position (I64.v position)})
  (scores : larray f32 (score_elements (I64.v position))) (values : larray bf16 cache_elements)
  (head : sz{head < query_heads}) (output : larray f32 head_width)
  (#sv : erased (lseq f32 (score_elements (I64.v position))))
  (#vv : erased (lseq bf16 cache_elements)) (#old : erased (lseq f32 head_width)) (#fs #fv : perm)
  preserves gpu ** scores |-> Frac fs sv ** values |-> Frac fv vv
  requires output |-> old
  ensures exists* (after : lseq f32 head_width). output |-> after **
    pure (forall (model : online_interpretation (I64.v position)). online_inputs sv vv model ==>
      after %~ attention_head (I64.v position) sv (fst model) (snd model) (SZ.v head))
{
  let mut maxima = [| zero #f32 #_; 16sz |];
  let mut denominators = [| zero #f32 #_; 16sz |];
  let mut partial = [| zero #f32 #_; 4096sz |];
  let mut total_mass32 = [| zero #f32 #_; 1sz |];
  collect_partitions position scores values head maxima denominators partial;
  let maximum = select_maximum maxima;
  merge_partitions position head maximum maxima denominators partial total_mass32 output #sv #vv;
  let denominator = total_mass32.(0sz);
  divide_attention position head maximum denominator output #sv #vv;
}

inline_for_extraction noextract
let gated_interpretation (position : position_index) =
  online_interpretation position & real_vector query_gate_width

inline_for_extraction noextract
let gated_inputs (#position : position_index) (scores : lseq f32 (score_elements position))
  (values : lseq bf16 cache_elements) (gates : lseq f32 query_gate_width)
  (model : gated_interpretation position) : prop =
  online_inputs scores values (fst model) /\ gates %~ snd model

inline_for_extraction noextract
let gated_head (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (model : gated_interpretation position)
  : GTot (real_vector head_width) =
  let attended = attention_head position controls (fst (fst model)) (snd (fst model)) head in
  S.init_ghost head_width (fun i ->
    S.index attended i *. sigmoid (S.index (snd model) (head * (2 * head_width) + head_width + i)))

inline_for_extraction noextract
let gated_attention (position : position_index) (controls : lseq f32 (score_elements position))
  (model : gated_interpretation position) : GTot (real_vector activation_width) =
  S.init_ghost activation_width (fun i ->
    S.index (gated_head position controls (i / head_width) model) (i % head_width))

inline_for_extraction noextract
let gated_head_slice (position : position_index) (controls : lseq f32 (score_elements position))
  (head : nat{head < query_heads}) (model : gated_interpretation position)
  : Lemma (vector_slice (gated_attention position controls model) (head * head_width) head_width ==
      gated_head position controls head model)
  = S.lemma_eq_elim (vector_slice (gated_attention position controls model) (head * head_width) head_width)
      (gated_head position controls head model)

inline_for_extraction noextract
fn gate_attention_head (position : i64{valid_position (I64.v position)})
  (gates : larray f32 query_gate_width) (output : larray f32 head_width)
  (head : sz{head < query_heads})
  (#controls : erased (lseq f32 (score_elements (I64.v position))))
  (#values : erased (lseq bf16 cache_elements)) (#gv : erased (lseq f32 query_gate_width))
  (#before : erased (lseq f32 head_width)) (#fg : perm)
  preserves gpu ** gates |-> Frac fg gv
  requires output |-> before **
    pure (forall (model : gated_interpretation (I64.v position)). gated_inputs controls values gv model ==>
      before %~ attention_head (I64.v position) controls (fst (fst model)) (snd (fst model)) (SZ.v head))
  ensures exists* (after : lseq f32 head_width). output |-> after **
    pure (forall (model : gated_interpretation (I64.v position)). gated_inputs controls values gv model ==>
      after %~ gated_head (I64.v position) controls (SZ.v head) model)
{
  let mut coordinate = 0sz;
  while (!coordinate <^ 256sz)
    invariant exists* (i : sz{i <= head_width}) (current : lseq f32 head_width).
      coordinate |-> i ** output |-> current **
      pure (forall (j : nat{SZ.v i <= j /\ j < head_width}). S.index current j == S.index before j) **
      pure (forall (model : gated_interpretation (I64.v position)). gated_inputs controls values gv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index current j %~ S.index (gated_head (I64.v position) controls (SZ.v head) model) j))
    decreases (head_width - !coordinate)
  {
    let i = !coordinate;
    let prior = output.(i);
    let gate = gates.(head *^ 512sz +^ 256sz +^ i);
    let activated = sigmoid32 gate;
    let result = F32.mul_rn_ftz prior activated;
    lift_mul #(gated_interpretation (I64.v position)) (gated_inputs #(I64.v position) controls values gv)
      (fun model -> S.index (attention_head (I64.v position) controls (fst (fst model)) (snd (fst model))
        (SZ.v head)) (SZ.v i))
      (fun model -> sigmoid (S.index (snd model) (SZ.v head * (2 * head_width) + head_width + SZ.v i)))
      prior activated result;
    output.(i) <- result;
    coordinate := i +^ 1sz;
  }
}

inline_for_extraction noextract
let attended_lift (position : position_index) (controls : lseq f32 (score_elements position))
  (values : lseq bf16 cache_elements) (gates : lseq f32 query_gate_width)
  (head : nat{head < query_heads}) (output : lseq f32 head_width)
  : Lemma
      (requires forall (model : online_interpretation position). online_inputs controls values model ==>
        output %~ attention_head position controls (fst model) (snd model) head)
      (ensures forall (model : gated_interpretation position). gated_inputs controls values gates model ==>
        output %~ attention_head position controls (fst (fst model)) (snd (fst model)) head)
  = let related (model : gated_interpretation position{gated_inputs controls values gates model})
      : Lemma (output %~ attention_head position controls (fst (fst model)) (snd (fst model)) head)
      = assert (online_inputs controls values (fst model))
    in C.forall_intro related

[@@CPrologue "__device__"]
fn attend_all (position : i64{valid_position (I64.v position)})
  (scores : larray f32 (score_elements (I64.v position))) (values : larray bf16 cache_elements)
  (gates : larray f32 query_gate_width) (output : larray f32 activation_width)
  (#sv : erased (lseq f32 (score_elements (I64.v position))))
  (#vv : erased (lseq bf16 cache_elements)) (#gv : erased (lseq f32 query_gate_width))
  (#old : erased (lseq f32 activation_width)) (#fs #fv #fg : perm)
  preserves gpu ** scores |-> Frac fs sv ** values |-> Frac fv vv ** gates |-> Frac fg gv
  requires output |-> old
  ensures exists* (after : lseq f32 activation_width). output |-> after **
    pure (forall (model : gated_interpretation (I64.v position)). gated_inputs sv vv gv model ==>
      after %~ gated_attention (I64.v position) sv model)

inline_for_extraction noextract
let tail_normalized (model : matrix_interpretation hidden_width) : GTot (real_vector hidden_width) =
  normalization (fst model) post_norm_offset (snd model)
inline_for_extraction noextract
let tail_gate (model : matrix_interpretation hidden_width) : GTot (real_vector mlp_width) =
  linear (fst model) mlp_gate_offset mlp_width hidden_width (tail_normalized model)
inline_for_extraction noextract
let tail_up (model : matrix_interpretation hidden_width) : GTot (real_vector mlp_width) =
  linear (fst model) mlp_up_offset mlp_width hidden_width (tail_normalized model)
inline_for_extraction noextract
let tail_activation (model : matrix_interpretation hidden_width) : GTot (real_vector mlp_width) =
  mlp_vector (tail_gate model) (tail_up model)
inline_for_extraction noextract
let tail_down (model : matrix_interpretation hidden_width) : GTot (real_vector hidden_width) =
  linear (fst model) mlp_down_offset hidden_width mlp_width (tail_activation model)
inline_for_extraction noextract
let tail_result (model : matrix_interpretation hidden_width) : GTot (real_vector hidden_width) =
  residual_vector (snd model) (tail_down model)

inline_for_extraction noextract
let tail_alignment (model : matrix_interpretation hidden_width)
  : Lemma (tail_result model == mlp_residual (fst model) (snd model))
  = assert (tail_result model == mlp_residual (fst model) (snd model))
    by (FStar.Tactics.V2.norm [delta_only [
      `%tail_result; `%tail_down; `%tail_activation; `%tail_gate; `%tail_up; `%tail_normalized;
      `%residual_vector; `%mlp_vector; `%mlp_residual]; zeta; iota];
      FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let tail_post_equiv (weights : lseq bf16 weight_elements) (before after : lseq bf16 hidden_width)
  : Lemma (
      (forall (model : matrix_interpretation hidden_width).
        normalization_inputs weights before model ==>
        after %~ residual_vector (snd model) (tail_down model)) ==
      (forall (model : matrix_interpretation hidden_width).
        normalization_inputs weights before model ==> after %~ mlp_residual (fst model) (snd model))
  )
  = assert (
      (forall (model : matrix_interpretation hidden_width).
        normalization_inputs weights before model ==>
        after %~ residual_vector (snd model) (tail_down model)) ==
      (forall (model : matrix_interpretation hidden_width).
        normalization_inputs weights before model ==> after %~ mlp_residual (fst model) (snd model)))
    by (FStar.Tactics.V2.norm [delta_only [
      `%tail_down; `%tail_activation; `%tail_gate; `%tail_up; `%tail_normalized;
      `%residual_vector; `%mlp_vector; `%mlp_residual]; zeta; iota];
      FStar.Tactics.V2.trefl ())

[@@CPrologue "__device__"]
fn finish_mlp (weights : larray bf16 weight_elements) (hidden : larray bf16 hidden_width)
  (#wv : erased (lseq bf16 weight_elements)) (#before : erased (lseq bf16 hidden_width)) (#fw : perm)
  preserves gpu ** weights |-> Frac fw wv
  requires hidden |-> before
  ensures exists* (after : lseq bf16 hidden_width). hidden |-> after **
    pure (forall (model : matrix_interpretation hidden_width). normalization_inputs wv before model ==>
      after %~ mlp_residual (fst model) (snd model))

noeq noextract
type layer_interpretation (position : position_index) = {
  layer_weights : real_vector weight_elements;
  layer_hidden : real_vector hidden_width;
  layer_previous_keys : real_vector (previous_elements position);
  layer_previous_values : real_vector (previous_elements position)
}

inline_for_extraction noextract
let layer_inputs (#position : position_index)
  (weights : lseq bf16 weight_elements) (hidden : lseq bf16 hidden_width)
  (keys values : lseq bf16 (previous_elements position)) (model : layer_interpretation position)
  : prop = weights %~ model.layer_weights /\ hidden %~ model.layer_hidden /\
    keys %~ model.layer_previous_keys /\ values %~ model.layer_previous_values

inline_for_extraction noextract
let layer_normalized (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector hidden_width) =
  normalization model.layer_weights input_norm_offset model.layer_hidden
inline_for_extraction noextract
let layer_query_gate (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector query_gate_width) =
  linear model.layer_weights query_gate_offset query_gate_width hidden_width (layer_normalized model)
inline_for_extraction noextract
let layer_key_projection (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector kv_width) =
  linear model.layer_weights key_offset kv_width hidden_width (layer_normalized model)
inline_for_extraction noextract
let layer_value_projection (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector kv_width) =
  linear model.layer_weights value_offset kv_width hidden_width (layer_normalized model)
inline_for_extraction noextract
let layer_current_keys (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector kv_width) =
  transformed_heads false position (model.layer_weights, layer_key_projection model)
inline_for_extraction noextract
let layer_queries (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector activation_width) =
  transformed_heads true position (model.layer_weights, layer_query_gate model)
inline_for_extraction noextract
let layer_keys (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector cache_elements) =
  cache_update position model.layer_previous_keys (layer_current_keys model)
inline_for_extraction noextract
let layer_values (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector cache_elements) =
  cache_update position model.layer_previous_values (layer_value_projection model)
inline_for_extraction noextract
let layer_scores (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector (score_elements position)) =
  attention_scores position (layer_queries model) (layer_keys model)
inline_for_extraction noextract
let layer_attention_inputs (#position : position_index) (model : layer_interpretation position)
  : GTot (gated_interpretation position) =
  ((layer_scores model, layer_values model), layer_query_gate model)
inline_for_extraction noextract
let layer_activation (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : GTot (real_vector activation_width) =
  gated_attention position controls (layer_attention_inputs model)
inline_for_extraction noextract
let layer_projected (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : GTot (real_vector hidden_width) =
  linear model.layer_weights output_offset hidden_width activation_width (layer_activation controls model)
inline_for_extraction noextract
let layer_residual (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : GTot (real_vector hidden_width) =
  residual_vector model.layer_hidden (layer_projected controls model)
inline_for_extraction noextract
let layer_result (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : GTot (real_vector hidden_width) =
  mlp_residual model.layer_weights (layer_residual controls model)

inline_for_extraction noextract
let lift_transform (#p : Type) (query : bool) (position : position_index)
  (premise : p -> prop) (wm : p -> GTot (real_vector weight_elements))
  (xm : p -> GTot (real_vector (transform_inputs query)))
  (wv : lseq bf16 weight_elements) (xv : lseq f32 (transform_inputs query))
  (out : lseq f32 (transform_outputs query))
  : Lemma
      (requires (forall (model : p). premise model ==> wv %~ wm model /\ xv %~ xm model) /\
        (forall (model : matrix_interpretation (transform_inputs query)). matrix_inputs wv xv model ==>
          out %~ transformed_heads query position model))
      (ensures forall (model : p). premise model ==>
        out %~ transformed_heads query position (wm model, xm model))
  = let related (model : p{premise model})
      : Lemma (out %~ transformed_heads query position (wm model, xm model))
      = assert (matrix_inputs wv xv (wm model, xm model))
    in C.forall_intro related

inline_for_extraction noextract
let lift_cache (#p : Type) (position : position_index) (premise : p -> prop)
  (previous_model : p -> GTot (real_vector (previous_elements position)))
  (current_model : p -> GTot (real_vector kv_width))
  (previous : lseq bf16 (previous_elements position)) (current : lseq f32 kv_width)
  : Lemma
      (requires forall (model : p). premise model ==>
        previous %~ previous_model model /\ current %~ current_model model)
      (ensures forall (model : p). premise model ==>
        physical_cache position previous current %~ cache_update position (previous_model model) (current_model model))
  = physical_cache_relation position previous current;
    let related (model : p{premise model})
      : Lemma (physical_cache position previous current %~
          cache_update position (previous_model model) (current_model model)) = ()
    in C.forall_intro related

inline_for_extraction noextract
let lift_scores (#p : Type) (position : position_index) (premise : p -> prop)
  (qm : p -> GTot (real_vector activation_width)) (km : p -> GTot (real_vector cache_elements))
  (qv : lseq f32 activation_width) (kv : lseq bf16 cache_elements)
  (out : lseq f32 (score_elements position))
  : Lemma
      (requires (forall (model : p). premise model ==> qv %~ qm model /\ kv %~ km model) /\
        (forall (model : score_interpretation). score_inputs qv kv model ==>
          out %~ attention_scores position (fst model) (snd model)))
      (ensures forall (model : p). premise model ==> out %~ attention_scores position (qm model) (km model))
  = let related (model : p{premise model})
      : Lemma (out %~ attention_scores position (qm model) (km model))
      = assert (score_inputs qv kv (qm model, km model))
    in C.forall_intro related

inline_for_extraction noextract
let lift_tail (#p : Type) (premise : p -> prop)
  (wm : p -> GTot (real_vector weight_elements)) (hm : p -> GTot (real_vector hidden_width))
  (wv : lseq bf16 weight_elements) (hv out : lseq bf16 hidden_width)
  : Lemma
      (requires (forall (model : p). premise model ==> wv %~ wm model /\ hv %~ hm model) /\
        (forall (model : matrix_interpretation hidden_width). normalization_inputs wv hv model ==>
          out %~ mlp_residual (fst model) (snd model)))
      (ensures forall (model : p). premise model ==> out %~ mlp_residual (wm model) (hm model))
  = let related (model : p{premise model})
      : Lemma (out %~ mlp_residual (wm model) (hm model))
      = assert (normalization_inputs wv hv (wm model, hm model))
    in C.forall_intro related

inline_for_extraction noextract
let model_current_keys (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector kv_width) =
  let key_heads = S.init_ghost kv_heads (fun head ->
      key_transform model.layer_weights position
        (vector_slice (layer_key_projection model) (head * head_width) head_width)) in
  S.init_ghost kv_width (fun i -> S.index (S.index key_heads (i / head_width)) (i % head_width))

inline_for_extraction noextract
let model_queries (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector activation_width) =
  let query_heads_values = S.init_ghost query_heads (fun head ->
      query_transform model.layer_weights position
        (vector_slice (layer_query_gate model) (head * (2 * head_width)) head_width)) in
  S.init_ghost activation_width (fun i ->
    S.index (S.index query_heads_values (i / head_width)) (i % head_width))

inline_for_extraction noextract
let model_keys (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector cache_elements) =
  cache_update position model.layer_previous_keys (model_current_keys model)

inline_for_extraction noextract
let model_scores (#position : position_index) (model : layer_interpretation position)
  : GTot (real_vector (score_elements position)) =
  attention_scores position (model_queries model) (model_keys model)

inline_for_extraction noextract
let model_activation (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : GTot (real_vector activation_width) =
  let attended = S.init_ghost query_heads (fun head ->
      attention_head position controls (model_scores model) (layer_values model) head) in
  S.init_ghost activation_width (fun i ->
      S.index (S.index attended (i / head_width)) (i % head_width) *.
        sigmoid (S.index (layer_query_gate model)
          ((i / head_width) * (2 * head_width) + head_width + i % head_width)))

inline_for_extraction noextract
let model_hidden (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : GTot (real_vector hidden_width) =
  mlp_residual model.layer_weights (residual_vector model.layer_hidden
    (linear model.layer_weights output_offset hidden_width activation_width (model_activation controls model)))

inline_for_extraction noextract
let reconstructed_model (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : GTot (attention_values position) = {
    score_result = model_scores model;
    hidden_result = model_hidden controls model;
    key_result = model_keys model;
    value_result = layer_values model
  }

inline_for_extraction noextract
let full_reconstruction (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : Lemma (execution_model model.layer_weights model.layer_hidden position
      model.layer_previous_keys model.layer_previous_values controls == reconstructed_model controls model)
  = assert (execution_model model.layer_weights model.layer_hidden position
      model.layer_previous_keys model.layer_previous_values controls == reconstructed_model controls model)
    by (FStar.Tactics.V2.norm [delta_only [
      `%execution_model; `%decoder_prefix_model; `%decoder_activation; `%reconstructed_model; `%model_hidden; `%residual_vector;
      `%model_activation; `%model_scores; `%model_keys; `%model_queries; `%model_current_keys;
      `%layer_values; `%layer_value_projection; `%layer_key_projection; `%layer_query_gate;
      `%layer_normalized]; zeta; iota];
      FStar.Tactics.V2.trefl ())

#push-options "--fuel 6 --z3rlimit 60"
inline_for_extraction noextract
let nested_transforms_alignment (#position : position_index) (model : layer_interpretation position)
  : Lemma (model_current_keys model == layer_current_keys model /\ model_queries model == layer_queries model)
  = S.lemma_eq_elim (model_current_keys model) (layer_current_keys model);
    S.lemma_eq_elim (model_queries model) (layer_queries model)
#pop-options

inline_for_extraction noextract
let model_prefix_alignment (#position : position_index) (model : layer_interpretation position)
  : Lemma (model_keys model == layer_keys model /\ model_scores model == layer_scores model)
  = nested_transforms_alignment model

#push-options "--fuel 6 --z3rlimit 60"
inline_for_extraction noextract
let model_activation_alignment (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : Lemma (model_activation controls model == layer_activation controls model)
  = model_prefix_alignment model;
    FStar.Pervasives.norm_spec [delta_only [
      `%layer_activation; `%layer_attention_inputs; `%gated_attention; `%gated_head]; zeta; iota]
      (layer_activation controls model);
    S.lemma_eq_elim (model_activation controls model) (layer_activation controls model)
#pop-options

inline_for_extraction noextract
let model_hidden_alignment (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : Lemma (model_hidden controls model == layer_result controls model)
  = model_activation_alignment controls model;
    FStar.Pervasives.norm_spec [delta_only [`%model_hidden]; zeta; iota] (model_hidden controls model);
    FStar.Pervasives.norm_spec [delta_only [`%layer_result; `%layer_residual; `%layer_projected]; zeta; iota]
      (layer_result controls model)

inline_for_extraction noextract
let layer_model_alignment (#position : position_index)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : Lemma (
      let full = execution_model model.layer_weights model.layer_hidden position
        model.layer_previous_keys model.layer_previous_values controls in
      full.score_result == layer_scores model /\
      full.key_result == layer_keys model /\
      full.value_result == layer_values model /\
      full.hidden_result == layer_result controls model)
  = full_reconstruction controls model;
    model_prefix_alignment model;
    model_hidden_alignment controls model

inline_for_extraction noextract
let layer_queries_prefix (#position:position_index) (model:layer_interpretation position)
  : Lemma (
      layer_queries model ==
      (decoder_prefix_model model.layer_weights model.layer_hidden position
        model.layer_previous_keys model.layer_previous_values).prefix_queries) =
  nested_transforms_alignment model;
  assert (
      model_queries model ==
      (decoder_prefix_model model.layer_weights model.layer_hidden position
        model.layer_previous_keys model.layer_previous_values).prefix_queries)
    by (Proof.norm [delta_only [
          `%decoder_prefix_model; `%model_queries; `%layer_query_gate; `%layer_normalized]; zeta; iota];
        Proof.trefl ())

noextract
let numerical_post (position : position_index)
  (weights : lseq bf16 weight_elements) (hidden : lseq bf16 hidden_width)
  (previous_keys previous_values : lseq bf16 (previous_elements position))
  (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
  (controls : lseq f32 (score_elements position)) : prop =
  forall (rw : real_vector weight_elements) (rh : real_vector hidden_width)
    (rk rv : real_vector (previous_elements position)).
    weights %~ rw /\ hidden %~ rh /\ previous_keys %~ rk /\ previous_values %~ rv ==>
    (let full = execution_model rw rh position rk rv controls in
      controls %~ full.score_result /\ result %~ full.hidden_result /\
      keys %~ full.key_result /\ values %~ full.value_result)

(* Keep source provenance intact without expanding it during kernel capture. *)
noextract
let observed_execution_post (position:position_index)
  (weights:bfseq weight_elements) (hidden:bfseq hidden_width)
  (previous_keys previous_values:bfseq (previous_elements position))
  (result:bfseq hidden_width) (keys values:bfseq cache_elements)
  (observation:score_observation position) : prop =
  retained_cache_post position previous_keys keys /\
  retained_cache_post position previous_values values /\
  observation.observed_keys == keys /\
  numerical_post position weights hidden previous_keys previous_values result keys values
    observation.observed_scores /\
  (forall (model:layer_interpretation position).
    layer_inputs weights hidden previous_keys previous_values model ==>
    observation.observed_queries %~ layer_queries model)

inline_for_extraction noextract
let layer_values_approx (#position : position_index)
  (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : prop =
  controls %~ layer_scores model /\ result %~ layer_result controls model /\
  keys %~ layer_keys model /\ values %~ layer_values model

inline_for_extraction noextract
let layer_values_post (#position : position_index)
  (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : prop =
  let full = execution_model model.layer_weights model.layer_hidden position
    model.layer_previous_keys model.layer_previous_values controls in
  controls %~ full.score_result /\ result %~ full.hidden_result /\
  keys %~ full.key_result /\ values %~ full.value_result

inline_for_extraction noextract
let layer_post_alignment (#position : position_index)
  (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
  (controls : lseq f32 (score_elements position)) (model : layer_interpretation position)
  : Lemma (layer_values_approx result keys values controls model ==>
      layer_values_post result keys values controls model)
  = layer_model_alignment controls model

inline_for_extraction noextract
let forall_layer_values (position : position_index) (pred : layer_interpretation position -> prop)
  : Lemma
      (requires forall (model : layer_interpretation position). pred model)
      (ensures forall (rw : real_vector weight_elements) (rh : real_vector hidden_width)
        (rk rv : real_vector (previous_elements position)).
        pred ({ layer_weights = rw; layer_hidden = rh;
          layer_previous_keys = rk; layer_previous_values = rv } <: layer_interpretation position))
  = let related (rw : real_vector weight_elements) (rh : real_vector hidden_width)
      (rk rv : real_vector (previous_elements position))
      : Lemma (pred ({ layer_weights = rw; layer_hidden = rh;
          layer_previous_keys = rk; layer_previous_values = rv } <: layer_interpretation position))
      = ()
    in C.forall_intro_4 related

inline_for_extraction noextract
let layer_record_post (position : position_index)
  (weights : lseq bf16 weight_elements) (hidden : lseq bf16 hidden_width)
  (previous_keys previous_values : lseq bf16 (previous_elements position))
  (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
  (controls : lseq f32 (score_elements position)) : prop =
  forall (rw : real_vector weight_elements) (rh : real_vector hidden_width)
    (rk rv : real_vector (previous_elements position)).
    let model : layer_interpretation position = {
      layer_weights = rw; layer_hidden = rh; layer_previous_keys = rk; layer_previous_values = rv
    } in
    layer_inputs weights hidden previous_keys previous_values model ==>
    layer_values_post result keys values controls model

inline_for_extraction noextract
let layer_record_post_equiv (position : position_index)
  (weights : lseq bf16 weight_elements) (hidden : lseq bf16 hidden_width)
  (previous_keys previous_values : lseq bf16 (previous_elements position))
  (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
  (controls : lseq f32 (score_elements position))
  : Lemma (layer_record_post position weights hidden previous_keys previous_values result keys values controls ==
      numerical_post position weights hidden previous_keys previous_values result keys values controls)
  = assert (layer_record_post position weights hidden previous_keys previous_values result keys values controls ==
      numerical_post position weights hidden previous_keys previous_values result keys values controls)
    by (FStar.Tactics.V2.norm [delta_only [
      `%layer_record_post; `%layer_inputs; `%layer_values_post; `%numerical_post]; zeta; iota];
      FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let layer_finish (position : position_index)
  (weights : lseq bf16 weight_elements) (hidden : lseq bf16 hidden_width)
  (previous_keys previous_values : lseq bf16 (previous_elements position))
  (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
  (controls : lseq f32 (score_elements position))
  : Lemma
      (requires retained_cache_post position previous_keys keys /\
        retained_cache_post position previous_values values /\
        (forall (model : layer_interpretation position).
          layer_inputs weights hidden previous_keys previous_values model ==>
          controls %~ layer_scores model /\ result %~ layer_result controls model /\
          keys %~ layer_keys model /\ values %~ layer_values model))
      (ensures execution_post weights hidden position previous_keys previous_values result keys values /\
        numerical_post position weights hidden previous_keys previous_values result keys values controls)
  = C.forall_intro (layer_post_alignment #position result keys values controls);
    compose_relations #(layer_interpretation position)
      (layer_inputs #position weights hidden previous_keys previous_values)
      (layer_values_approx #position result keys values controls)
      (layer_values_post #position result keys values controls);
    forall_layer_values position (fun model ->
      layer_inputs #position weights hidden previous_keys previous_values model ==>
      layer_values_post #position result keys values controls model);
    assert (layer_record_post position weights hidden previous_keys previous_values result keys values controls);
    layer_record_post_equiv position weights hidden previous_keys previous_values result keys values controls;
    C.exists_intro #(lseq f32 (score_elements position))
      (numerical_post position weights hidden previous_keys previous_values result keys values) controls

noextract
let source_bound_post
  (input:logical_inputs)
  (weights:bfseq weight_elements) (hidden:bfseq hidden_width)
  (previous_keys previous_values:bfseq (previous_elements input.call.position))
  (result:bfseq hidden_width) (keys values:bfseq cache_elements)
  (observation:score_observation input.call.position)
  : Lemma
      (requires input_representation input weights hidden previous_keys previous_values /\
        observed_execution_post input.call.position weights hidden previous_keys previous_values
          result keys values observation)
      (ensures source_score_post input previous_keys previous_values result keys values
        observation.observed_scores) =
  FStar.Pervasives.norm_spec [delta_only [`%observed_execution_post]; zeta; iota]
    (observed_execution_post input.call.position weights hidden previous_keys previous_values
      result keys values observation);
  representation_sequence weight_elements weights input.packed_weights;
  representation_sequence hidden_width hidden input.input_hidden;
  representation_sequence (previous_elements input.call.position) previous_keys input.old_keys;
  representation_sequence (previous_elements input.call.position) previous_values input.old_values;
  let model:layer_interpretation input.call.position = {
    layer_weights = sequence_of input.packed_weights;
    layer_hidden = sequence_of input.input_hidden;
    layer_previous_keys = sequence_of input.old_keys;
    layer_previous_values = sequence_of input.old_values
  } in
  assert (layer_inputs weights hidden previous_keys previous_values model);
  layer_queries_prefix model;
  decoder_prefix_shared input;
  let query (h:Sh.index query_heads) (d:Sh.index head_width)
    : Lemma (v_approximates (S.index observation.observed_queries (h * head_width + d))
        (shared_queries input h d)) = ()
  in
  C.forall_intro_2 query;
  unconditional_state_post input previous_keys previous_values keys values;
  let numerical (_:bilateral_live_scores_finite input.call.position observation.observed_scores)
    : Lemma (semantic_postcondition input previous_keys previous_values result keys values) =
    semantic_from_model input previous_keys previous_values result keys values observation.observed_scores
  in C.impl_intro numerical

(* Keep the complete device body out of host kernel-capture normalization. *)
[@@CPrologue "__device__"]
fn family_kernel
  (#input:erased logical_inputs)
  (input0 : larray bf16 weight_elements{is_global_array input0})
  (input1 : larray bf16 hidden_width{is_global_array input1})
  (input2:array bf16{state_pointer input2 input.call.position})
  (input3:array bf16{state_pointer input3 input.call.position})
  (input4:i64{I64.v input4 == input.call.position})
  (output0 : larray bf16 hidden_width{is_global_array output0})
  (output1 : larray bf16 cache_elements{is_global_array output1})
  (output2 : larray bf16 cache_elements{is_global_array output2})
  (scores:larray f32 (score_elements (I64.v input4)){is_global_array scores})
  (#weights : erased (lseq bf16 weight_elements)) (#hidden : erased (lseq bf16 hidden_width))
  (#previous_keys #previous_values:erased (bfseq (previous_elements (I64.v input4))))
  (#old_scores:erased (lseq f32 (score_elements (I64.v input4))))
  (#fweights #fhidden #fkeys #fvalues : perm)
  preserves gpu ** input0 |-> Frac fweights weights ** input1 |-> Frac fhidden hidden **
    state_ownership input2 (I64.v input4) fkeys previous_keys **
    state_ownership input3 (I64.v input4) fvalues previous_values
  requires pure (input_representation input weights hidden previous_keys previous_values)
  requires A.pts_to_uninit output0 hidden_width **
    A.pts_to_uninit output1 cache_elements ** A.pts_to_uninit output2 cache_elements **
    scores |-> old_scores
  ensures exists* (result : lseq bf16 hidden_width) (keys values : lseq bf16 cache_elements)
    (live_scores:lseq f32 (score_elements input.call.position)).
    output0 |-> result ** output1 |-> keys ** output2 |-> values **
    scores |-> live_scores **
    pure (source_score_post input previous_keys previous_values result keys values live_scores)
{
  let mut normalized = [| zero #f32 #_; 1024sz |];
  let mut query_gate = [| zero #f32 #_; 4096sz |];
  let mut key_projection = [| zero #f32 #_; 512sz |];
  let mut value_projection = [| zero #f32 #_; 512sz |];
  let mut current_keys = [| zero #f32 #_; 512sz |];
  let mut queries = [| zero #f32 #_; 2048sz |];
  let mut activation = [| zero #f32 #_; 2048sz |];
  let mut projection = [| zero #f32 #_; 1024sz |];
  normalize_hidden input0 input1 normalized 0sz;
  with nv. assert normalized |-> nv;
  lift_normalization #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) (fun model -> model.layer_hidden) input_norm_offset weights hidden nv;
  matrix_project #hidden_width #query_gate_width input0 normalized query_gate
    1024sz 4096sz (SZ.uint32_to_sizet 11012096ul) true;
  with qg. assert query_gate |-> qg;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #query_gate_width
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) layer_normalized query_gate_offset weights nv qg;
  matrix_project #hidden_width #kv_width input0 normalized key_projection
    1024sz 512sz (SZ.uint32_to_sizet 15206400ul) true;
  with kp. assert key_projection |-> kp;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #kv_width
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) layer_normalized key_offset weights nv kp;
  matrix_project #hidden_width #kv_width input0 normalized value_projection
    1024sz 512sz (SZ.uint32_to_sizet 15730688ul) true;
  with vp. assert value_projection |-> vp;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #kv_width
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) layer_normalized value_offset weights nv vp;
  transform_all_heads false input0 key_projection current_keys input4;
  with ck. assert current_keys |-> ck;
  lift_transform #(layer_interpretation (I64.v input4)) false (I64.v input4)
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) layer_key_projection weights kp ck;
  store_cache input2 output1 current_keys input4;
  store_cache input3 output2 value_projection input4;
  lift_cache #(layer_interpretation (I64.v input4)) (I64.v input4)
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_previous_keys) layer_current_keys previous_keys ck;
  lift_cache #(layer_interpretation (I64.v input4)) (I64.v input4)
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_previous_values) layer_value_projection previous_values vp;
  physical_cache_relation (I64.v input4) previous_keys ck;
  physical_cache_relation (I64.v input4) previous_values vp;
  with keys. assert output1 |-> keys;
  with values. assert output2 |-> values;
  transform_all_heads true input0 query_gate queries input4;
  with qv. assert queries |-> qv;
  lift_transform #(layer_interpretation (I64.v input4)) true (I64.v input4)
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) layer_query_gate weights qg qv;
  fill_scores input4 queries output1 scores #qv #keys #old_scores;
  with controls. assert scores |-> controls;
  lift_scores #(layer_interpretation (I64.v input4)) (I64.v input4)
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    layer_queries layer_keys qv keys controls;
  attend_all input4 scores output2 query_gate activation;
  with av. assert activation |-> av;
  map_relations #(layer_interpretation (I64.v input4)) #(gated_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (gated_inputs #(I64.v input4) controls values qg)
    (fun model -> av %~ gated_attention (I64.v input4) controls model)
    (layer_attention_inputs #(I64.v input4));
  matrix_project #activation_width #hidden_width input0 activation projection
    2048sz 1024sz (SZ.uint32_to_sizet 16255488ul) false;
  with projected. assert projection |-> projected;
  lift_matrix #(layer_interpretation (I64.v input4)) #activation_width #hidden_width
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) (layer_activation controls) output_offset weights av projected;
  copy_hidden input1 output0;
  add_residual output0 projection;
  with residual. assert output0 |-> residual;
  lift_residual #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_hidden) (layer_projected controls) hidden projected residual;
  finish_mlp input0 output0;
  with result. assert output0 |-> result;
  lift_tail #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_keys previous_values)
    (fun model -> model.layer_weights) (layer_residual controls) weights residual result;
  layer_finish (I64.v input4) weights hidden previous_keys previous_values result keys values controls;
  assert pure (observed_execution_post (I64.v input4) weights hidden
    previous_keys previous_values result keys values
    { observed_queries = qv; observed_keys = keys; observed_scores = controls });
  let _ = source_bound_post input weights hidden previous_keys previous_values result keys values
    { observed_queries = qv; observed_keys = keys; observed_scores = controls };
  assert pure (source_score_post input previous_keys previous_values result keys values controls);
}

(* A concrete public definition, not an implementer-chosen observation rule. *)
fn kplay_run
  (#input:erased logical_inputs)
  (input0:larray bf16 weight_elements{is_global_array input0})
  (input1:larray bf16 hidden_width{is_global_array input1})
  (input2:array bf16{state_pointer input2 input.call.position})
  (input3:array bf16{state_pointer input3 input.call.position})
  (input4:i64{I64.v input4 == input.call.position})
  (output0:larray bf16 hidden_width{is_global_array output0})
  (output1:larray bf16 cache_elements{is_global_array output1})
  (output2:larray bf16 cache_elements{is_global_array output2})
  (#weights:erased (bfseq weight_elements))
  (#hidden:erased (bfseq hidden_width))
  (#previous_keys #previous_values:erased (bfseq (previous_elements (I64.v input4))))
  (#fweights #fhidden #fkeys #fvalues:perm)
  preserves cpu
  preserves on gpu_loc (
    input0 |-> Frac fweights weights **
    input1 |-> Frac fhidden hidden **
    state_ownership input2 (I64.v input4) fkeys previous_keys **
    state_ownership input3 (I64.v input4) fvalues previous_values)
  requires pure (input_representation input weights hidden previous_keys previous_values)
  requires on gpu_loc (
    A.pts_to_uninit output0 hidden_width **
    A.pts_to_uninit output1 cache_elements **
    A.pts_to_uninit output2 cache_elements)
  returns observed_scores:erased (lseq f32 (score_elements input.call.position))
  ensures exists* (result:bfseq hidden_width) (keys values:bfseq cache_elements).
    on gpu_loc (output0 |-> result ** output1 |-> keys ** output2 |-> values) **
    pure (source_score_post input previous_keys previous_values result keys values observed_scores)
{
  Sh.full_profile_position_bound input.call;
  let count = 8sz *^ (SZ.uint32_to_sizet (IC.int64_to_uint32 input4) +^ 1sz);
  let scores = gpu_array_alloc #f32 count;
  with old_scores. assert on gpu_loc (scores |-> old_scores);
  let initial_scores:erased (lseq f32 (score_elements (I64.v input4))) =
    FStar.Ghost.hide #(lseq f32 (score_elements (I64.v input4)))
      (FStar.Ghost.reveal #(seq f32) old_scores);
  launch_kernel_1 (fun () ->
    family_kernel #input input0 input1 input2 input3 input4 output0 output1 output2 scores
      #weights #hidden #previous_keys #previous_values #initial_scores
      #fweights #fhidden #fkeys #fvalues);
  with result. assert on gpu_loc (output0 |-> result);
  with keys. assert on gpu_loc (output1 |-> keys);
  with values. assert on gpu_loc (output2 |-> values);
  with live_scores. assert on gpu_loc (scores |-> live_scores);
  gpu_array_free scores;
  FStar.Ghost.hide #(lseq f32 (score_elements input.call.position))
    (FStar.Ghost.reveal #(lseq f32 (score_elements input.call.position)) live_scores)
}
