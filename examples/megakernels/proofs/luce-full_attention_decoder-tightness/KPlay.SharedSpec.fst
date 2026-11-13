module KPlay.SharedSpec

open FStar.Real
module Exp = FStar.Math.Exp
module Pow = FStar.Math.Pow
module Sqrt = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry

(* Real-valued, layout-free contract for the selected eager reference.
   BF16/FP32 arithmetic and casts are interpreted through Kuiper's qualitative
   approximation laws, not as exact IEEE operations or error bounds. The only
   mathematical primitives used below come from the installed libraries.
   Integer indices, head selection, cache presence and append order are exact. *)

inline_for_extraction noextract
let index (n:nat) : Type0 = i:nat{i < n}

inline_for_extraction noextract
let vector (n:nat) : Type0 = index n -> real

inline_for_extraction noextract
let matrix (rows cols:nat) : Type0 = index rows -> vector cols

inline_for_extraction noextract
let layer_index : Type0 = index 24

inline_for_extraction noextract
let full_attention_layer : Type0 = l:layer_index{l % 4 = 3}

inline_for_extraction noextract
let decode_count : Type0 = d:nat{d = 1 \/ d = 5 \/ d = 30 \/ d = 128}

noextract
type invocation = {
  prompt_length: p:pos{p <= 520};
  decode_steps: decode_count;
  position: p:nat{p < prompt_length + decode_steps - 1};
  layer: full_attention_layer
}

noextract noeq
type layer_weights = {
  input_norm: vector 1024;
  post_attention_norm: vector 1024;
  query_gate_proj: matrix 4096 1024;
  key_proj: matrix 512 1024;
  value_proj: matrix 512 1024;
  query_norm: vector 256;
  key_norm: vector 256;
  output_proj: matrix 1024 2048;
  mlp_gate_proj: matrix 3584 1024;
  mlp_up_proj: matrix 3584 1024;
  mlp_down_proj: matrix 1024 3584
}

inline_for_extraction noextract
let kv_tensor (n:nat) : Type0 = index 2 -> index n -> vector 256

noextract noeq
type kv_history = {
  length: pos;
  keys: kv_tensor length;
  values: kv_tensor length
}

inline_for_extraction noextract
let convolution_state : Type0 = matrix 6144 4

inline_for_extraction noextract
let recurrent_state : Type0 = index 16 -> matrix 128 128

(* A slot pairs the synchronized key/value lists, not physical buffers.
   None is the reference's lazy, uninitialized cache, not a row of zeros.
   Unselected slots are explicit arbitrary frames; no earlier layer is rerun. *)
noextract noeq
type hybrid_cache = {
  kv: layer_index -> option kv_history;
  convolution: layer_index -> option convolution_state;
  recurrent: layer_index -> option recurrent_state
}

noextract noeq
type layer_result = {
  hidden: vector 1024;
  cache: hybrid_cache
}

inline_for_extraction noextract
let empty_cache () : hybrid_cache = {
  kv = (fun _ -> None);
  convolution = (fun _ -> None);
  recurrent = (fun _ -> None)
}

inline_for_extraction noextract
let history_length (old:option kv_history) : nat =
  match old with
  | None -> 0
  | Some history -> history.length

inline_for_extraction noextract
let domain (call:invocation) (before:hybrid_cache) : prop =
  history_length (before.kv call.layer) == call.position

inline_for_extraction noextract
let rec sum (n:nat) (terms:vector n) : Tot real (decreases n) =
  if n = 0 then 0.0R
  else sum (n - 1) (fun i -> terms i) +. terms (n - 1)

inline_for_extraction noextract
let rec sum_squares (n:nat) (x:vector n)
  : Tot Sqrt.rnonneg (decreases n) =
  if n = 0 then 0.0R
  else sum_squares (n - 1) (fun i -> x i) +. x (n - 1) *. x (n - 1)

inline_for_extraction noextract
let rec sum_positive (n:pos) (terms:index n -> Sqrt.rpos)
  : Tot Sqrt.rpos (decreases n) =
  if n = 1 then terms 0
  else sum_positive (n - 1) (fun i -> terms i) +. terms (n - 1)

inline_for_extraction noextract
let linear (rows cols:nat) (w:matrix rows cols) (x:vector cols)
  : vector rows =
  fun r -> sum cols (fun c -> w r c *. x c)

inline_for_extraction noextract
let vector_add (n:nat) (x y:vector n) : vector n =
  fun i -> x i +. y i

inline_for_extraction noextract
let rms_norm (n:pos) (weight x:vector n) : vector n =
  let mean_square = sum_squares n x /. of_int n in
  let inverse_rms = Sqrt.rsqrt (mean_square +. 0.000001R) in
  fun i -> (x i *. inverse_rms) *. (1.0R +. weight i)

inline_for_extraction noextract
let sigmoid (x:real) : real =
  let e = Exp.exp (0.0R -. x) in
  Exp.exp_positive (0.0R -. x);
  1.0R /. (1.0R +. e)

inline_for_extraction noextract
let silu (x:real) : real = x *. sigmoid x

inline_for_extraction noextract
let softmax (n:pos) (scores:vector n) : vector n =
  let exponentials : index n -> Sqrt.rpos =
    fun i ->
      Exp.exp_positive (scores i);
      Exp.exp (scores i)
  in
  let denominator = sum_positive n exponentials in
  fun i -> exponentials i /. denominator

(* Default RoPE has no dynamic frequency update. Text execution expands the
   same position to all three axes; interleaving still keeps frequency j. *)
inline_for_extraction noextract
let text_axis_position (position:nat) (_axis:index 3) : nat = position

inline_for_extraction noextract
let mrope_axis (j:index 32) : index 3 =
  if j % 3 = 1 && j < 33 then 1
  else if j % 3 = 2 && j < 30 then 2
  else 0

inline_for_extraction noextract
let inverse_frequency (j:index 32) : real =
  1.0R /. Pow.pow 10000000.0R (of_int (2 * j) /. 64.0R)

inline_for_extraction noextract
let rope_angle (position:nat) (j:index 32) : real =
  inverse_frequency j *. of_int (text_axis_position position (mrope_axis j))

inline_for_extraction noextract
let rope_cos (position:nat) (j:index 32) : real =
  Trig.cos (rope_angle position j) *. 1.0R

inline_for_extraction noextract
let rope_sin (position:nat) (j:index 32) : real =
  Trig.sin (rope_angle position j) *. 1.0R

inline_for_extraction noextract
let rotate_partial (position:nat) (x:vector 256) : vector 256 =
  fun d ->
    if d < 64 then
      let j = if d < 32 then d else d - 32 in
      let half_rotated =
        if d < 32 then 0.0R -. x (d + 32) else x (d - 32)
      in
      x d *. rope_cos position j +. half_rotated *. rope_sin position j
    else x d

(* q_proj is reshaped to [8,512] before its last axis is chunked.
   It is not split into a global query half and a global gate half. *)
inline_for_extraction noextract
let split_queries (projected:vector 4096) : matrix 8 256 =
  fun h d -> projected (512 * h + d)

inline_for_extraction noextract
let split_gates (projected:vector 4096) : matrix 8 256 =
  fun h d -> projected (512 * h + 256 + d)

inline_for_extraction noextract
let reshape_kv (projected:vector 512) : matrix 2 256 =
  fun h d -> projected (256 * h + d)

inline_for_extraction noextract
let append_kv
  (position:nat)
  (old:option kv_history{history_length old == position})
  (new_key new_value:matrix 2 256)
  : h:kv_history{h.length == position + 1} =
  match old with
  | None -> {
      length = 1;
      keys = (fun h _t d -> new_key h d);
      values = (fun h _t d -> new_value h d)
    }
  | Some history -> {
      length = history.length + 1;
      keys = (fun h t d ->
        if t < history.length then history.keys h t d else new_key h d);
      values = (fun h t d ->
        if t < history.length then history.values h t d else new_value h d)
    }

inline_for_extraction noextract
let replace_kv (before:hybrid_cache) (layer:layer_index) (next:kv_history)
  : hybrid_cache = {
  kv = (fun l -> if l = layer then Some next else before.kv l);
  convolution = before.convolution;
  recurrent = before.recurrent
}

inline_for_extraction noextract
let kv_head (query_head:index 8) : index 2 = query_head / 4

(* The eager mask has kv_length=position+1, offset=0 and no padding.
   Every stored key satisfies t<=position, so every additive entry is zero.
   The reference's finfo(dtype).min branch is unreachable in this call domain. *)
inline_for_extraction noextract
let causal_bias (position:nat) (t:index (position + 1)) : real =
  assert (t <= position);
  0.0R

inline_for_extraction noextract
let attention
  (position:nat)
  (queries:matrix 8 256)
  (history:kv_history{history.length == position + 1})
  : matrix 8 256 =
  fun h ->
    let scores : vector (position + 1) =
      fun t ->
        let dot = sum 256 (fun d -> queries h d *. history.keys (kv_head h) t d) in
        dot *. (1.0R /. 16.0R) +. causal_bias position t
    in
    let probabilities = softmax (position + 1) scores in
    fun d ->
      sum (position + 1)
        (fun t -> probabilities t *. history.values (kv_head h) t d)

noextract
let full_attention_decoder
  (call:invocation)
  (w:layer_weights)
  (x:vector 1024)
  (before:hybrid_cache{domain call before})
  : layer_result =
  let normalized = rms_norm 1024 w.input_norm x in
  let projected_qg = linear 4096 1024 w.query_gate_proj normalized in
  let queries = split_queries projected_qg in
  let gates = split_gates projected_qg in
  let keys = reshape_kv (linear 512 1024 w.key_proj normalized) in
  let values = reshape_kv (linear 512 1024 w.value_proj normalized) in
  let rotated_queries : matrix 8 256 =
    fun h -> rotate_partial call.position (rms_norm 256 w.query_norm (queries h))
  in
  let rotated_keys : matrix 2 256 =
    fun h -> rotate_partial call.position (rms_norm 256 w.key_norm (keys h))
  in
  let next = append_kv call.position (before.kv call.layer) rotated_keys values in
  let context = attention call.position rotated_queries next in
  let gated_context : vector 2048 =
    fun j ->
      context (j / 256) (j % 256) *. sigmoid (gates (j / 256) (j % 256))
  in
  let mixed = linear 1024 2048 w.output_proj gated_context in
  let after_attention = vector_add 1024 x mixed in
  let mlp_input = rms_norm 1024 w.post_attention_norm after_attention in
  let mlp_gate = linear 3584 1024 w.mlp_gate_proj mlp_input in
  let mlp_up = linear 3584 1024 w.mlp_up_proj mlp_input in
  let mlp_product : vector 3584 = fun i -> silu (mlp_gate i) *. mlp_up i in
  let mlp_output = linear 1024 3584 w.mlp_down_proj mlp_product in
  {
    hidden = vector_add 1024 after_attention mlp_output;
    cache = replace_kv before call.layer next
  }

noextract
let postcondition
  (call:invocation)
  (w:layer_weights)
  (x:vector 1024)
  (before:hybrid_cache{domain call before})
  (result:layer_result)
  : prop =
  let expected = full_attention_decoder call w x before in
  (forall (i:index 1024). result.hidden i == expected.hidden i) /\
  result.cache == expected.cache

inline_for_extraction noextract
let full_profile_position_bound (call:invocation)
  : Lemma (call.position <= 646 /\ call.position + 1 <= 2048) = ()

inline_for_extraction noextract
let zero_position_rotation (x:vector 256) (d:index 256)
  : Lemma (rotate_partial 0 x d == x d) =
  Trig.sin_zero ();
  Trig.cos_zero ()

inline_for_extraction noextract
let cache_frame (before:hybrid_cache) (layer:layer_index) (next:kv_history)
  : Lemma (
      let after = replace_kv before layer next in
      after.kv layer == Some next /\
      (forall (l:layer_index). l <> layer ==> after.kv l == before.kv l) /\
      after.convolution == before.convolution /\
      after.recurrent == before.recurrent) = ()

inline_for_extraction noextract
let append_preserves_prefix
  (old:kv_history)
  (new_key new_value:matrix 2 256)
  (h:index 2) (t:index old.length) (d:index 256)
  : Lemma (
      let next = append_kv old.length (Some old) new_key new_value in
      next.keys h t d == old.keys h t d /\
      next.values h t d == old.values h t d) = ()

inline_for_extraction noextract
let append_includes_current
  (position:nat)
  (old:option kv_history{history_length old == position})
  (new_key new_value:matrix 2 256)
  (h:index 2) (d:index 256)
  : Lemma (
      let next = append_kv position old new_key new_value in
      next.keys h position d == new_key h d /\
      next.values h position d == new_value h d) =
  match old with
  | None -> ()
  | Some _ -> ()
