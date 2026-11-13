module KPlay.SharedSpec

open FStar.Real

module Exp = FStar.Math.Exp
module Pow = FStar.Math.Pow
module Root = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry
module Sum = Kuiper.Sum

(* The same complete 2026-10-13 structural-causal decoder algebra and cache
   transition, newly scoped by the 2026-10-15 BILATERAL NO-OVERFLOW authorization.
   M.json places actual native no-overflow/no-invalid premises in each role's
   source-bound representation/refinement theorem, on the same native inputs.
   They are not shared inputs or Boolean guards on this concrete postcondition.
   BF16/FP32 operations and casts retain Kuiper's v_approximates abstraction,
   not native bitwise equality, an error bound, or reference-defined implying
   candidate-defined. Native domain satisfaction is assumed, not proved here.
   Safety, initialization, input preservation and physical/cache frame
   obligations remain separate from the conditional numerical refinement.
   Integer indices and cache constructors retain their exact meanings.
   The singleton batch axis is implicit; function arguments are logical axes,
   not storage offsets. No checkpoint parameter values are fixed here. *)

type index (n:nat) = i:nat{i < n}
type vector (n:nat) = index n -> GTot real
type matrix (rows cols:nat) = index rows -> GTot (vector cols)
type hidden_states = matrix 7 1024
type heads (count:nat) = index 7 -> GTot (matrix count 128)
type attention_rows = index 7 -> GTot (matrix 16 7)

inline_for_extraction noextract
let sum (n:nat) (f:vector n) : GTot real =
  Sum.sum 0 n f

inline_for_extraction noextract
let rec sum_nonnegative (n:nat) (f:vector n)
  : Lemma
      (requires forall (i:index n). f i >=. 0.0R)
      (ensures sum n f >=. 0.0R)
      (decreases n)
  =
  if n > 0 then (
    sum_nonnegative (n - 1) f;
    Sum.sum_pop_right 0 n f
  )

inline_for_extraction noextract
let sum_positive (n:pos) (f:vector n)
  : Lemma
      (requires forall (i:index n). f i >. 0.0R)
      (ensures sum n f >. 0.0R)
  =
  sum_nonnegative (n - 1) f;
  Sum.sum_pop_right 0 n f

inline_for_extraction noextract
let dot (#n:nat) (x y:vector n) : GTot real =
  sum n (fun i -> x i *. y i)

inline_for_extraction noextract
let rms_norm (#n:pos) (weight x:vector n) : GTot (vector n) =
  let squares : vector n = fun i -> x i *. x i in
  sum_nonnegative n squares;
  let mean_square = sum n squares /. of_int n in
  let inverse_rms = Root.rsqrt (mean_square +. 0.000001R) in
  fun i -> weight i *. (x i *. inverse_rms)

inline_for_extraction noextract
let linear (#src #dst:pos)
  (weight:matrix dst src) (x:vector src) : GTot (vector dst) =
  fun out_channel -> dot x (weight out_channel)

inline_for_extraction noextract
let linear_rows (#rows #src #dst:pos)
  (weight:matrix dst src) (x:matrix rows src) : GTot (matrix rows dst) =
  fun token -> linear weight (x token)

inline_for_extraction noextract
let normalize_hidden (weight:vector 1024) (x:hidden_states)
  : GTot hidden_states =
  fun token -> rms_norm weight (x token)

inline_for_extraction noextract
let split_heads (#count:pos) (x:matrix 7 (count * 128))
  : GTot (heads count) =
  fun token head lane -> x token (head * 128 + lane)

inline_for_extraction noextract
let normalize_heads (#count:pos) (weight:vector 128) (x:heads count)
  : GTot (heads count) =
  fun token head -> rms_norm weight (x token head)

inline_for_extraction noextract
let inverse_frequency (lane:index 128) : real =
  1.0R /. Pow.pow 1000000.0R (of_int (2 * (lane % 64)) /. 128.0R)

inline_for_extraction noextract
let rope_angle (position:int) (lane:index 128) : real =
  of_int position *. inverse_frequency lane

inline_for_extraction noextract
let prefill_cos (token:index 7) (lane:index 128) : real =
  Trig.cos (rope_angle token lane)

inline_for_extraction noextract
let prefill_sin (token:index 7) (lane:index 128) : real =
  Trig.sin (rope_angle token lane)

(* The patched source checks this finite-sentinel mask but never adds it to
   scores. Both attention reductions below select the prefix structurally. *)
inline_for_extraction noextract
let bf16_minimum () : real =
  of_int (-338953138925153547590470800371487866880)

inline_for_extraction noextract
let prefill_mask (query key:index 7) : real =
  if key <= query then 0.0R else bf16_minimum ()

[@@erasable]
noeq
type attention_data = {
  position_ids: index 7 -> int;
  cache_positions: index 7 -> int;
  cosine: matrix 7 128;
  sine: matrix 7 128;
  additive_mask: matrix 7 7
}

inline_for_extraction noextract
let prefill_attention_data () : attention_data = {
  position_ids = (fun token -> token);
  cache_positions = (fun token -> token);
  cosine = prefill_cos;
  sine = prefill_sin;
  additive_mask = prefill_mask
}

inline_for_extraction noextract
let valid_attention_data (data:attention_data) : prop =
  (forall (token:index 7).
    data.position_ids token == token /\
    data.cache_positions token == token) /\
  (forall (token:index 7) (lane:index 128).
    data.cosine token lane == prefill_cos token lane /\
    data.sine token lane == prefill_sin token lane) /\
  (forall (query key:index 7).
    data.additive_mask query key == prefill_mask query key)

inline_for_extraction noextract
let rotary (#count:pos) (data:attention_data) (x:heads count)
  : GTot (heads count) =
  fun token head lane ->
    let rotated_half =
      if lane < 64 then 0.0R -. x token head (lane + 64)
      else x token head (lane - 64)
    in
    x token head lane *. data.cosine token lane +.
      rotated_half *. data.sine token lane

inline_for_extraction noextract
let repeat_kv (x:heads 8) : GTot (heads 16) =
  fun token head lane -> x token (head / 2) lane

inline_for_extraction noextract
let softmax (#n:pos) (x:vector n) : GTot (vector n) =
  let numerators : vector n = fun i -> Exp.exp (x i) in
  sum_positive n numerators;
  let denominator = sum n numerators in
  fun i -> numerators i /. denominator

inline_for_extraction noextract
let softmax_positive (#n:pos) (x:vector n) (i:index n)
  : Lemma (softmax x i >. 0.0R)
  =
  sum_positive n (fun j -> Exp.exp (x j))

inline_for_extraction noextract
let prefix_softmax (token:index 7) (scores:vector (token + 1))
  : GTot (vector 7) =
  let probabilities = softmax scores in
  fun key_token ->
    if key_token <= token then probabilities key_token else 0.0R

inline_for_extraction noextract
let prefix_softmax_denominator_positive
  (token:index 7) (scores:vector (token + 1))
  : Lemma (sum (token + 1) (fun i -> Exp.exp (scores i)) >. 0.0R)
  =
  sum_positive (token + 1) (fun i -> Exp.exp (scores i))

inline_for_extraction noextract
let prefix_softmax_positive
  (token:index 7) (scores:vector (token + 1))
  (key_token:index 7{key_token <= token})
  : Lemma (prefix_softmax token scores key_token >. 0.0R)
  =
  softmax_positive scores key_token

inline_for_extraction noextract
let prefix_softmax_future_zero
  (token:index 7) (scores:vector (token + 1))
  (key_token:index 7{token < key_token})
  : Lemma (prefix_softmax token scores key_token == 0.0R)
  = ()

inline_for_extraction noextract
let attention_probabilities
  (data:attention_data) (query repeated_key:heads 16)
  : GTot attention_rows =
  let scaling = Root.rsqrt 128.0R in
  fun token head ->
    prefix_softmax token (fun key_token ->
      dot (query token head) (repeated_key key_token head) *. scaling)

inline_for_extraction noextract
let attention_prefix_positive
  (data:attention_data) (query repeated_key:heads 16)
  (token:index 7) (head:index 16) (key_token:index 7{key_token <= token})
  : Lemma (attention_probabilities data query repeated_key token head key_token >. 0.0R)
  =
  prefix_softmax_positive token
    (fun source_token ->
      dot (query token head) (repeated_key source_token head) *. Root.rsqrt 128.0R)
    key_token

inline_for_extraction noextract
let attention_future_zero
  (data:attention_data) (query repeated_key:heads 16)
  (token:index 7) (head:index 16) (key_token:index 7{token < key_token})
  : Lemma (attention_probabilities data query repeated_key token head key_token == 0.0R)
  = ()

inline_for_extraction noextract
let attention_values (probabilities:attention_rows) (repeated_value:heads 16)
  : GTot (heads 16) =
  fun token head lane ->
    sum (token + 1) (fun key_token ->
      probabilities token head key_token *. repeated_value key_token head lane)

inline_for_extraction noextract
let concatenate_heads (x:heads 16) : GTot (matrix 7 2048) =
  fun token channel -> x token (channel / 128) (channel % 128)

inline_for_extraction noextract
let residual_add (x y:hidden_states) : GTot hidden_states =
  fun token channel -> x token channel +. y token channel

inline_for_extraction noextract
let silu (x:real) : real =
  x /. (1.0R +. Exp.exp (0.0R -. x))

[@@erasable]
noeq
type parameters = {
  input_layernorm: vector 1024;
  q_proj: matrix 2048 1024;
  k_proj: matrix 1024 1024;
  v_proj: matrix 1024 1024;
  q_norm: vector 128;
  k_norm: vector 128;
  o_proj: matrix 1024 2048;
  post_attention_layernorm: vector 1024;
  gate_proj: matrix 3072 1024;
  up_proj: matrix 3072 1024;
  down_proj: matrix 1024 3072
}

inline_for_extraction noextract
let feed_forward (weight:parameters) (x:hidden_states) : GTot hidden_states =
  let gate = linear_rows weight.gate_proj x in
  let up = linear_rows weight.up_proj x in
  let gated : matrix 7 3072 =
    fun token channel -> silu (gate token channel) *. up token channel
  in
  linear_rows weight.down_proj gated

[@@erasable]
noeq
type cached_kv = {
  keys: heads 8;
  values: heads 8
}

(* In this fresh prefill, each DynamicLayer is either uninitialized/empty,
   or initialized with exactly seven positions. It stores normalized, rotated
   keys and unnormalized values BEFORE GQA replication. *)
[@@erasable]
noeq
type cache_slot =
  | Empty
  | Prefilled : contents:cached_kv -> cache_slot

type cache_state = index 28 -> GTot cache_slot

inline_for_extraction noextract
let cache_initialized (slot:cache_slot) : GTot bool =
  match slot with
  | Empty -> false
  | Prefilled _ -> true

inline_for_extraction noextract
let cache_length (slot:cache_slot) : GTot nat =
  match slot with
  | Empty -> 0
  | Prefilled _ -> 7

inline_for_extraction noextract
let valid_cache_before (layer:index 28) (before:cache_state) : prop =
  forall (other:index 28).
    if other < layer then cache_initialized (before other)
    else not (cache_initialized (before other))

inline_for_extraction noextract
let update_cache (before:cache_state) (layer:index 28) (key value:heads 8)
  : GTot cache_state =
  fun other ->
    if other = layer then Prefilled { keys = key; values = value }
    else before other

[@@erasable]
noeq
type layer_input = {
  hidden: hidden_states;
  weight: parameters;
  data: attention_data;
  layer: index 28;
  cache_before: cache_state
}

inline_for_extraction noextract
let valid_input (input:layer_input) : prop =
  valid_attention_data input.data /\
  valid_cache_before input.layer input.cache_before

type prefill_input = input:layer_input{valid_input input}

[@@erasable]
noeq
type layer_result = {
  hidden_after: hidden_states;
  cache_after: cache_state
}

inline_for_extraction noextract
let decoder_layer (input:prefill_input) : GTot layer_result =
  let weight = input.weight in
  let normalized = normalize_hidden weight.input_layernorm input.hidden in
  let query = normalize_heads weight.q_norm
    (split_heads #16 (linear_rows weight.q_proj normalized)) in
  let key = normalize_heads weight.k_norm
    (split_heads #8 (linear_rows weight.k_proj normalized)) in
  let value = split_heads #8 (linear_rows weight.v_proj normalized) in
  let rotated_query = rotary input.data query in
  let rotated_key = rotary input.data key in
  let cache_after = update_cache input.cache_before input.layer rotated_key value in
  let probabilities = attention_probabilities input.data
    rotated_query (repeat_kv rotated_key) in
  let mixed = attention_values probabilities (repeat_kv value) in
  let projected = linear_rows weight.o_proj (concatenate_heads mixed) in
  let after_attention = residual_add input.hidden projected in
  let mlp_input = normalize_hidden weight.post_attention_layernorm after_attention in
  let hidden_after = residual_add after_attention (feed_forward weight mlp_input) in
  { hidden_after = hidden_after; cache_after = cache_after }

inline_for_extraction noextract
let same_cache_slot (left right:cache_slot) : prop =
  match left, right with
  | Empty, Empty -> True
  | Prefilled l, Prefilled r ->
    forall (token:index 7) (head:index 8) (lane:index 128).
      l.keys token head lane == r.keys token head lane /\
      l.values token head lane == r.values token head lane
  | _ -> False

inline_for_extraction noextract
let same_result (left right:layer_result) : prop =
  (forall (token:index 7) (channel:index 1024).
    left.hidden_after token channel == right.hidden_after token channel) /\
  (forall (layer:index 28).
    same_cache_slot (left.cache_after layer) (right.cache_after layer))

inline_for_extraction noextract
let postcondition (input:prefill_input) (result:layer_result) : prop =
  same_result result (decoder_layer input)

inline_for_extraction noextract
let decoder_layer_satisfies_postcondition (input:prefill_input)
  : Lemma (postcondition input (decoder_layer input))
  = ()

inline_for_extraction noextract
let postcondition_determines_result
  (input:prefill_input) (left right:layer_result)
  : Lemma
      (requires postcondition input left /\ postcondition input right)
      (ensures same_result left right)
  = ()

inline_for_extraction noextract
let cache_frame
  (before:cache_state) (layer:index 28) (key value:heads 8)
  (other:index 28{other <> layer})
  : Lemma (update_cache before layer key value other == before other)
  = ()

inline_for_extraction noextract
let cache_target
  (before:cache_state) (layer:index 28) (key value:heads 8)
  : Lemma
      (cache_initialized (update_cache before layer key value layer) /\
       cache_length (update_cache before layer key value layer) == 7 /\
       same_cache_slot (update_cache before layer key value layer)
         (Prefilled { keys = key; values = value }))
  = ()

inline_for_extraction noextract
let canonical_attention_data_valid ()
  : Lemma (valid_attention_data (prefill_attention_data ()))
  = ()
