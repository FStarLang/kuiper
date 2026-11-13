module KPlay.Proof.Shared.Recurrent

open Kuiper.Real
open Kuiper.Sum

module R = FStar.Real
module E = FStar.Math.Exp
module Root = FStar.Math.Sqrt
module Sum = Kuiper.Sum

// Real-valued algebraic specification, not an IEEE or CUDA implementation.
// Arithmetic/cast approximation and concrete branch correspondence are separate
// obligations of each role. No comparison stability follows from approximation.

inline_for_extraction noextract
let fin (n:nat) = i:nat{i < n}

inline_for_extraction noextract
let vector (n:nat) = fin n -> GTot real

inline_for_extraction noextract
let matrix (m n:nat) = fin m -> GTot (vector n)

inline_for_extraction noextract
let tensor3 (a b c:nat) = fin a -> GTot (matrix b c)

inline_for_extraction noextract
let epsilon : r:real{r >. 0.0R} = 0.000001R

noextract
noeq type weights = {
  input_norm: vector 1024;
  post_attention_norm: vector 1024;
  in_qkv: matrix 6144 1024;
  in_z: matrix 2048 1024;
  in_beta: matrix 16 1024;
  in_a: matrix 16 1024;
  conv_kernel: matrix 6144 4;
  a_log: vector 16;
  dt_bias: vector 16;
  gated_norm: vector 128;
  out_projection: matrix 1024 2048;
  mlp_gate: matrix 3584 1024;
  mlp_up: matrix 3584 1024;
  mlp_down: matrix 1024 3584;
}

noextract
type invocation = {
  smoke: bool;
  prompt_length: p:nat{1 <= p /\ p <= 520};
  decode_steps: d:nat{
    if smoke then d == 1
    else d == 1 \/ d == 5 \/ d == 30 \/ d == 128
  };
  cache_position: t:nat{
    t < prompt_length + decode_steps - 1 /\
    prompt_length + decode_steps - 1 <= 2048
  };
  layer_index: l:nat{l < 24 /\ l % 4 < 3};
}

inline_for_extraction noextract
let convolution_state = matrix 6144 4

inline_for_extraction noextract
let recurrent_state = tensor3 16 128 128

noextract
noeq type kv_state = {
  kv_length: nat;
  keys: tensor3 2 kv_length 256;
  values: tensor3 2 kv_length 256;
}

// Cache axes are semantic indices, not buffer offsets. K/V entries are framed.
noextract
noeq type model_cache = {
  conv_slots: fin 24 -> GTot (option convolution_state);
  recurrent_slots: fin 24 -> GTot (option recurrent_state);
  kv_slots: fin 24 -> GTot (option kv_state);
}

inline_for_extraction noextract
let initial_cache () : model_cache = {
  conv_slots = (fun _ -> None);
  recurrent_slots = (fun _ -> None);
  kv_slots = (fun _ -> None);
}

inline_for_extraction noextract
let has_previous_state (cache:model_cache) : GTot bool =
  Some? (cache.conv_slots 22)

inline_for_extraction noextract
let cache_domain (call:invocation) (cache:model_cache) : GTot prop =
  if call.cache_position = 0 then
    not (has_previous_state cache) /\
    not (Some? (cache.conv_slots call.layer_index)) /\
    not (Some? (cache.recurrent_slots call.layer_index))
  else
    has_previous_state cache /\
    Some? (cache.conv_slots call.layer_index) /\
    Some? (cache.recurrent_slots call.layer_index)

inline_for_extraction noextract
let present_value (#a:Type0) (x:option a{Some? x}) : GTot a =
  match x with
  | Some value -> value

inline_for_extraction noextract
let zero_convolution () : convolution_state = fun _ _ -> 0.0R

inline_for_extraction noextract
let zero_recurrent () : recurrent_state = fun _ _ _ -> 0.0R

inline_for_extraction noextract
let rec sum_nonnegative
  (n:nat)
  (f:fin n -> GTot real)
  : Lemma
      (requires forall (i:fin n). f i >=. 0.0R)
      (ensures Sum.sum 0 n f >=. 0.0R)
      (decreases n)
  =
  if n > 0 then (
    sum_nonnegative (n - 1) f;
    Sum.sum_pop_right 0 n f;
    assert (Sum.sum 0 n f == Sum.sum 0 (n - 1) f +. f (n - 1));
    assert (f (n - 1) >=. 0.0R)
  )

inline_for_extraction noextract
let sum_squares (#n:nat) (x:vector n)
  : GTot (r:real{r >=. 0.0R})
  =
  let squared:vector n = fun i -> x i *. x i in
  sum_nonnegative n squared;
  Sum.sum 0 n squared

inline_for_extraction noextract
let project (#rows #columns:nat)
  (w:matrix rows columns) (x:vector columns)
  : GTot (vector rows)
  =
  fun row -> Sum.sum 0 columns (fun column -> w row column *. x column)

inline_for_extraction noextract
let add_vectors (#n:nat) (x y:vector n) : vector n =
  fun i -> x i +. y i

inline_for_extraction noextract
let rms_scale (#n:pos) (x:vector n)
  : GTot (r:real{r >. 0.0R})
  =
  Root.rsqrt (sum_squares x /. R.of_int n +. epsilon)

inline_for_extraction noextract
let centered_rms_norm (#n:pos) (w x:vector n) : GTot (vector n) =
  let scale = rms_scale x in
  fun i -> (x i *. scale) *. (1.0R +. w i)

inline_for_extraction noextract
let l2_normalize (x:vector 128) : GTot (vector 128) =
  let scale = Root.rsqrt (sum_squares x +. epsilon) in
  fun i -> x i *. scale

inline_for_extraction noextract
let sigmoid (x:real) : real =
  E.exp_positive (0.0R -. x);
  1.0R /. (1.0R +. E.exp (0.0R -. x))

inline_for_extraction noextract
let silu (x:real) : real = x *. sigmoid x

// The default softplus uses beta=1, threshold=20, with a strict comparison.
inline_for_extraction noextract
let softplus (x:real) : GTot real =
  if t2b (x >. 20.0R) then x
  else (
    E.exp_positive x;
    E.log (1.0R +. E.exp x)
  )

inline_for_extraction noextract
let shift_convolution
  (old:convolution_state) (raw_qkv:vector 6144)
  : convolution_state
  =
  fun channel tap ->
    if tap < 3 then old channel (tap + 1) else raw_qkv channel

inline_for_extraction noextract
let convolve (w:matrix 6144 4) (next:convolution_state)
  : vector 6144
  =
  fun channel ->
    silu (Sum.sum 0 4 (fun tap -> w channel tap *. next channel tap))

noextract
noeq type mixer_result = {
  mixed_hidden: vector 1024;
  next_convolution: convolution_state;
  next_recurrent: recurrent_state;
}

inline_for_extraction noextract
let token_mixer
  (w:weights)
  (normalized_hidden:vector 1024)
  (old_convolution:convolution_state)
  (old_recurrent:recurrent_state)
  : GTot mixer_result
  =
  let raw_qkv = project w.in_qkv normalized_hidden in
  let z = project w.in_z normalized_hidden in
  let b = project w.in_beta normalized_hidden in
  let a = project w.in_a normalized_hidden in
  let next_convolution = shift_convolution old_convolution raw_qkv in
  let qkv = convolve w.conv_kernel next_convolution in
  let query:matrix 16 128 =
    fun head -> l2_normalize (fun i -> qkv (head * 128 + i))
  in
  let key:matrix 16 128 =
    fun head -> l2_normalize (fun i -> qkv (2048 + head * 128 + i))
  in
  let value:matrix 16 128 =
    fun head i -> qkv (4096 + head * 128 + i)
  in
  let beta:vector 16 = fun head -> sigmoid (b head) in
  let decay:vector 16 =
    fun head ->
      let g = (0.0R -. E.exp (w.a_log head)) *.
              softplus (a head +. w.dt_bias head)
      in
      E.exp g
  in
  let decayed:recurrent_state =
    fun head i j -> old_recurrent head i j *. decay head
  in
  let memory:matrix 16 128 =
    fun head j ->
      Sum.sum 0 128 (fun i -> decayed head i j *. key head i)
  in
  let delta:matrix 16 128 =
    fun head j -> (value head j -. memory head j) *. beta head
  in
  let next_recurrent:recurrent_state =
    fun head i j -> decayed head i j +. key head i *. delta head j
  in
  let query_scale = Root.rsqrt 128.0R in
  let core:matrix 16 128 =
    fun head j ->
      Sum.sum 0 128
        (fun i -> next_recurrent head i j *. (query head i *. query_scale))
  in
  let gated:matrix 16 128 =
    fun head ->
      let scale = rms_scale (core head) in
      fun j ->
        (w.gated_norm j *. (core head j *. scale)) *.
        silu (z (head * 128 + j))
  in
  let flattened:vector 2048 = fun i -> gated (i / 128) (i % 128) in
  {
    mixed_hidden = project w.out_projection flattened;
    next_convolution;
    next_recurrent;
  }

inline_for_extraction noextract
let feed_forward (w:weights) (x:vector 1024) : GTot (vector 1024) =
  let gate = project w.mlp_gate x in
  let up = project w.mlp_up x in
  let product:vector 3584 = fun i -> silu (gate i) *. up i in
  project w.mlp_down product

noextract
noeq type layer_values = {
  hidden_value: vector 1024;
  convolution_value: convolution_state;
  recurrent_value: recurrent_state;
}

inline_for_extraction noextract
let decoder_values
  (call:invocation)
  (w:weights)
  (hidden:vector 1024)
  (before:model_cache{cache_domain call before})
  : GTot layer_values
  =
  let use_previous = has_previous_state before in
  let old_convolution =
    if use_previous then present_value (before.conv_slots call.layer_index)
    else zero_convolution ()
  in
  let old_recurrent =
    if use_previous then present_value (before.recurrent_slots call.layer_index)
    else zero_recurrent ()
  in
  let normalized = centered_rms_norm w.input_norm hidden in
  let mixed = token_mixer w normalized old_convolution old_recurrent in
  let residual = add_vectors hidden mixed.mixed_hidden in
  let mlp_input = centered_rms_norm w.post_attention_norm residual in
  {
    hidden_value = add_vectors residual (feed_forward w mlp_input);
    convolution_value = mixed.next_convolution;
    recurrent_value = mixed.next_recurrent;
  }

inline_for_extraction noextract
let install_state
  (layer:fin 24) (before:model_cache) (computed:layer_values)
  : model_cache
  =
  {
    conv_slots = (fun i ->
      if i = layer then Some computed.convolution_value else before.conv_slots i);
    recurrent_slots = (fun i ->
      if i = layer then Some computed.recurrent_value else before.recurrent_slots i);
    kv_slots = before.kv_slots;
  }

noextract
noeq type result = {
  hidden: vector 1024;
  cache: model_cache;
}

inline_for_extraction noextract
let recurrent_decoder
  (call:invocation)
  (w:weights)
  (hidden:vector 1024)
  (before:model_cache{cache_domain call before})
  : GTot result
  =
  let computed = decoder_values call w hidden before in
  {
    hidden = computed.hidden_value;
    cache = install_state call.layer_index before computed;
  }

inline_for_extraction noextract
let cache_frame (layer:fin 24) (before after:model_cache) : GTot prop =
  (forall (i:fin 24). i =!= layer ==>
    after.conv_slots i == before.conv_slots i /\
    after.recurrent_slots i == before.recurrent_slots i) /\
  (forall (i:fin 24). after.kv_slots i == before.kv_slots i)

inline_for_extraction noextract
let postcondition
  (call:invocation)
  (w:weights)
  (hidden:vector 1024)
  (before:model_cache{cache_domain call before})
  (actual:result)
  : GTot prop
  =
  let expected = decoder_values call w hidden before in
  (forall (i:fin 1024). actual.hidden i == expected.hidden_value i) /\
  (match actual.cache.conv_slots call.layer_index,
         actual.cache.recurrent_slots call.layer_index with
   | Some conv, Some recurrent ->
       (forall (channel:fin 6144) (tap:fin 4).
         conv channel tap == expected.convolution_value channel tap) /\
       (forall (head:fin 16) (i j:fin 128).
         recurrent head i j == expected.recurrent_value head i j)
   | _ -> False) /\
  cache_frame call.layer_index before actual.cache

inline_for_extraction noextract
let initial_cache_is_valid
  (call:invocation{call.cache_position == 0})
  : Lemma (cache_domain call (initial_cache ()))
  = ()

inline_for_extraction noextract
let recurrent_decoder_satisfies_postcondition
  (call:invocation)
  (w:weights)
  (hidden:vector 1024)
  (before:model_cache{cache_domain call before})
  : Lemma
      (postcondition call w hidden before
        (recurrent_decoder call w hidden before))
  = ()
