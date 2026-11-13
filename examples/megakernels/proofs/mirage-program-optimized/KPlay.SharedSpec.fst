module KPlay.SharedSpec

open FStar.Real

module S = FStar.Seq.Base
module C = FStar.Classical
module Exp = FStar.Math.Exp
module Pow = FStar.Math.Pow
module Root = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry
module Sum = Kuiper.Sum
module Sc = Kuiper.Scalars
module N = Kuiper.Approximates.Base
module F = Kuiper.Floating.Base
module BF16 = Kuiper.BFloat16

(* Fixed reference mathematics; numerical observations use the existing
   approximation relation. Native score order is deliberately a separate view.
   The scalar parameters below change representations, never model operations.
   Source binding and bilateral numerical premises are specified in M.json. *)

inline_for_extraction noextract
let prompt_length : pos = 7

inline_for_extraction noextract
let hidden_size : pos = 1024

inline_for_extraction noextract
let vocab_size : pos = 151936

inline_for_extraction noextract
let layer_count : pos = 28

inline_for_extraction noextract
type index (n:nat) = i:nat{i < n}

inline_for_extraction noextract
type vector (a:Type0) (n:nat) = index n -> GTot a

inline_for_extraction noextract
type matrix (a:Type0) (rows cols:nat) =
  index rows -> GTot (vector a cols)

inline_for_extraction noextract
type hidden (a:Type0) = matrix a 7 1024

inline_for_extraction noextract
type heads (a:Type0) (count:nat) = index 7 -> GTot (matrix a count 128)

inline_for_extraction noextract
type token_id = index vocab_size

inline_for_extraction noextract
type prompt_tokens = p:S.seq token_id{S.length p == prompt_length}

inline_for_extraction noextract
type returned_tokens = p:S.seq token_id{S.length p == prompt_length + 1}

inline_for_extraction noextract
noeq type parameters (a:Type0) = {
  input_layernorm: vector a 1024;
  q_proj: matrix a 2048 1024;
  k_proj: matrix a 1024 1024;
  v_proj: matrix a 1024 1024;
  q_norm: vector a 128;
  k_norm: vector a 128;
  o_proj: matrix a 1024 2048;
  post_attention_layernorm: vector a 1024;
  gate_proj: matrix a 3072 1024;
  up_proj: matrix a 3072 1024;
  down_proj: matrix a 1024 3072
}

inline_for_extraction noextract
noeq type model_weights (a:Type0) = {
  embedding: matrix a 151936 1024;
  final_norm: vector a 1024;
  layers: index 28 -> GTot (parameters a)
}

inline_for_extraction noextract
noeq type inputs = {
  weight: model_weights BF16.t;
  prompt: prompt_tokens;
  decode_steps: n:nat{n == 1}
}

inline_for_extraction noextract
noeq type rotary_tables (a:Type0) = {
  cosine: matrix a 7 128;
  sine: matrix a 7 128
}

inline_for_extraction noextract
noeq type layer_values (a:Type0) = {
  hidden_after: hidden a;
  keys: heads a 8;
  values: heads a 8
}

inline_for_extraction noextract
let sum (n:nat) (f:vector real n) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 n f

inline_for_extraction noextract
let rec sum_nonnegative (n:nat) (f:vector real n)
  : Lemma
      (requires forall (i:index n). f i >=. 0.0R)
      (ensures sum n f >=. 0.0R)
      (decreases n)
  =
  if n > 0 then (
    sum_nonnegative (n - 1) f;
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 n f)

inline_for_extraction noextract
let sum_positive (n:pos) (f:vector real n)
  : Lemma
      (requires forall (i:index n). f i >. 0.0R)
      (ensures sum n f >. 0.0R)
  =
  sum_nonnegative (n - 1) f;
  Sum.sum_pop_right #real #Sum.real_add_semigroup 0 n f

inline_for_extraction noextract
let dot (#n:nat) (x y:vector real n) : GTot real =
  sum n (fun i -> x i *. y i)

inline_for_extraction noextract
let rms_norm (#n:pos) (weight x:vector real n) : GTot (vector real n) =
  let squares : vector real n = fun i -> x i *. x i in
  sum_nonnegative n squares;
  let mean_square = sum n squares /. of_int n in
  let scale = Root.rsqrt (mean_square +. 0.000001R) in
  fun i -> weight i *. (x i *. scale)

inline_for_extraction noextract
let linear (#src #dst:pos)
  (weight:matrix real dst src) (x:vector real src)
  : GTot (vector real dst) =
  fun channel -> dot x (weight channel)

inline_for_extraction noextract
let linear_rows (#rows #src #dst:pos)
  (weight:matrix real dst src) (x:matrix real rows src)
  : GTot (matrix real rows dst) =
  fun token -> linear weight (x token)

inline_for_extraction noextract
let normalize_hidden (weight:vector real 1024) (x:hidden real)
  : GTot (hidden real) =
  fun token -> rms_norm weight (x token)

inline_for_extraction noextract
let split_heads (#a:Type0) (#count:pos) (x:matrix a 7 (count * 128))
  : GTot (heads a count) =
  fun token head lane -> x token (head * 128 + lane)

inline_for_extraction noextract
let normalize_heads (#count:pos) (weight:vector real 128) (x:heads real count)
  : GTot (heads real count) =
  fun token head -> rms_norm weight (x token head)

inline_for_extraction noextract
let inverse_frequency (r:index 64) : real =
  1.0R /. Pow.pow 1000000.0R (of_int (2 * r) /. 128.0R)

inline_for_extraction noextract
let tables_from_frequency (frequency:vector real 64) (factor:real)
  : GTot (rotary_tables real) = {
  cosine = (fun token lane ->
    Trig.cos (of_int token *. frequency (lane % 64)) *. factor);
  sine = (fun token lane ->
    Trig.sin (of_int token *. frequency (lane % 64)) *. factor)
}

inline_for_extraction noextract
let canonical_rotary () : GTot (rotary_tables real) =
  tables_from_frequency inverse_frequency 1.0R

inline_for_extraction noextract
let prefill_mask (query key:index 7) : real =
  if key <= query then 0.0R
  else of_int (-338953138925153547590470800371487866880)

inline_for_extraction noextract
let rotary (#count:pos) (data:rotary_tables real) (x:heads real count)
  : GTot (heads real count) =
  fun token head lane ->
    let rotated_half =
      if lane < 64 then 0.0R -. x token head (lane + 64)
      else x token head (lane - 64)
    in
    x token head lane *. data.cosine token lane +.
      rotated_half *. data.sine token lane

inline_for_extraction noextract
let repeat_kv (#a:Type0) (x:heads a 8) : GTot (heads a 16) =
  fun token head lane -> x token (head / 2) lane

inline_for_extraction noextract
let softmax (#n:pos) (scores:vector real n) : GTot (vector real n) =
  let numerator : vector real n = fun i -> Exp.exp (scores i) in
  sum_positive n numerator;
  let denominator = sum n numerator in
  fun i -> numerator i /. denominator

inline_for_extraction noextract
let prefix_softmax (token:index 7) (scores:vector real (token + 1))
  : GTot (vector real 7) =
  let probabilities = softmax scores in
  fun key -> if key <= token then probabilities key else 0.0R

inline_for_extraction noextract
let attention_probabilities (query repeated_key:heads real 16)
  : GTot (index 7 -> GTot (matrix real 16 7)) =
  fun token head ->
    prefix_softmax token (fun key ->
      dot (query token head) (repeated_key key head) *. Root.rsqrt 128.0R)

inline_for_extraction noextract
let attention_values
  (probabilities:index 7 -> GTot (matrix real 16 7))
  (repeated_value:heads real 16) : GTot (heads real 16) =
  fun token head lane ->
    sum (token + 1) (fun key ->
      probabilities token head key *. repeated_value key head lane)

inline_for_extraction noextract
let concatenate_heads (#a:Type0) (x:heads a 16) : GTot (matrix a 7 2048) =
  fun token channel -> x token (channel / 128) (channel % 128)

inline_for_extraction noextract
let residual_add (x y:hidden real) : GTot (hidden real) =
  fun token channel -> x token channel +. y token channel

inline_for_extraction noextract
let silu (x:real) : real =
  x /. (1.0R +. Exp.exp (0.0R -. x))

inline_for_extraction noextract
let feed_forward (weight:parameters real) (x:hidden real)
  : GTot (hidden real) =
  let gate = linear_rows weight.gate_proj x in
  let up = linear_rows weight.up_proj x in
  let gated : matrix real 7 3072 =
    fun token channel -> silu (gate token channel) *. up token channel
  in
  linear_rows weight.down_proj gated

(* Reused decoder equations. The checked mask is not an arithmetic operand.
   K/V are the computed unreplicated cache values, not independent outputs. *)
inline_for_extraction noextract
let decoder_layer (weight:parameters real) (data:rotary_tables real)
  (x:hidden real) : GTot (layer_values real) =
  let normalized = normalize_hidden weight.input_layernorm x in
  let query = normalize_heads weight.q_norm
    (split_heads #real #16 (linear_rows weight.q_proj normalized)) in
  let key = normalize_heads weight.k_norm
    (split_heads #real #8 (linear_rows weight.k_proj normalized)) in
  let value = split_heads #real #8 (linear_rows weight.v_proj normalized) in
  let rotated_query = rotary data query in
  let rotated_key = rotary data key in
  let probabilities = attention_probabilities rotated_query (repeat_kv rotated_key) in
  let mixed = attention_values probabilities (repeat_kv value) in
  let projected = linear_rows weight.o_proj (concatenate_heads mixed) in
  let after_attention = residual_add x projected in
  let mlp_input = normalize_hidden weight.post_attention_layernorm after_attention in
  {
    hidden_after = residual_add after_attention (feed_forward weight mlp_input);
    keys = rotated_key;
    values = value
  }

inline_for_extraction noextract
let embedding_lookup (#a:Type0) (weight:model_weights a) (prompt:prompt_tokens)
  : GTot (hidden a) =
  fun token channel -> weight.embedding (S.index prompt token) channel

inline_for_extraction noextract
let rec hidden_prefix (weight:model_weights real) (prompt:prompt_tokens)
  (count:nat{count <= 28}) : GTot (hidden real) (decreases count) =
  if count = 0 then embedding_lookup weight prompt
  else
    (decoder_layer (weight.layers (count - 1)) (canonical_rotary ())
      (hidden_prefix weight prompt (count - 1))).hidden_after

inline_for_extraction noextract
let mathematical_layer (weight:model_weights real) (prompt:prompt_tokens)
  (layer:index 28) : GTot (layer_values real) =
  decoder_layer (weight.layers layer) (canonical_rotary ())
    (hidden_prefix weight prompt layer)

inline_for_extraction noextract
let mathematical_hidden (weight:model_weights real) (prompt:prompt_tokens)
  : GTot (hidden real) =
  normalize_hidden weight.final_norm (hidden_prefix weight prompt 28)

inline_for_extraction noextract
let mathematical_logits (weight:model_weights real) (prompt:prompt_tokens)
  : GTot (vector real 151936) =
  linear weight.embedding (mathematical_hidden weight prompt 6)

inline_for_extraction noextract
let vector_approximates (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (#n:nat) (actual:vector a n) (ideal:vector real n) : prop =
  forall (i:index n). N.v_approximates (actual i) (ideal i)

inline_for_extraction noextract
let matrix_approximates (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (#rows #cols:nat) (actual:matrix a rows cols) (ideal:matrix real rows cols) : prop =
  forall (row:index rows). vector_approximates (actual row) (ideal row)

inline_for_extraction noextract
let heads_approximates (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (#count:nat) (actual:heads a count) (ideal:heads real count) : prop =
  forall (token:index 7). matrix_approximates (actual token) (ideal token)

inline_for_extraction noextract
let parameters_approximates (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (actual:parameters a) (ideal:parameters real) : prop =
  vector_approximates actual.input_layernorm ideal.input_layernorm /\
  matrix_approximates actual.q_proj ideal.q_proj /\
  matrix_approximates actual.k_proj ideal.k_proj /\
  matrix_approximates actual.v_proj ideal.v_proj /\
  vector_approximates actual.q_norm ideal.q_norm /\
  vector_approximates actual.k_norm ideal.k_norm /\
  matrix_approximates actual.o_proj ideal.o_proj /\
  vector_approximates actual.post_attention_layernorm ideal.post_attention_layernorm /\
  matrix_approximates actual.gate_proj ideal.gate_proj /\
  matrix_approximates actual.up_proj ideal.up_proj /\
  matrix_approximates actual.down_proj ideal.down_proj

inline_for_extraction noextract
let weights_approximates (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (actual:model_weights a) (ideal:model_weights real) : prop =
  matrix_approximates actual.embedding ideal.embedding /\
  vector_approximates actual.final_norm ideal.final_norm /\
  (forall (layer:index 28).
    parameters_approximates (actual.layers layer) (ideal.layers layer))

inline_for_extraction noextract
let rotary_approximates (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (actual:rotary_tables a) (ideal:rotary_tables real) : prop =
  matrix_approximates actual.cosine ideal.cosine /\
  matrix_approximates actual.sine ideal.sine

inline_for_extraction noextract
let layer_approximates (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (actual:layer_values a) (ideal:layer_values real) : prop =
  matrix_approximates actual.hidden_after ideal.hidden_after /\
  heads_approximates actual.keys ideal.keys /\
  heads_approximates actual.values ideal.values

inline_for_extraction noextract
let real_vector (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (#n:nat) (x:vector a n) : GTot (vector real n) =
  fun i -> N.to_real (x i)

inline_for_extraction noextract
let real_matrix (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (#rows #cols:nat) (x:matrix a rows cols) : GTot (matrix real rows cols) =
  fun row -> real_vector (x row)

inline_for_extraction noextract
let real_parameters (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (x:parameters a) : GTot (parameters real) = {
  input_layernorm = real_vector x.input_layernorm;
  q_proj = real_matrix x.q_proj;
  k_proj = real_matrix x.k_proj;
  v_proj = real_matrix x.v_proj;
  q_norm = real_vector x.q_norm;
  k_norm = real_vector x.k_norm;
  o_proj = real_matrix x.o_proj;
  post_attention_layernorm = real_vector x.post_attention_layernorm;
  gate_proj = real_matrix x.gate_proj;
  up_proj = real_matrix x.up_proj;
  down_proj = real_matrix x.down_proj
}

inline_for_extraction noextract
let real_weights (weight:model_weights BF16.t) : GTot (model_weights real) = {
  embedding = real_matrix weight.embedding;
  final_norm = real_vector weight.final_norm;
  layers = (fun layer -> real_parameters (weight.layers layer))
}

inline_for_extraction noextract
let real_vector_related (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (#n:nat) (x:vector a n)
  : Lemma (vector_approximates x (real_vector x)) =
  C.forall_intro #(index n)
    #(fun i -> N.v_approximates (x i) (N.to_real (x i)))
    (fun i -> N.to_real_ok (x i))

inline_for_extraction noextract
let real_matrix_related (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (#rows #cols:nat) (x:matrix a rows cols)
  : Lemma (matrix_approximates x (real_matrix x)) =
  C.forall_intro #(index rows)
    #(fun row -> vector_approximates (x row) (real_vector (x row)))
    (fun row -> real_vector_related (x row))

inline_for_extraction noextract
let real_parameters_related (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (x:parameters a)
  : Lemma (parameters_approximates x (real_parameters x)) =
  real_vector_related x.input_layernorm;
  real_matrix_related x.q_proj;
  real_matrix_related x.k_proj;
  real_matrix_related x.v_proj;
  real_vector_related x.q_norm;
  real_vector_related x.k_norm;
  real_matrix_related x.o_proj;
  real_vector_related x.post_attention_layernorm;
  real_matrix_related x.gate_proj;
  real_matrix_related x.up_proj;
  real_matrix_related x.down_proj

inline_for_extraction noextract
let real_weights_related (weight:model_weights BF16.t)
  : Lemma (weights_approximates weight (real_weights weight)) =
  real_matrix_related weight.embedding;
  real_vector_related weight.final_norm;
  C.forall_intro #(index 28)
    #(fun layer ->
      parameters_approximates (weight.layers layer) (real_parameters (weight.layers layer)))
    (fun layer -> real_parameters_related (weight.layers layer))

(* These are definitions of relational liftings of fixed equations, not
   assumed operations or caller-supplied correctness predicates. Quantifying
   over related operands permits composition without injective approximation
   or equating to_real with a native score-order interpretation. *)
inline_for_extraction noextract
let decoder_relation (#x #a:Type0)
  {| Sc.scalar x, N.real_like x, Sc.scalar a, N.real_like a |}
  (weight:parameters BF16.t) (input:hidden x)
  (data:rotary_tables a) (output:layer_values a) : prop =
  forall (wr:parameters real) (xr:hidden real) (dr:rotary_tables real).
    parameters_approximates weight wr /\
    matrix_approximates input xr /\
    rotary_approximates data dr ==>
    layer_approximates output (decoder_layer wr dr xr)

inline_for_extraction noextract
let final_norm_relation (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (weight:vector BF16.t 1024) (input:hidden a) (output:hidden BF16.t) : prop =
  forall (wr:vector real 1024) (xr:hidden real).
    vector_approximates weight wr /\ matrix_approximates input xr ==>
    matrix_approximates output (normalize_hidden wr xr)

inline_for_extraction noextract
let projection_relation (#score:Type0) {| Sc.scalar score, N.real_like score |}
  (embedding:matrix BF16.t 151936 1024) (normalized:hidden BF16.t)
  (scores:vector score 151936) : prop =
  forall (er:matrix real 151936 1024) (yr:hidden real).
    matrix_approximates embedding er /\ matrix_approximates normalized yr ==>
    vector_approximates scores (linear er (yr 6))

inline_for_extraction noextract
let same_vector (#a:Type0) (#n:nat) (left right:vector a n) : prop =
  forall (i:index n). left i == right i

inline_for_extraction noextract
let same_matrix (#a:Type0) (#rows #cols:nat)
  (left right:matrix a rows cols) : prop =
  forall (row:index rows). same_vector (left row) (right row)

inline_for_extraction noextract
let same_heads (#a:Type0) (#count:nat) (left right:heads a count) : prop =
  forall (token:index 7). same_matrix (left token) (right token)

inline_for_extraction noextract
let same_parameters (#a:Type0) (left right:parameters a) : prop =
  same_vector left.input_layernorm right.input_layernorm /\
  same_matrix left.q_proj right.q_proj /\
  same_matrix left.k_proj right.k_proj /\
  same_matrix left.v_proj right.v_proj /\
  same_vector left.q_norm right.q_norm /\
  same_vector left.k_norm right.k_norm /\
  same_matrix left.o_proj right.o_proj /\
  same_vector left.post_attention_layernorm right.post_attention_layernorm /\
  same_matrix left.gate_proj right.gate_proj /\
  same_matrix left.up_proj right.up_proj /\
  same_matrix left.down_proj right.down_proj

inline_for_extraction noextract
let same_inputs (left right:inputs) : prop =
  left.prompt == right.prompt /\
  left.decode_steps == right.decode_steps /\
  same_matrix left.weight.embedding right.weight.embedding /\
  same_vector left.weight.final_norm right.weight.final_norm /\
  (forall (layer:index 28).
    same_parameters (left.weight.layers layer) (right.weight.layers layer))

inline_for_extraction noextract
noeq type cached_kv (a:Type0) = {
  cached_keys: heads a 8;
  cached_values: heads a 8
}

inline_for_extraction noextract
noeq type cache_slot (a:Type0) =
  | Empty : cache_slot a
  | Prefilled : contents:cached_kv a -> cache_slot a

inline_for_extraction noextract
type cache_state (a:Type0) = index 28 -> GTot (cache_slot a)

inline_for_extraction noextract
let same_cache_slot (#a:Type0) (left right:cache_slot a) : prop =
  match left, right with
  | Empty, Empty -> True
  | Prefilled l, Prefilled r ->
    same_heads l.cached_keys r.cached_keys /\
    same_heads l.cached_values r.cached_values
  | _ -> False

inline_for_extraction noextract
let same_cache (#a:Type0) (left right:cache_state a) : prop =
  forall (layer:index 28). same_cache_slot (left layer) (right layer)

inline_for_extraction noextract
let update_cache (#a:Type0) (before:cache_state a) (layer:index 28)
  (values:layer_values a) : GTot (cache_state a) =
  fun other ->
    if other = layer then
      Prefilled { cached_keys = values.keys; cached_values = values.values }
    else before other

inline_for_extraction noextract
let cache_length (#a:Type0) (slot:cache_slot a) : GTot nat =
  match slot with
  | Empty -> 0
  | Prefilled _ -> 7

inline_for_extraction noextract
let cache_frame (#a:Type0) (before:cache_state a) (layer:index 28)
  (values:layer_values a) (other:index 28{other <> layer})
  : Lemma (update_cache before layer values other == before other) = ()

inline_for_extraction noextract
noeq type rotary_state (a:Type0) = {
  inv_freq: vector a 64;
  original_inv_freq: vector a 64;
  attention_scaling: a;
  max_seq_len_cached: nat;
  original_max_seq_len: nat
}

inline_for_extraction noextract
let same_rotary_state (#a:Type0) (left right:rotary_state a) : prop =
  same_vector left.inv_freq right.inv_freq /\
  same_vector left.original_inv_freq right.original_inv_freq /\
  left.attention_scaling == right.attention_scaling /\
  left.max_seq_len_cached == right.max_seq_len_cached /\
  left.original_max_seq_len == right.original_max_seq_len

inline_for_extraction noextract
let rotary_initialized (#a:Type0) {| Sc.scalar a |} (state:rotary_state a) : prop =
  same_vector state.original_inv_freq state.inv_freq /\
  state.attention_scaling == Sc.one /\
  state.max_seq_len_cached == 40960 /\
  state.original_max_seq_len == 40960

inline_for_extraction noextract
let rotary_generation (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (state:rotary_state a) (tables:rotary_tables a) : prop =
  vector_approximates state.inv_freq inverse_frequency /\
  N.v_approximates state.attention_scaling 1.0R /\
  (forall (frequency:vector real 64) (factor:real).
    vector_approximates state.inv_freq frequency /\
    N.v_approximates state.attention_scaling factor ==>
    rotary_approximates tables (tables_from_frequency frequency factor))

inline_for_extraction noextract
noeq type execution (a score:Type0) = {
  embedded: hidden BF16.t;
  position_ids: index 7 -> int;
  cache_positions: index 7 -> int;
  checked_mask: matrix real 7 7;
  rotary_initial: rotary_state a;
  rotary_final: rotary_state a;
  rope: rotary_tables a;
  layer: index 28 -> GTot (layer_values a);
  cache_at: index 29 -> GTot (cache_state a);
  normalized: hidden BF16.t;
  scores: vector score 151936;
  selected: token_id
}

inline_for_extraction noextract
noeq type outputs = {
  tokens: returned_tokens;
  normalized_hidden: hidden BF16.t
}

inline_for_extraction noextract
noeq type post_state (a ambient:Type0) = {
  owned_inputs: inputs;
  history: returned_tokens;
  cache: cache_state a;
  rotary: rotary_state a;
  frame: ambient
}

inline_for_extraction noextract
let append_token (prompt:prompt_tokens) (selected:token_id) : Tot returned_tokens =
  S.append prompt (S.create 1 selected)

inline_for_extraction noextract
let embedding_copy (#a #score:Type0) (input:inputs) (run:execution a score) : prop =
  same_matrix run.embedded (embedding_lookup input.weight input.prompt)

inline_for_extraction noextract
let cache_execution (#a #score:Type0) (run:execution a score) : prop =
  (forall (layer:index 28). same_cache_slot (run.cache_at 0 layer) Empty) /\
  (forall (layer:index 28).
    same_cache (run.cache_at (layer + 1))
      (update_cache (run.cache_at layer) layer (run.layer layer)))

(* This contract is owed even when numerical-domain hypotheses fail.
   Equality here is datum equality, not IEEE numerical equality or approximation.
   Native safety, ownership, initialization and frame proofs remain additional
   representation obligations; a pure relation cannot assert memory safety. *)
inline_for_extraction noextract
let structural_postcondition (#a #score #ambient:Type0) {| Sc.scalar a |}
  (input:inputs) (before:ambient) (run:execution a score)
  (output:outputs) (after:post_state a ambient) : prop =
  same_inputs input after.owned_inputs /\
  embedding_copy input run /\
  (forall (token:index 7).
    run.position_ids token == token /\ run.cache_positions token == token) /\
  (forall (query key:index 7). run.checked_mask query key == prefill_mask query key) /\
  rotary_initialized run.rotary_initial /\
  same_rotary_state run.rotary_initial run.rotary_final /\
  cache_execution run /\
  same_cache after.cache (run.cache_at 28) /\
  same_rotary_state after.rotary run.rotary_final /\
  output.tokens == append_token input.prompt run.selected /\
  same_matrix output.normalized_hidden run.normalized /\
  after.history == output.tokens /\
  after.frame == before

(* The producer-consumer edges are explicit: lookup -> layer 0 -> ... -> 27
   -> BF16 final norm -> tied projection -> selector. Neither the final
   mathematical result nor a freely chosen operation predicate is a premise. *)
inline_for_extraction noextract
let numerical_execution (#a #score:Type0)
  {| Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:execution a score) : prop =
  rotary_generation run.rotary_initial run.rope /\
  decoder_relation (input.weight.layers 0) run.embedded run.rope (run.layer 0) /\
  (forall (layer:index 28{0 < layer}).
    decoder_relation (input.weight.layers layer)
      (run.layer (layer - 1)).hidden_after run.rope (run.layer layer)) /\
  final_norm_relation input.weight.final_norm (run.layer 27).hidden_after run.normalized /\
  projection_relation input.weight.embedding run.normalized run.scores

inline_for_extraction noextract
let selector_domain (#score:Type0) {| F.floating score |}
  (scores:vector score 151936) : prop =
  forall (token:token_id). F.kind (scores token) == F.Finite

inline_for_extraction noextract
let maximal_actual_index (#score:Type0) {| F.floating score |}
  (scores:vector score 151936) (selected:token_id) : prop =
  forall (token:token_id). F.lte (scores token) (scores selected)

inline_for_extraction noextract
let postcondition (#a #score #ambient:Type0)
  {| Sc.scalar a, N.real_like a, F.floating score, N.real_like score |}
  (input:inputs) (before:ambient) (run:execution a score)
  (output:outputs) (after:post_state a ambient) : prop =
  structural_postcondition input before run output after /\
  numerical_execution input run /\
  selector_domain run.scores /\
  maximal_actual_index run.scores run.selected

inline_for_extraction noextract
let numerical_denotation (#a #score:Type0)
  {| Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:execution a score) : prop =
  let weight = real_weights input.weight in
  (forall (layer:index 28).
    layer_approximates (run.layer layer) (mathematical_layer weight input.prompt layer)) /\
  matrix_approximates run.normalized (mathematical_hidden weight input.prompt) /\
  vector_approximates run.scores (mathematical_logits weight input.prompt)

inline_for_extraction noextract
let rotary_generation_canonical (#a:Type0) {| Sc.scalar a, N.real_like a |}
  (state:rotary_state a) (tables:rotary_tables a)
  : Lemma
      (requires rotary_generation state tables)
      (ensures rotary_approximates tables (canonical_rotary ())) = ()

inline_for_extraction noextract
let embedding_copy_related (#a #score:Type0) (input:inputs) (run:execution a score)
  : Lemma
      (requires embedding_copy input run)
      (ensures matrix_approximates run.embedded
        (embedding_lookup (real_weights input.weight) input.prompt)) =
  C.forall_intro_2 #(index 7) #(fun _ -> index 1024)
    #(fun token channel ->
      N.v_approximates (run.embedded token channel)
        (embedding_lookup (real_weights input.weight) input.prompt token channel))
    (fun token channel ->
      N.to_real_ok (input.weight.embedding (S.index input.prompt token) channel))

inline_for_extraction noextract
let decoder_relation_apply (#x #a:Type0)
  {| Sc.scalar x, N.real_like x, Sc.scalar a, N.real_like a |}
  (weight:parameters BF16.t) (input:hidden x) (data:rotary_tables a)
  (output:layer_values a)
  (wr:parameters real) (xr:hidden real) (dr:rotary_tables real)
  : Lemma
      (requires decoder_relation weight input data output /\
        parameters_approximates weight wr /\ matrix_approximates input xr /\
        rotary_approximates data dr)
      (ensures layer_approximates output (decoder_layer wr dr xr)) = ()

inline_for_extraction noextract
let rec decoder_chain_related (#a #score:Type0)
  {| Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:execution a score) (count:pos{count <= 28})
  : Lemma
      (requires embedding_copy input run /\ numerical_execution input run)
      (ensures layer_approximates (run.layer (count - 1))
        (mathematical_layer (real_weights input.weight) input.prompt (count - 1)))
      (decreases count)
  =
  let weight = real_weights input.weight in
  real_weights_related input.weight;
  rotary_generation_canonical run.rotary_initial run.rope;
  if count = 1 then (
    embedding_copy_related input run;
    decoder_relation_apply (input.weight.layers 0) run.embedded run.rope
      (run.layer 0) (weight.layers 0)
      (embedding_lookup weight input.prompt) (canonical_rotary ())
  ) else (
    decoder_chain_related input run (count - 1);
    decoder_relation_apply (input.weight.layers (count - 1))
      (run.layer (count - 2)).hidden_after run.rope (run.layer (count - 1))
      (weight.layers (count - 1)) (hidden_prefix weight input.prompt (count - 1))
      (canonical_rotary ()))

inline_for_extraction noextract
let numerical_execution_related (#a #score:Type0)
  {| Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:execution a score)
  : Lemma
      (requires embedding_copy input run /\ numerical_execution input run)
      (ensures numerical_denotation input run)
  =
  real_weights_related input.weight;
  decoder_chain_related input run 28;
  let weight = real_weights input.weight in
  assert (matrix_approximates run.normalized (mathematical_hidden weight input.prompt));
  assert (vector_approximates run.scores (mathematical_logits weight input.prompt));
  C.forall_intro #(index 28)
    #(fun layer ->
      layer_approximates (run.layer layer) (mathematical_layer weight input.prompt layer))
    (fun layer -> decoder_chain_related input run (layer + 1))

inline_for_extraction noextract
let structural_outputs (#a #score #ambient:Type0) {| Sc.scalar a |}
  (input:inputs) (before:ambient) (run:execution a score)
  (output:outputs) (after:post_state a ambient)
  : Lemma
      (requires structural_postcondition input before run output after)
      (ensures
        same_inputs input after.owned_inputs /\
        S.length output.tokens == 8 /\
        (forall (token:index 7). S.index output.tokens token == S.index input.prompt token) /\
        S.index output.tokens 7 == run.selected /\
        same_matrix output.normalized_hidden run.normalized /\
        after.history == output.tokens /\ after.frame == before) = ()

inline_for_extraction noextract
let same_cache_slot_transitive (#a:Type0) (left middle right:cache_slot a)
  : Lemma
      (requires same_cache_slot left middle /\ same_cache_slot middle right)
      (ensures same_cache_slot left right) = ()

inline_for_extraction noextract
let rec cache_prefix_exact (#a #score:Type0)
  (run:execution a score) (count:nat{count <= 28})
  : Lemma
      (requires cache_execution run)
      (ensures forall (layer:index 28).
        same_cache_slot (run.cache_at count layer)
          (if layer < count then
            Prefilled {
              cached_keys = (run.layer layer).keys;
              cached_values = (run.layer layer).values
            }
          else Empty))
      (decreases count)
  =
  if count > 0 then (
    cache_prefix_exact run (count - 1);
    C.forall_intro #(index 28)
      #(fun layer ->
        same_cache_slot (run.cache_at count layer)
          (if layer < count then
            Prefilled {
              cached_keys = (run.layer layer).keys;
              cached_values = (run.layer layer).values
            }
          else Empty))
      (fun layer ->
        if layer = count - 1 then ()
        else (
          cache_frame (run.cache_at (count - 1)) (count - 1)
            (run.layer (count - 1)) layer;
          same_cache_slot_transitive (run.cache_at count layer)
            (run.cache_at (count - 1) layer)
            (if layer < count - 1 then
              Prefilled {
                cached_keys = (run.layer layer).keys;
                cached_values = (run.layer layer).values
              }
            else Empty))))

inline_for_extraction noextract
let structural_cache_contents (#a #score #ambient:Type0) {| Sc.scalar a |}
  (input:inputs) (before:ambient) (run:execution a score)
  (output:outputs) (after:post_state a ambient)
  : Lemma
      (requires structural_postcondition input before run output after)
      (ensures forall (layer:index 28).
        cache_length (after.cache layer) == 7 /\
        same_cache_slot (after.cache layer)
          (Prefilled {
            cached_keys = (run.layer layer).keys;
            cached_values = (run.layer layer).values
          }))
  =
  cache_prefix_exact run 28;
  C.forall_intro #(index 28)
    #(fun layer ->
      cache_length (after.cache layer) == 7 /\
      same_cache_slot (after.cache layer)
        (Prefilled {
          cached_keys = (run.layer layer).keys;
          cached_values = (run.layer layer).values
        }))
    (fun layer ->
      same_cache_slot_transitive (after.cache layer) (run.cache_at 28 layer)
        (Prefilled {
          cached_keys = (run.layer layer).keys;
          cached_values = (run.layer layer).values
        }))

inline_for_extraction noextract
let postcondition_meaning (#a #score #ambient:Type0)
  {| Sc.scalar a, N.real_like a, F.floating score, N.real_like score |}
  (input:inputs) (before:ambient) (run:execution a score)
  (output:outputs) (after:post_state a ambient)
  : Lemma
      (requires postcondition input before run output after)
      (ensures
        numerical_denotation input run /\
        matrix_approximates output.normalized_hidden
          (mathematical_hidden (real_weights input.weight) input.prompt) /\
        selector_domain run.scores /\
        maximal_actual_index run.scores (S.index output.tokens 7) /\
        (forall (token:index 7). S.index output.tokens token == S.index input.prompt token))
  =
  numerical_execution_related input run;
  structural_outputs input before run output after

inline_for_extraction noextract
let any_maximum_allowed (#a #score #ambient:Type0)
  {| Sc.scalar a, N.real_like a, F.floating score, N.real_like score |}
  (input:inputs) (before:ambient) (run:execution a score)
  (output:outputs) (after:post_state a ambient) (selected:token_id)
  : Lemma
      (requires postcondition input before run output after /\
        maximal_actual_index run.scores selected)
      (ensures
        postcondition input before { run with selected = selected }
          { output with tokens = append_token input.prompt selected }
          { after with history = append_token input.prompt selected }) = ()

inline_for_extraction noextract
let distinct_selected_histories (prompt:prompt_tokens) (left right:token_id)
  : Lemma
      (requires left <> right)
      (ensures append_token prompt left =!= append_token prompt right) =
  assert (S.index (append_token prompt left) 7 == left);
  assert (S.index (append_token prompt right) 7 == right)

inline_for_extraction noextract
let prefix_softmax_future_zero
  (token:index 7) (scores:vector real (token + 1))
  (key:index 7{token < key})
  : Lemma (prefix_softmax token scores key == 0.0R) = ()

inline_for_extraction noextract
let prefix_softmax_denominator_positive
  (token:index 7) (scores:vector real (token + 1))
  : Lemma (sum (token + 1) (fun key -> Exp.exp (scores key)) >. 0.0R) =
  sum_positive (token + 1) (fun key -> Exp.exp (scores key))

(* Only the observation types are separated; all equations and scalar
   instances below come from the unchanged accepted specification. *)
inline_for_extraction noextract
let mixed_rotary_generation (#metadata #a:Type0)
  {| Sc.scalar metadata, N.real_like metadata, Sc.scalar a, N.real_like a |}
  (state:rotary_state metadata) (tables:rotary_tables a) : prop =
  vector_approximates state.inv_freq inverse_frequency /\
  N.v_approximates state.attention_scaling 1.0R /\
  (forall (frequency:vector real 64) (factor:real).
    vector_approximates state.inv_freq frequency /\
    N.v_approximates state.attention_scaling factor ==>
    rotary_approximates tables (tables_from_frequency frequency factor))

inline_for_extraction noextract
noeq type mixed_execution (metadata a score:Type0) = {
  embedded: hidden BF16.t;
  position_ids: index 7 -> int;
  cache_positions: index 7 -> int;
  checked_mask: matrix real 7 7;
  rotary_initial: rotary_state metadata;
  rotary_final: rotary_state metadata;
  rope: rotary_tables a;
  layer: index 28 -> GTot (layer_values a);
  cache_at: index 29 -> GTot (cache_state a);
  normalized: hidden BF16.t;
  scores: vector score 151936;
  selected: token_id
}

inline_for_extraction noextract
noeq type mixed_post_state (metadata a ambient:Type0) = {
  owned_inputs: inputs;
  history: returned_tokens;
  cache: cache_state a;
  rotary: rotary_state metadata;
  frame: ambient
}

inline_for_extraction noextract
let mixed_embedding_copy (#metadata #a #score:Type0)
  (input:inputs) (run:mixed_execution metadata a score) : prop =
  same_matrix run.embedded (embedding_lookup input.weight input.prompt)

inline_for_extraction noextract
let mixed_cache_execution (#metadata #a #score:Type0) (run:mixed_execution metadata a score) : prop =
  (forall (layer:index 28). same_cache_slot (run.cache_at 0 layer) Empty) /\
  (forall (layer:index 28).
    same_cache (run.cache_at (layer + 1))
      (update_cache (run.cache_at layer) layer (run.layer layer)))

inline_for_extraction noextract
let mixed_structural_postcondition (#metadata #a #score #ambient:Type0) {| Sc.scalar metadata |}
  (input:inputs) (before:ambient) (run:mixed_execution metadata a score)
  (output:outputs) (after:mixed_post_state metadata a ambient) : prop =
  same_inputs input after.owned_inputs /\
  mixed_embedding_copy input run /\
  (forall (token:index 7).
    run.position_ids token == token /\ run.cache_positions token == token) /\
  (forall (query key:index 7). run.checked_mask query key == prefill_mask query key) /\
  rotary_initialized run.rotary_initial /\
  same_rotary_state run.rotary_initial run.rotary_final /\
  mixed_cache_execution run /\
  same_cache after.cache (run.cache_at 28) /\
  same_rotary_state after.rotary run.rotary_final /\
  output.tokens == append_token input.prompt run.selected /\
  same_matrix output.normalized_hidden run.normalized /\
  after.history == output.tokens /\
  after.frame == before

inline_for_extraction noextract
let mixed_numerical_execution (#metadata #a #score:Type0)
  {| Sc.scalar metadata, N.real_like metadata,
     Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:mixed_execution metadata a score) : prop =
  mixed_rotary_generation run.rotary_initial run.rope /\
  decoder_relation (input.weight.layers 0) run.embedded run.rope (run.layer 0) /\
  (forall (layer:index 28{0 < layer}).
    decoder_relation (input.weight.layers layer)
      (run.layer (layer - 1)).hidden_after run.rope (run.layer layer)) /\
  final_norm_relation input.weight.final_norm (run.layer 27).hidden_after run.normalized /\
  projection_relation input.weight.embedding run.normalized run.scores

inline_for_extraction noextract
let mixed_numerical_denotation (#metadata #a #score:Type0)
  {| Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:mixed_execution metadata a score) : prop =
  let weight = real_weights input.weight in
  (forall (layer:index 28).
    layer_approximates (run.layer layer)
      (mathematical_layer weight input.prompt layer)) /\
  matrix_approximates run.normalized (mathematical_hidden weight input.prompt) /\
  vector_approximates run.scores (mathematical_logits weight input.prompt)

inline_for_extraction noextract
let mixed_postcondition (#metadata #a #score #ambient:Type0)
  {| Sc.scalar metadata, N.real_like metadata,
     Sc.scalar a, N.real_like a, F.floating score, N.real_like score |}
  (input:inputs) (before:ambient) (run:mixed_execution metadata a score)
  (output:outputs) (after:mixed_post_state metadata a ambient) : prop =
  mixed_structural_postcondition input before run output after /\
  mixed_numerical_execution input run /\
  selector_domain run.scores /\
  maximal_actual_index run.scores run.selected

inline_for_extraction noextract
let mixed_rotary_generation_canonical (#metadata #a:Type0)
  {| Sc.scalar metadata, N.real_like metadata, Sc.scalar a, N.real_like a |}
  (state:rotary_state metadata) (tables:rotary_tables a)
  : Lemma
      (requires mixed_rotary_generation state tables)
      (ensures rotary_approximates tables (canonical_rotary ())) = ()

inline_for_extraction noextract
let mixed_embedding_copy_related (#metadata #a #score:Type0)
  (input:inputs) (run:mixed_execution metadata a score)
  : Lemma
      (requires mixed_embedding_copy input run)
      (ensures matrix_approximates run.embedded
        (embedding_lookup (real_weights input.weight) input.prompt)) =
  C.forall_intro_2 #(index 7) #(fun _ -> index 1024)
    #(fun token channel ->
      N.v_approximates (run.embedded token channel)
        (embedding_lookup (real_weights input.weight) input.prompt token channel))
    (fun token channel ->
      N.to_real_ok (input.weight.embedding (S.index input.prompt token) channel))

inline_for_extraction noextract
let rec mixed_decoder_chain_related (#metadata #a #score:Type0)
  {| Sc.scalar metadata, N.real_like metadata,
     Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:mixed_execution metadata a score) (count:pos{count <= 28})
  : Lemma
      (requires mixed_embedding_copy input run /\ mixed_numerical_execution input run)
      (ensures layer_approximates (run.layer (count - 1))
        (mathematical_layer (real_weights input.weight) input.prompt (count - 1)))
      (decreases count)
  =
  let weight = real_weights input.weight in
  real_weights_related input.weight;
  mixed_rotary_generation_canonical run.rotary_initial run.rope;
  if count = 1 then (
    mixed_embedding_copy_related input run;
    decoder_relation_apply (input.weight.layers 0) run.embedded run.rope
      (run.layer 0) (weight.layers 0)
      (embedding_lookup weight input.prompt) (canonical_rotary ())
  ) else (
    mixed_decoder_chain_related input run (count - 1);
    decoder_relation_apply (input.weight.layers (count - 1))
      (run.layer (count - 2)).hidden_after run.rope (run.layer (count - 1))
      (weight.layers (count - 1)) (hidden_prefix weight input.prompt (count - 1))
      (canonical_rotary ()))

inline_for_extraction noextract
let mixed_numerical_execution_related (#metadata #a #score:Type0)
  {| Sc.scalar metadata, N.real_like metadata,
     Sc.scalar a, N.real_like a, Sc.scalar score, N.real_like score |}
  (input:inputs) (run:mixed_execution metadata a score)
  : Lemma
      (requires mixed_embedding_copy input run /\ mixed_numerical_execution input run)
      (ensures mixed_numerical_denotation input run) =
  real_weights_related input.weight;
  mixed_decoder_chain_related input run 28;
  let weight = real_weights input.weight in
  assert (matrix_approximates run.normalized (mathematical_hidden weight input.prompt));
  assert (vector_approximates run.scores (mathematical_logits weight input.prompt));
  C.forall_intro #(index 28)
    #(fun layer ->
      layer_approximates (run.layer layer)
        (mathematical_layer weight input.prompt layer))
    (fun layer -> mixed_decoder_chain_related input run (layer + 1))

inline_for_extraction noextract
let mixed_postcondition_meaning (#metadata #a #score #ambient:Type0)
  {| Sc.scalar metadata, N.real_like metadata,
     Sc.scalar a, N.real_like a, F.floating score, N.real_like score |}
  (input:inputs) (before:ambient) (run:mixed_execution metadata a score)
  (output:outputs) (after:mixed_post_state metadata a ambient)
  : Lemma
      (requires mixed_postcondition input before run output after)
      (ensures mixed_numerical_denotation input run /\
        matrix_approximates output.normalized_hidden
          (mathematical_hidden (real_weights input.weight) input.prompt) /\
        selector_domain run.scores /\
        maximal_actual_index run.scores (S.index output.tokens 7) /\
        (forall (token:index 7). S.index output.tokens token == S.index input.prompt token)) =
  mixed_numerical_execution_related input run

inline_for_extraction noextract
let mixed_legacy_execution (#a #score:Type0) (run:execution a score)
  : GTot (mixed_execution a a score) = {
  embedded = run.embedded; position_ids = run.position_ids;
  cache_positions = run.cache_positions; checked_mask = run.checked_mask;
  rotary_initial = run.rotary_initial; rotary_final = run.rotary_final;
  rope = run.rope; layer = run.layer; cache_at = run.cache_at;
  normalized = run.normalized; scores = run.scores; selected = run.selected
}

inline_for_extraction noextract
let mixed_legacy_state (#a #ambient:Type0) (after:post_state a ambient)
  : GTot (mixed_post_state a a ambient) = {
  owned_inputs = after.owned_inputs; history = after.history;
  cache = after.cache; rotary = after.rotary; frame = after.frame
}

inline_for_extraction noextract
let mixed_legacy_postcondition_equivalence (#a #score #ambient:Type0)
  {| Sc.scalar a, N.real_like a, F.floating score, N.real_like score |}
  (input:inputs) (before:ambient) (run:execution a score)
  (output:outputs) (after:post_state a ambient)
  : Lemma (
      mixed_postcondition input before (mixed_legacy_execution run) output (mixed_legacy_state after)
      <==> postcondition input before run output after) = ()

