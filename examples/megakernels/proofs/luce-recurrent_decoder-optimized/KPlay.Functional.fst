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
module LocalSoftplus = Luce.Softplus.Fast
module Shared = KPlay.SharedSpec
module Sum = Kuiper.Sum

#set-options "--fuel 2 --ifuel 1 --z3rlimit 30"

inline_for_extraction noextract
let head_count : pos = heads
inline_for_extraction noextract
let tap_count : pos = conv_width
inline_for_extraction noextract
let mlp_width : pos = intermediate_width
inline_for_extraction noextract
let value_width : pos = mixer_width
inline_for_extraction noextract
let channel_count : pos = qkv_width
inline_for_extraction noextract
let head_state_elements : pos = head_width * head_width
inline_for_extraction noextract
let state_elements : pos = recurrent_count
inline_for_extraction noextract
let convolution_elements : pos = convolution_count
inline_for_extraction noextract
let output_gate_offset : nat = z_offset
inline_for_extraction noextract
let alpha_offset : nat = a_offset
inline_for_extraction noextract
let convolution_offset : nat = conv_kernel_offset
inline_for_extraction noextract
let log_decay_offset : nat = a_log_offset
inline_for_extraction noextract
let alpha_bias_offset : nat = dt_bias_offset
inline_for_extraction noextract
let output_offset : nat = out_projection_offset
inline_for_extraction noextract
let weight_elements : pos = weight_count
inline_for_extraction noextract
let epsilon : Sqrt.rpos = Shared.epsilon
inline_for_extraction noextract
let real_vector (count : nat) = lseq real count

let vector_slice
  (#n : nat)
  (values : real_vector n)
  (offset count : nat{offset + count <= n})
  : GTot (real_vector count)
  = S.init_ghost count (fun i -> S.index values (offset + i))

let dot
  (#n : nat)
  (left right : real_vector n)
  : GTot real
  = rsum (S.init_ghost n (fun i -> S.index left i *. S.index right i))

let rec sum_squares
  (#n : nat)
  (values : real_vector n)
  (count : nat{count <= n})
  : GTot Sqrt.rnonneg (decreases count)
  = if count = 0 then 0.0R
    else
      let x = S.index values (count - 1) in
      sum_squares values (count - 1) +. x *. x

let rms_scale
  (#n : pos)
  (values : real_vector n)
  : GTot Sqrt.rpos
  = Sqrt.rsqrt (sum_squares values n /. R.of_int n +. epsilon)

let linear
  (weights : real_vector weight_elements)
  (offset : nat)
  (rows columns : pos{offset + rows * columns <= weight_elements})
  (values : real_vector columns)
  : GTot (real_vector rows)
  = S.init_ghost rows (fun row ->
      rsum (S.init_ghost columns (fun column ->
        S.index weights (offset + row * columns + column) *.
          S.index values column)))

let hidden_normalization
  (weights : real_vector weight_elements)
  (offset : nat{offset + hidden_width <= weight_elements})
  (hidden : real_vector hidden_width)
  : GTot (real_vector hidden_width)
  = let scale = rms_scale hidden in
    S.init_ghost hidden_width (fun i ->
      S.index hidden i *. scale *. (1.0R +. S.index weights (offset + i)))

let sigmoid (x : real) : GTot real =
  1.0R /. (1.0R +. exp (0.0R -. x))

let silu (x : real) : GTot real =
  x *. sigmoid x

inline_for_extraction noextract
let softplus_linear_branch (biased_alpha : f32) : bool =
  (of_int 20L <: f32) `lt` biased_alpha

let softplus (biased_alpha : f32) (x : real) : GTot real =
  Shared.softplus x

let initial_state
  (position : int)
  (previous : real_vector (previous_elements position state_elements))
  : GTot (real_vector state_elements)
  = S.init_ghost state_elements (fun i ->
      if position = 0 then 0.0R else S.index previous i)

let next_convolution
  (position : int)
  (previous : real_vector (previous_elements position convolution_elements))
  (projected : real_vector channel_count)
  : GTot (real_vector convolution_elements)
  = S.init_ghost convolution_elements (fun i ->
      let channel = i / tap_count in
      let tap = i % tap_count in
      if tap = tap_count - 1 then S.index projected channel
      else if position = 0 then 0.0R
      else S.index previous (channel * tap_count + tap + 1))

let convolved_head
  (weights : real_vector weight_elements)
  (convolution : real_vector convolution_elements)
  (region : nat{region < 3})
  (head : nat{head < head_count})
  : GTot (real_vector head_width)
  = S.init_ghost head_width (fun i ->
      let channel = region * value_width + head * head_width + i in
      silu (rsum (S.init_ghost tap_count (fun tap ->
        S.index convolution (channel * tap_count + tap) *.
          S.index weights (convolution_offset + channel * tap_count + tap)))))

let normalized_key (values : real_vector head_width)
  : GTot (real_vector head_width)
  = let scale = Sqrt.rsqrt (sum_squares values head_width +. epsilon) in
    S.init_ghost head_width (fun i -> S.index values i *. scale)

let normalized_query (values : real_vector head_width)
  : GTot (real_vector head_width)
  = let key_normalized = normalized_key values in
    S.init_ghost head_width (fun i ->
      S.index key_normalized i /. Sqrt.sqrt (R.of_int head_width))

let delta_state
  (previous : real_vector head_state_elements)
  (key value : real_vector head_width)
  (beta decay : real)
  : GTot (real_vector head_state_elements)
  = S.init_ghost head_state_elements (fun index ->
      let row = index / head_width in
      let column = index % head_width in
      let predicted = dot (vector_slice previous (row * head_width) head_width) key in
      let error = beta *. (S.index value row -. decay *. predicted) in
      decay *. S.index previous index +. error *. S.index key column)

let state_readout
  (states : real_vector head_state_elements)
  (query : real_vector head_width)
  : GTot (real_vector head_width)
  = S.init_ghost head_width (fun row ->
      dot (vector_slice states (row * head_width) head_width) query)

let gated_readout
  (weights : real_vector weight_elements)
  (readout gate : real_vector head_width)
  : GTot (real_vector head_width)
  = let scale = rms_scale readout in
    S.init_ghost head_width (fun i ->
      S.index readout i *. scale *.
        S.index weights (gated_norm_offset + i) *. silu (S.index gate i))

let head_transition
  (weights : real_vector weight_elements)
  (convolution : real_vector convolution_elements)
  (previous : real_vector state_elements)
  (gates : real_vector value_width)
  (beta_projection alpha_argument : real_vector head_count)
  (controls : lseq f32 head_count)
  (head : nat{head < head_count})
  : GTot (real_vector head_state_elements & real_vector head_width)
  = let query = normalized_query (convolved_head weights convolution 0 head) in
    let key = normalized_key (convolved_head weights convolution 1 head) in
    let value = convolved_head weights convolution 2 head in
    let beta = sigmoid (S.index beta_projection head) in
    let sp = softplus (S.index controls head) (S.index alpha_argument head) in
    let decay = exp ((0.0R -. exp (S.index weights (log_decay_offset + head))) *. sp) in
    let old_head = vector_slice previous (head * head_state_elements) head_state_elements in
    let new_head = delta_state old_head key value beta decay in
    let readout = state_readout new_head query in
    let gate = vector_slice gates (head * head_width) head_width in
    (new_head, gated_readout weights readout gate)

noeq noextract
type layer_values = {
  alpha_argument : real_vector head_count;
  hidden_result : real_vector hidden_width;
  state_result : real_vector state_elements;
  convolution_result : real_vector convolution_elements
}

let recurrent_model
  (weights : real_vector weight_elements)
  (hidden : real_vector hidden_width)
  (position : int)
  (previous_states : real_vector (previous_elements position state_elements))
  (previous_convolution : real_vector (previous_elements position convolution_elements))
  (controls : lseq f32 head_count)
  : GTot layer_values
  = let normalized = hidden_normalization weights input_norm_offset hidden in
    let projected = linear weights qkv_offset channel_count hidden_width normalized in
    let gates = linear weights output_gate_offset value_width hidden_width normalized in
    let beta_projection = linear weights beta_offset head_count hidden_width normalized in
    let alpha_projection = linear weights alpha_offset head_count hidden_width normalized in
    let alpha_argument = S.init_ghost head_count (fun head ->
      S.index alpha_projection head +. S.index weights (alpha_bias_offset + head)) in
    let convolution = next_convolution position previous_convolution projected in
    let previous = initial_state position previous_states in
    let heads = S.init_ghost head_count (fun head ->
      head_transition weights convolution previous gates beta_projection
        alpha_argument controls head) in
    let states = S.init_ghost state_elements (fun i ->
      let (state, _) = S.index heads (i / head_state_elements) in
      S.index state (i % head_state_elements)) in
    let activation = S.init_ghost value_width (fun i ->
      let (_, values) = S.index heads (i / head_width) in
      S.index values (i % head_width)) in
    let attention_output = linear weights output_offset hidden_width value_width activation in
    let residual = S.init_ghost hidden_width (fun i ->
      S.index hidden i +. S.index attention_output i) in
    let post_normalized = hidden_normalization weights post_norm_offset residual in
    let mlp_gate = linear weights mlp_gate_offset mlp_width hidden_width post_normalized in
    let mlp_up = linear weights mlp_up_offset mlp_width hidden_width post_normalized in
    let mlp_activation = S.init_ghost mlp_width (fun i ->
      silu (S.index mlp_gate i) *. S.index mlp_up i) in
    let mlp_output = linear weights mlp_down_offset hidden_width mlp_width mlp_activation in
    let result = S.init_ghost hidden_width (fun i ->
      S.index residual i +. S.index mlp_output i) in
    {
      alpha_argument = alpha_argument;
      hidden_result = result;
      state_result = states;
      convolution_result = convolution
    }

let retained_taps_post
  (position : int)
  (previous : lseq f32 (previous_elements position convolution_elements))
  (convolution : lseq f32 convolution_elements)
  : prop
  = forall (channel : nat{channel < channel_count})
           (tap : nat{tap < tap_count - 1}).
      S.index convolution (channel * tap_count + tap) ==
        (if position = 0 then (zero <: f32)
         else S.index previous (channel * tap_count + tap + 1))

let recurrent_decoder_post
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (position : int)
  (previous_states : lseq f32 (previous_elements position state_elements))
  (previous_convolution : lseq f32 (previous_elements position convolution_elements))
  (result : lseq bf16 hidden_width)
  (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements)
  : prop
  = retained_taps_post position previous_convolution convolution /\
    (exists (controls : lseq f32 head_count).
      forall (real_weights : real_vector weight_elements)
             (real_hidden : real_vector hidden_width)
             (real_states : real_vector (previous_elements position state_elements))
             (real_convolution : real_vector (previous_elements position convolution_elements)).
        weights %~ real_weights /\ hidden %~ real_hidden /\
        previous_states %~ real_states /\ previous_convolution %~ real_convolution ==>
        (let model = recurrent_model real_weights real_hidden position
           real_states real_convolution controls in
         controls %~ model.alpha_argument /\
         result %~ model.hidden_result /\
         states %~ model.state_result /\
         convolution %~ model.convolution_result))

let rec prefix (n : nat) (f : nat -> GTot real)
  : GTot real (decreases n)
  = if n = 0 then 0.0R else prefix (n - 1) f +. f (n - 1)

let prefix_step (n : nat) (f : nat -> GTot real)
  : Lemma (prefix (n + 1) f == prefix n f +. f n)
  = ()

let rec prefix_ext (n : nat) (f g : nat -> GTot real)
  : Lemma
      (requires forall (i : nat). i < n ==> f i == g i)
      (ensures prefix n f == prefix n g)
      (decreases n)
  = if n > 0 then prefix_ext (n - 1) f g

let rec prefix_add (n : nat) (f g : nat -> GTot real)
  : Lemma
      (ensures prefix n (fun i -> f i +. g i) == prefix n f +. prefix n g)
      (decreases n)
  = if n > 0 then prefix_add (n - 1) f g

let rec prefix_zero (n : nat)
  : Lemma (ensures prefix n (fun _ -> 0.0R) == 0.0R) (decreases n)
  = if n > 0 then prefix_zero (n - 1)

let rec prefix_split (n m : nat) (f : nat -> GTot real)
  : Lemma
      (ensures prefix (n + m) f ==
        prefix n f +. prefix m (fun i -> f (n + i)))
      (decreases m)
  = if m > 0 then prefix_split n (m - 1) f

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

let seq_value (s : seq real) (i : nat) : GTot real =
  if i < S.length s then S.index s i else 0.0R

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

let prefix_init (n : nat) (f : nat -> GTot real)
  : Lemma (prefix n f == rsum (S.init_ghost n f))
  = let s = S.init_ghost n f in
    prefix_ext n f (seq_value s);
    prefix_rsum s n;
    assert (S.equal (S.slice s 0 n) s)

let pack_vector (#n:nat) (values:Shared.vector n) : GTot (real_vector n) =
  S.init_ghost n values

let pack_convolution (values:Shared.convolution_state)
  : GTot (real_vector convolution_count) =
  S.init_ghost convolution_count (fun i -> values (i / conv_width) (i % conv_width))

let pack_recurrent (values:Shared.recurrent_state)
  : GTot (real_vector recurrent_count) =
  S.init_ghost recurrent_count (fun i ->
    values (i / head_state_elements) (i % head_width) ((i / head_width) % head_width))

let pack_head (values:Shared.matrix head_width head_width)
  : GTot (real_vector head_state_elements) =
  S.init_ghost head_state_elements (fun i -> values (i % head_width) (i / head_width))

let pack_previous (#n:nat) (position:int) (values:real_vector n)
  : GTot (real_vector (previous_elements position n)) =
  if position = 0 then S.empty else values

let packed_weight (w:Shared.weights) (i:Shared.fin weight_count) : GTot real =
  if i < post_norm_offset then w.input_norm i
  else if i < mlp_gate_offset then w.post_attention_norm (i - post_norm_offset)
  else if i < mlp_up_offset then
    w.mlp_gate ((i - mlp_gate_offset) / hidden_width) ((i - mlp_gate_offset) % hidden_width)
  else if i < mlp_down_offset then
    w.mlp_up ((i - mlp_up_offset) / hidden_width) ((i - mlp_up_offset) % hidden_width)
  else if i < qkv_offset then
    w.mlp_down ((i - mlp_down_offset) / mlp_width) ((i - mlp_down_offset) % mlp_width)
  else if i < z_offset then
    w.in_qkv ((i - qkv_offset) / hidden_width) ((i - qkv_offset) % hidden_width)
  else if i < beta_offset then
    w.in_z ((i - z_offset) / hidden_width) ((i - z_offset) % hidden_width)
  else if i < a_offset then
    w.in_beta ((i - beta_offset) / hidden_width) ((i - beta_offset) % hidden_width)
  else if i < conv_kernel_offset then
    w.in_a ((i - a_offset) / hidden_width) ((i - a_offset) % hidden_width)
  else if i < a_log_offset then
    w.conv_kernel ((i - conv_kernel_offset) / conv_width) ((i - conv_kernel_offset) % conv_width)
  else if i < dt_bias_offset then w.a_log (i - a_log_offset)
  else if i < gated_norm_offset then w.dt_bias (i - dt_bias_offset)
  else if i < out_projection_offset then w.gated_norm (i - gated_norm_offset)
  else
    w.out_projection ((i - out_projection_offset) / mixer_width)
      ((i - out_projection_offset) % mixer_width)

let pack_weights (w:Shared.weights) : GTot (real_vector weight_count) =
  pack_vector (packed_weight w)

let real_weight_vector (width:nat) (offset:nat{offset + width <= weight_count})
  (physical:real_vector weight_count) (logical:Shared.vector width) : GTot prop =
  forall (i:Shared.fin width). S.index physical (offset + i) == logical i

let real_weight_matrix (rows:nat) (columns:pos)
  (offset:nat{offset + rows * columns <= weight_count})
  (physical:real_vector weight_count) (logical:Shared.matrix rows columns) : GTot prop =
  forall (row:Shared.fin rows) (column:Shared.fin columns).
    S.index physical (offset + row * columns + column) == logical row column

let real_weight_views (rw:real_vector weight_count) (w:Shared.weights) : GTot prop =
  real_weight_vector hidden_width input_norm_offset rw w.input_norm /\
  real_weight_vector hidden_width post_norm_offset rw w.post_attention_norm /\
  real_weight_matrix qkv_width hidden_width qkv_offset rw w.in_qkv /\
  real_weight_matrix mixer_width hidden_width z_offset rw w.in_z /\
  real_weight_matrix heads hidden_width beta_offset rw w.in_beta /\
  real_weight_matrix heads hidden_width a_offset rw w.in_a /\
  real_weight_matrix qkv_width conv_width conv_kernel_offset rw w.conv_kernel /\
  real_weight_vector heads a_log_offset rw w.a_log /\
  real_weight_vector heads dt_bias_offset rw w.dt_bias /\
  real_weight_vector head_width gated_norm_offset rw w.gated_norm /\
  real_weight_matrix hidden_width mixer_width out_projection_offset rw w.out_projection /\
  real_weight_matrix mlp_width hidden_width mlp_gate_offset rw w.mlp_gate /\
  real_weight_matrix mlp_width hidden_width mlp_up_offset rw w.mlp_up /\
  real_weight_matrix hidden_width mlp_width mlp_down_offset rw w.mlp_down

#push-options "--fuel 5 --z3rlimit 100"
let packed_weight_views (w:Shared.weights)
  : Lemma (real_weight_views (pack_weights w) w) = ()

let packed_weights_approx (physical:lseq bf16 weight_count) (w:Shared.weights)
  : Lemma
      (requires weights_representation physical w)
      (ensures physical %~ pack_weights w) =
  let point (i:Shared.fin weight_count)
    : Lemma (S.index physical i %~ packed_weight w i) =
    if i < post_norm_offset then ()
    else if i < mlp_gate_offset then ()
    else if i < mlp_up_offset then ()
    else if i < mlp_down_offset then ()
    else if i < qkv_offset then ()
    else if i < z_offset then ()
    else if i < beta_offset then ()
    else if i < a_offset then ()
    else if i < conv_kernel_offset then ()
    else if i < a_log_offset then ()
    else if i < dt_bias_offset then ()
    else if i < gated_norm_offset then ()
    else if i < out_projection_offset then ()
    else ()
  in
  C.forall_intro point
#pop-options

let rec prefix_shared (n:nat) (f:nat -> GTot real) (g:Shared.vector n)
  : Lemma
      (requires forall (i:Shared.fin n). f i == g i)
      (ensures prefix n f == Sum.sum 0 n g)
      (decreases n) =
  if n > 0 then (
    prefix_shared (n - 1) f g;
    Sum.sum_pop_right 0 n g
  )

let sequence_sum_shared (#n:nat) (values:real_vector n)
  (logical:Kuiper.Common.between 0 n -> GTot real)
  : Lemma
      (requires forall (i:Shared.fin n). S.index values i == logical i)
      (ensures rsum values == Sum.sum 0 n logical) =
  prefix_rsum values n;
  prefix_shared n (seq_value values) logical;
  assert (S.equal (S.slice values 0 n) values)

let scale_distributes (a b c:real)
  : Lemma (a *. (b +. c) == a *. b +. a *. c) = ()

let rec shared_sum_scaled (n:nat) (scale:real) (f g:Shared.vector n)
  : Lemma
      (requires forall (i:Shared.fin n). g i == scale *. f i)
      (ensures Sum.sum 0 n g == scale *. Sum.sum 0 n f)
      (decreases n) =
  if n > 0 then (
    shared_sum_scaled (n - 1) scale f g;
    Sum.sum_pop_right 0 n f;
    Sum.sum_pop_right 0 n g;
    assert (Sum.sum 0 n f == Sum.sum 0 (n - 1) f +. f (n - 1));
    assert (Sum.sum 0 n g == Sum.sum 0 (n - 1) g +. g (n - 1));
    assert (g (n - 1) == scale *. f (n - 1));
    scale_distributes scale (Sum.sum 0 (n - 1) f) (f (n - 1))
  )

let rec shared_sum_ext (n:nat) (f g:Shared.vector n)
  : Lemma
      (requires forall (i:Shared.fin n). f i == g i)
      (ensures Sum.sum 0 n f == Sum.sum 0 n g)
      (decreases n) =
  if n > 0 then (
    shared_sum_ext (n - 1) f g;
    Sum.sum_pop_right 0 n f;
    Sum.sum_pop_right 0 n g
  )

let shared_square_terms (#n:nat) (x:Shared.vector n) : Shared.vector n =
  fun i -> x i *. x i

let rec squares_shared_prefix (#n:nat) (x:Shared.vector n) (count:nat{count <= n})
  : Lemma
      (ensures sum_squares (pack_vector x) count ==
        Sum.sum 0 count (shared_square_terms x))
      (decreases count) =
  if count > 0 then (
    squares_shared_prefix x (count - 1);
    Sum.sum_pop_right 0 count (shared_square_terms x);
    assert (S.index (pack_vector x) (count - 1) == x (count - 1));
    assert (sum_squares (pack_vector x) count ==
      sum_squares (pack_vector x) (count - 1) +. x (count - 1) *. x (count - 1));
    assert (Sum.sum 0 count (shared_square_terms x) ==
      Sum.sum 0 (count - 1) (shared_square_terms x) +. x (count - 1) *. x (count - 1))
  )

let squares_shared (#n:nat) (x:Shared.vector n)
  : Lemma (sum_squares (pack_vector x) n == Shared.sum_squares x) =
  squares_shared_prefix x n;
  shared_sum_ext n (shared_square_terms x) (fun i -> x i *. x i)

let rms_shared (#n:pos) (x:Shared.vector n)
  : Lemma (rms_scale (pack_vector x) == Shared.rms_scale x) =
  squares_shared x

let pack_slice (#n:nat) (x:Shared.vector n)
  (offset count:nat{offset + count <= n})
  : Lemma (vector_slice (pack_vector x) offset count ==
      pack_vector #count (fun i -> x (offset + i))) =
  assert (S.equal (vector_slice (pack_vector x) offset count)
    (pack_vector #count (fun i -> x (offset + i))))

let pack_head_slice (x:Shared.recurrent_state) (head:Shared.fin heads)
  : Lemma (vector_slice (pack_recurrent x) (head * head_state_elements) head_state_elements ==
      pack_head (x head)) =
  assert (S.equal
    (vector_slice (pack_recurrent x) (head * head_state_elements) head_state_elements)
    (pack_head (x head)))

let normalization_shared
  (rw:real_vector weight_count) (offset:nat{offset + hidden_width <= weight_count})
  (w x:Shared.vector hidden_width)
  : Lemma
      (requires real_weight_vector hidden_width offset rw w)
      (ensures hidden_normalization rw offset (pack_vector x) ==
        pack_vector (Shared.centered_rms_norm w x)) =
  rms_shared x;
  assert (S.equal (hidden_normalization rw offset (pack_vector x))
    (pack_vector (Shared.centered_rms_norm w x)))

#push-options "--fuel 4 --z3rlimit 60"
let linear_shared (rw:real_vector weight_count) (offset:nat)
  (rows columns:pos{offset + rows * columns <= weight_count})
  (w:Shared.matrix rows columns) (x:Shared.vector columns)
  : Lemma
      (requires real_weight_matrix rows columns offset rw w)
      (ensures linear rw offset rows columns (pack_vector x) ==
        pack_vector (Shared.project w x)) =
  let point (row:Shared.fin rows)
    : Lemma (S.index (linear rw offset rows columns (pack_vector x)) row ==
        Shared.project w x row) =
    norm_spec [delta_only [`%linear]; zeta; iota]
      (linear rw offset rows columns (pack_vector x));
    norm_spec [delta_only [`%Shared.project]; zeta; iota] (Shared.project w x);
    let terms = S.init_ghost columns (fun column ->
      S.index rw (offset + row * columns + column) *. S.index (pack_vector x) column) in
    sequence_sum_shared #columns
      terms (fun column -> w row column *. x column);
    assert (S.index (linear rw offset rows columns (pack_vector x)) row == rsum terms);
    assert (Shared.project w x row == Sum.sum 0 columns (fun column -> w row column *. x column))
  in
  C.forall_intro point;
  assert (S.equal (linear rw offset rows columns (pack_vector x))
    (pack_vector (Shared.project w x)))

let initial_state_shared (position:int) (old:Shared.recurrent_state)
  : Lemma
      (requires position = 0 ==> (forall (h:Shared.fin heads) (i j:Shared.fin head_width).
        old h i j == 0.0R))
      (ensures initial_state position (pack_previous position (pack_recurrent old)) ==
        pack_recurrent old) =
  assert (S.equal
    (initial_state position (pack_previous position (pack_recurrent old)))
    (pack_recurrent old))

let next_convolution_shared (position:int) (old:Shared.convolution_state)
  (raw:Shared.vector qkv_width)
  : Lemma
      (requires position = 0 ==> (forall (c:Shared.fin qkv_width) (t:Shared.fin conv_width).
        old c t == 0.0R))
      (ensures next_convolution position (pack_previous position (pack_convolution old))
        (pack_vector raw) == pack_convolution (Shared.shift_convolution old raw)) =
  assert (S.equal
    (next_convolution position (pack_previous position (pack_convolution old)) (pack_vector raw))
    (pack_convolution (Shared.shift_convolution old raw)))

let activation_shared (x:real)
  : Lemma (sigmoid x == Shared.sigmoid x /\ silu x == Shared.silu x) = ()

let convolution_sum (rw:real_vector weight_count) (conv:real_vector convolution_count)
  (channel:Shared.fin qkv_width) : GTot real =
  rsum (S.init_ghost tap_count (fun tap ->
    S.index conv (channel * tap_count + tap) *.
      S.index rw (convolution_offset + channel * tap_count + tap)))

let convolved_head_at (rw:real_vector weight_count) (conv:real_vector convolution_count)
  (region:nat{region < 3}) (head:Shared.fin heads) (i:Shared.fin head_width)
  : Lemma (S.index (convolved_head rw conv region head) i ==
      silu (convolution_sum rw conv (region * value_width + head * head_width + i))) =
  physical_extents ();
  norm_spec [delta_only [`%convolved_head]; zeta; iota] (convolved_head rw conv region head);
  S.init_ghost_index_ head_width (fun coordinate ->
    let channel = region * value_width + head * head_width + coordinate in
    silu (rsum (S.init_ghost tap_count (fun tap ->
      S.index conv (channel * tap_count + tap) *.
        S.index rw (convolution_offset + channel * tap_count + tap))))) i;
  norm_spec [delta_only [`%convolution_sum]; zeta; iota]
    (convolution_sum rw conv (region * value_width + head * head_width + i))

let convolution_sum_shared (rw:real_vector weight_count)
  (w:Shared.matrix qkv_width conv_width) (conv:Shared.convolution_state)
  (channel:Shared.fin qkv_width)
  : Lemma
      (requires real_weight_matrix qkv_width conv_width conv_kernel_offset rw w)
      (ensures convolution_sum rw (pack_convolution conv) channel ==
        Sum.sum 0 4 (fun tap -> w channel tap *. conv channel tap)) =
  sequence_sum_shared #4
    (S.init_ghost tap_count (fun tap ->
      S.index (pack_convolution conv) (channel * tap_count + tap) *.
        S.index rw (convolution_offset + channel * tap_count + tap)))
    (fun tap -> w channel tap *. conv channel tap)

let shared_convolve_at (w:Shared.matrix qkv_width conv_width)
  (conv:Shared.convolution_state) (channel:Shared.fin qkv_width)
  : Lemma (Shared.convolve w conv channel ==
      Shared.silu (Sum.sum 0 4 (fun tap -> w channel tap *. conv channel tap))) = ()

#push-options "--fuel 1 --z3rlimit 30"
let convolved_head_shared (rw:real_vector weight_count)
  (w:Shared.matrix qkv_width conv_width) (conv:Shared.convolution_state)
  (region:nat{region < 3}) (head:Shared.fin heads)
  : Lemma
      (requires real_weight_matrix qkv_width conv_width conv_kernel_offset rw w)
      (ensures convolved_head rw (pack_convolution conv) region head ==
        pack_vector #head_width (fun i -> Shared.convolve w conv (region * mixer_width + head * head_width + i))) =
  let point (i:Shared.fin head_width)
    : Lemma (S.index (convolved_head rw (pack_convolution conv) region head) i ==
        Shared.convolve w conv (region * mixer_width + head * head_width + i)) =
    let channel = region * value_width + head * head_width + i in
    convolved_head_at rw (pack_convolution conv) region head i;
    convolution_sum_shared rw w conv channel;
    activation_shared (convolution_sum rw (pack_convolution conv) channel);
    shared_convolve_at w conv channel
  in
  C.forall_intro point;
  assert (S.equal (convolved_head rw (pack_convolution conv) region head)
    (pack_vector #head_width (fun i -> Shared.convolve w conv (region * mixer_width + head * head_width + i))))
#pop-options

let normalized_key_shared (x:Shared.vector head_width)
  : Lemma (normalized_key (pack_vector x) == pack_vector (Shared.l2_normalize x)) =
  squares_shared x;
  assert (S.equal (normalized_key (pack_vector x)) (pack_vector (Shared.l2_normalize x)))

let normalized_query_shared (x:Shared.vector head_width)
  : Lemma (normalized_query (pack_vector x) ==
      pack_vector #head_width (fun i -> Shared.l2_normalize x i *. Sqrt.rsqrt 128.0R)) =
  normalized_key_shared x;
  Sqrt.sqrt_positive 128.0R;
  assert (S.equal (normalized_query (pack_vector x))
    (pack_vector #head_width (fun i -> Shared.l2_normalize x i *. Sqrt.rsqrt 128.0R)))
#pop-options

let dot_shared (#n:nat) (x y:Shared.vector n)
  : Lemma (dot #n (pack_vector x) (pack_vector y) ==
      Sum.sum 0 n (fun i -> x i *. y i)) =
  norm_spec [delta_only [`%dot]; zeta; iota] (dot #n (pack_vector x) (pack_vector y));
  sequence_sum_shared #n
    (S.init_ghost n (fun i -> S.index (pack_vector x) i *. S.index (pack_vector y) i))
    (fun i -> x i *. y i)

let pack_head_row (old:Shared.matrix head_width head_width) (row:Shared.fin head_width)
  : Lemma (vector_slice (pack_head old) (row * head_width) head_width ==
      pack_vector #head_width (fun i -> old i row)) =
  assert (S.equal (vector_slice (pack_head old) (row * head_width) head_width)
    (pack_vector #head_width (fun i -> old i row)))

let product_reorder (a b c:real)
  : Lemma ((a *. b) *. c == b *. (a *. c)) = ()

let delta_reorder (old value key predicted beta decay:real)
  : Lemma (
      decay *. old +. (beta *. (value -. decay *. predicted)) *. key ==
      old *. decay +. key *. ((value -. decay *. predicted) *. beta)) = ()

let shared_update
  (old:Shared.matrix head_width head_width) (key value:Shared.vector head_width)
  (beta decay:real) : Shared.matrix head_width head_width =
  fun i j ->
    old i j *. decay +. key i *.
      ((value j -. Sum.sum 0 head_width (fun k -> (old k j *. decay) *. key k)) *. beta)

let shared_readout
  (state:Shared.matrix head_width head_width) (query:Shared.vector head_width)
  : Shared.vector head_width =
  fun j -> Sum.sum 0 head_width (fun i -> state i j *. query i)

#push-options "--fuel 4 --z3rlimit 60"
let delta_state_shared
  (old:Shared.matrix head_width head_width) (key value:Shared.vector head_width)
  (beta decay:real)
  : Lemma (delta_state (pack_head old) (pack_vector key) (pack_vector value) beta decay ==
      pack_head (shared_update old key value beta decay)) =
  let point (index:Shared.fin head_state_elements)
    : Lemma (
        S.index (delta_state (pack_head old) (pack_vector key) (pack_vector value) beta decay) index ==
        shared_update old key value beta decay (index % head_width) (index / head_width)) =
    let row = index / head_width in
    let column = index % head_width in
    let terms:Kuiper.Common.between 0 head_width -> GTot real = fun i -> old i row *. key i in
    let scaled:Kuiper.Common.between 0 head_width -> GTot real =
      fun i -> (old i row *. decay) *. key i in
    pack_head_row old row;
    dot_shared #head_width (fun i -> old i row) key;
    let scale_point (i:Shared.fin head_width)
      : Lemma (scaled i == decay *. terms i) =
      product_reorder (old i row) decay (key i)
    in
    C.forall_intro scale_point;
    shared_sum_scaled head_width decay terms scaled;
    delta_reorder (old column row) (value row) (key column)
      (Sum.sum 0 head_width terms) beta decay;
    norm_spec [delta_only [`%delta_state]; zeta; iota]
      (delta_state (pack_head old) (pack_vector key) (pack_vector value) beta decay);
    ()
  in
  C.forall_intro point;
  assert (S.equal
    (delta_state (pack_head old) (pack_vector key) (pack_vector value) beta decay)
    (pack_head (shared_update old key value beta decay)))

let state_readout_shared
  (state:Shared.matrix head_width head_width) (query:Shared.vector head_width)
  : Lemma (state_readout (pack_head state) (pack_vector query) ==
      pack_vector (shared_readout state query)) =
  let point (row:Shared.fin head_width)
    : Lemma (S.index (state_readout (pack_head state) (pack_vector query)) row ==
        shared_readout state query row) =
    pack_head_row state row;
    dot_shared #head_width (fun i -> state i row) query
  in
  C.forall_intro point;
  assert (S.equal (state_readout (pack_head state) (pack_vector query))
    (pack_vector (shared_readout state query)))

let gated_readout_shared (rw:real_vector weight_count)
  (w readout gate:Shared.vector head_width)
  : Lemma
      (requires real_weight_vector head_width gated_norm_offset rw w)
      (ensures gated_readout rw (pack_vector readout) (pack_vector gate) ==
        pack_vector #head_width (fun i ->
          (w i *. (readout i *. Shared.rms_scale readout)) *. Shared.silu (gate i))) =
  rms_shared readout;
  let point (i:Shared.fin head_width)
    : Lemma (S.index (gated_readout rw (pack_vector readout) (pack_vector gate)) i ==
        (w i *. (readout i *. Shared.rms_scale readout)) *. Shared.silu (gate i)) =
    activation_shared (gate i);
    product_reorder (readout i *. Shared.rms_scale readout) (w i) (Shared.silu (gate i))
  in
  C.forall_intro point;
  assert (S.equal (gated_readout rw (pack_vector readout) (pack_vector gate))
    (pack_vector #head_width (fun i ->
      (w i *. (readout i *. Shared.rms_scale readout)) *. Shared.silu (gate i))))
#pop-options

noextract
noeq type shared_head_values = {
  ideal_query_input:Shared.vector head_width;
  ideal_key_input:Shared.vector head_width;
  ideal_value:Shared.vector head_width;
  ideal_query:Shared.vector head_width;
  ideal_key:Shared.vector head_width;
  ideal_beta:real;
  ideal_decay:real;
  ideal_state:Shared.matrix head_width head_width;
  ideal_core:Shared.vector head_width;
  ideal_gated:Shared.vector head_width;
}

#push-options "--fuel 1 --z3rlimit 40"
[@@"opaque_to_smt"]
noextract
let shared_head (w:Shared.weights) (qkv:Shared.vector qkv_width)
  (z:Shared.vector mixer_width) (b a:Shared.vector heads)
  (old:Shared.recurrent_state) (head:Shared.fin heads)
  : GTot (p:shared_head_values{
      p.ideal_query_input == (fun (i:Shared.fin head_width) -> qkv (head * head_width + i)) /\
      p.ideal_key_input == (fun (i:Shared.fin head_width) -> qkv (mixer_width + head * head_width + i)) /\
      p.ideal_value == (fun (i:Shared.fin head_width) -> qkv (2 * mixer_width + head * head_width + i)) /\
      (forall (i:Shared.fin head_width).
        p.ideal_query i == Shared.l2_normalize p.ideal_query_input i *. Sqrt.rsqrt 128.0R) /\
      p.ideal_key == Shared.l2_normalize p.ideal_key_input /\
      p.ideal_beta == Shared.sigmoid (b head) /\
      p.ideal_decay == exp ((0.0R -. exp (w.a_log head)) *. Shared.softplus (a head +. w.dt_bias head)) /\
      p.ideal_state == shared_update (old head) p.ideal_key p.ideal_value p.ideal_beta p.ideal_decay /\
      p.ideal_core == shared_readout p.ideal_state p.ideal_query /\
      (forall (i:Shared.fin head_width).
        p.ideal_gated i == (w.gated_norm i *. (p.ideal_core i *. Shared.rms_scale p.ideal_core)) *.
          Shared.silu (z (head * head_width + i)))}) =
  let qinput:Shared.vector head_width = fun i -> qkv (head * head_width + i) in
  let kinput:Shared.vector head_width = fun i -> qkv (mixer_width + head * head_width + i) in
  let value:Shared.vector head_width = fun i -> qkv (2 * mixer_width + head * head_width + i) in
  let query:Shared.vector head_width = fun i -> Shared.l2_normalize qinput i *. Sqrt.rsqrt 128.0R in
  let key = Shared.l2_normalize kinput in
  let beta = Shared.sigmoid (b head) in
  let decay = exp ((0.0R -. exp (w.a_log head)) *. Shared.softplus (a head +. w.dt_bias head)) in
  let state = shared_update (old head) key value beta decay in
  let core = shared_readout state query in
  {
    ideal_query_input = qinput;
    ideal_key_input = kinput;
    ideal_value = value;
    ideal_query = query;
    ideal_key = key;
    ideal_beta = beta;
    ideal_decay = decay;
    ideal_state = state;
    ideal_core = core;
    ideal_gated = (fun i ->
      (w.gated_norm i *. (core i *. Shared.rms_scale core)) *. Shared.silu (z (head * head_width + i)));
  }
#pop-options

#push-options "--fuel 3 --z3rlimit 100"
let head_transition_shared (rw:real_vector weight_count) (w:Shared.weights)
  (conv:Shared.convolution_state) (old:Shared.recurrent_state)
  (z:Shared.vector mixer_width) (b a:Shared.vector heads)
  (controls:lseq f32 head_count) (head:Shared.fin heads)
  : Lemma
      (requires real_weight_views rw w)
      (ensures (
        let actual = head_transition rw (pack_convolution conv) (pack_recurrent old)
          (pack_vector z) (pack_vector b)
          (pack_vector #heads (fun h -> a h +. w.dt_bias h)) controls head in
        let expected = shared_head w (Shared.convolve w.conv_kernel conv) z b a old head in
        fst actual == pack_head expected.ideal_state /\
        snd actual == pack_vector expected.ideal_gated)) =
  let expected = shared_head w (Shared.convolve w.conv_kernel conv) z b a old head in
  convolved_head_shared rw w.conv_kernel conv 0 head;
  convolved_head_shared rw w.conv_kernel conv 1 head;
  convolved_head_shared rw w.conv_kernel conv 2 head;
  assert (S.equal (convolved_head rw (pack_convolution conv) 0 head) (pack_vector expected.ideal_query_input));
  assert (S.equal (convolved_head rw (pack_convolution conv) 1 head) (pack_vector expected.ideal_key_input));
  assert (S.equal (convolved_head rw (pack_convolution conv) 2 head) (pack_vector expected.ideal_value));
  normalized_query_shared expected.ideal_query_input;
  normalized_key_shared expected.ideal_key_input;
  assert (S.equal (normalized_query (pack_vector expected.ideal_query_input))
    (pack_vector expected.ideal_query));
  activation_shared (b head);
  assert (S.index rw (log_decay_offset + head) == w.a_log head);
  assert (S.index (pack_vector #heads (fun h -> a h +. w.dt_bias h)) head ==
    a head +. w.dt_bias head);
  pack_head_slice old head;
  delta_state_shared (old head) expected.ideal_key expected.ideal_value expected.ideal_beta expected.ideal_decay;
  state_readout_shared expected.ideal_state expected.ideal_query;
  pack_slice z (head * head_width) head_width;
  gated_readout_shared rw w.gated_norm expected.ideal_core (fun i -> z (head * head_width + i));
  norm_spec [delta_only [`%head_transition]; zeta; iota]
    (head_transition rw (pack_convolution conv) (pack_recurrent old)
      (pack_vector z) (pack_vector b)
      (pack_vector #heads (fun h -> a h +. w.dt_bias h)) controls head);
  assert (S.equal
    (snd (head_transition rw (pack_convolution conv) (pack_recurrent old)
      (pack_vector z) (pack_vector b)
      (pack_vector #heads (fun h -> a h +. w.dt_bias h)) controls head))
    (pack_vector expected.ideal_gated))
#pop-options

let shared_states (values:Shared.fin heads -> GTot shared_head_values)
  : Shared.recurrent_state =
  fun h i j -> (values h).ideal_state i j

let shared_activation (values:Shared.fin heads -> GTot shared_head_values)
  : Shared.vector mixer_width =
  fun i -> (values (i / head_width)).ideal_gated (i % head_width)

let shared_mlp_product (gate up:Shared.vector mlp_width) : Shared.vector mlp_width =
  fun i -> Shared.silu (gate i) *. up i

[@@"opaque_to_smt"]
noextract
let assembled_shared_values
  (call:Shared.invocation) (w:Shared.weights) (hidden:Shared.vector hidden_width)
  (before:Shared.model_cache{Shared.cache_domain call before})
  : GTot Shared.layer_values =
  let normalized = Shared.centered_rms_norm w.input_norm hidden in
  let raw = Shared.project w.in_qkv normalized in
  let z = Shared.project w.in_z normalized in
  let b = Shared.project w.in_beta normalized in
  let a = Shared.project w.in_a normalized in
  let convolution = Shared.shift_convolution (old_convolution call before) raw in
  let qkv = Shared.convolve w.conv_kernel convolution in
  let head_values:Shared.fin heads -> GTot shared_head_values =
    shared_head w qkv z b a (old_recurrent call before) in
  let activation = shared_activation head_values in
  let attention = Shared.project w.out_projection activation in
  let residual = Shared.add_vectors hidden attention in
  let postnorm = Shared.centered_rms_norm w.post_attention_norm residual in
  let gate = Shared.project w.mlp_gate postnorm in
  let up = Shared.project w.mlp_up postnorm in
  let product = shared_mlp_product gate up in
  let down = Shared.project w.mlp_down product in
  {
    Shared.hidden_value = Shared.add_vectors residual down;
    Shared.convolution_value = convolution;
    Shared.recurrent_value = shared_states head_values;
  }

let assembled_shared_matches
  (call:Shared.invocation) (w:Shared.weights) (hidden:Shared.vector hidden_width)
  (before:Shared.model_cache{Shared.cache_domain call before})
  : Lemma (
      (forall (i:Shared.fin hidden_width).
        (assembled_shared_values call w hidden before).hidden_value i ==
        (Shared.decoder_values call w hidden before).hidden_value i) /\
      (forall (c:Shared.fin qkv_width) (t:Shared.fin conv_width).
        (assembled_shared_values call w hidden before).convolution_value c t ==
        (Shared.decoder_values call w hidden before).convolution_value c t) /\
      (forall (h:Shared.fin heads) (i j:Shared.fin head_width).
        (assembled_shared_values call w hidden before).recurrent_value h i j ==
        (Shared.decoder_values call w hidden before).recurrent_value h i j)) =
  assert (
      (forall (i:Shared.fin hidden_width).
        (assembled_shared_values call w hidden before).hidden_value i ==
        (Shared.decoder_values call w hidden before).hidden_value i) /\
      (forall (c:Shared.fin qkv_width) (t:Shared.fin conv_width).
        (assembled_shared_values call w hidden before).convolution_value c t ==
        (Shared.decoder_values call w hidden before).convolution_value c t) /\
      (forall (h:Shared.fin heads) (i j:Shared.fin head_width).
        (assembled_shared_values call w hidden before).recurrent_value h i j ==
        (Shared.decoder_values call w hidden before).recurrent_value h i j))
    by (FStar.Tactics.V2.norm [
      iota; zeta; reduce_projections; unascribe; primops; simplify;
      delta_only [
        `%assembled_shared_values; `%shared_head; `%shared_update; `%shared_readout;
        `%shared_states; `%shared_activation; `%shared_mlp_product;
        `%old_convolution; `%old_recurrent;
        `%Shared.decoder_values; `%Shared.token_mixer; `%Shared.feed_forward;
        `%hidden_width; `%head_width; `%heads; `%mixer_width; `%qkv_width;
        `%conv_width; `%mlp_width; `%intermediate_width]];
      FStar.Tactics.V2.trivial ())

let packed_inputs
  (call:Shared.invocation) (w:Shared.weights) (hidden:Shared.vector hidden_width)
  (before:Shared.model_cache{Shared.cache_domain call before})
  (wv:lseq bf16 weight_count) (hv:lseq bf16 hidden_width) (position:int)
  (sv:lseq f32 (previous_elements position recurrent_count))
  (cv:lseq f32 (previous_elements position convolution_count))
  : Lemma
      (requires input_representation call w hidden before wv hv position sv cv)
      (ensures
        wv %~ pack_weights w /\ hv %~ pack_vector hidden /\
        sv %~ pack_previous position (pack_recurrent (old_recurrent call before)) /\
        cv %~ pack_previous position (pack_convolution (old_convolution call before))) =
  packed_weights_approx wv w;
  if position = 0 then ()
  else (
    let recurrent_point (i:Shared.fin recurrent_count)
      : Lemma (S.index sv i %~
          S.index (pack_recurrent (old_recurrent call before)) i) =
      recurrent_index_covers i
    in
    let convolution_point (i:Shared.fin convolution_count)
      : Lemma (S.index cv i %~
          S.index (pack_convolution (old_convolution call before)) i) =
      convolution_index_covers i
    in
    C.forall_intro recurrent_point;
    C.forall_intro convolution_point
  )

let packed_outputs
  (out:lseq bf16 hidden_width) (states:lseq f32 recurrent_count)
  (convolution:lseq f32 convolution_count) (values:Shared.layer_values)
  : Lemma
      (requires
        out %~ pack_vector values.hidden_value /\
        states %~ pack_recurrent values.recurrent_value /\
        convolution %~ pack_convolution values.convolution_value)
      (ensures output_representation out states convolution values) =
  let recurrent_point (h:Shared.fin heads) (key value:Shared.fin head_width)
    : Lemma (S.index states (recurrent_index h key value) %~ values.recurrent_value h key value) =
    assert (S.index (pack_recurrent values.recurrent_value) (recurrent_index h key value) ==
      values.recurrent_value h key value)
  in
  let convolution_point (c:Shared.fin qkv_width) (t:Shared.fin conv_width)
    : Lemma (S.index convolution (convolution_index c t) %~ values.convolution_value c t) =
    assert (S.index (pack_convolution values.convolution_value) (convolution_index c t) ==
      values.convolution_value c t)
  in
  C.forall_intro_3 recurrent_point;
  C.forall_intro_2 convolution_point

let tree_width (n : nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32

let rec tree_value
  (f : nat -> GTot real)
  (width : nat{tree_width width})
  (lane : nat)
  : GTot real (decreases 32 - width)
  = if width = 32 then f lane
    else tree_value f (2 * width) lane +.
      tree_value f (2 * width) (lane + width)

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

let tree_root (f : nat -> GTot real)
  : Lemma (tree_value f 1 0 == prefix 32 f)
  = tree_total f 1

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

let tree_invariant
  (original current : lseq f32 32)
  (width : nat{tree_width width})
  : prop
  = forall (model : lseq real 32).
      original %~ model ==>
      (forall (i : nat{i < width}).
        S.index current i %~ tree_value (seq_value model) width i)

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

let tree_initial (values : lseq f32 32)
  : Lemma (tree_invariant values values 32)
  = ()

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

let square_cell (hidden : lseq real hidden_width) (i : nat) : GTot real =
  if i < hidden_width then
    let x = S.index hidden i in x *. x
  else 0.0R

let square_pair (hidden : lseq real hidden_width) (warp lane : nat) : GTot real =
  square_cell hidden (warp * 32 + lane) +.
  square_cell hidden (512 + (warp * 32 + lane))

let warp_square_sum (hidden : lseq real hidden_width) (warp : nat) : GTot real =
  if warp < 16 then prefix 32 (square_pair hidden warp) else 0.0R

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

[@@CPrologue "__device__"]
fn native_log (x : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (r : real{r >. 0.0R}).
    Kuiper.Approximates.Base.v_approximates x r ==>
    Kuiper.Approximates.Base.v_approximates result (Kuiper.Real.log r))
{
  Fast.log x
}

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

let log_argument (x exponential argument : f32)
  : Lemma
      (requires
        (forall (r : real). x %~ r ==> exponential %~ Kuiper.Real.exp r) /\
        (forall (a b : real).
          (one #f32) %~ a /\ exponential %~ b ==> argument %~ (a +. b)))
      (ensures forall (r : real).
        x %~ r ==> argument %~ (1.0R +. Kuiper.Real.exp r) /\
          1.0R +. Kuiper.Real.exp r >. 0.0R)
  = let related (r : real{x %~ r})
      : Lemma (argument %~ (1.0R +. Kuiper.Real.exp r) /\
          1.0R +. Kuiper.Real.exp r >. 0.0R)
      = exp_positive r
    in FStar.Classical.forall_intro related

[@@CPrologue "__device__"]
fn softplus32 (x : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (r : real).
    x %~ r ==> result %~ softplus x r)
{
  LocalSoftplus.softplus20 x
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
let query_factor32 () : Pure f32
  (requires True)
  (ensures fun result -> result %~ (1.0R /. FStar.Math.Sqrt.sqrt 128.0R))
  = let width : f32 = of_int 128L in
    let _ = of_int_approx #f32 128L in
    sqrt_approx width 128.0R;
    let root = sqrt width in
    div_approx (one #f32) root 1.0R (FStar.Math.Sqrt.sqrt 128.0R);
    div (one #f32) root

[@@CPrologue "__device__"]
fn query_factor ()
  preserves gpu
  returns result : f32
  ensures pure (result %~ (1.0R /. FStar.Math.Sqrt.sqrt 128.0R))
{
  query_factor32 ()
}

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

let control_arguments (weights : real_vector weight_elements) (alpha : real_vector head_count)
  : GTot (real_vector head_count)
  = S.init_ghost head_count (fun h -> S.index alpha h +. S.index weights (alpha_bias_offset + h))

let matrix_inputs
  (#n : nat)
  (weights : lseq bf16 weight_elements)
  (values : lseq f32 n)
  (model : matrix_interpretation n)
  : prop = weights %~ fst model /\ values %~ snd model

let matrix_term
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (column : nat)
  : GTot real
  = if column < n then
      S.index (fst model) (offset + column) *. S.index (snd model) column
    else 0.0R

let matrix_block
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (lane group : nat)
  : GTot real
  = prefix 8 (fun j -> matrix_term model offset (group * 256 + (lane * 8 + j)))

let matrix_lane
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (lane : nat)
  : GTot real
  = prefix (n / 256) (matrix_block model offset lane)

let matrix_partial
  (#n : nat)
  (model : matrix_interpretation n)
  (offset : nat{offset + n <= weight_elements})
  (lane group count : nat)
  (chunks : bool)
  : GTot real
  = (if chunks then 0.0R else prefix group (matrix_block model offset lane)) +.
      prefix count (fun j -> matrix_term model offset (group * 256 + (lane * 8 + j)))

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
  (#n : pos{n % 256 = 0 /\ n <= 8192})
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
  (#n : pos{n % 256 = 0 /\ n <= 8192})
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

inline_for_extraction noextract
fn matrix_project
  (#n : pos{n % 256 = 0 /\ n <= 8192})
  (#m : pos{m <= 8192})
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

let normalization_inputs
  (weights : lseq bf16 weight_elements)
  (hidden : lseq bf16 hidden_width)
  (model : matrix_interpretation hidden_width)
  : prop = weights %~ fst model /\ hidden %~ snd model

inline_for_extraction noextract
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
      after %~ hidden_normalization (fst model) (SZ.v offset) (snd model))
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
          S.index (hidden_normalization (fst model) (SZ.v offset) (snd model)) j))
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

let rec prefix_scale (n : nat) (a : real) (f : nat -> GTot real)
  : Lemma (ensures prefix n (fun i -> a *. f i) == a *. prefix n f)
      (decreases n)
  = if n > 0 then prefix_scale (n - 1) a f

let head_term
  (#n : nat)
  (model : real_vector n & real_vector head_width)
  (offset : nat{offset + head_width <= n})
  (i : nat)
  : GTot real
  = if i < head_width then
      S.index (fst model) (offset + i) *. S.index (snd model) i
    else 0.0R

let head_lane
  (#n : nat)
  (model : real_vector n & real_vector head_width)
  (offset : nat{offset + head_width <= n})
  (lane : nat)
  : GTot real
  = prefix 4 (fun group -> head_term model offset (group * 32 + lane))

let head_dot_model
  (#n : nat)
  (offset : nat{offset + head_width <= n})
  (model : real_vector n & real_vector head_width)
  : Lemma (prefix 32 (head_lane model offset) ==
      dot (vector_slice (fst model) offset head_width) (snd model))
  = prefix_ext 32 (head_lane model offset)
      (fun lane -> prefix 4 (fun group -> head_term model offset (group * 32 + lane)));
    prefix_swap 32 4 (fun lane group -> head_term model offset (group * 32 + lane));
    prefix_blocks 4 32 (head_term model offset);
    prefix_init 128 (head_term model offset);
    FStar.Pervasives.norm_spec [delta_only [`%dot]; zeta; iota]
      (dot (vector_slice (fst model) offset head_width) (snd model));
    S.lemma_eq_elim (S.init_ghost 128 (head_term model offset))
      (S.init_ghost head_width (fun i ->
        S.index (vector_slice (fst model) offset head_width) i *. S.index (snd model) i))

let head_dot_inputs
  (#n : nat)
  (left : lseq f32 n)
  (right : lseq f32 head_width)
  (model : real_vector n & real_vector head_width)
  : prop = left %~ fst model /\ right %~ snd model

let head_lane_unfold
  (#n : nat)
  (offset : nat{offset + head_width <= n}) (lane : nat)
  (model : real_vector n & real_vector head_width)
  : Lemma (head_lane model offset lane ==
      prefix 4 (fun group -> head_term model offset (group * 32 + lane)))
  = ()

inline_for_extraction noextract
fn strided_lane
  (#n : pos)
  (left : larray f32 n)
  (right : larray f32 head_width)
  (offset : sz{SZ.v offset + head_width <= n})
  (lane : sz{lane < 32})
  (#lv : erased (lseq f32 n))
  (#rv : erased (lseq f32 head_width))
  (#fl #fr : perm)
  preserves gpu ** left |-> Frac fl lv ** right |-> Frac fr rv
  returns result : f32
  ensures pure (forall (model : real_vector n & real_vector head_width).
    head_dot_inputs lv rv model ==> result %~ head_lane model (SZ.v offset) (SZ.v lane))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut group = 0sz;
  let mut accumulator : f32 = zero;
  while (!group <^ 4sz)
    invariant exists* (g : sz{g <= 4}) (value : f32).
      group |-> g ** accumulator |-> value **
      pure (forall (model : real_vector n & real_vector head_width).
        head_dot_inputs lv rv model ==>
        value %~ prefix (SZ.v g)
          (fun j -> head_term model (SZ.v offset) (j * 32 + SZ.v lane)))
    decreases (4 - !group)
  {
    let g = !group;
    let i = g *^ 32sz +^ lane;
    let a = left.(offset +^ i);
    let b = right.(i);
    let previous = !accumulator;
    assert pure (forall (model : real_vector n & real_vector head_width).
      head_dot_inputs lv rv model ==>
      a %~ S.index (fst model) (SZ.v offset + SZ.v i) /\
      b %~ S.index (snd model) (SZ.v i));
    let value = F32.fma_rn_ftz a b previous;
    lift_fma #(real_vector n & real_vector head_width) (head_dot_inputs #n lv rv)
      (fun model -> S.index (fst model) (SZ.v offset + SZ.v i))
      (fun model -> S.index (snd model) (SZ.v i))
      (fun model -> prefix (SZ.v g)
        (fun j -> head_term #n model (SZ.v offset) (j * 32 + SZ.v lane)))
      a b previous value;
    accumulator := value;
    group := g +^ 1sz;
  };
  C.forall_intro (head_lane_unfold #n (SZ.v offset) (SZ.v lane));
  !accumulator
}

inline_for_extraction noextract
fn strided_dot
  (#n : pos)
  (left : larray f32 n)
  (right : larray f32 head_width)
  (offset : sz{SZ.v offset + head_width <= n})
  (#lv : erased (lseq f32 n))
  (#rv : erased (lseq f32 head_width))
  (#fl #fr : perm)
  preserves gpu ** left |-> Frac fl lv ** right |-> Frac fr rv
  returns result : f32
  ensures pure (forall (model : real_vector n & real_vector head_width).
    head_dot_inputs lv rv model ==>
    result %~ dot (vector_slice (fst model) (SZ.v offset) head_width) (snd model))
{
  let mut partial = [| zero #f32 #_; 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i : sz{i <= 32}) (values : lseq f32 32).
      lane |-> i ** partial |-> values **
      pure (forall (model : real_vector n & real_vector head_width).
        head_dot_inputs lv rv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~ head_lane model (SZ.v offset) j))
    decreases (32 - !lane)
  {
    let i = !lane;
    let value = strided_lane left right offset i;
    partial.(i) <- value;
    lane := i +^ 1sz;
  };
  with values. assert partial |-> values;
  let result = sum32 partial;
  lift_sum32 #(real_vector n & real_vector head_width) (head_dot_inputs #n lv rv)
    (fun model -> head_lane #n model (SZ.v offset)) values result;
  C.forall_intro (head_dot_model #n (SZ.v offset));
  result
}

let head_square (values : real_vector head_width) (i : nat) : GTot real =
  if i < head_width then S.index values i *. S.index values i else 0.0R

let head_square_lane (values : real_vector head_width) (lane group : nat) : GTot real =
  head_square values (group * 32 + lane)

let head_square_lane_step (lane group : nat) (model : real_vector head_width)
  : Lemma (prefix (group + 1) (head_square_lane model lane) ==
      prefix group (head_square_lane model lane) +. head_square model (group * 32 + lane))
  = prefix_step group (head_square_lane model lane)

let rec head_squares_prefix (values : real_vector head_width)
  (count : nat{count <= head_width})
  : Lemma (ensures prefix count (head_square values) == sum_squares values count)
      (decreases count)
  = if count > 0 then head_squares_prefix values (count - 1)

let square_strided_model (values : real_vector head_width)
  : Lemma (prefix 32 (fun lane ->
      prefix 4 (head_square_lane values lane)) ==
      sum_squares values head_width)
  = let expand_lane (lane : nat)
      : Lemma (prefix 4 (head_square_lane values lane) ==
          prefix 4 (fun group -> head_square values (group * 32 + lane)))
      = prefix_ext 4 (head_square_lane values lane)
          (fun group -> head_square values (group * 32 + lane))
    in
    C.forall_intro expand_lane;
    prefix_ext 32 (fun lane -> prefix 4 (head_square_lane values lane))
      (fun lane -> prefix 4 (fun group -> head_square values (group * 32 + lane)));
    prefix_swap 32 4 (fun lane group -> head_square values (group * 32 + lane));
    prefix_blocks 4 32 (head_square values);
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
          S.index values j %~ prefix 4 (head_square_lane model j)))
    decreases (32 - !lane)
  {
    let i = !lane;
    let mut group = 0sz;
    let mut accumulator : f32 = zero;
    while (!group <^ 4sz)
      invariant exists* (g : sz{g <= 4}) (value : f32).
        group |-> g ** accumulator |-> value **
        pure (forall (model : real_vector head_width). before %~ model ==>
          value %~ prefix (SZ.v g) (head_square_lane model (SZ.v i)))
      decreases (4 - !group)
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
    (fun model lane -> prefix 4 (head_square_lane model lane)) values result;
  C.forall_intro square_strided_model;
  result
}

inline_for_extraction noextract
fn query_scale32 (scale : f32) (query : bool)
  preserves gpu
  returns result : f32
  ensures pure (forall (r : real). scale %~ r ==>
    result %~ (if query then r /. Sqrt.sqrt 128.0R else r))
{
  if query {
    let factor = query_factor32 ();
    assert pure (factor %~ (1.0R /. Sqrt.sqrt 128.0R));
    let result = F32.mul_rn_ftz scale factor;
    lift_mul (fun (r : real) -> scale %~ r) (fun r -> r)
      (fun _ -> 1.0R /. Sqrt.sqrt 128.0R) scale factor result;
    result
  } else {
    scale
  }
}

let head_normalization_scale (model : real_vector head_width) (query : bool) : GTot real =
  let scale = Sqrt.rsqrt (sum_squares model head_width +. epsilon) in
  if query then scale /. Sqrt.sqrt 128.0R else scale

let head_normalization_value (model : real_vector head_width) (query : bool)
  : GTot (real_vector head_width)
  = if query then normalized_query model else normalized_key model

let head_scale_finish
  (before : lseq f32 head_width)
  (argument scale : f32)
  : Lemma
      (requires
        (forall (model : real_vector head_width). before %~ model ==>
          argument %~ (sum_squares model head_width +. epsilon <: real)) /\
        (forall (r : Sqrt.rpos). argument %~ (r <: real) ==> scale %~ (1.0R /. Sqrt.sqrt r)))
      (ensures forall (model : real_vector head_width). before %~ model ==>
        scale %~ (Sqrt.rsqrt (sum_squares model head_width +. epsilon) <: real))
  = let related (model : real_vector head_width{before %~ model})
      : Lemma (scale %~ (Sqrt.rsqrt (sum_squares model head_width +. epsilon) <: real))
      = ()
    in C.forall_intro related

#push-options "--fuel 4 --z3rlimit 60"
let head_normalization_cell (query : bool) (i : nat{i < head_width})
  (model : real_vector head_width)
  : Lemma (S.index (head_normalization_value model query) i ==
      S.index model i *. head_normalization_scale model query)
  = ()
#pop-options

inline_for_extraction noextract
fn normalize_head
  (input : larray f32 head_width)
  (query : bool)
  (#before : erased (lseq f32 head_width))
  preserves gpu
  requires input |-> before
  ensures exists* (after : lseq f32 head_width). input |-> after **
    pure (forall (model : real_vector head_width). before %~ model ==>
      after %~ head_normalization_value model query)
{
  let squares = square_strided input;
  let eps = epsilon32 ();
  let argument = F32.add_rn_ftz squares eps;
  lift_add (fun (model : real_vector head_width) -> before %~ model)
    (fun model -> (sum_squares model head_width <: real))
    (fun _ -> (epsilon <: real)) squares eps argument;
  let scale = F32.rsqrt_approx_ftz argument;
  head_scale_finish before argument scale;
  let factor = query_scale32 scale query;
  assert pure (forall (model : real_vector head_width). before %~ model ==>
    factor %~ head_normalization_scale model query);
  let mut column = 0sz;
  while (!column <^ 128sz)
    invariant exists* (i : sz{i <= 128}) (current : lseq f32 head_width).
      column |-> i ** input |-> current **
      pure (forall (j : nat{SZ.v i <= j /\ j < head_width}).
        S.index current j == S.index before j) **
      pure (forall (model : real_vector head_width). before %~ model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index current j %~ S.index (head_normalization_value model query) j))
    decreases (128 - !column)
  {
    let i = !column;
    let value = input.(i);
    let result = F32.mul_rn_ftz value factor;
    lift_mul (fun (model : real_vector head_width) -> before %~ model)
      (fun model -> S.index model (SZ.v i))
      (fun model -> head_normalization_scale model query) value factor result;
    C.forall_intro (head_normalization_cell query (SZ.v i));
    input.(i) <- result;
    column := i +^ 1sz;
  }
}

let physical_initial
  (#n : nat)
  (position : int)
  (previous : lseq f32 (previous_elements position n))
  : GTot (lseq f32 n)
  = S.init_ghost n (fun i -> if position = 0 then (zero <: f32) else S.index previous i)

#push-options "--fuel 4"
inline_for_extraction noextract
let initial_position (position : i64) : b:bool{b <==> I64.v position = 0} =
  I64.eq position 0L
#pop-options

inline_for_extraction noextract
fn initial_cell
  (#n : pos)
  (input : array f32)
  (position : i64)
  (index : sz{index < n})
  (#previous : erased (lseq f32 (previous_elements (I64.v position) n)))
  (#fraction : perm)
  preserves gpu ** state_ownership input (I64.v position) n fraction previous
  returns result : f32
  ensures pure (result == S.index (physical_initial #n (I64.v position) previous) (SZ.v index))
{
  if (initial_position position) {
    zero #f32
  } else {
    rewrite state_ownership input (I64.v position) n fraction previous as
      (pure (is_global_array input) ** input |-> Frac fraction previous);
    gpu_pts_to_ref input;
    let result = input.(index);
    rewrite (pure (is_global_array input) ** input |-> Frac fraction previous) as
      state_ownership input (I64.v position) n fraction previous;
    result
  }
}

inline_for_extraction noextract
fn initialize_state
  (input : array f32)
  (output : larray f32 state_elements)
  (position : i64)
  (#previous : erased (lseq f32 (previous_elements (I64.v position) state_elements)))
  (#old : erased (lseq f32 state_elements))
  (#fraction : perm)
  preserves gpu ** state_ownership input (I64.v position) state_elements fraction previous
  requires output |-> old
  ensures output |-> physical_initial (I64.v position) previous
{
  let mut index = 0sz;
  while (!index <^ SZ.uint32_to_sizet 262144ul)
    invariant exists* (i : sz{i <= state_elements}) (values : lseq f32 state_elements).
      index |-> i ** output |-> values **
      pure (forall (j : nat{j < SZ.v i}).
        S.index values j == S.index (physical_initial #state_elements (I64.v position) previous) j)
    decreases (state_elements - !index)
  {
    let i = !index;
    let result = initial_cell #state_elements input position i;
    output.(i) <- result;
    index := i +^ 1sz;
  };
  with values. assert output |-> values;
  assert pure (S.equal values (physical_initial #state_elements (I64.v position) previous));
}

let initial_state_relation
  (position : int)
  (previous : lseq f32 (previous_elements position state_elements))
  (model : real_vector (previous_elements position state_elements))
  : Lemma
      (requires previous %~ model)
      (ensures physical_initial #state_elements position previous %~ initial_state position model)
  = let _ = Approx.a0 #f32 in ()

let physical_convolution
  (position : int)
  (previous : lseq f32 (previous_elements position convolution_elements))
  (projected : lseq f32 channel_count)
  : GTot (lseq f32 convolution_elements)
  = S.init_ghost convolution_elements (fun i ->
      if i % tap_count = tap_count - 1 then S.index projected (i / tap_count)
      else if position = 0 then (zero <: f32)
      else S.index previous (i + 1))

inline_for_extraction noextract
fn convolution_cell
  (previous : array f32)
  (projected : larray f32 channel_count)
  (position : i64)
  (index : sz{index < convolution_elements})
  (#cv : erased (lseq f32 (previous_elements (I64.v position) convolution_elements)))
  (#pv : erased (lseq f32 channel_count))
  (#fc #fp : perm)
  preserves gpu ** projected |-> Frac fp pv **
    state_ownership previous (I64.v position) convolution_elements fc cv
  returns result : f32
  ensures pure (result == S.index (physical_convolution (I64.v position) cv pv) (SZ.v index))
{
  if (index %^ 4sz >=^ 3sz) {
    projected.(index /^ 4sz)
  } else if (initial_position position) {
    zero #f32
  } else {
    rewrite state_ownership previous (I64.v position) convolution_elements fc cv as
      (pure (is_global_array previous) ** previous |-> Frac fc cv);
    gpu_pts_to_ref previous;
    let result = previous.(index +^ 1sz);
    rewrite (pure (is_global_array previous) ** previous |-> Frac fc cv) as
      state_ownership previous (I64.v position) convolution_elements fc cv;
    result
  }
}

let physical_convolution_relation
  (position : int)
  (previous : lseq f32 (previous_elements position convolution_elements))
  (projected : lseq f32 channel_count)
  : Lemma (
      retained_taps_post position previous (physical_convolution position previous projected) /\
      (forall (real_previous : real_vector (previous_elements position convolution_elements))
              (real_projected : real_vector channel_count).
        previous %~ real_previous /\ projected %~ real_projected ==>
        physical_convolution position previous projected %~
          next_convolution position real_previous real_projected))
  = let _ = Approx.a0 #f32 in ()

inline_for_extraction noextract
fn shift_convolution
  (previous : array f32)
  (projected : larray f32 channel_count)
  (output : larray f32 convolution_elements)
  (position : i64)
  (#cv : erased (lseq f32 (previous_elements (I64.v position) convolution_elements)))
  (#pv : erased (lseq f32 channel_count))
  (#old : erased (lseq f32 convolution_elements))
  (#fc #fp : perm)
  preserves gpu ** projected |-> Frac fp pv **
    state_ownership previous (I64.v position) convolution_elements fc cv
  requires output |-> old
  ensures output |-> physical_convolution (I64.v position) cv pv **
    pure (retained_taps_post (I64.v position) cv (physical_convolution (I64.v position) cv pv))
{
  let mut index = 0sz;
  while (!index <^ 24576sz)
    invariant exists* (i : sz{i <= convolution_elements}) (values : lseq f32 convolution_elements).
      index |-> i ** output |-> values **
      pure (forall (j : nat{j < SZ.v i}).
        S.index values j == S.index (physical_convolution (I64.v position) cv pv) j)
    decreases (convolution_elements - !index)
  {
    let i = !index;
    let value = convolution_cell previous projected position i;
    output.(i) <- value;
    index := i +^ 1sz;
  };
  with values. assert output |-> values;
  assert pure (S.equal values (physical_convolution (I64.v position) cv pv));
  physical_convolution_relation (I64.v position) cv pv;
}

inline_for_extraction noextract
let convolution_interpretation =
  real_vector weight_elements & real_vector convolution_elements

let convolution_inputs
  (weights : lseq bf16 weight_elements) (convolution : lseq f32 convolution_elements)
  (model : convolution_interpretation)
  : prop = weights %~ fst model /\ convolution %~ snd model

let convolution_term
  (model : convolution_interpretation)
  (channel : nat{channel < channel_count}) (tap : nat)
  : GTot real =
    if tap < tap_count then
      S.index (snd model) (channel * tap_count + tap) *.
        S.index (fst model) (convolution_offset + channel * tap_count + tap)
    else 0.0R

#push-options "--fuel 4 --z3rlimit 60"
let convolution_cell_model
  (region : nat{region < 3}) (head : nat{head < head_count}) (i : nat{i < head_width})
  (model : convolution_interpretation)
  : Lemma (
      silu (prefix tap_count (convolution_term model (region * value_width + head * head_width + i))) ==
      S.index (convolved_head (fst model) (snd model) region head) i)
  = let channel = region * value_width + head * head_width + i in
    convolved_head_at (fst model) (snd model) region head i;
    prefix_init tap_count (convolution_term model channel);
    norm_spec [delta_only [`%convolution_sum]; zeta; iota]
      (convolution_sum (fst model) (snd model) channel);
    S.lemma_eq_elim (S.init_ghost tap_count (convolution_term model channel))
      (S.init_ghost tap_count (fun tap ->
        S.index (snd model) (channel * tap_count + tap) *.
          S.index (fst model) (convolution_offset + channel * tap_count + tap)))
#pop-options

inline_for_extraction noextract
fn convolve_channel
  (weights : larray bf16 weight_elements)
  (convolution : larray f32 convolution_elements)
  (channel : sz{channel < channel_count})
  (#wv : erased (lseq bf16 weight_elements))
  (#cv : erased (lseq f32 convolution_elements))
  (#fw #fc : perm)
  preserves gpu ** weights |-> Frac fw wv ** convolution |-> Frac fc cv
  returns result : f32
  ensures pure (forall (model : convolution_interpretation).
    convolution_inputs wv cv model ==>
    result %~ silu (prefix tap_count (convolution_term model (SZ.v channel))))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut tap = 0sz;
  let mut accumulator : f32 = zero;
  while (!tap <^ 4sz)
    invariant exists* (t : sz{t <= 4}) (value : f32).
      tap |-> t ** accumulator |-> value **
      pure (forall (model : convolution_interpretation).
        convolution_inputs wv cv model ==>
        value %~ prefix (SZ.v t) (convolution_term model (SZ.v channel)))
    decreases (4 - !tap)
  {
    let t = !tap;
    let offset = channel *^ 4sz +^ t;
    let x = convolution.(offset);
    let w : f32 = Cast.fcast (weights.(SZ.uint32_to_sizet 19433472ul +^ offset));
    let previous = !accumulator;
    let value = F32.fma_rn_ftz x w previous;
    lift_fma #convolution_interpretation (convolution_inputs wv cv)
      (fun model -> S.index (snd model) (SZ.v offset))
      (fun model -> S.index (fst model) (convolution_offset + SZ.v offset))
      (fun model -> prefix (SZ.v t) (convolution_term model (SZ.v channel)))
      x w previous value;
    accumulator := value;
    tap := t +^ 1sz;
  };
  let value = !accumulator;
  silu32 value
}

inline_for_extraction noextract
fn convolve_head
  (weights : larray bf16 weight_elements)
  (convolution : larray f32 convolution_elements)
  (output : larray f32 head_width)
  (region : sz{region < 3}) (head : sz{head < head_count})
  (#wv : erased (lseq bf16 weight_elements))
  (#cv : erased (lseq f32 convolution_elements))
  (#old : erased (lseq f32 head_width))
  (#fw #fc : perm)
  preserves gpu ** weights |-> Frac fw wv ** convolution |-> Frac fc cv
  requires output |-> old
  ensures exists* (after : lseq f32 head_width). output |-> after **
    pure (forall (model : convolution_interpretation).
      convolution_inputs wv cv model ==>
      after %~ convolved_head (fst model) (snd model) (SZ.v region) (SZ.v head))
{
  let mut column = 0sz;
  while (!column <^ 128sz)
    invariant exists* (i : sz{i <= 128}) (values : lseq f32 head_width).
      column |-> i ** output |-> values **
      pure (forall (model : convolution_interpretation).
        convolution_inputs wv cv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~
            S.index (convolved_head (fst model) (snd model) (SZ.v region) (SZ.v head)) j))
    decreases (128 - !column)
  {
    let i = !column;
    let channel = region *^ 2048sz +^ head *^ 128sz +^ i;
    let result = convolve_channel weights convolution channel;
    C.forall_intro (convolution_cell_model (SZ.v region) (SZ.v head) (SZ.v i));
    output.(i) <- result;
    column := i +^ 1sz;
  }
}

inline_for_extraction noextract
let inverse_head_width () : Pure f32
  (requires True)
  (ensures fun result -> result %~ (1.0R /. 128.0R))
  = let n : f32 = of_int 128L in
    let _ = of_int_approx #f32 128L in
    div_approx (one #f32) n 1.0R 128.0R;
    div (one #f32) n

let gated_square_term (model : real_vector head_width) (warp lane : nat) : GTot real =
  head_square model (warp * 32 + lane)

let gated_warp_square (model : real_vector head_width) (warp : nat) : GTot real =
  if warp < 4 then prefix 32 (gated_square_term model warp) else 0.0R

let gated_warp_model (model : real_vector head_width) (warp : nat{warp < 4})
  : Lemma (gated_warp_square model warp ==
      prefix 32 (fun lane -> head_square model (warp * 32 + lane)))
  = prefix_ext 32 (gated_square_term model warp)
      (fun lane -> head_square model (warp * 32 + lane))

let gated_squares_model (model : real_vector head_width)
  : Lemma (prefix 32 (gated_warp_square model) == sum_squares model head_width)
  = C.forall_intro (gated_warp_model model);
    prefix_ext 4 (gated_warp_square model)
      (fun warp -> prefix 32 (fun lane -> head_square model (warp * 32 + lane)));
    prefix_blocks 4 32 (head_square model);
    head_squares_prefix model head_width;
    prefix_split 4 28 (gated_warp_square model);
    prefix_ext 28 (fun i -> gated_warp_square model (4 + i)) (fun _ -> 0.0R);
    prefix_zero 28

let gated_rms_finish
  (before : lseq f32 head_width) (argument result : f32)
  : Lemma
      (requires
        (forall (model : real_vector head_width). before %~ model ==>
          argument %~ (sum_squares model head_width /. 128.0R +. epsilon <: real)) /\
        (forall (r : Sqrt.rpos). argument %~ (r <: real) ==>
          result %~ (1.0R /. Sqrt.sqrt r)))
      (ensures forall (model : real_vector head_width). before %~ model ==>
        result %~ (rms_scale #head_width model <: real))
  = let related (model : real_vector head_width{before %~ model})
      : Lemma (result %~ (rms_scale #head_width model <: real))
      = ()
    in C.forall_intro related

inline_for_extraction noextract
fn gated_rstd
  (input : larray f32 head_width)
  (#before : erased (lseq f32 head_width))
  (#fraction : perm)
  preserves gpu ** input |-> Frac fraction before
  returns result : f32
  ensures pure (forall (model : real_vector head_width). before %~ model ==>
    result %~ (rms_scale #head_width model <: real))
{
  assert pure ((zero #f32) %~ 0.0R);
  let mut lanes = [| zero #f32 #_; 32sz |];
  let mut warps = [| zero #f32 #_; 32sz |];
  let mut warp = 0sz;
  while (!warp <^ 4sz)
    invariant exists* (w : sz{w <= 4}) (values : lseq f32 32).
      warp |-> w ** warps |-> values **
      pure (forall (model : real_vector head_width). before %~ model ==>
        (forall (i : nat{i < 32}).
          if i < SZ.v w then S.index values i %~ gated_warp_square model i
          else S.index values i == (zero <: f32)))
    invariant exists* (values : lseq f32 32). lanes |-> values
    decreases (4 - !warp)
  {
    let w = !warp;
    let mut lane = 0sz;
    while (!lane <^ 32sz)
      invariant exists* (i : sz{i <= 32}) (values : lseq f32 32).
        lane |-> i ** lanes |-> values **
        pure (forall (model : real_vector head_width). before %~ model ==>
          (forall (j : nat{j < SZ.v i}).
            S.index values j %~ gated_square_term model (SZ.v w) j))
      decreases (32 - !lane)
    {
      let i = !lane;
      let x = input.(w *^ 32sz +^ i);
      let squared = F32.fma_rn_ftz x x (zero #f32);
      lift_fma (fun (model : real_vector head_width) -> before %~ model)
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i))
        (fun model -> S.index model (SZ.v w * 32 + SZ.v i))
        (fun _ -> 0.0R) x x (zero #f32) squared;
      lanes.(i) <- squared;
      lane := i +^ 1sz;
    };
    with values. assert lanes |-> values;
    let warp_total = sum32 lanes;
    lift_sum32 (fun (model : real_vector head_width) -> before %~ model)
      (fun model -> gated_square_term model (SZ.v w)) values warp_total;
    warps.(w) <- warp_total;
    warp := w +^ 1sz;
  };
  with values. assert warps |-> values;
  let squares = sum32 warps;
  lift_sum32 (fun (model : real_vector head_width) -> before %~ model)
    gated_warp_square values squares;
  C.forall_intro gated_squares_model;
  let factor = inverse_head_width ();
  let eps = epsilon32 ();
  assert pure (factor %~ (1.0R /. 128.0R) /\ eps %~ (epsilon <: real));
  let argument = F32.fma_rn_ftz squares factor eps;
  lift_fma (fun (model : real_vector head_width) -> before %~ model)
    (fun model -> (sum_squares model head_width <: real))
    (fun _ -> 1.0R /. 128.0R) (fun _ -> (epsilon <: real))
    squares factor eps argument;
  let result = F32.rsqrt_approx_ftz argument;
  gated_rms_finish before argument result;
  result
}

inline_for_extraction noextract
let gated_interpretation =
  real_vector weight_elements & (real_vector head_width & real_vector value_width)

let gated_inputs
  (weights : lseq bf16 weight_elements)
  (readout : lseq f32 head_width) (gates : lseq f32 value_width)
  (model : gated_interpretation)
  : prop = weights %~ fst model /\ readout %~ fst (snd model) /\ gates %~ snd (snd model)

let gated_model (model : gated_interpretation) (head : nat{head < head_count})
  : GTot (real_vector head_width)
  = gated_readout (fst model) (fst (snd model))
      (vector_slice (snd (snd model)) (head * head_width) head_width)

let lift_gated
  (#p : Type) (premise : p -> prop)
  (wm : p -> GTot (real_vector weight_elements))
  (rm : p -> GTot (real_vector head_width))
  (gm : p -> GTot (real_vector value_width))
  (head : nat{head < head_count})
  (wv : lseq bf16 weight_elements) (rv : lseq f32 head_width)
  (gv : lseq f32 value_width) (after : lseq f32 head_width)
  : Lemma
      (requires
        (forall (v : p). premise v ==> wv %~ wm v /\ rv %~ rm v /\ gv %~ gm v) /\
        (forall (model : gated_interpretation). gated_inputs wv rv gv model ==>
          after %~ gated_model model head))
      (ensures forall (v : p). premise v ==>
        after %~ gated_readout (wm v) (rm v) (vector_slice (gm v) (head * head_width) head_width))
  = let related (v : p{premise v})
      : Lemma (
          after %~ gated_readout (wm v) (rm v) (vector_slice (gm v) (head * head_width) head_width))
      = let interpretation : gated_interpretation = (wm v, (rm v, gm v)) in
        assert (gated_inputs wv rv gv interpretation);
        assert (after %~ gated_model interpretation head)
    in C.forall_intro related

#push-options "--fuel 4 --z3rlimit 60"
let gated_cell_model
  (head : nat{head < head_count}) (i : nat{i < head_width}) (model : gated_interpretation)
  : Lemma (S.index (gated_model model head) i ==
      S.index (fst (snd model)) i *. rms_scale #head_width (fst (snd model)) *.
      S.index (fst model) (gated_norm_offset + i) *.
      silu (S.index (snd (snd model)) (head * head_width + i)))
  = ()
#pop-options

inline_for_extraction noextract
fn gate_readout
  (weights : larray bf16 weight_elements)
  (readout : larray f32 head_width)
  (gates : larray f32 value_width)
  (head : sz{head < head_count})
  (#wv : erased (lseq bf16 weight_elements))
  (#rv : erased (lseq f32 head_width))
  (#gv : erased (lseq f32 value_width))
  (#fw #fg : perm)
  preserves gpu ** weights |-> Frac fw wv ** gates |-> Frac fg gv
  requires readout |-> rv
  ensures exists* (after : lseq f32 head_width). readout |-> after **
    pure (forall (model : gated_interpretation). gated_inputs wv rv gv model ==>
      after %~ gated_model model (SZ.v head))
{
  let scale = gated_rstd readout;
  let mut column = 0sz;
  while (!column <^ 128sz)
    invariant exists* (i : sz{i <= 128}) (values : lseq f32 head_width).
      column |-> i ** readout |-> values **
      pure (forall (j : nat{SZ.v i <= j /\ j < head_width}).
        S.index values j == S.index rv j) **
      pure (forall (model : gated_interpretation). gated_inputs wv rv gv model ==>
        (forall (j : nat{j < SZ.v i}).
          S.index values j %~ S.index (gated_model model (SZ.v head)) j))
    decreases (128 - !column)
  {
    let i = !column;
    let value = readout.(i);
    let gain : f32 = Cast.fcast (weights.(SZ.uint32_to_sizet 19458080ul +^ i));
    let gate = gates.(head *^ 128sz +^ i);
    assert pure (forall (model : gated_interpretation). gated_inputs wv rv gv model ==>
      value %~ S.index (fst (snd model)) (SZ.v i) /\
      scale %~ (rms_scale #head_width (fst (snd model)) <: real) /\
      gain %~ S.index (fst model) (gated_norm_offset + SZ.v i) /\
      gate %~ S.index (snd (snd model)) (SZ.v head * head_width + SZ.v i));
    let scaled = F32.mul_rn_ftz value scale;
    lift_mul #gated_interpretation (gated_inputs wv rv gv)
      (fun model -> S.index (fst (snd model)) (SZ.v i))
      (fun model -> (rms_scale #head_width (fst (snd model)) <: real)) value scale scaled;
    let normed = F32.mul_rn_ftz scaled gain;
    lift_mul #gated_interpretation (gated_inputs wv rv gv)
      (fun model -> S.index (fst (snd model)) (SZ.v i) *. rms_scale #head_width (fst (snd model)))
      (fun model -> S.index (fst model) (gated_norm_offset + SZ.v i)) scaled gain normed;
    let gate_value = silu32 gate;
    assert pure (forall (model : gated_interpretation). gated_inputs wv rv gv model ==>
      gate_value %~ silu (S.index (snd (snd model)) (SZ.v head * head_width + SZ.v i)));
    let result = F32.mul_rn_ftz normed gate_value;
    lift_mul #gated_interpretation (gated_inputs wv rv gv)
      (fun model -> S.index (fst (snd model)) (SZ.v i) *. rms_scale #head_width (fst (snd model)) *.
        S.index (fst model) (gated_norm_offset + SZ.v i))
      (fun model -> silu (S.index (snd (snd model)) (SZ.v head * head_width + SZ.v i)))
      normed gate_value result;
    C.forall_intro (gated_cell_model (SZ.v head) (SZ.v i));
    readout.(i) <- result;
    column := i +^ 1sz;
  }
}

let dot_vector_term (a b : real_vector head_width) (i : nat) : GTot real =
  if i < head_width then S.index a i *. S.index b i else 0.0R

let dot_vector_prefix (a b : real_vector head_width)
  : Lemma (dot a b == prefix head_width (dot_vector_term a b))
  = prefix_init head_width (dot_vector_term a b);
    FStar.Pervasives.norm_spec [delta_only [`%dot]; zeta; iota] (dot a b);
    S.lemma_eq_elim (S.init_ghost head_width (dot_vector_term a b))
      (S.init_ghost head_width (fun i -> S.index a i *. S.index b i))

let delta_error (old key : real_vector head_width) (value beta decay : real) : GTot real =
  beta *. (value -. decay *. dot old key)

let delta_row_model (old key : real_vector head_width) (value beta decay : real)
  : GTot (real_vector head_width)
  = let error = delta_error old key value beta decay in
    S.init_ghost head_width (fun i -> decay *. S.index old i +. error *. S.index key i)

#push-options "--fuel 4 --z3rlimit 60"
let delta_row_cell
  (old key : real_vector head_width) (value beta decay : real) (i : nat{i < head_width})
  : Lemma (S.index (delta_row_model old key value beta decay) i ==
      decay *. S.index old i +. delta_error old key value beta decay *. S.index key i)
  = ()
#pop-options

#push-options "--fuel 0 --ifuel 0"
let real_mul_assoc (a b c : real) : Lemma ((a *. b) *. c == a *. (b *. c)) = ()

let real_mul_distribute (a b c : real)
  : Lemma ((a +. b) *. c == a *. c +. b *. c) = ()

let distribute_delta_dot (decay old error key query : real)
  : Lemma ((decay *. old +. error *. key) *. query ==
      decay *. (old *. query) +. error *. (key *. query))
  = real_mul_distribute (decay *. old) (error *. key) query;
    real_mul_assoc decay old query;
    real_mul_assoc error key query
#pop-options

#push-options "--fuel 0 --ifuel 0"
let delta_row_dot_term
  (old key query : real_vector head_width) (value beta decay : real) (i : nat{i < head_width})
  : Lemma (dot_vector_term (delta_row_model old key value beta decay) query i ==
      decay *. dot_vector_term old query i +.
      delta_error old key value beta decay *. dot_vector_term key query i)
  = delta_row_cell old key value beta decay i;
    FStar.Pervasives.norm_spec [delta_only [`%dot_vector_term]; zeta; iota]
      (dot_vector_term (delta_row_model old key value beta decay) query i);
    FStar.Pervasives.norm_spec [delta_only [`%dot_vector_term]; zeta; iota]
      (dot_vector_term old query i);
    FStar.Pervasives.norm_spec [delta_only [`%dot_vector_term]; zeta; iota]
      (dot_vector_term key query i);
    distribute_delta_dot decay (S.index old i)
      (delta_error old key value beta decay) (S.index key i) (S.index query i)
#pop-options

let delta_row_dot
  (old key query : real_vector head_width) (value beta decay : real)
  : Lemma (dot (delta_row_model old key value beta decay) query ==
      decay *. dot old query +. delta_error old key value beta decay *. dot key query)
  = let error = delta_error old key value beta decay in
    let updated = delta_row_model old key value beta decay in
    dot_vector_prefix updated query;
    dot_vector_prefix old query;
    dot_vector_prefix key query;
    C.forall_intro (delta_row_dot_term old key query value beta decay);
    prefix_ext head_width (dot_vector_term updated query)
      (fun i -> decay *. dot_vector_term old query i +. error *. dot_vector_term key query i);
    prefix_add head_width
      (fun i -> decay *. dot_vector_term old query i)
      (fun i -> error *. dot_vector_term key query i);
    prefix_scale head_width decay (dot_vector_term old query);
    prefix_scale head_width error (dot_vector_term key query)

noeq noextract
type row_interpretation = {
  row_old : real_vector head_width;
  row_key : real_vector head_width;
  row_query : real_vector head_width;
  row_value : real;
  row_beta : real;
  row_decay : real
}

let row_model (model : row_interpretation) : GTot (real_vector head_width) =
  delta_row_model model.row_old model.row_key model.row_value model.row_beta model.row_decay

let row_error (model : row_interpretation) : GTot real =
  delta_error model.row_old model.row_key model.row_value model.row_beta model.row_decay

let row_inputs
  (old key query : lseq f32 head_width)
  (value beta decay kq : f32)
  (model : row_interpretation)
  : prop =
    old %~ model.row_old /\ key %~ model.row_key /\ query %~ model.row_query /\
    value %~ model.row_value /\ beta %~ model.row_beta /\ decay %~ model.row_decay /\
    kq %~ dot model.row_key model.row_query

let row_readout_model (model : row_interpretation)
  : Lemma (dot (row_model model) model.row_query ==
      model.row_decay *. dot model.row_old model.row_query +.
      row_error model *. dot model.row_key model.row_query)
  = delta_row_dot model.row_old model.row_key model.row_query
      model.row_value model.row_beta model.row_decay

let row_dot_lift
  (old key query : lseq f32 head_width)
  (value beta decay kq stk sqv : f32)
  : Lemma
      (requires
        (forall (model : real_vector head_width & real_vector head_width).
          head_dot_inputs old key model ==>
          stk %~ dot (vector_slice (fst model) 0 head_width) (snd model)) /\
        (forall (model : real_vector head_width & real_vector head_width).
          head_dot_inputs old query model ==>
          sqv %~ dot (vector_slice (fst model) 0 head_width) (snd model)))
      (ensures forall (model : row_interpretation). row_inputs old key query value beta decay kq model ==>
        stk %~ dot model.row_old model.row_key /\ sqv %~ dot model.row_old model.row_query)
  = let related (model : row_interpretation{row_inputs old key query value beta decay kq model})
      : Lemma (stk %~ dot model.row_old model.row_key /\ sqv %~ dot model.row_old model.row_query)
      = S.lemma_eq_elim (vector_slice model.row_old 0 head_width) model.row_old;
        assert (head_dot_inputs old key (model.row_old, model.row_key));
        assert (head_dot_inputs old query (model.row_old, model.row_query));
        assert (stk %~ dot (vector_slice model.row_old 0 head_width) model.row_key);
        assert (sqv %~ dot (vector_slice model.row_old 0 head_width) model.row_query)
    in C.forall_intro related

#push-options "--fuel 4"
let row_cell_model (i : nat{i < head_width}) (model : row_interpretation)
  : Lemma (S.index (row_model model) i ==
      model.row_decay *. S.index model.row_old i +. row_error model *. S.index model.row_key i)
  = ()
#pop-options

#push-options "--z3rlimit 120"
inline_for_extraction noextract
fn update_delta_row
  (row key query : larray f32 head_width)
  (value beta decay kq : f32)
  (#old #kv #qv : erased (lseq f32 head_width))
  (#fk #fq : perm)
  preserves gpu ** key |-> Frac fk kv ** query |-> Frac fq qv
  requires row |-> old
  returns result : f32
  ensures exists* (after : lseq f32 head_width). row |-> after **
    pure (forall (model : row_interpretation). row_inputs old kv qv value beta decay kq model ==>
      after %~ row_model model /\ result %~ dot (row_model model) model.row_query)
{
  let stk = strided_dot row key 0sz;
  let sqv = strided_dot row query 0sz;
  row_dot_lift old kv qv value beta decay kq stk sqv;
  let negative_decay = negate32 decay;
  let residual = F32.fma_rn_ftz negative_decay stk value;
  lift_fma #row_interpretation (row_inputs old kv qv value beta decay kq)
    (fun model -> 0.0R -. model.row_decay)
    (fun model -> dot model.row_old model.row_key)
    (fun model -> model.row_value)
    negative_decay stk value residual;
  let error = F32.mul_rn_ftz residual beta;
  lift_mul #row_interpretation (row_inputs old kv qv value beta decay kq)
    (fun model -> (0.0R -. model.row_decay) *. dot model.row_old model.row_key +. model.row_value)
    (fun model -> model.row_beta) residual beta error;
  assert pure (forall (model : row_interpretation). row_inputs old kv qv value beta decay kq model ==>
    error %~ row_error model);
  let correction = F32.mul_rn_ftz error kq;
  lift_mul #row_interpretation (row_inputs old kv qv value beta decay kq)
    row_error (fun model -> dot model.row_key model.row_query) error kq correction;
  let readout = F32.fma_rn_ftz decay sqv correction;
  lift_fma #row_interpretation (row_inputs old kv qv value beta decay kq)
    (fun model -> model.row_decay)
    (fun model -> dot model.row_old model.row_query)
    (fun model -> row_error model *. dot model.row_key model.row_query)
    decay sqv correction readout;
  C.forall_intro row_readout_model;
  let mut column = 0sz;
  while (!column <^ 128sz)
    invariant exists* (i : sz{i <= 128}) (values : lseq f32 head_width).
      column |-> i ** row |-> values **
      pure (forall (j : nat{SZ.v i <= j /\ j < head_width}).
        S.index values j == S.index old j) **
      pure (forall (model : row_interpretation). row_inputs old kv qv value beta decay kq model ==>
        (forall (j : nat{j < SZ.v i}). S.index values j %~ S.index (row_model model) j))
    decreases (128 - !column)
  {
    let i = !column;
    let prior = row.(i);
    let k = key.(i);
    let change = F32.mul_rn_ftz k error;
    lift_mul #row_interpretation (row_inputs old kv qv value beta decay kq)
      (fun model -> S.index model.row_key (SZ.v i)) row_error k error change;
    let updated = F32.fma_rn_ftz prior decay change;
    lift_fma #row_interpretation (row_inputs old kv qv value beta decay kq)
      (fun model -> S.index model.row_old (SZ.v i))
      (fun model -> model.row_decay)
      (fun model -> S.index model.row_key (SZ.v i) *. row_error model)
      prior decay change updated;
    C.forall_intro (row_cell_model (SZ.v i));
    row.(i) <- updated;
    column := i +^ 1sz;
  };
  readout
}
#pop-options

let replace_slice
  (#a : Type) (#n #m : nat)
  (before : lseq a n) (offset : nat{offset + m <= n}) (patch : lseq a m)
  : GTot (lseq a n)
  = S.init_ghost n (fun i ->
      if offset <= i /\ i < offset + m then S.index patch (i - offset) else S.index before i)

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

noeq noextract
type update_interpretation = {
  update_previous : real_vector head_state_elements;
  update_key : real_vector head_width;
  update_query : real_vector head_width;
  update_value : real_vector head_width;
  update_beta : real;
  update_decay : real
}

let update_inputs
  (previous : lseq f32 head_state_elements)
  (key query value : lseq f32 head_width) (beta decay : f32)
  (model : update_interpretation)
  : prop =
    previous %~ model.update_previous /\ key %~ model.update_key /\
    query %~ model.update_query /\ value %~ model.update_value /\
    beta %~ model.update_beta /\ decay %~ model.update_decay

let updated_head
  (model : update_interpretation)
  : GTot (real_vector head_state_elements)
  = delta_state model.update_previous
      model.update_key model.update_value model.update_beta model.update_decay

let updated_readout
  (model : update_interpretation)
  : GTot (real_vector head_width)
  = state_readout (updated_head model) model.update_query

let update_row_interpretation
  (row : nat{row < head_width})
  (model : update_interpretation)
  : GTot row_interpretation
  = {
      row_old = vector_slice model.update_previous
        (row * head_width) head_width;
      row_key = model.update_key;
      row_query = model.update_query;
      row_value = S.index model.update_value row;
      row_beta = model.update_beta;
      row_decay = model.update_decay
    }

let slice_composition
  (#n : nat) (values : real_vector n)
  (start count : nat{start + count <= n})
  (offset length : nat{offset + length <= count})
  : Lemma (vector_slice (vector_slice values start count) offset length ==
      vector_slice values (start + offset) length)
  = assert (S.equal (vector_slice (vector_slice values start count) offset length)
      (vector_slice values (start + offset) length))

#push-options "--fuel 4 --z3rlimit 60"
let update_row_model
  (row : nat{row < head_width})
  (model : update_interpretation)
  : Lemma (
      row_model (update_row_interpretation row model) ==
        vector_slice (updated_head model) (row * head_width) head_width /\
      dot (row_model (update_row_interpretation row model)) model.update_query ==
        S.index (updated_readout model) row)
  = assert (S.equal
      (row_model (update_row_interpretation row model))
      (vector_slice (updated_head model) (row * head_width) head_width))
#pop-options

let update_row_lift
  (previous : lseq f32 head_state_elements)
  (key query value : lseq f32 head_width) (beta decay kq v readout : f32)
  (row : nat{row < head_width})
  (after : lseq f32 head_width)
  : Lemma
      (requires
        v == S.index value row /\
        (forall (kr qr : real_vector head_width).
          key %~ kr /\ query %~ qr ==> kq %~ dot kr qr) /\
        (forall (model : row_interpretation).
          row_inputs
            (S.slice previous (row * head_width) ((row + 1) * head_width))
            key query v beta decay kq model ==>
          after %~ row_model model /\ readout %~ dot (row_model model) model.row_query))
      (ensures forall (model : update_interpretation).
        update_inputs previous key query value beta decay model ==>
        after %~ vector_slice (updated_head model) (row * head_width) head_width /\
        readout %~ S.index (updated_readout model) row)
  = let related
      (model : update_interpretation{update_inputs previous key query value beta decay model})
      : Lemma (
        after %~ vector_slice (updated_head model) (row * head_width) head_width /\
        readout %~ S.index (updated_readout model) row)
      = f32_slice_relation previous model.update_previous
          (row * head_width) head_width;
        update_row_model row model;
        assert (row_inputs
          (S.slice previous (row * head_width) ((row + 1) * head_width))
          key query v beta decay kq (update_row_interpretation row model));
        assert (after %~ row_model (update_row_interpretation row model))
    in C.forall_intro related

let update_prefix
  (previous current : lseq f32 head_state_elements)
  (key query value : lseq f32 head_width) (beta decay : f32)
  (rows : nat{rows <= head_width})
  : prop
  = (forall (i : nat{i < head_state_elements}).
      rows * head_width <= i ==>
      S.index current i == S.index previous i) /\
    (forall (model : update_interpretation).
      update_inputs previous key query value beta decay model ==>
      (forall (i : nat{i < rows * head_width}).
        S.index current i %~ S.index (updated_head model) i))

let readout_prefix
  (previous : lseq f32 head_state_elements)
  (key query value : lseq f32 head_width) (beta decay : f32)
  (rows : nat{rows <= head_width})
  (readout : lseq f32 head_width)
  : prop
  = forall (model : update_interpretation).
      update_inputs previous key query value beta decay model ==>
      (forall (i : nat{i < rows}). S.index readout i %~ S.index (updated_readout model) i)

let update_prefix_step
  (previous current : lseq f32 head_state_elements)
  (key query value : lseq f32 head_width) (beta decay : f32)
  (row : nat{row < head_width})
  (after : lseq f32 head_width)
  : Lemma
      (requires update_prefix previous current key query value beta decay row /\
        (forall (model : update_interpretation).
          update_inputs previous key query value beta decay model ==>
          after %~ vector_slice (updated_head model) (row * head_width) head_width))
      (ensures update_prefix previous
        (replace_slice current (row * head_width) after)
        key query value beta decay (row + 1))
  = ()

let kq_model
  (key query : lseq f32 head_width) (kq : f32)
  : Lemma
      (requires forall (model : real_vector head_width & real_vector head_width).
        head_dot_inputs key query model ==>
        kq %~ dot (vector_slice (fst model) 0 head_width) (snd model))
      (ensures forall (kr qr : real_vector head_width).
        key %~ kr /\ query %~ qr ==> kq %~ dot kr qr)
  = let related (kr qr : real_vector head_width)
      : Lemma (key %~ kr /\ query %~ qr ==> kq %~ dot kr qr)
      = S.lemma_eq_elim (vector_slice kr 0 head_width) kr;
        assert (key %~ kr /\ query %~ qr ==> head_dot_inputs key query (kr, qr));
        assert (key %~ kr /\ query %~ qr ==> kq %~ dot (vector_slice kr 0 head_width) qr)
    in C.forall_intro_2 related

inline_for_extraction noextract
fn update_head
  (states : larray f32 head_state_elements)
  (key query value : larray f32 head_width)
  (readout : larray f32 head_width)
  (beta decay : f32)
  (#previous : erased (lseq f32 head_state_elements))
  (#kv #qv #vv #old_readout : erased (lseq f32 head_width))
  (#fk #fq #fv : perm)
  preserves gpu ** key |-> Frac fk kv ** query |-> Frac fq qv ** value |-> Frac fv vv
  requires states |-> previous ** readout |-> old_readout
  ensures exists* (after : lseq f32 head_state_elements) (out : lseq f32 head_width).
    states |-> after ** readout |-> out **
    pure (forall (model : update_interpretation).
      update_inputs previous kv qv vv beta decay model ==>
      after %~ updated_head model /\ out %~ updated_readout model)
{
  let kq = strided_dot key query 0sz;
  kq_model kv qv kq;
  let mut row_values = [| zero #f32 #_; 128sz |];
  let mut row = 0sz;
  while (!row <^ 128sz)
    invariant exists* (j : sz{j <= 128}) (current : lseq f32 head_state_elements)
      (out : lseq f32 head_width).
      row |-> j ** states |-> current ** readout |-> out **
      pure (update_prefix previous current kv qv vv beta decay (SZ.v j) /\
        readout_prefix previous kv qv vv beta decay (SZ.v j) out)
    invariant exists* (values : lseq f32 head_width). row_values |-> values
    decreases (128 - !row)
  {
    let j = !row;
    let offset = j *^ 128sz;
    with current. assert states |-> current;
    copy_f32_slice states row_values offset 128sz;
    assert pure (S.equal
      (S.slice current (SZ.v offset) (SZ.v offset + head_width))
      (S.slice previous (SZ.v offset) (SZ.v offset + head_width)));
    let v = value.(j);
    let result = update_delta_row row_values key query v beta decay kq;
    with after. assert row_values |-> after;
    update_row_lift previous kv qv vv beta decay kq v result (SZ.v j) after;
    update_prefix_step previous current kv qv vv beta decay (SZ.v j) after;
    write_f32_slice states row_values offset 128sz;
    readout.(j) <- result;
    row := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn bias_controls
  (weights : larray bf16 weight_elements)
  (alpha controls : larray f32 head_count)
  (#wv : erased (lseq bf16 weight_elements))
  (#av #old : erased (lseq f32 head_count))
  (#fw #fa : perm)
  preserves gpu ** weights |-> Frac fw wv ** alpha |-> Frac fa av
  requires controls |-> old
  ensures exists* (after : lseq f32 head_count). controls |-> after **
    pure (forall (model : matrix_interpretation head_count). matrix_inputs wv av model ==>
      after %~ control_arguments (fst model) (snd model))
{
  let mut head = 0sz;
  while (!head <^ 16sz)
    invariant exists* (h : sz{h <= 16}) (values : lseq f32 head_count).
      head |-> h ** controls |-> values **
      pure (forall (model : matrix_interpretation head_count). matrix_inputs wv av model ==>
        (forall (i : nat{i < SZ.v h}).
          S.index values i %~ S.index (control_arguments (fst model) (snd model)) i))
    decreases (16 - !head)
  {
    let h = !head;
    let a = alpha.(h);
    let bias : f32 = Cast.fcast (weights.(SZ.uint32_to_sizet 19458064ul +^ h));
    let value = F32.add_rn_ftz a bias;
    lift_add #(matrix_interpretation head_count) (matrix_inputs #head_count wv av)
      (fun model -> S.index (snd model) (SZ.v h))
      (fun model -> S.index (fst model) (alpha_bias_offset + SZ.v h))
      a bias value;
    controls.(h) <- value;
    head := h +^ 1sz;
  }
}

inline_for_extraction noextract
fn decay32 (weight control : f32)
  preserves gpu
  returns result : f32
  ensures pure (forall (w x : real). weight %~ w /\ control %~ x ==>
    result %~ exp ((0.0R -. exp w) *. softplus control x))
{
  let scale = exponential_ftz weight;
  let negative_scale = negate32 scale;
  let sp = softplus32 control;
  let argument = F32.mul_rn_ftz negative_scale sp;
  lift_mul #(real & real) (fun p -> weight %~ fst p /\ control %~ snd p)
    (fun p -> 0.0R -. exp (fst p))
    (fun p -> softplus control (snd p)) negative_scale sp argument;
  exponential_ftz argument
}

noeq noextract
type heads_interpretation = {
  heads_weights : real_vector weight_elements;
  heads_convolution : real_vector convolution_elements;
  heads_previous : real_vector state_elements;
  heads_gates : real_vector value_width;
  heads_beta : real_vector head_count;
  heads_alpha : real_vector head_count
}

let heads_inputs
  (weights : lseq bf16 weight_elements)
  (convolution : lseq f32 convolution_elements)
  (previous : lseq f32 state_elements)
  (gates : lseq f32 value_width) (beta controls : lseq f32 head_count)
  (model : heads_interpretation)
  : prop =
    weights %~ model.heads_weights /\ convolution %~ model.heads_convolution /\
    previous %~ model.heads_previous /\ gates %~ model.heads_gates /\
    beta %~ model.heads_beta /\ controls %~ model.heads_alpha

let head_local_inputs
  (weights : lseq bf16 weight_elements)
  (convolution : lseq f32 convolution_elements)
  (previous : lseq f32 head_state_elements)
  (gates : lseq f32 value_width) (beta controls : lseq f32 head_count)
  (head : nat{head < head_count}) (model : heads_interpretation)
  : prop =
    weights %~ model.heads_weights /\ convolution %~ model.heads_convolution /\
    previous %~ vector_slice model.heads_previous (head * head_state_elements) head_state_elements /\
    gates %~ model.heads_gates /\ beta %~ model.heads_beta /\ controls %~ model.heads_alpha

let head_result
  (model : heads_interpretation) (controls : lseq f32 head_count) (head : nat{head < head_count})
  : GTot (real_vector head_state_elements & real_vector head_width)
  = head_transition model.heads_weights model.heads_convolution model.heads_previous
      model.heads_gates model.heads_beta model.heads_alpha controls head

let head_update_model
  (model : heads_interpretation) (controls : lseq f32 head_count) (head : nat{head < head_count})
  : GTot update_interpretation
  = {
      update_previous = vector_slice model.heads_previous (head * head_state_elements) head_state_elements;
      update_key = normalized_key (convolved_head model.heads_weights model.heads_convolution 1 head);
      update_query = normalized_query (convolved_head model.heads_weights model.heads_convolution 0 head);
      update_value = convolved_head model.heads_weights model.heads_convolution 2 head;
      update_beta = sigmoid (S.index model.heads_beta head);
      update_decay = exp ((0.0R -. exp (S.index model.heads_weights (log_decay_offset + head))) *.
        softplus (S.index controls head) (S.index model.heads_alpha head))
    }

#push-options "--fuel 6 --z3rlimit 60"
let head_result_model
  (model : heads_interpretation) (controls : lseq f32 head_count) (head : nat{head < head_count})
  : Lemma (
      fst (head_result model controls head) == updated_head (head_update_model model controls head) /\
      snd (head_result model controls head) ==
        gated_readout model.heads_weights (updated_readout (head_update_model model controls head))
          (vector_slice model.heads_gates (head * head_width) head_width))
  = ()
#pop-options

let head_prepared_relation
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 head_state_elements) (gv : lseq f32 value_width)
  (bv controls : lseq f32 head_count)
  (head : nat{head < head_count})
  (query_before key_before query key value : lseq f32 head_width)
  : Lemma
      (requires
        (forall (model : convolution_interpretation). convolution_inputs wv cv model ==>
          query_before %~ convolved_head (fst model) (snd model) 0 head /\
          key_before %~ convolved_head (fst model) (snd model) 1 head /\
          value %~ convolved_head (fst model) (snd model) 2 head) /\
        (forall (model : real_vector head_width). query_before %~ model ==>
          query %~ head_normalization_value model true) /\
        (forall (model : real_vector head_width). key_before %~ model ==>
          key %~ head_normalization_value model false))
      (ensures forall (model : heads_interpretation).
        head_local_inputs wv cv previous gv bv controls head model ==>
        query %~ (head_update_model model controls head).update_query /\
        key %~ (head_update_model model controls head).update_key /\
        value %~ (head_update_model model controls head).update_value)
  = let related
      (model : heads_interpretation{head_local_inputs wv cv previous gv bv controls head model})
      : Lemma (
        query %~ (head_update_model model controls head).update_query /\
        key %~ (head_update_model model controls head).update_key /\
        value %~ (head_update_model model controls head).update_value)
      = assert (convolution_inputs wv cv (model.heads_weights, model.heads_convolution));
        assert (query_before %~ convolved_head model.heads_weights model.heads_convolution 0 head);
        assert (key_before %~ convolved_head model.heads_weights model.heads_convolution 1 head)
    in C.forall_intro related

let head_update_lift
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 head_state_elements) (gv : lseq f32 value_width)
  (bv controls : lseq f32 head_count)
  (head : nat{head < head_count})
  (key query value : lseq f32 head_width) (beta decay : f32)
  (after : lseq f32 head_state_elements) (readout : lseq f32 head_width)
  : Lemma
      (requires
        (forall (model : heads_interpretation).
          head_local_inputs wv cv previous gv bv controls head model ==>
          update_inputs previous key query value beta decay (head_update_model model controls head)) /\
        (forall (model : update_interpretation).
          update_inputs previous key query value beta decay model ==>
          after %~ updated_head model /\ readout %~ updated_readout model))
      (ensures forall (model : heads_interpretation).
        head_local_inputs wv cv previous gv bv controls head model ==>
        after %~ fst (head_result model controls head) /\
        readout %~ updated_readout (head_update_model model controls head))
  = let related
      (model : heads_interpretation{head_local_inputs wv cv previous gv bv controls head model})
      : Lemma (
        after %~ fst (head_result model controls head) /\
        readout %~ updated_readout (head_update_model model controls head))
      = head_result_model model controls head
    in C.forall_intro related

#push-options "--fuel 4 --z3rlimit 60"
let head_parameters_lift
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 head_state_elements) (gv : lseq f32 value_width)
  (bv controls : lseq f32 head_count) (head : nat{head < head_count})
  (key query value : lseq f32 head_width)
  (beta_input control weight beta decay : f32)
  : Lemma
      (requires
        beta_input == S.index bv head /\ control == S.index controls head /\
        (forall (wr : real_vector weight_elements). wv %~ wr ==>
          weight %~ S.index wr (log_decay_offset + head)) /\
        (forall (r : real). beta_input %~ r ==> beta %~ sigmoid r) /\
        (forall (w x : real). weight %~ w /\ control %~ x ==>
          decay %~ exp ((0.0R -. exp w) *. softplus control x)) /\
        (forall (model : heads_interpretation).
          head_local_inputs wv cv previous gv bv controls head model ==>
          query %~ (head_update_model model controls head).update_query /\
          key %~ (head_update_model model controls head).update_key /\
          value %~ (head_update_model model controls head).update_value))
      (ensures forall (model : heads_interpretation).
        head_local_inputs wv cv previous gv bv controls head model ==>
        update_inputs previous key query value beta decay (head_update_model model controls head))
  = let related
      (model : heads_interpretation{head_local_inputs wv cv previous gv bv controls head model})
      : Lemma (update_inputs previous key query value beta decay (head_update_model model controls head))
      = assert (beta_input %~ S.index model.heads_beta head);
        assert (control %~ S.index model.heads_alpha head);
        assert (weight %~ S.index model.heads_weights (log_decay_offset + head));
        assert (beta %~ sigmoid (S.index model.heads_beta head));
        assert (decay %~ exp
          ((0.0R -. exp (S.index model.heads_weights (log_decay_offset + head))) *.
            softplus (S.index controls head) (S.index model.heads_alpha head)))
    in C.forall_intro related
#pop-options

let head_gating_lift
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 head_state_elements) (gv : lseq f32 value_width)
  (bv controls : lseq f32 head_count) (head : nat{head < head_count})
  (readout after : lseq f32 head_width)
  : Lemma
      (requires
        (forall (model : heads_interpretation).
          head_local_inputs wv cv previous gv bv controls head model ==>
          readout %~ updated_readout (head_update_model model controls head)) /\
        (forall (model : gated_interpretation). gated_inputs wv readout gv model ==>
          after %~ gated_model model head))
      (ensures forall (model : heads_interpretation).
        head_local_inputs wv cv previous gv bv controls head model ==>
        after %~ snd (head_result model controls head))
  = lift_gated #heads_interpretation
      (head_local_inputs wv cv previous gv bv controls head)
      (fun model -> model.heads_weights)
      (fun model -> updated_readout (head_update_model model controls head))
      (fun model -> model.heads_gates) head wv readout gv after;
    let aligned (model : heads_interpretation)
      : Lemma (snd (head_result model controls head) ==
          gated_readout model.heads_weights (updated_readout (head_update_model model controls head))
            (vector_slice model.heads_gates (head * head_width) head_width))
      = head_result_model model controls head
    in C.forall_intro aligned

inline_for_extraction noextract
fn evaluate_head
  (weights : larray bf16 weight_elements)
  (convolution : larray f32 convolution_elements)
  (states : larray f32 head_state_elements)
  (gates : larray f32 value_width)
  (betas controls : larray f32 head_count)
  (readout : larray f32 head_width)
  (head : sz{head < head_count})
  (#wv : erased (lseq bf16 weight_elements))
  (#cv : erased (lseq f32 convolution_elements))
  (#previous : erased (lseq f32 head_state_elements))
  (#gv : erased (lseq f32 value_width))
  (#bv #av : erased (lseq f32 head_count))
  (#old_readout : erased (lseq f32 head_width))
  (#fw #fc #fg #fb #fa : perm)
  preserves gpu ** weights |-> Frac fw wv ** convolution |-> Frac fc cv **
    gates |-> Frac fg gv ** betas |-> Frac fb bv ** controls |-> Frac fa av
  requires states |-> previous ** readout |-> old_readout
  ensures exists* (after : lseq f32 head_state_elements) (out : lseq f32 head_width).
    states |-> after ** readout |-> out **
    pure (forall (model : heads_interpretation).
      head_local_inputs wv cv previous gv bv av (SZ.v head) model ==>
      after %~ fst (head_result model av (SZ.v head)) /\
      out %~ snd (head_result model av (SZ.v head)))
{
  let mut query = [| zero #f32 #_; 128sz |];
  let mut key = [| zero #f32 #_; 128sz |];
  let mut value = [| zero #f32 #_; 128sz |];
  convolve_head weights convolution query 0sz head;
  convolve_head weights convolution key 1sz head;
  convolve_head weights convolution value 2sz head;
  with query_before. assert query |-> query_before;
  with key_before. assert key |-> key_before;
  with values. assert value |-> values;
  normalize_head query true;
  normalize_head key false;
  with queries. assert query |-> queries;
  with keys. assert key |-> keys;
  head_prepared_relation wv cv previous gv bv av (SZ.v head)
    query_before key_before queries keys values;
  let beta_input = betas.(head);
  let beta = sigmoid32 beta_input;
  let control = controls.(head);
  let weight : f32 = Cast.fcast (weights.(SZ.uint32_to_sizet 19458048ul +^ head));
  assert pure (forall (wr : real_vector weight_elements). wv %~ wr ==>
    weight %~ S.index wr (log_decay_offset + SZ.v head));
  let decay = decay32 weight control;
  head_parameters_lift wv cv previous gv bv av (SZ.v head) keys queries values
    beta_input control weight beta decay;
  update_head states key query value readout beta decay;
  with after. assert states |-> after;
  with raw_readout. assert readout |-> raw_readout;
  head_update_lift wv cv previous gv bv av (SZ.v head)
    keys queries values beta decay after raw_readout;
  gate_readout weights readout gates head;
  with out. assert readout |-> out;
  head_gating_lift wv cv previous gv bv av (SZ.v head) raw_readout out;
  assert pure (forall (model : heads_interpretation).
    head_local_inputs wv cv previous gv bv av (SZ.v head) model ==>
    after %~ fst (head_result model av (SZ.v head)) /\
    out %~ snd (head_result model av (SZ.v head)));
}

let all_states (model : heads_interpretation) (controls : lseq f32 head_count)
  : GTot (real_vector state_elements)
  = S.init_ghost state_elements (fun i ->
      S.index (fst (head_result model controls (i / head_state_elements))) (i % head_state_elements))

let all_activations (model : heads_interpretation) (controls : lseq f32 head_count)
  : GTot (real_vector value_width)
  = S.init_ghost value_width (fun i ->
      S.index (snd (head_result model controls (i / head_width))) (i % head_width))

let flatten_heads
  (width : pos) (values : (h:nat{h < head_count} -> GTot (real_vector width)))
  : GTot (real_vector (head_count * width))
  = S.init_ghost (head_count * width) (fun i ->
      S.index (values (i / width)) (i % width))

let flatten_heads_slice
  (width : pos) (values : (h:nat{h < head_count} -> GTot (real_vector width)))
  (head : nat{head < head_count})
  : Lemma (vector_slice (flatten_heads width values) (head * width) width == values head)
  = let cell (i : nat{i < width})
      : Lemma (
          S.index (vector_slice (flatten_heads width values) (head * width) width) i ==
          S.index (values head) i)
      = FStar.Math.Lemmas.lemma_div_plus i head width;
        FStar.Math.Lemmas.lemma_mod_plus i head width;
        FStar.Math.Lemmas.small_div i width;
        FStar.Math.Lemmas.small_mod i width
    in
    C.forall_intro cell;
    S.lemma_eq_elim (vector_slice (flatten_heads width values) (head * width) width) (values head)

let all_states_head
  (head : nat{head < head_count}) (model : heads_interpretation) (controls : lseq f32 head_count)
  : Lemma (
      vector_slice (all_states model controls) (head * head_state_elements) head_state_elements ==
        fst (head_result model controls head))
  = assert (all_states model controls ==
      flatten_heads head_state_elements (fun h -> fst (head_result model controls h)))
      by (FStar.Tactics.V2.norm [delta_only [`%all_states; `%flatten_heads; `%state_elements]; zeta; iota];
          FStar.Tactics.V2.trefl ());
    flatten_heads_slice head_state_elements (fun h -> fst (head_result model controls h)) head;
    ()

let all_activations_head
  (head : nat{head < head_count}) (model : heads_interpretation) (controls : lseq f32 head_count)
  : Lemma (
      vector_slice (all_activations model controls) (head * head_width) head_width ==
        snd (head_result model controls head))
  = assert (all_activations model controls ==
      flatten_heads head_width (fun h -> snd (head_result model controls h)))
      by (FStar.Tactics.V2.norm [delta_only [`%all_activations; `%flatten_heads; `%value_width]; zeta; iota];
          FStar.Tactics.V2.trefl ());
    flatten_heads_slice head_width (fun h -> snd (head_result model controls h)) head;
    ()

let heads_prefix
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 state_elements) (gv : lseq f32 value_width) (bv av : lseq f32 head_count)
  (states : lseq f32 state_elements) (activation : lseq f32 value_width)
  (done : nat{done <= head_count})
  : prop =
    (forall (i : nat{done * head_state_elements <= i /\ i < state_elements}).
      S.index states i == S.index previous i) /\
    (forall (model : heads_interpretation). heads_inputs wv cv previous gv bv av model ==>
      (forall (i : nat{i < done * head_state_elements}).
        S.index states i %~ S.index (all_states model av) i) /\
      (forall (i : nat{i < done * head_width}).
        S.index activation i %~ S.index (all_activations model av) i))

let compose_relations
  (#p : Type) (before middle after : p -> prop)
  : Lemma
      (requires (forall (model : p). before model ==> middle model) /\
        (forall (model : p). middle model ==> after model))
      (ensures forall (model : p). before model ==> after model)
  = let related (model : p{before model}) : Lemma (after model) = ()
    in C.forall_intro related

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

let heads_slice_inputs
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 state_elements) (gv : lseq f32 value_width) (bv av : lseq f32 head_count)
  (head : nat{head < head_count}) (local_previous : lseq f32 head_state_elements)
  : Lemma
      (requires local_previous ==
        S.slice previous (head * head_state_elements) ((head + 1) * head_state_elements))
      (ensures forall (model : heads_interpretation).
        heads_inputs wv cv previous gv bv av model ==>
        head_local_inputs wv cv local_previous gv bv av head model)
  = let related (model : heads_interpretation{heads_inputs wv cv previous gv bv av model})
      : Lemma (head_local_inputs wv cv local_previous gv bv av head model)
      = f32_slice_relation #state_elements previous model.heads_previous
          (head * head_state_elements) head_state_elements;
        ()
    in C.forall_intro related

let heads_result_lift
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 state_elements) (gv : lseq f32 value_width) (bv av : lseq f32 head_count)
  (head : nat{head < head_count})
  (local_previous : lseq f32 head_state_elements)
  (states : lseq f32 head_state_elements) (activation : lseq f32 head_width)
  : Lemma
      (requires
        local_previous == S.slice previous (head * head_state_elements) ((head + 1) * head_state_elements) /\
        (forall (model : heads_interpretation).
          head_local_inputs wv cv local_previous gv bv av head model ==>
          states %~ fst (head_result model av head) /\ activation %~ snd (head_result model av head)))
      (ensures forall (model : heads_interpretation). heads_inputs wv cv previous gv bv av model ==>
        states %~ fst (head_result model av head) /\ activation %~ snd (head_result model av head))
  = heads_slice_inputs wv cv previous gv bv av head local_previous;
    compose_relations #heads_interpretation
      (heads_inputs wv cv previous gv bv av)
      (head_local_inputs wv cv local_previous gv bv av head)
      (fun model -> states %~ fst (head_result model av head) /\
        activation %~ snd (head_result model av head));
    ()

let heads_prefix_step
  (wv : lseq bf16 weight_elements) (cv : lseq f32 convolution_elements)
  (previous : lseq f32 state_elements) (gv : lseq f32 value_width) (bv av : lseq f32 head_count)
  (states : lseq f32 state_elements) (activation : lseq f32 value_width)
  (head : nat{head < head_count})
  (state_head : lseq f32 head_state_elements) (activation_head : lseq f32 head_width)
  : Lemma
      (requires heads_prefix wv cv previous gv bv av states activation head /\
        (forall (model : heads_interpretation). heads_inputs wv cv previous gv bv av model ==>
          state_head %~ fst (head_result model av head) /\
          activation_head %~ snd (head_result model av head)))
      (ensures heads_prefix wv cv previous gv bv av
        (replace_slice states (head * head_state_elements) state_head)
        (replace_slice activation (head * head_width) activation_head) (head + 1))
  = let aligned (model : heads_interpretation)
      : Lemma (
          vector_slice (all_states model av) (head * head_state_elements) head_state_elements ==
            fst (head_result model av head) /\
          vector_slice (all_activations model av) (head * head_width) head_width ==
            snd (head_result model av head))
      = all_states_head head model av;
        all_activations_head head model av;
        ()
    in
    C.forall_intro aligned;
    lift_replace_prefix #heads_interpretation #state_elements #head_state_elements
      (heads_inputs wv cv previous gv bv av) (fun model -> all_states model av)
      states (head * head_state_elements) state_head;
    lift_replace_prefix #heads_interpretation #value_width #head_width
      (heads_inputs wv cv previous gv bv av) (fun model -> all_activations model av)
      activation (head * head_width) activation_head;
    ()

inline_for_extraction noextract
fn recurrent_heads
  (weights : larray bf16 weight_elements)
  (convolution : larray f32 convolution_elements)
  (states : larray f32 state_elements)
  (gates : larray f32 value_width)
  (betas controls : larray f32 head_count)
  (activation : larray f32 value_width)
  (#wv : erased (lseq bf16 weight_elements))
  (#cv : erased (lseq f32 convolution_elements))
  (#previous : erased (lseq f32 state_elements))
  (#gv #old_activation : erased (lseq f32 value_width))
  (#bv #av : erased (lseq f32 head_count))
  (#fw #fc #fg #fb #fa : perm)
  preserves gpu ** weights |-> Frac fw wv ** convolution |-> Frac fc cv **
    gates |-> Frac fg gv ** betas |-> Frac fb bv ** controls |-> Frac fa av
  requires states |-> previous ** activation |-> old_activation
  ensures exists* (after : lseq f32 state_elements) (out : lseq f32 value_width).
    states |-> after ** activation |-> out **
    pure (forall (model : heads_interpretation). heads_inputs wv cv previous gv bv av model ==>
      after %~ all_states model av /\ out %~ all_activations model av)
{
  let mut head_states = [| zero #f32 #_; 16384sz |];
  let mut head_activation = [| zero #f32 #_; 128sz |];
  let mut head = 0sz;
  while (!head <^ 16sz)
    invariant exists* (h : sz{h <= 16}) (current : lseq f32 state_elements)
      (out : lseq f32 value_width).
      head |-> h ** states |-> current ** activation |-> out **
      pure (heads_prefix wv cv previous gv bv av current out (SZ.v h))
    invariant exists* (values : lseq f32 head_state_elements). head_states |-> values
    invariant exists* (values : lseq f32 head_width). head_activation |-> values
    decreases (16 - !head)
  {
    let h = !head;
    let offset = h *^ 16384sz;
    with current. assert states |-> current;
    with out. assert activation |-> out;
    copy_f32_slice states head_states offset 16384sz;
    assert pure (S.equal
      (S.slice current (SZ.v offset) (SZ.v offset + head_state_elements))
      (S.slice previous (SZ.v offset) (SZ.v offset + head_state_elements)));
    with local_previous. assert head_states |-> local_previous;
    evaluate_head weights convolution head_states gates betas controls head_activation h;
    with state_head. assert head_states |-> state_head;
    with activation_head. assert head_activation |-> activation_head;
    heads_result_lift wv cv previous gv bv av (SZ.v h) local_previous state_head activation_head;
    heads_prefix_step wv cv previous gv bv av current out (SZ.v h) state_head activation_head;
    write_f32_slice states head_states offset 16384sz;
    write_f32_slice activation head_activation (h *^ 128sz) 128sz;
    head := h +^ 1sz;
  }
}

let residual_vector (hidden projected : real_vector hidden_width)
  : GTot (real_vector hidden_width)
  = S.init_ghost hidden_width (fun i -> S.index hidden i +. S.index projected i)

let mlp_vector (gate up : real_vector mlp_width)
  : GTot (real_vector mlp_width)
  = S.init_ghost mlp_width (fun i -> silu (S.index gate i) *. S.index up i)

inline_for_extraction noextract
fn copy_hidden
  (input output : larray bf16 hidden_width)
  (#before #old : erased (lseq bf16 hidden_width))
  (#fraction : perm)
  preserves gpu ** input |-> Frac fraction before
  requires output |-> old
  ensures output |-> before
{
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i : sz{i <= 1024}) (values : lseq bf16 hidden_width).
      index |-> i ** output |-> values **
      pure (forall (j : nat{j < SZ.v i}). S.index values j == S.index before j)
    decreases (1024 - !index)
  {
    let i = !index;
    output.(i) <- input.(i);
    index := i +^ 1sz;
  };
  with values. assert output |-> values;
  assert pure (S.equal values before);
}

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

let lift_normalization
  (#p : Type) (premise : p -> prop)
  (wm : p -> GTot (real_vector weight_elements)) (hm : p -> GTot (real_vector hidden_width))
  (offset : nat{offset + hidden_width <= weight_elements})
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width) (out : lseq f32 hidden_width)
  : Lemma
      (requires
        (forall (v : p). premise v ==> wv %~ wm v /\ hv %~ hm v) /\
        (forall (model : matrix_interpretation hidden_width). normalization_inputs wv hv model ==>
          out %~ hidden_normalization (fst model) offset (snd model)))
      (ensures forall (v : p). premise v ==> out %~ hidden_normalization (wm v) offset (hm v))
  = let related (v : p{premise v})
      : Lemma (out %~ hidden_normalization (wm v) offset (hm v))
      = assert (normalization_inputs wv hv (wm v, hm v))
    in C.forall_intro related

let lift_controls
  (#p : Type) (premise : p -> prop)
  (wm : p -> GTot (real_vector weight_elements)) (am : p -> GTot (real_vector head_count))
  (wv : lseq bf16 weight_elements) (av controls : lseq f32 head_count)
  : Lemma
      (requires
        (forall (v : p). premise v ==> wv %~ wm v /\ av %~ am v) /\
        (forall (model : matrix_interpretation head_count). matrix_inputs wv av model ==>
          controls %~ control_arguments (fst model) (snd model)))
      (ensures forall (v : p). premise v ==> controls %~ control_arguments (wm v) (am v))
  = let related (v : p{premise v})
      : Lemma (controls %~ control_arguments (wm v) (am v))
      = assert (matrix_inputs #head_count wv av (wm v, am v))
    in C.forall_intro related

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

noeq noextract
type layer_interpretation (position : int) = {
  layer_weights : real_vector weight_elements;
  layer_hidden : real_vector hidden_width;
  layer_previous : real_vector (previous_elements position state_elements);
  layer_convolution : real_vector (previous_elements position convolution_elements)
}

let layer_inputs
  (#position : int)
  (weights : lseq bf16 weight_elements) (hidden : lseq bf16 hidden_width)
  (previous : lseq f32 (previous_elements position state_elements))
  (convolution : lseq f32 (previous_elements position convolution_elements))
  (model : layer_interpretation position)
  : prop = weights %~ model.layer_weights /\ hidden %~ model.layer_hidden /\
    previous %~ model.layer_previous /\ convolution %~ model.layer_convolution

let layer_normalized (#position : int) (model : layer_interpretation position)
  : GTot (real_vector hidden_width)
  = hidden_normalization model.layer_weights input_norm_offset model.layer_hidden

let layer_projected (#position : int) (model : layer_interpretation position)
  : GTot (real_vector channel_count)
  = linear model.layer_weights qkv_offset channel_count hidden_width (layer_normalized model)

let layer_gates (#position : int) (model : layer_interpretation position)
  : GTot (real_vector value_width)
  = linear model.layer_weights output_gate_offset value_width hidden_width (layer_normalized model)

let layer_beta (#position : int) (model : layer_interpretation position)
  : GTot (real_vector head_count)
  = linear model.layer_weights beta_offset head_count hidden_width (layer_normalized model)

let layer_alpha (#position : int) (model : layer_interpretation position)
  : GTot (real_vector head_count)
  = linear model.layer_weights alpha_offset head_count hidden_width (layer_normalized model)

let layer_controls (#position : int) (model : layer_interpretation position)
  : GTot (real_vector head_count)
  = control_arguments model.layer_weights (layer_alpha model)

let layer_next_convolution (#position : int) (model : layer_interpretation position)
  : GTot (real_vector convolution_elements)
  = next_convolution position model.layer_convolution (layer_projected model)

let layer_heads (#position : int) (model : layer_interpretation position)
  : GTot heads_interpretation
  = {
      heads_weights = model.layer_weights;
      heads_convolution = layer_next_convolution model;
      heads_previous = initial_state position model.layer_previous;
      heads_gates = layer_gates model;
      heads_beta = layer_beta model;
      heads_alpha = layer_controls model
    }

let layer_activation (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector value_width)
  = all_activations (layer_heads model) controls

let layer_attention (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector hidden_width)
  = linear model.layer_weights output_offset hidden_width value_width (layer_activation controls model)

let layer_residual (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector hidden_width)
  = residual_vector model.layer_hidden (layer_attention controls model)

let layer_post_normalized (#position : int)
  (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector hidden_width)
  = hidden_normalization model.layer_weights post_norm_offset (layer_residual controls model)

let layer_mlp_gate (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector mlp_width)
  = linear model.layer_weights mlp_gate_offset mlp_width hidden_width (layer_post_normalized controls model)

let layer_mlp_up (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector mlp_width)
  = linear model.layer_weights mlp_up_offset mlp_width hidden_width (layer_post_normalized controls model)

let layer_mlp_activation (#position : int)
  (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector mlp_width)
  = mlp_vector (layer_mlp_gate controls model) (layer_mlp_up controls model)

let layer_mlp_output (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector hidden_width)
  = linear model.layer_weights mlp_down_offset hidden_width mlp_width (layer_mlp_activation controls model)

let layer_result (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector hidden_width)
  = residual_vector (layer_residual controls model) (layer_mlp_output controls model)

let model_heads (weights : real_vector weight_elements) (hidden : real_vector hidden_width)
  (position : int)
  (previous : real_vector (previous_elements position state_elements))
  (convolution : real_vector (previous_elements position convolution_elements))
  (controls : lseq f32 head_count)
  : GTot (lseq (real_vector head_state_elements & real_vector head_width) head_count)
  = let normalized = hidden_normalization weights input_norm_offset hidden in
    let projected = linear weights qkv_offset channel_count hidden_width normalized in
    let gates = linear weights output_gate_offset value_width hidden_width normalized in
    let beta = linear weights beta_offset head_count hidden_width normalized in
    let alpha = linear weights alpha_offset head_count hidden_width normalized in
    let argument = control_arguments weights alpha in
    S.init_ghost head_count (fun head ->
      head_transition weights (next_convolution position convolution projected)
        (initial_state position previous) gates beta argument controls head)

let model_activation
  (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : GTot (real_vector value_width)
  = let heads = model_heads model.layer_weights model.layer_hidden position
      model.layer_previous model.layer_convolution controls in
    S.init_ghost value_width (fun i ->
      let (_, values) = S.index heads (i / head_width) in
      S.index values (i % head_width))

let model_tail (weights : real_vector weight_elements) (hidden : real_vector hidden_width)
  (activation : real_vector value_width) : GTot (real_vector hidden_width)
  = let attention_output = linear weights output_offset hidden_width value_width activation in
    let residual = S.init_ghost hidden_width (fun i ->
      S.index hidden i +. S.index attention_output i) in
    let post_normalized = hidden_normalization weights post_norm_offset residual in
    let mlp_gate = linear weights mlp_gate_offset mlp_width hidden_width post_normalized in
    let mlp_up = linear weights mlp_up_offset mlp_width hidden_width post_normalized in
    let mlp_activation = S.init_ghost mlp_width (fun i ->
      silu (S.index mlp_gate i) *. S.index mlp_up i) in
    let mlp_output = linear weights mlp_down_offset hidden_width mlp_width mlp_activation in
    S.init_ghost hidden_width (fun i ->
      S.index residual i +. S.index mlp_output i)

#push-options "--fuel 8 --z3rlimit 60"
let layer_heads_alignment
  (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : Lemma (
      model_activation controls model == layer_activation controls model)
  = assert (S.equal
      (model_activation controls model)
      (layer_activation controls model))
#pop-options

let layer_tail_alignment
  (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : Lemma (layer_result controls model ==
      model_tail model.layer_weights model.layer_hidden (layer_activation controls model))
  = assert (layer_result controls model ==
      model_tail model.layer_weights model.layer_hidden (layer_activation controls model))
    by (FStar.Tactics.V2.norm [delta_only [
      `%layer_result; `%layer_mlp_output; `%layer_mlp_activation; `%layer_mlp_gate;
      `%layer_mlp_up; `%layer_post_normalized; `%layer_residual; `%layer_attention;
      `%residual_vector; `%mlp_vector; `%model_tail]; zeta; iota];
      FStar.Tactics.V2.trefl ())

let recurrent_tail_alignment
  (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : Lemma (
      (recurrent_model model.layer_weights model.layer_hidden position
        model.layer_previous model.layer_convolution controls).hidden_result ==
      model_tail model.layer_weights model.layer_hidden (model_activation controls model))
  = assert (
      (recurrent_model model.layer_weights model.layer_hidden position
        model.layer_previous model.layer_convolution controls).hidden_result ==
      model_tail model.layer_weights model.layer_hidden (model_activation controls model))
    by (FStar.Tactics.V2.norm [delta_only [
      `%recurrent_model; `%model_tail; `%model_activation; `%model_heads;
      `%control_arguments]; zeta; iota];
      FStar.Tactics.V2.trefl ())

#push-options "--fuel 8 --z3rlimit 100"
let layer_model_alignment
  (#position : int) (controls : lseq f32 head_count) (model : layer_interpretation position)
  : Lemma (
      let result = recurrent_model model.layer_weights model.layer_hidden position
        model.layer_previous model.layer_convolution controls in
      result.alpha_argument == layer_controls model /\
      result.convolution_result == layer_next_convolution model /\
      result.state_result == all_states (layer_heads model) controls /\
      result.hidden_result == layer_result controls model)
  = let result = recurrent_model model.layer_weights model.layer_hidden position
      model.layer_previous model.layer_convolution controls in
    layer_heads_alignment controls model;
    layer_tail_alignment controls model;
    recurrent_tail_alignment controls model;
    FStar.Pervasives.norm_spec [delta_only [`%recurrent_model]; zeta; iota]
      (recurrent_model model.layer_weights model.layer_hidden position
        model.layer_previous model.layer_convolution controls);
    assert (S.equal result.alpha_argument (layer_controls model));
    assert (S.equal result.convolution_result (layer_next_convolution model));
    assert (S.equal result.state_result (all_states (layer_heads model) controls));
    assert (S.equal result.hidden_result (layer_result controls model))
#pop-options

let control_arguments_shared (rw:real_vector weight_count)
  (bias alpha:Shared.vector heads)
  : Lemma
      (requires real_weight_vector heads dt_bias_offset rw bias)
      (ensures control_arguments rw (pack_vector alpha) ==
        pack_vector #heads (fun h -> alpha h +. bias h)) =
  assert (S.equal (control_arguments rw (pack_vector alpha))
    (pack_vector #heads (fun h -> alpha h +. bias h)))

let residual_shared (x y:Shared.vector hidden_width)
  : Lemma (residual_vector (pack_vector x) (pack_vector y) ==
      pack_vector (Shared.add_vectors x y)) =
  assert (S.equal (residual_vector (pack_vector x) (pack_vector y))
    (pack_vector (Shared.add_vectors x y)))

let mlp_shared (gate up:Shared.vector mlp_width)
  : Lemma (mlp_vector (pack_vector gate) (pack_vector up) ==
      pack_vector (shared_mlp_product gate up)) =
  let point (i:Shared.fin mlp_width)
    : Lemma (S.index (mlp_vector (pack_vector gate) (pack_vector up)) i ==
        Shared.silu (gate i) *. up i) =
    activation_shared (gate i)
  in
  C.forall_intro point;
  assert (S.equal (mlp_vector (pack_vector gate) (pack_vector up))
    (pack_vector (shared_mlp_product gate up)))

let state_flat_coordinates (i:Shared.fin recurrent_count)
  : Lemma (
      (i % head_state_elements) % head_width == i % head_width /\
      (i % head_state_elements) / head_width == (i / head_width) % head_width) = ()

let heads_packing (model:heads_interpretation) (controls:lseq f32 head_count)
  (values:Shared.fin heads -> GTot shared_head_values)
  : Lemma
      (requires forall (h:Shared.fin heads).
        fst (head_result model controls h) == pack_head (values h).ideal_state /\
        snd (head_result model controls h) == pack_vector (values h).ideal_gated)
      (ensures
        all_states model controls ==
          pack_recurrent (shared_states values) /\
        all_activations model controls ==
          pack_vector (shared_activation values)) =
  let state_point (i:Shared.fin recurrent_count)
    : Lemma (S.index (all_states model controls) i ==
        (values (i / head_state_elements)).ideal_state (i % head_width) ((i / head_width) % head_width)) =
    state_flat_coordinates i;
    assert (fst (head_result model controls (i / head_state_elements)) ==
      pack_head (values (i / head_state_elements)).ideal_state)
  in
  C.forall_intro state_point;
  assert (S.equal (all_states model controls)
    (pack_recurrent (shared_states values)));
  assert (S.equal (all_activations model controls)
    (pack_vector (shared_activation values)))

#push-options "--fuel 3 --z3rlimit 100"
let heads_shared (model:heads_interpretation) (w:Shared.weights)
  (conv:Shared.convolution_state) (old:Shared.recurrent_state)
  (z:Shared.vector mixer_width) (b a:Shared.vector heads) (controls:lseq f32 head_count)
  : Lemma
      (requires
        real_weight_views model.heads_weights w /\
        model.heads_convolution == pack_convolution conv /\
        model.heads_previous == pack_recurrent old /\
        model.heads_gates == pack_vector z /\
        model.heads_beta == pack_vector b /\
        model.heads_alpha == pack_vector #heads (fun h -> a h +. w.dt_bias h))
      (ensures (
        let values:Shared.fin heads -> GTot shared_head_values =
          shared_head w (Shared.convolve w.conv_kernel conv) z b a old in
        all_states model controls ==
          pack_recurrent (shared_states values) /\
        all_activations model controls ==
          pack_vector (shared_activation values))) =
  let values:Shared.fin heads -> GTot shared_head_values =
    shared_head w (Shared.convolve w.conv_kernel conv) z b a old in
  let point (h:Shared.fin heads)
    : Lemma (
        fst (head_result model controls h) == pack_head (values h).ideal_state /\
        snd (head_result model controls h) == pack_vector (values h).ideal_gated) =
    head_transition_shared model.heads_weights w conv old z b a controls h
  in
  C.forall_intro point;
  heads_packing model controls values
#pop-options

#push-options "--fuel 2 --z3rlimit 100"
let internal_shared_composition
  (call:Shared.invocation) (w:Shared.weights) (hidden:Shared.vector hidden_width)
  (before:Shared.model_cache{Shared.cache_domain call before}) (controls:lseq f32 head_count)
  : Lemma (
      let actual = recurrent_model (pack_weights w) (pack_vector hidden) call.cache_position
        (pack_previous call.cache_position (pack_recurrent (old_recurrent call before)))
        (pack_previous call.cache_position (pack_convolution (old_convolution call before))) controls in
      let expected = assembled_shared_values call w hidden before in
      actual.hidden_result == pack_vector expected.hidden_value /\
      actual.state_result == pack_recurrent expected.recurrent_value /\
      actual.convolution_result == pack_convolution expected.convolution_value) =
  let rw = pack_weights w in
  let old_r = old_recurrent call before in
  let old_c = old_convolution call before in
  let model:layer_interpretation call.cache_position = {
    layer_weights = rw;
    layer_hidden = pack_vector hidden;
    layer_previous = pack_previous call.cache_position (pack_recurrent old_r);
    layer_convolution = pack_previous call.cache_position (pack_convolution old_c);
  } in
  let normalized = Shared.centered_rms_norm w.input_norm hidden in
  let raw = Shared.project w.in_qkv normalized in
  let z = Shared.project w.in_z normalized in
  let b = Shared.project w.in_beta normalized in
  let a = Shared.project w.in_a normalized in
  let conv = Shared.shift_convolution old_c raw in
  let qkv = Shared.convolve w.conv_kernel conv in
  let values:Shared.fin heads -> GTot shared_head_values = shared_head w qkv z b a old_r in
  let activation = shared_activation values in
  let attention = Shared.project w.out_projection activation in
  let residual = Shared.add_vectors hidden attention in
  let postnorm = Shared.centered_rms_norm w.post_attention_norm residual in
  let gate = Shared.project w.mlp_gate postnorm in
  let up = Shared.project w.mlp_up postnorm in
  let product = shared_mlp_product gate up in
  let down = Shared.project w.mlp_down product in
  packed_weight_views w;
  normalization_shared rw input_norm_offset w.input_norm hidden;
  assert (layer_normalized model == pack_vector normalized);
  linear_shared rw qkv_offset qkv_width hidden_width w.in_qkv normalized;
  linear_shared rw z_offset mixer_width hidden_width w.in_z normalized;
  linear_shared rw beta_offset heads hidden_width w.in_beta normalized;
  linear_shared rw a_offset heads hidden_width w.in_a normalized;
  assert (layer_projected model == pack_vector raw);
  assert (layer_gates model == pack_vector z);
  assert (layer_beta model == pack_vector b);
  assert (layer_alpha model == pack_vector a);
  control_arguments_shared rw w.dt_bias a;
  assert (layer_controls model == pack_vector #heads (fun h -> a h +. w.dt_bias h));
  if call.cache_position = 0 then first_use_is_zero call before;
  initial_state_shared call.cache_position old_r;
  next_convolution_shared call.cache_position old_c raw;
  assert (layer_next_convolution model == pack_convolution conv);
  norm_spec [delta_only [`%layer_heads]; zeta; iota] (layer_heads model);
  heads_shared (layer_heads model) w conv old_r z b a controls;
  assert (layer_activation controls model == pack_vector activation);
  linear_shared rw out_projection_offset hidden_width mixer_width w.out_projection activation;
  assert (layer_attention controls model == pack_vector attention);
  residual_shared hidden attention;
  assert (layer_residual controls model == pack_vector residual);
  normalization_shared rw post_norm_offset w.post_attention_norm residual;
  assert (layer_post_normalized controls model == pack_vector postnorm);
  linear_shared rw mlp_gate_offset mlp_width hidden_width w.mlp_gate postnorm;
  linear_shared rw mlp_up_offset mlp_width hidden_width w.mlp_up postnorm;
  assert (layer_mlp_gate controls model == pack_vector gate);
  assert (layer_mlp_up controls model == pack_vector up);
  mlp_shared gate up;
  assert (layer_mlp_activation controls model == pack_vector product);
  linear_shared rw mlp_down_offset hidden_width mlp_width w.mlp_down product;
  assert (layer_mlp_output controls model == pack_vector down);
  residual_shared residual down;
  assert (layer_result controls model == pack_vector (Shared.add_vectors residual down));
  layer_model_alignment controls model;
  norm_spec [delta_only [`%assembled_shared_values]; zeta; iota]
    (assembled_shared_values call w hidden before);
  ()
#pop-options

#push-options "--fuel 2 --z3rlimit 100"
let establish_shared_post
  (position:int)
  (wv:lseq bf16 weight_count) (hv:lseq bf16 hidden_width)
  (sv:lseq f32 (previous_elements position recurrent_count))
  (cv:lseq f32 (previous_elements position convolution_count))
  (out:lseq bf16 hidden_width) (states:lseq f32 recurrent_count)
  (convolution:lseq f32 convolution_count) (controls:lseq f32 head_count)
  : Lemma
      (requires
        forall (rw:real_vector weight_count) (rh:real_vector hidden_width)
          (rs:real_vector (previous_elements position recurrent_count))
          (rc:real_vector (previous_elements position convolution_count)).
          wv %~ rw /\ hv %~ rh /\ sv %~ rs /\ cv %~ rc ==>
          (let model = recurrent_model rw rh position rs rc controls in
           out %~ model.hidden_result /\ states %~ model.state_result /\
           convolution %~ model.convolution_result))
      (ensures semantic_postcondition wv hv position sv cv out states convolution) =
  let point (call:Shared.invocation) (w:Shared.weights) (hidden:Shared.vector hidden_width)
    (before:Shared.model_cache{Shared.cache_domain call before /\
      input_representation call w hidden before wv hv position sv cv})
    : Lemma (call_postcondition call w hidden before out states convolution) =
    packed_inputs call w hidden before wv hv position sv cv;
    let rw = pack_weights w in
    let rh = pack_vector hidden in
    let rs = pack_previous position (pack_recurrent (old_recurrent call before)) in
    let rc = pack_previous position (pack_convolution (old_convolution call before)) in
    let actual = recurrent_model rw rh position rs rc controls in
    assert (out %~ actual.hidden_result /\ states %~ actual.state_result /\
      convolution %~ actual.convolution_result);
    internal_shared_composition call w hidden before controls;
    let assembled = assembled_shared_values call w hidden before in
    assert (out %~ pack_vector assembled.hidden_value /\
      states %~ pack_recurrent assembled.recurrent_value /\
      convolution %~ pack_convolution assembled.convolution_value);
    assembled_shared_matches call w hidden before;
    let expected = Shared.decoder_values call w hidden before in
    assert (S.equal (pack_vector assembled.hidden_value) (pack_vector expected.hidden_value));
    assert (S.equal (pack_recurrent assembled.recurrent_value) (pack_recurrent expected.recurrent_value));
    assert (S.equal (pack_convolution assembled.convolution_value) (pack_convolution expected.convolution_value));
    packed_outputs out states convolution expected;
    decoder_values_establish_call_postcondition call w hidden before out states convolution
  in
  C.forall_intro_4 point
#pop-options

let layer_state_initialization
  (#position : int)
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width)
  (sv : lseq f32 (previous_elements position state_elements))
  (cv : lseq f32 (previous_elements position convolution_elements))
  : Lemma (forall (model : layer_interpretation position).
      layer_inputs wv hv sv cv model ==>
      physical_initial #state_elements position sv %~ initial_state position model.layer_previous)
  = let related (model : layer_interpretation position{layer_inputs wv hv sv cv model})
      : Lemma (
          physical_initial #state_elements position sv %~ initial_state position model.layer_previous)
      = initial_state_relation position sv model.layer_previous
    in C.forall_intro related

let layer_convolution_lift
  (#position : int)
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width)
  (sv : lseq f32 (previous_elements position state_elements))
  (cv : lseq f32 (previous_elements position convolution_elements))
  (projected : lseq f32 channel_count)
  : Lemma
      (requires forall (model : layer_interpretation position).
        layer_inputs wv hv sv cv model ==> projected %~ layer_projected model)
      (ensures forall (model : layer_interpretation position).
        layer_inputs wv hv sv cv model ==>
        physical_convolution position cv projected %~ layer_next_convolution model)
  = physical_convolution_relation position cv projected;
    let related (model : layer_interpretation position{layer_inputs wv hv sv cv model})
      : Lemma (physical_convolution position cv projected %~ layer_next_convolution model)
      = ()
    in C.forall_intro related

let layer_heads_lift
  (#position : int)
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width)
  (sv : lseq f32 (previous_elements position state_elements))
  (cv : lseq f32 (previous_elements position convolution_elements))
  (convolution : lseq f32 convolution_elements) (previous : lseq f32 state_elements)
  (gates : lseq f32 value_width) (beta controls : lseq f32 head_count)
  (states : lseq f32 state_elements) (activation : lseq f32 value_width)
  : Lemma
      (requires
        (forall (model : layer_interpretation position). layer_inputs wv hv sv cv model ==>
          heads_inputs wv convolution previous gates beta controls (layer_heads model)) /\
        (forall (model : heads_interpretation). heads_inputs wv convolution previous gates beta controls model ==>
          states %~ all_states model controls /\ activation %~ all_activations model controls))
      (ensures forall (model : layer_interpretation position). layer_inputs wv hv sv cv model ==>
        states %~ all_states (layer_heads model) controls /\ activation %~ layer_activation controls model)
  = map_relations #(layer_interpretation position) #heads_interpretation
      (layer_inputs #position wv hv sv cv)
      (heads_inputs wv convolution previous gates beta controls)
      (fun model -> states %~ all_states model controls /\ activation %~ all_activations model controls)
      (layer_heads #position);
    ()

let layer_numerical_post
  (position : int)
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width)
  (sv : lseq f32 (previous_elements position state_elements))
  (cv : lseq f32 (previous_elements position convolution_elements))
  (out : lseq bf16 hidden_width) (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements) (controls : lseq f32 head_count)
  : prop =
    forall (rw : real_vector weight_elements) (rh : real_vector hidden_width)
           (rs : real_vector (previous_elements position state_elements))
           (rc : real_vector (previous_elements position convolution_elements)).
      wv %~ rw /\ hv %~ rh /\ sv %~ rs /\ cv %~ rc ==>
      (let model = recurrent_model rw rh position rs rc controls in
       controls %~ model.alpha_argument /\ out %~ model.hidden_result /\
       states %~ model.state_result /\ convolution %~ model.convolution_result)

let layer_values_approx
  (#position : int)
  (out : lseq bf16 hidden_width) (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements) (controls : lseq f32 head_count)
  (model : layer_interpretation position)
  : prop =
    controls %~ layer_controls model /\ out %~ layer_result controls model /\
    states %~ all_states (layer_heads model) controls /\ convolution %~ layer_next_convolution model

let layer_values_post
  (#position : int)
  (out : lseq bf16 hidden_width) (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements) (controls : lseq f32 head_count)
  (model : layer_interpretation position)
  : prop =
    let result = recurrent_model model.layer_weights model.layer_hidden position
      model.layer_previous model.layer_convolution controls in
    controls %~ result.alpha_argument /\ out %~ result.hidden_result /\
    states %~ result.state_result /\ convolution %~ result.convolution_result

let layer_post_alignment
  (#position : int)
  (out : lseq bf16 hidden_width) (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements) (controls : lseq f32 head_count)
  (model : layer_interpretation position)
  : Lemma (
      layer_values_approx out states convolution controls model ==>
      layer_values_post out states convolution controls model)
  = layer_model_alignment #position controls model;
    ()

let forall_layer_values
  (position : int) (pred : layer_interpretation position -> prop)
  : Lemma
      (requires forall (model : layer_interpretation position). pred model)
      (ensures
        forall (rw : real_vector weight_elements) (rh : real_vector hidden_width)
               (rs : real_vector (previous_elements position state_elements))
               (rc : real_vector (previous_elements position convolution_elements)).
          pred ({ layer_weights = rw; layer_hidden = rh;
            layer_previous = rs; layer_convolution = rc } <: layer_interpretation position))
  = let related
      (rw : real_vector weight_elements) (rh : real_vector hidden_width)
      (rs : real_vector (previous_elements position state_elements))
      (rc : real_vector (previous_elements position convolution_elements))
      : Lemma (pred ({ layer_weights = rw; layer_hidden = rh;
            layer_previous = rs; layer_convolution = rc } <: layer_interpretation position))
      = ()
    in C.forall_intro_4 related

let layer_record_post
  (position : int)
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width)
  (sv : lseq f32 (previous_elements position state_elements))
  (cv : lseq f32 (previous_elements position convolution_elements))
  (out : lseq bf16 hidden_width) (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements) (controls : lseq f32 head_count)
  : prop =
    forall (rw : real_vector weight_elements) (rh : real_vector hidden_width)
           (rs : real_vector (previous_elements position state_elements))
           (rc : real_vector (previous_elements position convolution_elements)).
      let model : layer_interpretation position = {
        layer_weights = rw; layer_hidden = rh; layer_previous = rs; layer_convolution = rc
      } in
      layer_inputs wv hv sv cv model ==> layer_values_post out states convolution controls model

let layer_record_post_equiv
  (position : int)
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width)
  (sv : lseq f32 (previous_elements position state_elements))
  (cv : lseq f32 (previous_elements position convolution_elements))
  (out : lseq bf16 hidden_width) (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements) (controls : lseq f32 head_count)
  : Lemma (
      layer_record_post position wv hv sv cv out states convolution controls ==
      layer_numerical_post position wv hv sv cv out states convolution controls)
  = assert (
      layer_record_post position wv hv sv cv out states convolution controls ==
      layer_numerical_post position wv hv sv cv out states convolution controls)
    by (FStar.Tactics.V2.norm [delta_only [
      `%layer_record_post; `%layer_inputs; `%layer_values_post; `%layer_numerical_post]; zeta; iota];
      FStar.Tactics.V2.trefl ())

let layer_finish
  (position : int)
  (wv : lseq bf16 weight_elements) (hv : lseq bf16 hidden_width)
  (sv : lseq f32 (previous_elements position state_elements))
  (cv : lseq f32 (previous_elements position convolution_elements))
  (out : lseq bf16 hidden_width) (states : lseq f32 state_elements)
  (convolution : lseq f32 convolution_elements) (controls : lseq f32 head_count)
  : Lemma
      (requires retained_taps_post position cv convolution /\
        (forall (model : layer_interpretation position). layer_inputs wv hv sv cv model ==>
          controls %~ layer_controls model /\ out %~ layer_result controls model /\
          states %~ all_states (layer_heads model) controls /\ convolution %~ layer_next_convolution model))
      (ensures recurrent_decoder_post wv hv position sv cv out states convolution /\
        semantic_postcondition wv hv position sv cv out states convolution)
  = C.forall_intro (layer_post_alignment #position out states convolution controls);
    compose_relations #(layer_interpretation position)
      (layer_inputs #position wv hv sv cv)
      (layer_values_approx #position out states convolution controls)
      (layer_values_post #position out states convolution controls);
    forall_layer_values position (fun model ->
      layer_inputs #position wv hv sv cv model ==>
      layer_values_post #position out states convolution controls model);
    assert (layer_record_post position wv hv sv cv out states convolution controls);
    layer_record_post_equiv position wv hv sv cv out states convolution controls;
    establish_shared_post position wv hv sv cv out states convolution controls;
    C.exists_intro #(lseq f32 head_count)
      (layer_numerical_post position wv hv sv cv out states convolution) controls;
    ()

inline_for_extraction noextract
fn family_kernel
  (input0 : larray bf16 weight_elements{is_global_array input0})
  (input1 : larray bf16 hidden_width{is_global_array input1})
  (input2 : array f32)
  (input3 : array f32)
  (input4 : i64{valid_position (I64.v input4)})
  (output0 : larray bf16 hidden_width{is_global_array output0})
  (output1 : larray f32 state_elements{is_global_array output1})
  (output2 : larray f32 convolution_elements{is_global_array output2})
  (#weights : erased (lseq bf16 weight_elements))
  (#hidden : erased (lseq bf16 hidden_width))
  (#previous_states : erased (lseq f32 (previous_elements (I64.v input4) state_elements)))
  (#previous_convolution : erased (lseq f32 (previous_elements (I64.v input4) convolution_elements)))
  (#old_hidden : erased (lseq bf16 hidden_width))
  (#old_states : erased (lseq f32 state_elements))
  (#old_convolution : erased (lseq f32 convolution_elements))
  (#fweights #fhidden #fstates #fconvolution : perm)
  preserves gpu ** input0 |-> Frac fweights weights ** input1 |-> Frac fhidden hidden **
    state_ownership input2 (I64.v input4) state_elements fstates previous_states **
    state_ownership input3 (I64.v input4) convolution_elements fconvolution previous_convolution
  requires output0 |-> old_hidden ** output1 |-> old_states ** output2 |-> old_convolution
  ensures exists* (result : lseq bf16 hidden_width) (states : lseq f32 state_elements)
    (convolution : lseq f32 convolution_elements).
    output0 |-> result ** output1 |-> states ** output2 |-> convolution **
    pure (recurrent_decoder_post weights hidden (I64.v input4)
      previous_states previous_convolution result states convolution /\
      semantic_postcondition weights hidden (I64.v input4)
        previous_states previous_convolution result states convolution)
{
  let mut normalized = [| zero #f32 #_; 1024sz |];
  let mut projected = [| zero #f32 #_; 6144sz |];
  let mut gates = [| zero #f32 #_; 2048sz |];
  let mut betas = [| zero #f32 #_; 16sz |];
  let mut alphas = [| zero #f32 #_; 16sz |];
  let mut controls = [| zero #f32 #_; 16sz |];
  let mut activation = [| zero #f32 #_; 2048sz |];
  let mut projection = [| zero #f32 #_; 1024sz |];
  let mut mlp_gate = [| zero #f32 #_; 3584sz |];
  let mut mlp_up = [| zero #f32 #_; 3584sz |];
  normalize_hidden input0 input1 normalized 0sz;
  with n0. assert normalized |-> n0;
  lift_normalization #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) (fun model -> model.layer_hidden)
    input_norm_offset weights hidden n0;
  matrix_project #hidden_width #channel_count input0 normalized projected
    1024sz 6144sz (SZ.uint32_to_sizet 11012096ul) true;
  with pv. assert projected |-> pv;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #channel_count
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) layer_normalized qkv_offset weights n0 pv;
  matrix_project #hidden_width #value_width input0 normalized gates
    1024sz 2048sz (SZ.uint32_to_sizet 17303552ul) true;
  with gv. assert gates |-> gv;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #value_width
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) layer_normalized output_gate_offset weights n0 gv;
  matrix_project #hidden_width #head_count input0 normalized betas
    1024sz 16sz (SZ.uint32_to_sizet 19400704ul) true;
  with bv. assert betas |-> bv;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #head_count
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) layer_normalized beta_offset weights n0 bv;
  matrix_project #hidden_width #head_count input0 normalized alphas
    1024sz 16sz (SZ.uint32_to_sizet 19417088ul) true;
  with av. assert alphas |-> av;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #head_count
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) layer_normalized alpha_offset weights n0 av;
  bias_controls input0 alphas controls;
  with cv. assert controls |-> cv;
  lift_controls #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) layer_alpha weights av cv;
  initialize_state input2 output1 input4;
  shift_convolution input3 projected output2 input4;
  layer_state_initialization #(I64.v input4) weights hidden previous_states previous_convolution;
  layer_convolution_lift #(I64.v input4) weights hidden previous_states previous_convolution pv;
  with initial_states. assert output1 |-> initial_states;
  with convolution. assert output2 |-> convolution;
  assert pure (forall (model : layer_interpretation (I64.v input4)).
    layer_inputs weights hidden previous_states previous_convolution model ==>
    heads_inputs weights convolution initial_states gv bv cv (layer_heads model));
  recurrent_heads input0 output2 output1 gates betas controls activation;
  with states. assert output1 |-> states;
  with activation_values. assert activation |-> activation_values;
  layer_heads_lift #(I64.v input4) weights hidden previous_states previous_convolution
    convolution initial_states gv bv cv states activation_values;
  matrix_project #value_width #hidden_width input0 activation projection
    2048sz 1024sz (SZ.uint32_to_sizet 19458208ul) false;
  with attention. assert projection |-> attention;
  lift_matrix #(layer_interpretation (I64.v input4)) #value_width #hidden_width
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) (layer_activation cv) output_offset weights activation_values attention;
  copy_hidden input1 output0;
  add_residual output0 projection;
  with residual. assert output0 |-> residual;
  lift_residual #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_hidden) (layer_attention cv) hidden attention residual;
  normalize_hidden input0 output0 normalized 1024sz;
  with n1. assert normalized |-> n1;
  lift_normalization #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) (layer_residual cv) post_norm_offset weights residual n1;
  matrix_project #hidden_width #mlp_width input0 normalized mlp_gate 1024sz 3584sz 2048sz true;
  with mg. assert mlp_gate |-> mg;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #mlp_width
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) (layer_post_normalized cv) mlp_gate_offset weights n1 mg;
  matrix_project #hidden_width #mlp_width input0 normalized mlp_up
    1024sz 3584sz (SZ.uint32_to_sizet 3672064ul) true;
  with mu. assert mlp_up |-> mu;
  lift_matrix #(layer_interpretation (I64.v input4)) #hidden_width #mlp_width
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) (layer_post_normalized cv) mlp_up_offset weights n1 mu;
  activate_mlp mlp_gate mlp_up;
  with ma. assert mlp_gate |-> ma;
  lift_mlp #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (layer_mlp_gate cv) (layer_mlp_up cv) mg mu ma;
  matrix_project #mlp_width #hidden_width input0 mlp_gate projection
    3584sz 1024sz (SZ.uint32_to_sizet 7342080ul) false;
  with mp. assert projection |-> mp;
  lift_matrix #(layer_interpretation (I64.v input4)) #mlp_width #hidden_width
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (fun model -> model.layer_weights) (layer_mlp_activation cv) mlp_down_offset weights ma mp;
  add_residual output0 projection;
  with result. assert output0 |-> result;
  lift_residual #(layer_interpretation (I64.v input4))
    (layer_inputs #(I64.v input4) weights hidden previous_states previous_convolution)
    (layer_residual cv) (layer_mlp_output cv) residual mp result;
  layer_finish (I64.v input4) weights hidden previous_states previous_convolution
    result states convolution cv;
  assert pure (recurrent_decoder_post weights hidden (I64.v input4)
    previous_states previous_convolution result states convolution /\
    semantic_postcondition weights hidden (I64.v input4)
      previous_states previous_convolution result states convolution);
}

instance send_global_array_contents
  (#a : Type0)
  (buffer : array a{is_global_array buffer})
  (#fraction : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to buffer #fraction contents)
  = Kuiper.Array.Core.is_send_pts_to buffer #fraction contents

ghost
fn move_state_ownership
  (buffer : array f32) (position : int) (count : nat) (fraction : perm)
  (contents : lseq f32 (previous_elements position count))
  (source : loc_id)
  (destination : loc_id{gpu_of source == gpu_of destination})
  requires on source (state_ownership buffer position count fraction contents)
  ensures on destination (state_ownership buffer position count fraction contents)
{
  if (position = 0) {
    map_loc source #(state_ownership buffer position count fraction contents) #emp
      fn () { rewrite state_ownership buffer position count fraction contents as emp };
    map_loc destination #emp #(state_ownership buffer position count fraction contents)
      fn () { rewrite emp as state_ownership buffer position count fraction contents };
  } else {
    map_loc source
      #(state_ownership buffer position count fraction contents)
      #(pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      fn () {
        rewrite state_ownership buffer position count fraction contents as
          (pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      };
    is_send_across_elim gpu_of (buffer |-> Frac fraction contents) destination;
    map_loc destination
      #(pure (is_global_array buffer) ** buffer |-> Frac fraction contents)
      #(state_ownership buffer position count fraction contents)
      fn () {
        rewrite (pure (is_global_array buffer) ** buffer |-> Frac fraction contents) as
          state_ownership buffer position count fraction contents
      };
  }
}

instance send_state_ownership
  (buffer : array f32) (position : int) (count : nat) (fraction : perm)
  (contents : lseq f32 (previous_elements position count))
  : is_send_across gpu_of (state_ownership buffer position count fraction contents)
  = fun source destination ->
      move_state_ownership buffer position count fraction contents source destination

fn kplay_run
  (input0 : larray bf16 weight_elements{is_global_array input0})
  (input1 : larray bf16 hidden_width{is_global_array input1})
  (input2 : array f32)
  (input3 : array f32)
  (input4 : i64{valid_position (I64.v input4)})
  (output0 : larray bf16 hidden_width{is_global_array output0})
  (output1 : larray f32 state_elements{is_global_array output1})
  (output2 : larray f32 convolution_elements{is_global_array output2})
  (#weights : erased (lseq bf16 weight_elements))
  (#hidden : erased (lseq bf16 hidden_width))
  (#previous_states : erased (lseq f32 (previous_elements (I64.v input4) state_elements)))
  (#previous_convolution : erased (lseq f32 (previous_elements (I64.v input4) convolution_elements)))
  (#fweights #fhidden #fstates #fconvolution : perm)
  preserves cpu
  preserves on gpu_loc (
    input0 |-> Frac fweights weights ** input1 |-> Frac fhidden hidden **
    state_ownership input2 (I64.v input4) state_elements fstates previous_states **
    state_ownership input3 (I64.v input4) convolution_elements fconvolution previous_convolution)
  requires exists* (old_hidden:lseq bf16 hidden_width)
    (old_states:lseq f32 state_elements) (old_convolution:lseq f32 convolution_elements).
    on gpu_loc (output0 |-> old_hidden ** output1 |-> old_states ** output2 |-> old_convolution)
  returns unit
  ensures exists* (result : lseq bf16 hidden_width) (states : lseq f32 state_elements)
    (convolution : lseq f32 convolution_elements).
    on gpu_loc (output0 |-> result ** output1 |-> states ** output2 |-> convolution) **
    pure (semantic_postcondition weights hidden (I64.v input4)
      previous_states previous_convolution result states convolution)
{
  with old_hidden old_states old_convolution.
    assert on gpu_loc (output0 |-> old_hidden ** output1 |-> old_states ** output2 |-> old_convolution);
  launch_kernel_1 (fun () ->
    family_kernel input0 input1 input2 input3 input4 output0 output1 output2
      #weights #hidden #previous_states #previous_convolution #old_hidden #old_states #old_convolution
      #fweights #fhidden #fstates #fconvolution);
}
