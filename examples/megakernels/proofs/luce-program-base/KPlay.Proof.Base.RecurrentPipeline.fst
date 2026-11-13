module KPlay.Proof.Base.RecurrentPipeline

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper.Real
module S = KPlay.Proof.Shared.Recurrent
module Sum = Kuiper.Sum
module R = FStar.Math.Sqrt
module E = FStar.Math.Exp

inline_for_extraction noextract
let rec native_sum_ext (n:nat) (f g:S.vector n)
  : Lemma
      (requires forall (i:S.fin n). f i == g i)
      (ensures Sum.sum 0 n f == Sum.sum 0 n g)
      (decreases n) =
  if n > 0 then (
    native_sum_ext (n - 1) f g;
    Sum.sum_pop_right 0 n f;
    Sum.sum_pop_right 0 n g;
    assert (Sum.sum 0 n f == Sum.sum 0 (n - 1) f +. f (n - 1));
    assert (Sum.sum 0 n g == Sum.sum 0 (n - 1) g +. g (n - 1))
  )

[@@"opaque_to_smt"]
noextract
let sum_n (n:nat) (f:S.vector n) : GTot real = Sum.sum 0 n (fun i -> f i)

inline_for_extraction noextract
let sum_unfold (n:nat) (f:S.vector n)
  : Lemma (sum_n n f == Sum.sum 0 n f) =
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.sum_n"]] (sum_n n f);
  native_sum_ext n (fun (i:Kuiper.Common.between 0 n) -> f i) f

inline_for_extraction noextract
let sum_zero (f:S.vector 0)
  : Lemma (ensures sum_n 0 f == 0.0R) [SMTPat (sum_n 0 f)] =
  sum_unfold 0 f

inline_for_extraction noextract
let sum_pop (n:pos) (f:S.vector n)
  : Lemma (sum_n n f == sum_n (n - 1) f +. f (n - 1)) =
  sum_unfold n f;
  sum_unfold (n - 1) f;
  Sum.sum_pop_right 0 n f

inline_for_extraction noextract
let rec sum_ext (n:nat) (f g:S.vector n)
  : Lemma
      (requires forall (i:S.fin n). f i == g i)
      (ensures sum_n n f == sum_n n g)
      (decreases n) =
  if n = 0 then (sum_zero f; sum_zero g)
  else (
    sum_ext (n - 1) f g;
    sum_pop n f;
    sum_pop n g
  )

inline_for_extraction noextract
let sum_step (n:pos) (f:S.vector n)
  : Lemma (sum_n n f == sum_n (n - 1) (fun i -> f i) +. f (n - 1)) =
  sum_pop n f;
  sum_ext (n - 1) f (fun i -> f i)

inline_for_extraction noextract
let sum_shared_ext (n:nat) (f:S.vector n)
  (g:Kuiper.Common.between 0 n -> GTot real)
  : Lemma
      (requires forall (i:S.fin n). f i == g i)
      (ensures sum_n n f == Sum.sum 0 n g) =
  sum_ext n f g;
  sum_unfold n g

noextract
noeq type head_model = {
  qinput:S.vector 128;
  kinput:S.vector 128;
  value:S.vector 128;
  z:S.vector 128;
  weight:S.vector 128;
  old:S.matrix 128 128;
  b:real;
  a:real;
  a_log:real;
  dt_bias:real;
  query:S.vector 128;
  key:S.vector 128;
  beta:real;
  decay:real;
  decayed:S.matrix 128 128;
  delta:S.vector 128;
  updated:S.matrix 128 128;
  core:S.vector 128;
  gated:S.vector 128;
}

[@@"opaque_to_smt"]
noextract
let valid_head (p:head_model) : GTot prop =
  (forall (i:S.fin 128). p.query i == S.l2_normalize p.qinput i *. R.rsqrt 128.0R) /\
  p.key == S.l2_normalize p.kinput /\
  p.beta == S.sigmoid p.b /\
  p.decay == E.exp ((0.0R -. E.exp p.a_log) *. S.softplus (p.a +. p.dt_bias)) /\
  (forall (i j:S.fin 128). p.decayed i j == p.old i j *. p.decay) /\
  (forall (j:S.fin 128). p.delta j ==
    (p.value j -. sum_n 128 (fun i -> p.decayed i j *. p.key i)) *. p.beta) /\
  (forall (i j:S.fin 128). p.updated i j == p.decayed i j +. p.key i *. p.delta j) /\
  (forall (j:S.fin 128). p.core j == sum_n 128 (fun i -> p.updated i j *. p.query i)) /\
  (forall (i:S.fin 128). p.gated i ==
    (p.weight i *. (p.core i *. S.rms_scale p.core)) *. S.silu (p.z i))

[@@"opaque_to_smt"]
noextract
let make_head (w:S.weights) (qkv:S.vector 6144) (z:S.vector 2048)
  (b a:S.vector 16) (old:S.recurrent_state) (h:S.fin 16)
  : GTot (p:head_model{
      valid_head p /\
      p.qinput == (fun (i:S.fin 128) -> qkv (h * 128 + i)) /\
      p.kinput == (fun (i:S.fin 128) -> qkv (2048 + h * 128 + i)) /\
      p.value == (fun (i:S.fin 128) -> qkv (4096 + h * 128 + i)) /\
      p.z == (fun (i:S.fin 128) -> z (h * 128 + i)) /\
      p.weight == w.gated_norm /\ p.old == old h /\
      p.b == b h /\ p.a == a h /\ p.a_log == w.a_log h /\ p.dt_bias == w.dt_bias h}) =
  let qinput:S.vector 128 = fun i -> qkv (h * 128 + i) in
  let kinput:S.vector 128 = fun i -> qkv (2048 + h * 128 + i) in
  let value:S.vector 128 = fun i -> qkv (4096 + h * 128 + i) in
  let query:S.vector 128 = fun i -> S.l2_normalize qinput i *. R.rsqrt 128.0R in
  let key = S.l2_normalize kinput in
  let beta = S.sigmoid (b h) in
  let decay = E.exp ((0.0R -. E.exp (w.a_log h)) *. S.softplus (a h +. w.dt_bias h)) in
  let decayed:S.matrix 128 128 = fun i j -> old h i j *. decay in
  let delta (j:S.fin 128)
    : GTot (r:real{r == (value j -. sum_n 128 (fun i -> decayed i j *. key i)) *. beta}) =
    (value j -. sum_n 128 (fun i -> decayed i j *. key i)) *. beta in
  let updated:S.matrix 128 128 = fun i j -> decayed i j +. key i *. delta j in
  let core (j:S.fin 128)
    : GTot (r:real{r == sum_n 128 (fun i -> updated i j *. query i)}) =
    sum_n 128 (fun i -> updated i j *. query i) in
  let gated:S.vector 128 = fun i ->
    (w.gated_norm i *. (core i *. S.rms_scale core)) *. S.silu (z (h * 128 + i)) in
  let p = {qinput; kinput; value; z=(fun (i:S.fin 128) -> z (h * 128 + i)); weight=w.gated_norm;
   old=old h; b=b h; a=a h; a_log=w.a_log h; dt_bias=w.dt_bias h;
   query; key; beta; decay; decayed; delta; updated; core; gated} in
  let delta_point (j:S.fin 128)
    : Lemma (p.delta j ==
      (p.value j -. sum_n 128 (fun i -> p.decayed i j *. p.key i)) *. p.beta) =
    sum_ext 128 (fun i -> p.decayed i j *. p.key i) (fun i -> decayed i j *. key i)
  in
  FStar.Classical.forall_intro delta_point;
  let core_point (j:S.fin 128)
    : Lemma (p.core j == sum_n 128 (fun i -> p.updated i j *. p.query i)) =
    sum_ext 128 (fun i -> p.updated i j *. p.query i) (fun i -> updated i j *. query i)
  in
  FStar.Classical.forall_intro core_point;
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.valid_head"]] (valid_head p);
  p

inline_for_extraction noextract
let old_convolution
  (call:S.invocation) (before:S.model_cache{S.cache_domain call before})
  : GTot S.convolution_state
  =
  if S.has_previous_state before
  then S.present_value (before.conv_slots call.layer_index)
  else S.zero_convolution ()

inline_for_extraction noextract
let old_recurrent
  (call:S.invocation) (before:S.model_cache{S.cache_domain call before})
  : GTot S.recurrent_state
  =
  if S.has_previous_state before
  then S.present_value (before.recurrent_slots call.layer_index)
  else S.zero_recurrent ()

noextract
noeq type logical_call = {
  call:S.invocation;
  weight:S.weights;
  input:S.vector 1024;
  before:cache:S.model_cache{S.cache_domain call cache};
}

noextract
noeq type decoder_stages = {
  normalized:S.vector 1024;
  raw:S.vector 6144;
  projected_z:S.vector 2048;
  projected_b:S.vector 16;
  projected_a:S.vector 16;
  next_conv:S.convolution_state;
  convolved:S.vector 6144;
  head_plans:S.fin 16 -> GTot head_model;
  gated_values:S.vector 2048;
  mixed:S.vector 1024;
  residual:S.vector 1024;
  postnorm:S.vector 1024;
  mlp_gate:S.vector 3584;
  mlp_up:S.vector 3584;
  product:S.vector 3584;
  down:S.vector 1024;
  final_hidden:S.vector 1024;
}

[@@"opaque_to_smt"]
noextract
let build_stages (c:logical_call)
  : GTot (s:decoder_stages{
      s.normalized == S.centered_rms_norm c.weight.input_norm c.input /\
      s.raw == S.project c.weight.in_qkv s.normalized /\
      s.projected_z == S.project c.weight.in_z s.normalized /\
      s.projected_b == S.project c.weight.in_beta s.normalized /\
      s.projected_a == S.project c.weight.in_a s.normalized /\
      s.next_conv == S.shift_convolution (old_convolution c.call c.before) s.raw /\
      s.convolved == S.convolve c.weight.conv_kernel s.next_conv /\
      (forall (h:S.fin 16). s.head_plans h == make_head c.weight s.convolved s.projected_z
        s.projected_b s.projected_a (old_recurrent c.call c.before) h) /\
      (forall (i:S.fin 2048). s.gated_values i == (s.head_plans (i / 128)).gated (i % 128)) /\
      s.mixed == S.project c.weight.out_projection s.gated_values /\
      s.residual == S.add_vectors c.input s.mixed /\
      s.postnorm == S.centered_rms_norm c.weight.post_attention_norm s.residual /\
      s.mlp_gate == S.project c.weight.mlp_gate s.postnorm /\
      s.mlp_up == S.project c.weight.mlp_up s.postnorm /\
      (forall (i:S.fin 3584). s.product i == S.silu (s.mlp_gate i) *. s.mlp_up i) /\
      s.down == S.project c.weight.mlp_down s.product /\
      s.final_hidden == S.add_vectors s.residual s.down}) =
  let w = c.weight in
  let normalized = S.centered_rms_norm w.input_norm c.input in
  let raw = S.project w.in_qkv normalized in
  let projected_z = S.project w.in_z normalized in
  let projected_b = S.project w.in_beta normalized in
  let projected_a = S.project w.in_a normalized in
  let next_conv = S.shift_convolution (old_convolution c.call c.before) raw in
  let convolved = S.convolve w.conv_kernel next_conv in
  let head_plans:S.fin 16 -> GTot head_model =
    fun h -> (make_head c.weight convolved projected_z projected_b projected_a
      (old_recurrent c.call c.before) h <: head_model) in
  let gated_values:S.vector 2048 = fun i -> (head_plans (i / 128)).gated (i % 128) in
  let mixed = S.project w.out_projection gated_values in
  let residual = S.add_vectors c.input mixed in
  let postnorm = S.centered_rms_norm w.post_attention_norm residual in
  let mlp_gate = S.project w.mlp_gate postnorm in
  let mlp_up = S.project w.mlp_up postnorm in
  let product:S.vector 3584 = fun i -> S.silu (mlp_gate i) *. mlp_up i in
  let down = S.project w.mlp_down product in
  let final_hidden = S.add_vectors residual down in
  {normalized; raw; projected_z; projected_b; projected_a; next_conv; convolved;
   head_plans; gated_values; mixed; residual; postnorm; mlp_gate; mlp_up; product; down; final_hidden}

inline_for_extraction noextract
let stage_head_view (c:logical_call) (h:S.fin 16) : GTot prop =
  let s = build_stages c in
  let p = s.head_plans h in
  valid_head p /\
  (forall (i:S.fin 128). p.qinput i == s.convolved (h * 128 + i)) /\
  (forall (i:S.fin 128). p.kinput i == s.convolved (2048 + h * 128 + i)) /\
  (forall (i:S.fin 128). p.value i == s.convolved (4096 + h * 128 + i)) /\
  (forall (i:S.fin 128). p.z i == s.projected_z (h * 128 + i)) /\
  p.weight == c.weight.gated_norm /\ p.old == old_recurrent c.call c.before h /\
  p.b == s.projected_b h /\ p.a == s.projected_a h /\
  p.a_log == c.weight.a_log h /\ p.dt_bias == c.weight.dt_bias h

inline_for_extraction noextract
let stage_head_view_lemma (c:logical_call) (h:S.fin 16) : Lemma (stage_head_view c h) =
  let s = build_stages c in
  let p = make_head c.weight s.convolved s.projected_z s.projected_b s.projected_a
    (old_recurrent c.call c.before) h in
  assert (s.head_plans h == p);
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.stage_head_view"]] (stage_head_view c h)

inline_for_extraction noextract
let models_match (c:logical_call)
  : Lemma (
      (forall (i:S.fin 1024).
        (build_stages c).final_hidden i ==
        (S.decoder_values c.call c.weight c.input c.before).hidden_value i) /\
      (forall (h:S.fin 16) (i j:S.fin 128).
        ((build_stages c).head_plans h).updated i j ==
        (S.decoder_values c.call c.weight c.input c.before).recurrent_value h i j) /\
      (forall (channel:S.fin 6144) (tap:S.fin 4).
        (build_stages c).next_conv channel tap ==
        (S.decoder_values c.call c.weight c.input c.before).convolution_value channel tap)) =
  assert (
      (forall (i:S.fin 1024).
        (build_stages c).final_hidden i ==
        (S.decoder_values c.call c.weight c.input c.before).hidden_value i) /\
      (forall (h:S.fin 16) (i j:S.fin 128).
        ((build_stages c).head_plans h).updated i j ==
        (S.decoder_values c.call c.weight c.input c.before).recurrent_value h i j) /\
      (forall (channel:S.fin 6144) (tap:S.fin 4).
        (build_stages c).next_conv channel tap ==
        (S.decoder_values c.call c.weight c.input c.before).convolution_value channel tap))
    by (FStar.Tactics.V2.norm [iota; reduce_projections; unascribe; primops; simplify; delta_only [
      "KPlay.Proof.Base.RecurrentPipeline.build_stages"; "KPlay.Proof.Base.RecurrentPipeline.make_head";
      "KPlay.Proof.Base.RecurrentPipeline.sum_n"; "KPlay.Proof.Base.RecurrentPipeline.old_convolution"; "KPlay.Proof.Base.RecurrentPipeline.old_recurrent";
      "KPlay.Proof.Shared.Recurrent.decoder_values"; "KPlay.Proof.Shared.Recurrent.token_mixer"; "KPlay.Proof.Shared.Recurrent.feed_forward"
    ]];
    FStar.Tactics.V2.trivial ())
