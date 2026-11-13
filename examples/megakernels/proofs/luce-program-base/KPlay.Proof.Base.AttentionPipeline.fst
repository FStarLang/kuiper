module KPlay.Proof.Base.AttentionPipeline

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module A = KPlay.Proof.Shared.Attention
module Math = KPlay.Proof.Base.AttentionAlgebra

noeq type logical_call = {
  call:A.invocation;
  weight:A.layer_weights;
  input:A.vector 1024;
  before:cache:A.hybrid_cache{A.domain call cache};
}

noeq type stages (position:nat) = {
  normalized:A.vector 1024;
  projected_qg:A.vector 4096;
  projected_k:A.vector 512;
  projected_v:A.vector 512;
  queries:A.matrix 8 256;
  gates:A.vector 2048;
  keys:A.matrix 2 256;
  values:A.matrix 2 256;
  history:h:A.kv_history{h.A.length == position + 1};
  context:A.matrix 8 256;
  gated:A.vector 2048;
  mixed:A.vector 1024;
  residual:A.vector 1024;
  postnorm:A.vector 1024;
  mlp_gate:A.vector 3584;
  mlp_up:A.vector 3584;
  product:A.vector 3584;
  down:A.vector 1024;
  final_hidden:A.vector 1024;
}

[@@"opaque_to_smt"]
inline_for_extraction noextract
let build (c:logical_call)
  : GTot (s:stages c.call.A.position{
      s.normalized == A.rms_norm 1024 c.weight.A.input_norm c.input /\
      s.projected_qg == A.linear 4096 1024 c.weight.A.query_gate_proj s.normalized /\
      s.projected_k == A.linear 512 1024 c.weight.A.key_proj s.normalized /\
      s.projected_v == A.linear 512 1024 c.weight.A.value_proj s.normalized /\
      s.queries == Math.rotated_matrix 8 c.call.A.position c.weight.A.query_norm (A.split_queries s.projected_qg) /\
      s.gates == Math.flat_heads 8 (A.split_gates s.projected_qg) /\
      s.keys == Math.rotated_matrix 2 c.call.A.position c.weight.A.key_norm (A.reshape_kv s.projected_k) /\
      s.values == A.reshape_kv s.projected_v /\
      s.history == A.append_kv c.call.A.position (c.before.A.kv c.call.A.layer) s.keys s.values /\
      s.context == A.attention c.call.A.position s.queries s.history /\
      s.gated == Math.activation_vector 2048 false s.gates (Math.flat_heads 8 s.context) /\
      s.mixed == A.linear 1024 2048 c.weight.A.output_proj s.gated /\
      s.residual == A.vector_add 1024 c.input s.mixed /\
      s.postnorm == A.rms_norm 1024 c.weight.A.post_attention_norm s.residual /\
      s.mlp_gate == A.linear 3584 1024 c.weight.A.mlp_gate_proj s.postnorm /\
      s.mlp_up == A.linear 3584 1024 c.weight.A.mlp_up_proj s.postnorm /\
      s.product == Math.activation_vector 3584 true s.mlp_gate s.mlp_up /\
      s.down == A.linear 1024 3584 c.weight.A.mlp_down_proj s.product /\
      s.final_hidden == A.vector_add 1024 s.residual s.down}) =
  let normalized = A.rms_norm 1024 c.weight.A.input_norm c.input in
  let projected_qg = A.linear 4096 1024 c.weight.A.query_gate_proj normalized in
  let projected_k = A.linear 512 1024 c.weight.A.key_proj normalized in
  let projected_v = A.linear 512 1024 c.weight.A.value_proj normalized in
  let queries = Math.rotated_matrix 8 c.call.A.position c.weight.A.query_norm (A.split_queries projected_qg) in
  let gates = Math.flat_heads 8 (A.split_gates projected_qg) in
  let keys = Math.rotated_matrix 2 c.call.A.position c.weight.A.key_norm (A.reshape_kv projected_k) in
  let values = A.reshape_kv projected_v in
  let history = A.append_kv c.call.A.position (c.before.A.kv c.call.A.layer) keys values in
  let context = A.attention c.call.A.position queries history in
  let gated = Math.activation_vector 2048 false gates (Math.flat_heads 8 context) in
  let mixed = A.linear 1024 2048 c.weight.A.output_proj gated in
  let residual = A.vector_add 1024 c.input mixed in
  let postnorm = A.rms_norm 1024 c.weight.A.post_attention_norm residual in
  let mlp_gate = A.linear 3584 1024 c.weight.A.mlp_gate_proj postnorm in
  let mlp_up = A.linear 3584 1024 c.weight.A.mlp_up_proj postnorm in
  let product = Math.activation_vector 3584 true mlp_gate mlp_up in
  let down = A.linear 1024 3584 c.weight.A.mlp_down_proj product in
  let final_hidden = A.vector_add 1024 residual down in
  {normalized; projected_qg; projected_k; projected_v; queries; gates; keys; values; history; context;
   gated; mixed; residual; postnorm; mlp_gate; mlp_up; product; down; final_hidden}

inline_for_extraction noextract
let reference_result (c:logical_call) (s:stages c.call.A.position)
  : Lemma
      (requires s == build c)
      (ensures
        s.final_hidden == (A.full_attention_decoder c.call c.weight c.input c.before).A.hidden /\
        A.replace_kv c.before c.call.A.layer s.history ==
          (A.full_attention_decoder c.call c.weight c.input c.before).A.cache) =
  let canonical = build c in
  Math.shared_decoder_hidden c.call c.weight c.input c.before;
  Math.shared_decoder_cache c.call c.weight c.input c.before
