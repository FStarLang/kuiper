module KPlay.Proof.Optimized.Layout

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module S = FStar.Seq
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module E = KPlay.Proof.Shared.Embedding
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Storage = KPlay.Proof.Shared.Storage
module Input = KPlay.Proof.NativeInput

inline_for_extraction noextract
let weight_count : pos = 752393024

inline_for_extraction noextract
let packed = s:S.seq Kuiper.BFloat16.t{S.length s == weight_count}

inline_for_extraction noextract
let layer_base (layer:A.layer_index) : nat =
  254280704 + (layer / 4) * 83018720 + (layer % 4) * 21555360

inline_for_extraction noextract
let layer_size (layer:A.layer_index) : pos =
  if layer % 4 < 3 then 21555360 else 18352640

let layer_partition (layer:A.layer_index)
  : Lemma (
      layer_base layer + layer_size layer <= weight_count /\
      (if layer < 23
       then layer_base layer + layer_size layer == layer_base (layer + 1)
       else layer_base layer + layer_size layer == weight_count)) = ()

inline_for_extraction noextract
let linear_weights (#a:Type0) (get:P.index weight_count -> GTot a)
  (layer:Model.linear_layer) : GTot (O.linear_weights a) =
  layer_partition layer;
  let base = layer_base layer in {
    O.input_norm = (fun i -> get (base + i));
    O.post_attention_norm = (fun i -> get (base + 1024 + i));
    O.in_qkv = (fun r c -> get (base + 11012096 + r * 1024 + c));
    O.in_z = (fun r c -> get (base + 17303552 + r * 1024 + c));
    O.in_beta = (fun r c -> get (base + 19400704 + r * 1024 + c));
    O.in_a = (fun r c -> get (base + 19417088 + r * 1024 + c));
    O.conv_kernel = (fun r c -> get (base + 19433472 + r * 4 + c));
    O.a_log = (fun i -> get (base + 19458048 + i));
    O.dt_bias = (fun i -> get (base + 19458064 + i));
    O.gated_norm = (fun i -> get (base + 19458080 + i));
    O.out_projection = (fun r c -> get (base + 19458208 + r * 2048 + c));
    O.mlp_gate = (fun r c -> get (base + 2048 + r * 1024 + c));
    O.mlp_up = (fun r c -> get (base + 3672064 + r * 1024 + c));
    O.mlp_down = (fun r c -> get (base + 7342080 + r * 3584 + c));
  }

inline_for_extraction noextract
let full_weights (#a:Type0) (get:P.index weight_count -> GTot a)
  (layer:A.full_attention_layer) : GTot (O.full_weights a) =
  layer_partition layer;
  let base = layer_base layer in {
    O.input_norm = (fun i -> get (base + i));
    O.post_attention_norm = (fun i -> get (base + (1024 + i)));
    O.query_gate_proj = (fun r c -> get (base + (11012096 + 1024 * r + c)));
    O.key_proj = (fun r c -> get (base + (15206400 + 1024 * r + c)));
    O.value_proj = (fun r c -> get (base + (15730688 + 1024 * r + c)));
    O.query_norm = (fun i -> get (base + (16254976 + i)));
    O.key_norm = (fun i -> get (base + (16255232 + i)));
    O.output_proj = (fun r c -> get (base + (16255488 + 2048 * r + c)));
    O.mlp_gate_proj = (fun r c -> get (base + (2048 + 1024 * r + c)));
    O.mlp_up_proj = (fun r c -> get (base + (3672064 + 1024 * r + c)));
    O.mlp_down_proj = (fun r c -> get (base + (7342080 + 3584 * r + c)));
  }

inline_for_extraction noextract
let parameters (#a:Type0) (get:P.index weight_count -> GTot a)
  : GTot (O.parameters a) = {
  O.embedding = S.init_ghost #(E.hidden_vector a) 248320 (fun t ->
    S.init_ghost #a 1024 (fun i -> get (t * 1024 + i)));
  O.final_norm = (fun i -> get (254279680 + i));
  O.linear = linear_weights get;
  O.full = full_weights get;
}

let embedding_row (#a:Type0) (get:P.index weight_count -> GTot a) (token:P.index 248320)
  : Lemma (
      S.index (parameters get).O.embedding token ==
        S.init_ghost 1024 (fun i -> get (token * 1024 + i))) =
  FStar.Pervasives.norm_spec [delta_only [`%parameters]; zeta; iota] (parameters get).O.embedding;
  S.init_ghost_index_ #(E.hidden_vector a) 248320
    (fun t -> S.init_ghost 1024 (fun i -> get (t * 1024 + i))) token

inline_for_extraction noextract
let input_addresses (#p:pos{p <= 520}) (inputs:Input.arguments p)
  : GTot (O.parameters (Storage.cell Kuiper.BFloat16.t)) =
  parameters (Input.cell inputs.weights)

let tied_input_cells (#p:pos{p <= 520}) (inputs:Input.arguments p)
  (token:P.index 248320) (i:P.index 1024)
  : Lemma (
      S.index (S.index (input_addresses inputs).embedding token) i ==
        Input.cell inputs.weights (token * 1024 + i) /\
      (input_addresses inputs).final_norm i ==
        Input.cell inputs.weights (254279680 + i)) = ()

inline_for_extraction noextract
let parameter_cells (#p:pos{p <= 520}) (inputs:Input.arguments p)
  : GTot (O.parameters (Storage.cell Kuiper.BFloat16.t)) =
  input_addresses inputs

inline_for_extraction noextract
let parameter_values (values:packed) : GTot (O.parameters Kuiper.BFloat16.t) =
  parameters (fun i -> S.index values i)

let layer_extent (layer:A.layer_index)
  : Lemma (
      layer_base layer + (if layer % 4 < 3 then 21555360 else 18352640) <= weight_count) =
  layer_partition layer

inline_for_extraction noextract
let prompt_storage (#p:pos{p <= 520}) (inputs:Input.arguments p)
  (values:(s:S.seq FStar.Int64.t{S.length s == p}))
  : GTot (Storage.token_storage FStar.Int64.t p) = {
  Storage.contents = values;
  Storage.addresses = (fun i -> Input.cell inputs.prompt i);
}
