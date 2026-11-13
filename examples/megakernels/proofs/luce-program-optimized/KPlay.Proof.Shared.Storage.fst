module KPlay.Proof.Shared.Storage

open Pulse.Lib.Core
module Native = Pulse.Lib.Array
module Mem = Kuiper.Array.Core
module Every = Kuiper.ForEvery
module Permission = PulseCore.FractionalPermission
module S = FStar.Seq
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module E = KPlay.Proof.Shared.Embedding
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module O = KPlay.Proof.Shared.Observations

(* Native handles, not ownership Booleans. A role supplies its actual cell map;
   full separating ownership excludes aliasing of simultaneously live cells.
   Maps may use any number of buffers, offsets, strides, and padding cells. *)
inline_for_extraction noextract
noeq type cell (a:Type0) = {
  buffer: Native.array a;
  offset: n:nat{n < Native.length buffer};
}

inline_for_extraction noextract
let cell_owns (#a:Type0) (permission:Permission.perm) (at:cell a) (value:a)
  : slprop =
  Mem.pts_to_cell at.buffer #permission at.offset value

inline_for_extraction noextract
let vector_owns (#a:Type0) (#n:nat) (permission:Permission.perm)
  (at:O.vector (cell a) n) (value:O.vector a n) : slprop =
  Every.forevery (P.index n) (fun i -> cell_owns permission (at i) (value i))

inline_for_extraction noextract
let matrix_owns (#a:Type0) (#m #n:nat) (permission:Permission.perm)
  (at:O.matrix (cell a) m n) (value:O.matrix a m n) : slprop =
  Every.forevery (P.index m) (fun i -> vector_owns permission (at i) (value i))

inline_for_extraction noextract
let tensor_owns (#a:Type0) (#h #m #n:nat) (permission:Permission.perm)
  (at:O.tensor (cell a) h m n) (value:O.tensor a h m n) : slprop =
  Every.forevery (P.index h) (fun i -> matrix_owns permission (at i) (value i))

inline_for_extraction noextract
let linear_weights_owns (#a:Type0) (permission:Permission.perm)
  (at:O.linear_weights (cell a)) (w:O.linear_weights a) : slprop =
  vector_owns permission at.input_norm w.input_norm **
  vector_owns permission at.post_attention_norm w.post_attention_norm **
  matrix_owns permission at.in_qkv w.in_qkv **
  matrix_owns permission at.in_z w.in_z **
  matrix_owns permission at.in_beta w.in_beta **
  matrix_owns permission at.in_a w.in_a **
  matrix_owns permission at.conv_kernel w.conv_kernel **
  vector_owns permission at.a_log w.a_log **
  vector_owns permission at.dt_bias w.dt_bias **
  vector_owns permission at.gated_norm w.gated_norm **
  matrix_owns permission at.out_projection w.out_projection **
  matrix_owns permission at.mlp_gate w.mlp_gate **
  matrix_owns permission at.mlp_up w.mlp_up **
  matrix_owns permission at.mlp_down w.mlp_down

inline_for_extraction noextract
let full_weights_owns (#a:Type0) (permission:Permission.perm)
  (at:O.full_weights (cell a)) (w:O.full_weights a) : slprop =
  vector_owns permission at.input_norm w.input_norm **
  vector_owns permission at.post_attention_norm w.post_attention_norm **
  matrix_owns permission at.query_gate_proj w.query_gate_proj **
  matrix_owns permission at.key_proj w.key_proj **
  matrix_owns permission at.value_proj w.value_proj **
  vector_owns permission at.query_norm w.query_norm **
  vector_owns permission at.key_norm w.key_norm **
  matrix_owns permission at.output_proj w.output_proj **
  matrix_owns permission at.mlp_gate_proj w.mlp_gate_proj **
  matrix_owns permission at.mlp_up_proj w.mlp_up_proj **
  matrix_owns permission at.mlp_down_proj w.mlp_down_proj

inline_for_extraction noextract
let parameters_owns (#a:Type0) (permission:Permission.perm)
  (at:O.parameters (cell a)) (w:O.parameters a) : slprop =
  Every.forevery (D.token 248320) (fun t ->
    vector_owns #a #1024 permission
      (fun i -> S.index (S.index at.embedding t) i)
      (fun i -> S.index (S.index w.embedding t) i)) **
  vector_owns permission at.final_norm w.final_norm **
  Every.forevery Model.linear_layer (fun l ->
    linear_weights_owns permission (at.linear l) (w.linear l)) **
  Every.forevery A.full_attention_layer (fun l ->
    full_weights_owns permission (at.full l) (w.full l))

inline_for_extraction noextract
noeq type cache_storage (#m #c #r #k:Type0) (snapshot:O.cache m c r k) = {
  convolution:
    (l:A.layer_index{Some? (snapshot.convolution l)}) ->
    GTot (O.matrix (cell c) 6144 4);
  recurrent:
    (l:A.layer_index{Some? (snapshot.recurrent l)}) ->
    GTot (O.tensor (cell r) 16 128 128);
  keys:
    (l:A.layer_index{Some? (snapshot.kv l)}) ->
    GTot (O.tensor (cell k) 2 (R.present_value (snapshot.kv l)).length 256);
  values:
    (l:A.layer_index{Some? (snapshot.kv l)}) ->
    GTot (O.tensor (cell k) 2 (R.present_value (snapshot.kv l)).length 256);
}

inline_for_extraction noextract
let cache_owns (#m #c #r #k:Type0) (snapshot:O.cache m c r k)
  (at:cache_storage snapshot) : slprop =
  Every.forevery A.layer_index (fun l ->
    (match snapshot.convolution l with
    | None -> emp
    | Some value -> matrix_owns 1.0R (at.convolution l) value) **
    (match snapshot.recurrent l with
    | None -> emp
    | Some value -> tensor_owns 1.0R (at.recurrent l) value) **
    (match snapshot.kv l with
    | None -> emp
    | Some value ->
        tensor_owns 1.0R (at.keys l) value.keys **
        tensor_owns 1.0R (at.values l) value.values))

inline_for_extraction noextract
noeq type token_storage (token_cell:Type0) (count:nat) = {
  contents: D.vec token_cell count;
  addresses: O.vector (cell token_cell) count;
}

inline_for_extraction noextract
let tokens_owns (#token_cell:Type0) (#count:nat)
  (decode:token_cell -> GTot int) (permission:Permission.perm)
  (at:token_storage token_cell count) (tokens:D.vec (D.token 248320) count) : slprop =
  pure (forall (i:P.index count). decode (S.index at.contents i) == S.index tokens i) **
  vector_owns permission at.addresses (fun i -> S.index at.contents i)

inline_for_extraction noextract
noeq type history_storage (token_cell h:Type0) (count:nat) = {
  tokens: token_storage token_cell count;
  hidden: O.matrix (cell h) count 1024;
}

inline_for_extraction noextract
let history_owns (#token_cell #h:Type0) (#count:nat)
  (decode:token_cell -> GTot int)
  (at:history_storage token_cell h count) (history:D.histories 248320 1024 h count)
  : slprop =
  tokens_owns decode 1.0R at.tokens history.generated_tokens **
  matrix_owns 1.0R at.hidden
    (fun i j -> S.index (S.index history.hidden_history i) j)

inline_for_extraction noextract
let inputs_owns (#token_cell #h:Type0)
  (call:Model.invocation) (decode:token_cell -> GTot int)
  (permission:Permission.perm)
  (w:O.parameters h) (weights_at:O.parameters (cell h))
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (prompt_at:token_storage token_cell call.prompt_length) : slprop =
  parameters_owns permission weights_at w **
  tokens_owns decode permission prompt_at prompt

inline_for_extraction noextract
let loop_owns (#token_cell #m #h #c #r #k:Type0)
  (call:Model.invocation) (decode:token_cell -> GTot int)
  (permission:Permission.perm)
  (w:O.parameters h) (weights_at:O.parameters (cell h))
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (prompt_at:token_storage token_cell call.prompt_length)
  (completed:nat)
  (snapshot:O.cache m c r k) (cache_at:cache_storage snapshot)
  (history:D.histories 248320 1024 h (D.retained_count call.prompt_length completed))
  (history_at:history_storage token_cell h (D.retained_count call.prompt_length completed))
  : slprop =
  inputs_owns call decode permission w weights_at prompt prompt_at **
  cache_owns snapshot cache_at **
  history_owns decode history_at history
