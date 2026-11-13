module KPlay.Proof.Base.Inputs

#lang-pulse

open Kuiper

module S = FStar.Seq
module A = Pulse.Lib.Array
module Mem = Kuiper.Array.Core
module Every = Kuiper.ForEvery
module Trade = Pulse.Lib.Trade
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module IntCast = FStar.Int.Cast
module BF = Kuiper.BFloat16
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module P = KPlay.Proof.Shared.Projection
module E = KPlay.Proof.Shared.Embedding
module Attention = KPlay.Proof.Shared.Attention
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module O = KPlay.Proof.Shared.Observations
module Storage = KPlay.Proof.Shared.Storage
module Embedding = KPlay.Proof.Base.Embedding
module Rotary = KPlay.Proof.Base.Rotary
module Transfer = KPlay.Proof.Base.Transfer
module Approx = Kuiper.Approximates.Base

inline_for_extraction noextract
let token_value = token:I64.t{0 <= I64.v token /\ I64.v token < 248320}

inline_for_extraction noextract
let layer_base (layer:Attention.layer_index) : nat =
  254280704 + (layer / 4) * 83018720 + (layer % 4) * 21555360

inline_for_extraction noextract
let layer_extent (layer:Attention.layer_index)
  : Lemma (
      layer_base layer +
        (if layer % 4 < 3 then 21555360 else 18352640) <= 752393024) = ()

(* The same packing map is used for values and native cells. *)
inline_for_extraction noextract
let parameters (#a:Type0) (at:P.vector a 752393024) : GTot (O.parameters a) = {
  O.embedding = S.init_ghost #(E.hidden_vector a) 248320 (fun t ->
    S.init_ghost #a 1024 (fun i -> at (t * 1024 + i)));
  O.final_norm = (fun i -> at (254279680 + i));
  O.linear = (fun layer ->
    layer_extent layer;
    let base = layer_base layer in
    {
      O.input_norm = (fun i -> at (base + i));
      O.post_attention_norm = (fun i -> at (base + 1024 + i));
      O.in_qkv = (fun r c -> at (base + 11012096 + r * 1024 + c));
      O.in_z = (fun r c -> at (base + 17303552 + r * 1024 + c));
      O.in_beta = (fun r c -> at (base + 19400704 + r * 1024 + c));
      O.in_a = (fun r c -> at (base + 19417088 + r * 1024 + c));
      O.conv_kernel = (fun r c -> at (base + 19433472 + r * 4 + c));
      O.a_log = (fun i -> at (base + 19458048 + i));
      O.dt_bias = (fun i -> at (base + 19458064 + i));
      O.gated_norm = (fun i -> at (base + 19458080 + i));
      O.out_projection = (fun r c -> at (base + 19458208 + r * 2048 + c));
      O.mlp_gate = (fun r c -> at (base + 2048 + r * 1024 + c));
      O.mlp_up = (fun r c -> at (base + 3672064 + r * 1024 + c));
      O.mlp_down = (fun r c -> at (base + 7342080 + r * 3584 + c));
    });
  O.full = (fun layer ->
    layer_extent layer;
    let base = layer_base layer in
    {
      O.input_norm = (fun i -> at (base + i));
      O.post_attention_norm = (fun i -> at (base + 1024 + i));
      O.query_gate_proj = (fun r c -> at (base + 11012096 + r * 1024 + c));
      O.key_proj = (fun r c -> at (base + 15206400 + r * 1024 + c));
      O.value_proj = (fun r c -> at (base + 15730688 + r * 1024 + c));
      O.query_norm = (fun i -> at (base + 16254976 + i));
      O.key_norm = (fun i -> at (base + 16255232 + i));
      O.output_proj = (fun r c -> at (base + 16255488 + r * 2048 + c));
      O.mlp_gate_proj = (fun r c -> at (base + 2048 + r * 1024 + c));
      O.mlp_up_proj = (fun r c -> at (base + 3672064 + r * 1024 + c));
      O.mlp_down_proj = (fun r c -> at (base + 7342080 + r * 3584 + c));
    });
}

inline_for_extraction noextract
let parameter_cells (#p:pos{p <= 520}) (inputs:NativeInput.arguments p)
  : GTot (O.parameters (Storage.cell BF.t)) =
  parameters (fun i -> NativeInput.cell inputs.weights i)

inline_for_extraction noextract
let parameter_values (values:Transfer.packed) : GTot (O.parameters BF.t) =
  parameters (fun i -> S.index values i)

inline_for_extraction noextract
let prompt_storage (#p:pos{p <= 520}) (inputs:NativeInput.arguments p)
  (values:lseq I64.t p) : GTot (Storage.token_storage I64.t p) = {
  Storage.contents = values;
  Storage.addresses = (fun i -> NativeInput.cell inputs.prompt i);
}

inline_for_extraction noextract
let embedding_cell_binding (#p:pos{p <= 520}) (inputs:NativeInput.arguments p)
  (token:D.token 248320) (i:P.index 1024)
  : Lemma (
      let at = S.index (S.index (parameter_cells inputs).O.embedding token) i in
      at.buffer == Input.data inputs.weights /\
      at.offset == token * 1024 + i /\
      A.base_of at.buffer == A.base_of (Input.backing inputs.weights) /\
      A.offset_of at.buffer + at.offset ==
        A.offset_of (Input.backing inputs.weights) + Input.offset inputs.weights +
        token * 1024 + i) =
  NativeInput.cell_binding inputs.weights (token * 1024 + i)

inline_for_extraction noextract
let embedding_model (values:Transfer.packed) (token:D.token 248320)
  : Lemma (
      E.embedding_post (parameter_values values).O.embedding token (Embedding.row values token) /\
      (forall (i:P.index 1024).
        Approx.v_approximates (S.index (Embedding.row values token) i)
          (S.index
            (E.algebraic_embedding (O.model_parameters (parameter_values values)).Model.embedding token)
            i))) =
  E.embedding_correct (parameter_values values).O.embedding token;
  O.embedding_models_input (parameter_values values) token

inline_for_extraction noextract
fn read_weight_cell
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p)
  (weights:larray BF.t 752393024{weights == Input.data inputs.weights})
  (token:sz{SZ.v token < 248320}) (i:sz{SZ.v i < 1024})
  (#values:Transfer.packed) (#fraction:perm)
  preserves gpu
  preserves Storage.parameters_owns fraction (parameter_cells inputs) (parameter_values values)
  returns value:BF.t
  ensures pure (value == S.index values (SZ.v token * 1024 + SZ.v i))
{
  unfold Storage.parameters_owns fraction (parameter_cells inputs) (parameter_values values);
  Every.forevery_extract #(D.token 248320) (SZ.v token)
    (fun t -> Storage.vector_owns #BF.t #1024 fraction
      (fun j -> S.index (S.index (parameter_cells inputs).O.embedding t) j)
      (fun j -> S.index (S.index (parameter_values values).O.embedding t) j));
  unfold Storage.vector_owns #BF.t #1024 fraction
    (fun j -> S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) j)
    (fun j -> S.index (S.index (parameter_values values).O.embedding (SZ.v token)) j);
  Every.forevery_extract #(P.index 1024) (SZ.v i)
    (fun j -> Storage.cell_owns fraction
      (S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) j)
      (S.index (S.index (parameter_values values).O.embedding (SZ.v token)) j));
  embedding_cell_binding inputs (SZ.v token) (SZ.v i);
  unfold Storage.cell_owns fraction
    (S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) (SZ.v i))
    (S.index (S.index (parameter_values values).O.embedding (SZ.v token)) (SZ.v i));
  rewrite each
    (S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) (SZ.v i)).Storage.buffer
    as weights;
  let value = Mem.slice_read weights (SZ.add (SZ.mul token 1024sz) i);
  rewrite each weights as
    (S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) (SZ.v i)).Storage.buffer;
  fold Storage.cell_owns fraction
    (S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) (SZ.v i))
    (S.index (S.index (parameter_values values).O.embedding (SZ.v token)) (SZ.v i));
  Trade.elim_trade
    (Storage.cell_owns fraction
      (S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) (SZ.v i))
      (S.index (S.index (parameter_values values).O.embedding (SZ.v token)) (SZ.v i)))
    (Every.forevery (P.index 1024) (fun j -> Storage.cell_owns fraction
      (S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) j)
      (S.index (S.index (parameter_values values).O.embedding (SZ.v token)) j)));
  fold Storage.vector_owns #BF.t #1024 fraction
    (fun j -> S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) j)
    (fun j -> S.index (S.index (parameter_values values).O.embedding (SZ.v token)) j);
  Trade.elim_trade
    (Storage.vector_owns #BF.t #1024 fraction
      (fun j -> S.index (S.index (parameter_cells inputs).O.embedding (SZ.v token)) j)
      (fun j -> S.index (S.index (parameter_values values).O.embedding (SZ.v token)) j))
    (Every.forevery (D.token 248320) (fun t -> Storage.vector_owns #BF.t #1024 fraction
      (fun j -> S.index (S.index (parameter_cells inputs).O.embedding t) j)
      (fun j -> S.index (S.index (parameter_values values).O.embedding t) j)));
  fold Storage.parameters_owns fraction (parameter_cells inputs) (parameter_values values);
  value
}

inline_for_extraction noextract
fn read_token
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.prompt_length})
  (prompt:larray I64.t call.prompt_length{prompt == Input.data inputs.prompt})
  (next:I64.t)
  (#values:lseq I64.t call.prompt_length)
  (#logical:D.vec (D.token 248320) call.prompt_length) (#fraction:perm)
  preserves gpu
  preserves Storage.tokens_owns I64.v fraction (prompt_storage inputs values) logical
  requires pure (SZ.v position < call.prompt_length \/
    (0 <= I64.v next /\ I64.v next < 248320))
  returns token:token_value
  ensures pure (token == Embedding.source_token call (SZ.v position) values next)
{
  if (position <^ prompt_length) {
    unfold Storage.tokens_owns I64.v fraction (prompt_storage inputs values) logical;
    unfold Storage.vector_owns #I64.t #call.prompt_length fraction
      (prompt_storage inputs values).Storage.addresses
      (fun j -> S.index (prompt_storage inputs values).Storage.contents j);
    Every.forevery_extract #(P.index call.prompt_length) (SZ.v position)
      (fun j -> Storage.cell_owns fraction (NativeInput.cell inputs.prompt j) (S.index values j));
    NativeInput.cell_binding inputs.prompt (SZ.v position);
    unfold Storage.cell_owns fraction (NativeInput.cell inputs.prompt (SZ.v position))
      (S.index values (SZ.v position));
    rewrite each (NativeInput.cell inputs.prompt (SZ.v position)).Storage.buffer as prompt;
    let token = Mem.slice_read prompt position;
    assert pure (token == S.index values (SZ.v position));
    assert pure (I64.v token == S.index #(D.token 248320) logical (SZ.v position));
    rewrite each prompt as (NativeInput.cell inputs.prompt (SZ.v position)).Storage.buffer;
    fold Storage.cell_owns fraction (NativeInput.cell inputs.prompt (SZ.v position))
      (S.index values (SZ.v position));
    Trade.elim_trade
      (Storage.cell_owns fraction (NativeInput.cell inputs.prompt (SZ.v position))
        (S.index values (SZ.v position)))
      (Every.forevery (P.index call.prompt_length)
        (fun j -> Storage.cell_owns fraction (NativeInput.cell inputs.prompt j) (S.index values j)));
    fold Storage.vector_owns #I64.t #call.prompt_length fraction
      (prompt_storage inputs values).Storage.addresses
      (fun j -> S.index (prompt_storage inputs values).Storage.contents j);
    fold Storage.tokens_owns I64.v fraction (prompt_storage inputs values) logical;
    token
  } else {
    next
  }
}

(* Lane-zero kernel.cu:230-231. Input cells are borrowed from the frozen
   logical ownership, never replaced by full-array or final-owner authority. *)
inline_for_extraction noextract
fn embedding_step
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.prompt_length})
  (weights:larray BF.t 752393024{weights == Input.data inputs.weights})
  (prompt:larray I64.t call.prompt_length{prompt == Input.data inputs.prompt})
  (next:I64.t) (hidden:larray BF.t 16000)
  (#values:Transfer.packed) (#prompt_values:lseq I64.t call.prompt_length)
  (#logical:D.vec (D.token 248320) call.prompt_length)
  (#before:Rotary.snapshot) (#fraction:perm)
  preserves gpu
  preserves Storage.inputs_owns call I64.v fraction (parameter_values values)
    (parameter_cells inputs) logical (prompt_storage inputs prompt_values)
  requires pure (SZ.v position < call.prompt_length \/
    (0 <= I64.v next /\ I64.v next < 248320))
  requires A.pts_to_mask hidden before Embedding.embedding_mask
  returns token:token_value
  ensures A.pts_to_mask hidden
    (Embedding.writes before values (I64.v token) 1024) Embedding.embedding_mask
  ensures pure (
    token == Embedding.source_token call (SZ.v position) prompt_values next /\
    E.embedding_post (parameter_values values).O.embedding (I64.v token)
      (Embedding.row values (I64.v token)))
{
  unfold Storage.inputs_owns call I64.v fraction (parameter_values values)
    (parameter_cells inputs) logical (prompt_storage inputs prompt_values);
  let token = read_token call inputs position prompt_length prompt next;
  let index = SZ.uint32_to_sizet (IntCast.int64_to_uint32 token);
  assert pure (SZ.v index == I64.v token);
  Embedding.writes_zero before values (I64.v token);
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (current:sz{SZ.v current <= 1024}).
      i |-> current ** A.pts_to_mask hidden
        (Embedding.writes before values (I64.v token) (SZ.v current)) Embedding.embedding_mask
    invariant gpu **
      Storage.parameters_owns fraction (parameter_cells inputs) (parameter_values values)
    decreases (1024 - SZ.v !i)
  {
    let current = !i;
    let value = read_weight_cell inputs weights index current;
    Embedding.writes_step before values (I64.v token) (SZ.v current);
    A.mask_write hidden current value;
    i := SZ.add current 1sz;
  };
  Embedding.copied_row before values (I64.v token);
  embedding_model values (I64.v token);
  fold Storage.inputs_owns call I64.v fraction (parameter_values values)
    (parameter_cells inputs) logical (prompt_storage inputs prompt_values);
  token
}
