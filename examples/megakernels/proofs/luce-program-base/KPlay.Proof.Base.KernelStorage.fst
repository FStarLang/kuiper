module KPlay.Proof.Base.KernelStorage

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module T = Kuiper.Tensor
module C = Kuiper.Chest
module Mem = Kuiper.Array.Core
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Shared = KPlay.SharedSpec
module Storage = KPlay.Proof.Shared.Storage
module K = KPlay.Proof.Base.KernelState
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Controls = KPlay.Proof.Base.KernelControls
module Layout = KPlay.Proof.Base.KernelSetup
module Result = KPlay.Proof.Base.KernelResult
module Native = KPlay.Proof.Base.NativeProjection
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Arrays = KPlay.Proof.Base.SourceArrays
module Packed = KPlay.Proof.Base.PackedOwnership
module Prompt = KPlay.Proof.Base.InputOwnership
module Adapters = KPlay.Proof.Base.StorageAdapters
module Init = KPlay.Proof.Base.Initialization
module Cache = KPlay.Proof.Base.CacheNative

inline_for_extraction noextract
let input_binding (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length) (args:Entry.arguments call) : prop =
  args.Entry.weight == Input.data inputs.NativeInput.weights /\ args.Entry.prompt == Input.data inputs.NativeInput.prompt

inline_for_extraction noextract
let tensor (values:Transfer.packed) : C.chest2 BF.t 752393024 1 =
  T.from_seq Native.weight_layout values

inline_for_extraction noextract
let tensor_values (values:Transfer.packed)
  : Lemma (Native.packed_values (tensor values) == values) =
  Arrays.flat_weights (tensor values);
  T.to_from Native.weight_layout values

inline_for_extraction noextract
let input_cells (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) : slprop =
  Shared.caller_precondition call I64.v fraction (Inputs.parameter_values values) (Inputs.parameter_cells inputs)
    logical (Inputs.prompt_storage inputs physical)

inline_for_extraction noextract
let output_at (call:Model.invocation) (args:Entry.arguments call) (tokens:lseq I64.t call.Model.decode_steps)
  : Storage.history_storage I64.t BF.t call.Model.decode_steps = {
  Storage.tokens = {Storage.contents = tokens; Storage.addresses = (fun i -> {Storage.buffer = args.Entry.tokens; Storage.offset = i})};
  Storage.hidden = (fun row i -> {Storage.buffer = args.Entry.hidden; Storage.offset = row * 1024 + i});
}

inline_for_extraction noextract
let private_arrays (call:Model.invocation) (args:Entry.arguments call) : slprop =
  (exists* (s:lseq BF.t 12582912). args.Entry.cache |-> s) **
  ((exists* (s:lseq BF.t 442368). args.Entry.convolution |-> s) **
   (exists* (s:lseq FP.t 4718592). args.Entry.state |-> s))

inline_for_extraction noextract
let exported (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length) (args:Entry.arguments call)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) : slprop =
  exists* (m:K.metadata call{m.K.completed == Model.calls call}) (tokens:lseq I64.t call.Model.decode_steps).
    Shared.caller_postcondition call I64.v (fun x -> x) fraction
      (Inputs.parameter_values values) (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs physical)
      (Result.run call values logical m) (Result.output call values logical m) (output_at call args tokens)
      (K.caller call values logical m)

ghost
fn prepare
  (call:Model.invocation) (refs:Owned.controls call) (inputs:NativeInput.arguments call.Model.prompt_length)
  (args:Entry.arguments call{input_binding call inputs args})
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) ()
  requires input_cells call inputs values physical logical fraction **
    Layout.allocation_contents call args ** Controls.parked call refs
  ensures Layout.pre call refs args (tensor values) physical logical fraction
{
  unfold input_cells call inputs values physical logical fraction;
  unfold Shared.caller_precondition call I64.v fraction (Inputs.parameter_values values) (Inputs.parameter_cells inputs)
    logical (Inputs.prompt_storage inputs physical);
  unfold Storage.inputs_owns call I64.v fraction (Inputs.parameter_values values) (Inputs.parameter_cells inputs)
    logical (Inputs.prompt_storage inputs physical);
  Packed.parameters_to_array inputs values fraction;
  Prompt.prompt_to_array inputs physical logical fraction;
  tensor_values values;
  T.lem_from_array_core #BF.t #2 #_ #Native.weight_layout args.Entry.weight;
  rewrite A.pts_to (Input.data inputs.NativeInput.weights) #fraction values
    as A.pts_to (T.core (Layout.weight call args)) #fraction (Native.packed_values (tensor values));
  Arrays.array_to_weight (Layout.weight call args) #(tensor values) #fraction;
  rewrite A.pts_to (Input.data inputs.NativeInput.prompt) #fraction physical as A.pts_to args.Entry.prompt #fraction physical;
  fold Layout.readers call args (tensor values) physical fraction;
  fold Layout.pre call refs args (tensor values) physical logical fraction;
}

inline_for_extraction noextract
let matches_output (call:Model.invocation) (values:Transfer.packed)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (m:K.metadata call) (kv:Cache.snapshot)
  (tokens:lseq I64.t call.Model.decode_steps) (hidden:lseq BF.t (call.Model.decode_steps * 1024))
  : Lemma
      (requires Result.complete call values logical m kv /\
        (forall (i:P.index call.Model.decode_steps). Some (Seq.index tokens i) == Seq.index m.K.memory.K.tokens i) /\
        (forall (i:P.index (call.Model.decode_steps * 1024)). Some (Seq.index hidden i) == Seq.index m.K.memory.K.hidden i))
      (ensures m.K.completed == Model.calls call /\
        (forall (i:P.index call.Model.decode_steps).
          I64.v (Seq.index tokens i) == Seq.index (Result.output call values logical m).D.generated_tokens i) /\
        (forall (row:P.index call.Model.decode_steps) (i:P.index 1024).
          Seq.index hidden (row * 1024 + i) == Seq.index (Seq.index (Result.output call values logical m).D.hidden_history row) i)) = ()

ghost
fn export
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (args:Entry.arguments call{input_binding call inputs args})
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) ()
  requires Layout.post call args (tensor values) physical logical fraction
  ensures private_arrays call args ** exported call inputs args values physical logical fraction
{
  unfold Layout.post call args (tensor values) physical logical fraction;
  unfold Layout.readers call args (tensor values) physical fraction;
  tensor_values values;
  Arrays.weight_to_array (Layout.weight call args) #(tensor values) #fraction;
  T.lem_from_array_core #BF.t #2 #_ #Native.weight_layout args.Entry.weight;
  rewrite A.pts_to (T.core (Layout.weight call args)) #fraction (Native.packed_values (tensor values))
    as A.pts_to (Input.data inputs.NativeInput.weights) #fraction values;
  Packed.array_to_parameters inputs values fraction;
  rewrite A.pts_to args.Entry.prompt #fraction physical as A.pts_to (Input.data inputs.NativeInput.prompt) #fraction physical;
  Prompt.array_to_prompt inputs physical logical fraction;
  rewrite Result.post call args (Native.packed_values (tensor values)) logical as Result.post call args values logical;
  unfold Result.post call args values logical;
  with m kv. assert args.Entry.state |-> (reveal #(K.metadata call) m).K.memory.K.state **
    A.pts_to args.Entry.convolution (reveal #(K.metadata call) m).K.memory.K.convolution **
    A.pts_to_mask args.Entry.cache kv Init.all_cells **
    A.pts_to_mask args.Entry.tokens (reveal #(K.metadata call) m).K.memory.K.tokens Init.all_cells **
    A.pts_to_mask args.Entry.hidden (reveal #(K.metadata call) m).K.memory.K.hidden Init.all_cells;
  Result.refines call values logical m kv;
  Result.output_cells call values logical m kv;
  A.from_mask args.Entry.cache;
  A.from_mask args.Entry.tokens;
  A.from_mask args.Entry.hidden;
  with cache_values token_values hidden_values.
    assert args.Entry.cache |-> cache_values ** args.Entry.tokens |-> token_values ** args.Entry.hidden |-> hidden_values;
  matches_output call values logical m kv token_values hidden_values;
  fold private_arrays call args;
  Mem.array_to_slice args.Entry.tokens #1.0R #token_values;
  drop_ (Mem.is_full_slice args.Entry.tokens call.Model.decode_steps);
  Seq.slice_length token_values;
  rewrite Mem.pts_to_slice args.Entry.tokens #1.0R 0 call.Model.decode_steps token_values
    as Mem.pts_to_slice args.Entry.tokens #1.0R 0 call.Model.decode_steps (Seq.slice token_values 0 call.Model.decode_steps);
  Adapters.slice_to_vector #I64.t #(call.Model.decode_steps) args.Entry.tokens 0 token_values
    (output_at call args token_values).Storage.tokens.Storage.addresses (fun i -> Seq.index token_values i) 1.0R;
  fold Storage.tokens_owns I64.v 1.0R (output_at call args token_values).Storage.tokens
    (Result.output call values logical m).D.generated_tokens;
  Mem.array_to_slice args.Entry.hidden #1.0R #hidden_values;
  drop_ (Mem.is_full_slice args.Entry.hidden (call.Model.decode_steps * 1024));
  Seq.slice_length hidden_values;
  rewrite Mem.pts_to_slice args.Entry.hidden #1.0R 0 (call.Model.decode_steps * 1024) hidden_values
    as Mem.pts_to_slice args.Entry.hidden #1.0R 0 (call.Model.decode_steps * 1024)
      (Seq.slice hidden_values 0 (call.Model.decode_steps * 1024));
  Adapters.slice_to_matrix #BF.t #(call.Model.decode_steps) #1024 args.Entry.hidden 0 hidden_values
    (output_at call args token_values).Storage.hidden
    (fun row i -> Seq.index (Seq.index (Result.output call values logical m).D.hidden_history row) i) 1.0R;
  fold Storage.history_owns I64.v (output_at call args token_values) (Result.output call values logical m);
  fold Storage.inputs_owns call I64.v fraction (Inputs.parameter_values values) (Inputs.parameter_cells inputs)
    logical (Inputs.prompt_storage inputs physical);
  fold Shared.caller_precondition call I64.v fraction (Inputs.parameter_values values) (Inputs.parameter_cells inputs)
    logical (Inputs.prompt_storage inputs physical);
  fold Shared.safety_caller_postcondition call I64.v (fun x -> x) fraction
    (Inputs.parameter_values values) (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs physical)
    (Result.run call values logical m) (Result.output call values logical m) (output_at call args token_values)
    (K.caller call values logical m);
  fold Shared.caller_postcondition call I64.v (fun x -> x) fraction
    (Inputs.parameter_values values) (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs physical)
    (Result.run call values logical m) (Result.output call values logical m) (output_at call args token_values)
    (K.caller call values logical m);
  fold exported call inputs args values physical logical fraction;
}
