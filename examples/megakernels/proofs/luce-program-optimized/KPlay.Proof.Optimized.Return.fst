module KPlay.Proof.Optimized.Return

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module Mem = Kuiper.Array.Core
module I64 = FStar.Int64
module C = FStar.Classical
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module D = KPlay.Proof.Shared.Selection
module P = KPlay.Proof.Shared.Projection
module Model = KPlay.Proof.Shared.Model
module Shared = KPlay.SharedSpec
module Storage = KPlay.Proof.Shared.Storage
module Layout = KPlay.Proof.Optimized.Layout
module Packed = KPlay.Proof.Optimized.PackedOwnership
module Inputs = KPlay.Proof.Optimized.InputOwnership
module Adapters = KPlay.Proof.Optimized.StorageAdapters
module Entry = KPlay.Proof.Optimized.Entry
module Kernel = KPlay.Proof.Optimized.Kernel
module Execution = KPlay.Proof.Optimized.Execution
module Math = KPlay.Proof.Optimized.ProgramMath
module Memory = KPlay.Proof.Optimized.Memory
module State = KPlay.Proof.Optimized.State

inline_for_extraction noextract
let input_cells (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) : slprop =
  Shared.caller_precondition call I64.v permission (Layout.parameter_values w)
    (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical)

inline_for_extraction noextract
let input_binding (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (args:Entry.arguments call) : prop =
  args.Entry.weights == Input.data inputs.NativeInput.weights /\
  args.Entry.prompt == Input.data inputs.NativeInput.prompt

ghost
fn prepare (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (args:Entry.arguments call{input_binding call inputs args})
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) ()
  requires input_cells call inputs w physical prompt permission ** Kernel.allocations call args
  ensures Kernel.pre call args w physical prompt permission
{
  unfold input_cells call inputs w physical prompt permission;
  unfold Shared.caller_precondition call I64.v permission (Layout.parameter_values w)
    (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical);
  unfold Storage.inputs_owns call I64.v permission (Layout.parameter_values w)
    (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical);
  Packed.parameters_to_array inputs w permission;
  Inputs.prompt_to_array inputs physical prompt permission;
  rewrite A.pts_to (Input.data inputs.NativeInput.weights) #permission w as A.pts_to args.Entry.weights #permission w;
  rewrite A.pts_to (Input.data inputs.NativeInput.prompt) #permission physical as A.pts_to args.Entry.prompt #permission physical;
  fold Kernel.readers call args w physical permission;
  fold Kernel.pre call args w physical prompt permission;
}

inline_for_extraction noextract
let output_addresses (call:Model.invocation) (args:Entry.arguments call)
  (tokens:lseq I64.t call.decode_steps) : Storage.history_storage I64.t bf16 call.decode_steps = {
  Storage.tokens = {
    Storage.contents = tokens;
    Storage.addresses = (fun i -> {Storage.buffer = args.Entry.tokens; Storage.offset = i});
  };
  Storage.hidden = (fun g i -> {Storage.buffer = args.Entry.hidden; Storage.offset = g * 1024 + i});
}

inline_for_extraction noextract
let private_arrays (call:Model.invocation) (args:Entry.arguments call) : slprop =
  (exists* (v:lseq bf16 6291456). args.Entry.values |-> v) **
  (exists* (k:lseq bf16 6291456). args.Entry.keys |-> k) **
  (exists* (c:lseq f32 442368). args.Entry.convolution |-> c) **
  (exists* (s:lseq f32 4718592). args.Entry.states |-> s)

inline_for_extraction noextract
let exported (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (args:Entry.arguments call) (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) : slprop =
  exists* (p:Execution.prefix call{p.Execution.count == Model.calls call})
          (tokens:lseq I64.t call.decode_steps).
    Shared.safety_caller_postcondition call I64.v (fun x -> x) permission
      (Layout.parameter_values w) (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical)
      (Execution.run w prompt p) (Execution.caller w prompt p p.Execution.count).D.returned
      (output_addresses call args tokens) (Execution.caller w prompt p p.Execution.count) **
    pure ((Math.family_dependencies w prompt p /\ Math.numerical_scope p) ==>
      Shared.numerical_prefix call (Layout.parameter_values w) prompt (Execution.run w prompt p) (Model.calls call))

let outputs_ready (call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (p:Execution.prefix call{p.Execution.count == Model.calls call})
  (tokens:lseq (option I64.t) call.decode_steps)
  (hidden:lseq (option bf16) (call.decode_steps * 1024))
  : Lemma
      (requires Execution.outputs_agree w prompt p tokens hidden)
      (ensures Memory.ready tokens 0 call.decode_steps /\ Memory.ready hidden 0 (call.decode_steps * 1024)) =
  let cell (i:nat{i < call.decode_steps * 1024}) : Lemma (Some? (S.index hidden i)) =
    assert (i == (i / 1024) * 1024 + i % 1024);
    assert (i / 1024 < D.retained_count call.prompt_length p.Execution.count)
  in C.forall_intro cell

let conditional_numerics (call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (p:Execution.prefix call{p.Execution.count == Model.calls call})
  : Lemma
      (requires Execution.well_formed w prompt p)
      (ensures (Math.family_dependencies w prompt p /\ Math.numerical_scope p) ==>
        Shared.numerical_prefix call (Layout.parameter_values w) prompt (Execution.run w prompt p) (Model.calls call)) =
  let established (_:squash (Math.family_dependencies w prompt p /\ Math.numerical_scope p))
    : Lemma (Shared.numerical_prefix call (Layout.parameter_values w) prompt (Execution.run w prompt p) (Model.calls call)) =
    Math.numerical_prefix w prompt p (Model.calls call)
  in C.impl_intro established

ghost
fn export (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (args:Entry.arguments call{input_binding call inputs args})
  (w:Layout.packed) (physical:lseq I64.t call.prompt_length)
  (prompt:D.vec (D.token 248320) call.prompt_length) (permission:perm) ()
  requires Kernel.post call args w physical prompt permission
  ensures private_arrays call args ** exported call inputs args w physical prompt permission
{
  unfold Kernel.post call args w physical prompt permission;
  unfold Kernel.readers call args w physical permission;
  unfold Kernel.result call args w prompt;
  with p tokens hidden.
    assert A.pts_to args.Entry.states (Execution.physical p).State.states **
      A.pts_to args.Entry.convolution (Execution.physical p).State.convolution **
      A.pts_to args.Entry.keys (Execution.physical p).State.keys **
      A.pts_to args.Entry.values (Execution.physical p).State.values **
      A.pts_to_mask args.Entry.tokens tokens Memory.all_cells **
      A.pts_to_mask args.Entry.hidden hidden Memory.all_cells;
  outputs_ready call w prompt p tokens hidden;
  A.from_mask args.Entry.tokens;
  A.from_mask args.Entry.hidden;
  with token_values. assert args.Entry.tokens |-> token_values;
  with hidden_values. assert args.Entry.hidden |-> hidden_values;
  A.pts_to_len args.Entry.tokens;
  A.pts_to_len args.Entry.hidden;
  Execution.full_safety w prompt p;
  conditional_numerics call w prompt p;
  rewrite A.pts_to args.Entry.weights #permission w as A.pts_to (Input.data inputs.NativeInput.weights) #permission w;
  rewrite A.pts_to args.Entry.prompt #permission physical as A.pts_to (Input.data inputs.NativeInput.prompt) #permission physical;
  Packed.array_to_parameters inputs w permission;
  Inputs.array_to_prompt inputs physical prompt permission;
  fold Storage.inputs_owns call I64.v permission (Layout.parameter_values w)
    (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical);
  fold Shared.caller_precondition call I64.v permission (Layout.parameter_values w)
    (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical);
  Mem.array_to_slice args.Entry.tokens #1.0R #token_values;
  Mem.array_to_slice args.Entry.hidden #1.0R #hidden_values;
  drop_ (Mem.is_full_slice args.Entry.tokens call.decode_steps);
  drop_ (Mem.is_full_slice args.Entry.hidden (call.decode_steps * 1024));
  S.slice_length token_values;
  S.slice_length hidden_values;
  Adapters.slice_to_vector #I64.t #call.decode_steps args.Entry.tokens 0 token_values
    (output_addresses call args token_values).Storage.tokens.Storage.addresses
    (fun i -> S.index token_values i) 1.0R;
  fold Storage.tokens_owns I64.v 1.0R (output_addresses call args token_values).Storage.tokens
    (Execution.caller w prompt p p.Execution.count).D.returned.D.generated_tokens;
  Adapters.slice_to_matrix #bf16 #call.decode_steps #1024 args.Entry.hidden 0 hidden_values
    (output_addresses call args token_values).Storage.hidden
    (fun g i -> S.index (S.index
      (Execution.caller w prompt p p.Execution.count).D.returned.D.hidden_history g) i) 1.0R;
  fold Storage.history_owns I64.v (output_addresses call args token_values)
    (Execution.caller w prompt p p.Execution.count).D.returned;
  fold Shared.safety_caller_postcondition call I64.v (fun x -> x) permission
    (Layout.parameter_values w) (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical)
    (Execution.run w prompt p) (Execution.caller w prompt p p.Execution.count).D.returned
    (output_addresses call args token_values) (Execution.caller w prompt p p.Execution.count);
  fold exported call inputs args w physical prompt permission;
  fold private_arrays call args;
}
