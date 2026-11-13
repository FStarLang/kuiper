module KPlay.Proof.Optimized.Program

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
open Pulse.Lib.Send
module I64 = FStar.Int64
module RT = Kuiper.Tensor.Runtime
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module K = Kuiper.Kernel
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Caller = KPlay.Proof.Optimized.Caller
module Entry = KPlay.Proof.Optimized.Entry
module Descriptor = KPlay.Proof.Optimized.Descriptor
module Queue = KPlay.Proof.Optimized.Queue
module Layout = KPlay.Proof.Optimized.Layout

noextract
noeq type outcome =
  | Failed : error:RT.allocation_error -> outcome
  | Returned : tokens:RT.tensor I64.t -> hidden:RT.tensor bf16 -> outcome

inline_for_extraction noextract
let pre (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length) (prompt:D.vec (D.token 248320) call.prompt_length)
  (permission:perm) : slprop =
  Stream.stream_live stream ** RT.runtime stream ** Epoch.epoch_live stream epoch **
  Pledge.pledge0 (Epoch.epoch_done stream epoch) (Queue.source_input call inputs w physical prompt permission)

inline_for_extraction noextract
let post (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t) (w:Layout.packed)
  (physical:lseq I64.t call.prompt_length) (prompt:D.vec (D.token 248320) call.prompt_length)
  (permission:perm) (result:outcome) : slprop =
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
  | Failed _ -> Epoch.epoch_live stream epoch **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (Queue.source_input call inputs w physical prompt permission)
  | Returned tokens hidden ->
      Epoch.epoch_live stream (Epoch.epoch_next epoch) ** RT.owner stream tokens ** RT.owner stream hidden **
      Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
        (on gpu_loc (Queue.result_cells call inputs tokens hidden w physical prompt permission)) **
      pure (RT.metadata tokens [call.decode_steps] /\ RT.metadata hidden [call.decode_steps; 1024]))

let argument_binding (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call)
  : Lemma
      (requires Caller.shapes call b /\ Entry.bound call inputs b args)
      (ensures Descriptor.device_arguments call args) =
  Input.view_metadata inputs.NativeInput.weights;
  Input.view_metadata inputs.NativeInput.prompt

noextract
fn scoped (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (steps:I64.t{I64.v steps == call.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#w:erased Layout.packed) (#physical:erased (lseq I64.t call.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.prompt_length)) (#permission:perm)
  preserves cpu ** Input.reference inputs.NativeInput.weights ** Input.reference inputs.NativeInput.prompt
  requires pre call inputs stream epoch w physical prompt permission
  returns result:outcome
  ensures post call inputs stream epoch w physical prompt permission result
{
  unfold pre call inputs stream epoch w physical prompt permission;
  let allocation = Caller.allocate call stream #epoch (Queue.source_input call inputs w physical prompt permission) ();
  match allocation {
    Caller.Failed error -> {
      fold post call inputs stream epoch w physical prompt permission (Failed error);
      Failed error
    }
    Caller.Allocated buffers -> {
      let args = Entry.query_arguments call inputs stream buffers steps;
      argument_binding call inputs buffers args;
      Queue.prepare_pledge call inputs buffers args stream epoch w physical prompt permission ();
      let descriptor = Descriptor.descriptor call args #w #physical #prompt #permission;
      K.launch_kernel_full descriptor stream #epoch;
      Queue.export_pledge call inputs buffers args stream (Epoch.epoch_next epoch) w physical prompt permission ();
      unfold Caller.owners stream call buffers;
      fold Entry.private_owners stream call buffers;
      Entry.release_private call stream #(Epoch.epoch_next epoch) buffers
        (on gpu_loc (Queue.result_cells call inputs buffers.Caller.tokens buffers.Caller.hidden w physical prompt permission)) ();
      fold post call inputs stream epoch w physical prompt permission (Returned buffers.Caller.tokens buffers.Caller.hidden);
      Returned buffers.Caller.tokens buffers.Caller.hidden
    }
  }
}

noextract
fn apply (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (steps:I64.t{I64.v steps == call.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#w:erased Layout.packed) (#physical:erased (lseq I64.t call.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.prompt_length)) (#permission:perm)
  preserves cpu
  requires pre call inputs stream epoch w physical prompt permission
  returns result:outcome
  ensures post call inputs stream epoch w physical prompt permission result
{
  Input.with_parameter #bf16 #outcome inputs.NativeInput.weights
    (pre call inputs stream epoch w physical prompt permission)
    (post call inputs stream epoch w physical prompt permission)
    fn () {
      Input.with_parameter #I64.t #outcome inputs.NativeInput.prompt
        (Input.reference inputs.NativeInput.weights ** pre call inputs stream epoch w physical prompt permission)
        (fun result -> Input.reference inputs.NativeInput.weights **
          post call inputs stream epoch w physical prompt permission result)
        fn () { scoped call inputs steps stream #epoch #w #physical #prompt #permission }
    }
}
