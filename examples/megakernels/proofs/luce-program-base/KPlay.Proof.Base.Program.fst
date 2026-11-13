module KPlay.Proof.Base.Program

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Send
open FStar.Ghost

module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module RT = Kuiper.Tensor.Runtime
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module Kernel = Kuiper.Kernel
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Caller = KPlay.Proof.Base.Caller
module Entry = KPlay.Proof.Base.Entry
module Controls = KPlay.Proof.Base.KernelControls
module Descriptor = KPlay.Proof.Base.KernelDescriptor
module Layout = KPlay.Proof.Base.KernelSetup
module Storage = KPlay.Proof.Base.KernelStorage
module Queue = KPlay.Proof.Base.KernelQueue
module Transfer = KPlay.Proof.Base.Transfer
module API = KPlay.Proof.Base.ProgramTypes

inline_for_extraction noextract
let outcome = API.outcome

inline_for_extraction noextract
let pre = API.pre

inline_for_extraction noextract
let post = API.post

inline_for_extraction noextract
let argument_binding (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (b:Caller.buffers call) (args:Entry.arguments call)
  : Lemma
      (requires Caller.shapes call b /\ Entry.bound call inputs b args)
      (ensures Layout.device_arguments call args /\ Storage.input_binding call inputs args) =
  Input.view_metadata inputs.NativeInput.weights;
  Input.view_metadata inputs.NativeInput.prompt

noextract
fn scoped
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (steps:I64.t{I64.v steps == call.Model.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#values:erased Transfer.packed) (#physical:erased (lseq I64.t call.Model.prompt_length))
  (#logical:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#fraction:perm)
  preserves cpu ** Input.reference inputs.NativeInput.weights ** Input.reference inputs.NativeInput.prompt
  requires pre call inputs stream epoch values physical logical fraction
  returns result:outcome
  ensures post call inputs stream epoch values physical logical fraction result
{
  unfold pre call inputs stream epoch values physical logical fraction;
  unfold API.pre call inputs stream epoch values physical logical fraction;
  let allocation = Caller.allocate call stream #epoch (Queue.source_input call inputs values physical logical fraction) ();
  match allocation {
    Caller.Failed error -> {
      fold API.post call inputs stream epoch values physical logical fraction (API.Failed error);
      fold post call inputs stream epoch values physical logical fraction (API.Failed error);
      API.Failed error
    }
    Caller.Allocated buffers -> {
      let args = Entry.query_arguments call inputs stream buffers steps;
      argument_binding call inputs buffers args;
      let refs = Controls.allocate call #values #logical;
      Queue.prepare_pledge call refs inputs buffers args stream epoch values physical logical fraction ();
      let kernel = Descriptor.descriptor call refs args #(Storage.tensor values) #physical #logical #fraction;
      Kernel.launch_kernel_full kernel stream #epoch;
      Queue.export_pledge call inputs buffers args stream (Epoch.epoch_next epoch) values physical logical fraction ();
      unfold Caller.owners stream call buffers;
      fold Entry.private_owners stream call buffers;
      Entry.release_private call stream #(Epoch.epoch_next epoch) buffers
        (on Kuiper.Locs.gpu_loc (Queue.result_cells call inputs buffers.Caller.tokens buffers.Caller.hidden values physical logical fraction)) ();
      fold API.post call inputs stream epoch values physical logical fraction (API.Returned buffers.Caller.tokens buffers.Caller.hidden);
      fold post call inputs stream epoch values physical logical fraction (API.Returned buffers.Caller.tokens buffers.Caller.hidden);
      API.Returned buffers.Caller.tokens buffers.Caller.hidden
    }
  }
}

noextract
fn apply
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (steps:I64.t{I64.v steps == call.Model.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#values:erased Transfer.packed) (#physical:erased (lseq I64.t call.Model.prompt_length))
  (#logical:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#fraction:perm)
  preserves cpu
  requires pre call inputs stream epoch values physical logical fraction
  returns result:outcome
  ensures post call inputs stream epoch values physical logical fraction result
{
  Input.with_parameter #BF.t #outcome inputs.NativeInput.weights
    (pre call inputs stream epoch values physical logical fraction)
    (post call inputs stream epoch values physical logical fraction)
    fn () {
      Input.with_parameter #I64.t #outcome inputs.NativeInput.prompt
        (Input.reference inputs.NativeInput.weights ** pre call inputs stream epoch values physical logical fraction)
        (fun result -> Input.reference inputs.NativeInput.weights ** post call inputs stream epoch values physical logical fraction result)
        fn () {
          scoped call inputs steps stream #epoch #values #physical #logical #fraction
        }
    }
}
