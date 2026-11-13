module KPlay.Proof.Base.Entry

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send

module A = Pulse.Lib.Array
module KL = Kuiper.Locs
module RT = Kuiper.Tensor.Runtime
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module Caller = KPlay.Proof.Base.Caller
module Model = KPlay.Proof.Shared.Model
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64

noextract
noeq type arguments (call:Model.invocation) = {
  weight: A.larray BF.t 752393024;
  prompt: A.larray I64.t call.prompt_length;
  state: A.larray FP.t 4718592;
  convolution: A.larray BF.t 442368;
  cache: A.larray BF.t 12582912;
  tokens: A.larray I64.t call.decode_steps;
  hidden: A.larray BF.t (call.decode_steps * 1024);
  prompt_length: value:I64.t{I64.v value == call.prompt_length};
  steps: value:I64.t{I64.v value == call.decode_steps};
}

inline_for_extraction noextract
let bound (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (b:Caller.buffers call) (args:arguments call) : prop =
  args.weight == Input.data inputs.weights /\
  args.prompt == Input.data inputs.prompt /\
  args.state == RT.data b.state /\
  args.convolution == RT.data b.convolution /\
  args.cache == RT.data b.cache /\
  args.tokens == RT.data b.tokens /\
  args.hidden == RT.data b.hidden

(* Native parameter references are retained by the enclosing caller, not
   discharged at the pointer-query boundary. None of these queries enqueues work. *)
noextract
fn query_arguments
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (stream:Stream.stream_t) (b:Caller.buffers call)
  (steps:I64.t{I64.v steps == call.decode_steps})
  preserves KL.cpu ** Input.reference inputs.weights ** Input.reference inputs.prompt
  preserves Caller.owners stream call b
  requires pure (Caller.shapes call b)
  returns args:arguments call
  ensures pure (bound call inputs b args)
{
  unfold Caller.owners stream call b;
  let weight = Input.data_ptr inputs.weights;
  let prompt = Input.data_ptr inputs.prompt;
  let state = RT.data_ptr stream b.state;
  let convolution = RT.data_ptr stream b.convolution;
  let cache = RT.data_ptr stream b.cache;
  let tokens = RT.data_ptr stream b.tokens;
  let hidden = RT.data_ptr stream b.hidden;
  let prompt_length = Input.numel inputs.prompt;
  Input.view_metadata inputs.weights;
  Input.view_metadata inputs.prompt;
  fold Caller.owners stream call b;
  {weight; prompt; state; convolution; cache; tokens; hidden; prompt_length; steps}
}

inline_for_extraction noextract
let private_owners (stream:Stream.stream_t) (call:Model.invocation) (b:Caller.buffers call)
  : slprop =
  RT.owner stream b.cache ** RT.owner stream b.convolution ** RT.owner stream b.state

inline_for_extraction noextract
let private_contents (call:Model.invocation) (b:Caller.buffers call) (frame:slprop)
  : slprop =
  RT.contents b.cache ** (RT.contents b.convolution ** (RT.contents b.state ** frame))

(* kernel.cu:411-412 transfers both result owners and then destroys the three
   private buffers in reverse declaration order, under the real queue tail. *)
noextract
fn release_private
  (call:Model.invocation) (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (b:Caller.buffers call) (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires private_owners stream call b
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) (private_contents call b frame)
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
{
  unfold private_owners stream call b;
  RT.release stream #epoch b.cache (RT.contents b.convolution ** (RT.contents b.state ** frame)) ();
  RT.release stream #epoch b.convolution (RT.contents b.state ** frame) ();
  RT.release stream #epoch b.state frame ();
}
