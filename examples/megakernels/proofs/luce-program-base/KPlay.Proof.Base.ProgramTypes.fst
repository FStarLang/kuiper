module KPlay.Proof.Base.ProgramTypes

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper

module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module RT = Kuiper.Tensor.Runtime
module NativeInput = KPlay.Proof.NativeInput
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Queue = KPlay.Proof.Base.KernelQueue
module Transfer = KPlay.Proof.Base.Transfer

noextract
noeq type outcome =
  | Failed : error:RT.allocation_error -> outcome
  | Returned : tokens:RT.tensor I64.t -> hidden:RT.tensor BF.t -> outcome

inline_for_extraction noextract
let pre (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) : slprop =
  Stream.stream_live stream ** RT.runtime stream ** Epoch.epoch_live stream epoch **
  Pledge.pledge0 (Epoch.epoch_done stream epoch) (Queue.source_input call inputs values physical logical fraction)

inline_for_extraction noextract
let post (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (values:Transfer.packed) (physical:lseq I64.t call.Model.prompt_length)
  (logical:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm) (result:outcome) : slprop =
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
  | Failed _ ->
      Epoch.epoch_live stream epoch **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (Queue.source_input call inputs values physical logical fraction)
  | Returned tokens hidden ->
      Epoch.epoch_live stream (Epoch.epoch_next epoch) ** RT.owner stream tokens ** RT.owner stream hidden **
      Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
        (on Kuiper.Locs.gpu_loc (Queue.result_cells call inputs tokens hidden values physical logical fraction)) **
      pure (RT.metadata tokens [call.Model.decode_steps] /\ RT.metadata hidden [call.Model.decode_steps; 1024]))
