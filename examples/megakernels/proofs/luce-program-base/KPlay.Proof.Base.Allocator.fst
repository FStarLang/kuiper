module KPlay.Proof.Base.Allocator
#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send
module RT = Kuiper.Tensor.Runtime
module Helpers = Kuiper.Tensor.Runtime.Helpers
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge
module A = Pulse.Lib.Array
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module I64 = FStar.Int64

let float32_kind : RT.dtype FP.t = RT.Float32
let bfloat16_kind : RT.dtype BF.t = RT.BFloat16
let int64_kind : RT.dtype I64.t = RT.Int64

let source_state_extent ()
  : Lemma (
      RT.numel [4718592] == 4718592 /\
      18 * 16 * 128 * 128 == 4718592 /\
      Kuiper.SizeT.v (Kuiper.Sized.size #FP.t) == 4 /\
      4718592 * 4 == 18874368) =
  FP.lem_sizeof ()

noextract
fn allocate_source_state
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns outcome:RT.allocation_result FP.t
  ensures (
    match outcome with
    | RT.Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | RT.Allocated value ->
      RT.owner stream value ** pure (RT.metadata value [4718592]) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (RT.contents value ** frame))
{
  RT.empty RT.Float32 stream #epoch [4718592] frame ()
}

noextract
fn typed_pointer
  (stream:Stream.stream_t) (value:RT.tensor FP.t)
  preserves KL.cpu ** RT.owner stream value
  returns address:A.array FP.t
  ensures pure (address == RT.data value)
{
  RT.data_ptr stream value
}

noextract
fn release_with_exact_frame
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t) (value:RT.tensor FP.t)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires RT.owner stream value **
    Pledge.pledge0 (Epoch.epoch_done stream epoch) (RT.contents value ** frame)
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
{
  RT.release stream #epoch value frame ();
}

noextract
ghost
fn access_after_real_completion
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t) (value:RT.tensor FP.t)
  preserves Epoch.epoch_done stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) (RT.contents value)
  ensures RT.contents value
{
  Pledge.redeem_pledge emp_inames (Epoch.epoch_done stream epoch) (RT.contents value);
}
