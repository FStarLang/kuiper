module KPlay.Proof.Base.Caller

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send

module RT = Kuiper.Tensor.Runtime
module Helpers = Kuiper.Tensor.Runtime.Helpers
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module Model = KPlay.Proof.Shared.Model

noextract
noeq type buffers (call:Model.invocation) = {
  state: RT.tensor FP.t;
  convolution: RT.tensor BF.t;
  cache: RT.tensor BF.t;
  tokens: RT.tensor I64.t;
  hidden: RT.tensor BF.t;
}

inline_for_extraction noextract
let shapes (call:Model.invocation) (b:buffers call) : prop =
  RT.metadata b.state [4718592] /\
  RT.metadata b.convolution [442368] /\
  RT.metadata b.cache [12582912] /\
  RT.metadata b.tokens [call.decode_steps] /\
  RT.metadata b.hidden [call.decode_steps; 1024]

inline_for_extraction noextract
let owners (stream:Stream.stream_t) (call:Model.invocation) (b:buffers call) : slprop =
  RT.owner stream b.state ** RT.owner stream b.convolution **
  RT.owner stream b.cache ** RT.owner stream b.tokens ** RT.owner stream b.hidden

inline_for_extraction noextract
let pending_contents (call:Model.invocation) (b:buffers call) (frame:slprop) : slprop =
  RT.contents b.hidden ** (RT.contents b.tokens ** (RT.contents b.cache **
    (RT.contents b.convolution ** (RT.contents b.state ** frame))))

noextract
noeq type allocation (call:Model.invocation) =
  | Failed : error:RT.allocation_error -> allocation call
  | Allocated : value:buffers call -> allocation call

(* Source lines 402-406, including reverse destruction on exceptional exits.
   No allocation branch grants cell contents before its existing dependency. *)
noextract
fn allocate
  (call:Model.invocation) (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns outcome:allocation call
  ensures (
    match outcome with
    | Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | Allocated b ->
      owners stream call b ** pure (shapes call b) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (pending_contents call b frame))
{
  let state_result = RT.empty RT.Float32 stream #epoch [4718592] frame ();
  match state_result {
    RT.Failed error -> { Failed error }
    RT.Allocated state -> {
      let convolution_result = RT.empty RT.BFloat16 stream #epoch [442368]
        (RT.contents state ** frame) ();
      match convolution_result {
        RT.Failed error -> {
          RT.release stream #epoch state frame ();
          Failed error
        }
        RT.Allocated convolution -> {
          let cache_result = RT.empty RT.BFloat16 stream #epoch [12582912]
            (RT.contents convolution ** (RT.contents state ** frame)) ();
          match cache_result {
            RT.Failed error -> {
              RT.release stream #epoch convolution (RT.contents state ** frame) ();
              RT.release stream #epoch state frame ();
              Failed error
            }
            RT.Allocated cache -> {
              let token_result = RT.empty RT.Int64 stream #epoch [call.decode_steps]
                (RT.contents cache ** (RT.contents convolution ** (RT.contents state ** frame))) ();
              match token_result {
                RT.Failed error -> {
                  RT.release stream #epoch cache
                    (RT.contents convolution ** (RT.contents state ** frame)) ();
                  RT.release stream #epoch convolution (RT.contents state ** frame) ();
                  RT.release stream #epoch state frame ();
                  Failed error
                }
                RT.Allocated tokens -> {
                  let hidden_result = RT.empty RT.BFloat16 stream #epoch [call.decode_steps; 1024]
                    (RT.contents tokens ** (RT.contents cache **
                      (RT.contents convolution ** (RT.contents state ** frame)))) ();
                  match hidden_result {
                    RT.Failed error -> {
                      RT.release stream #epoch tokens (RT.contents cache **
                        (RT.contents convolution ** (RT.contents state ** frame))) ();
                      RT.release stream #epoch cache
                        (RT.contents convolution ** (RT.contents state ** frame)) ();
                      RT.release stream #epoch convolution (RT.contents state ** frame) ();
                      RT.release stream #epoch state frame ();
                      Failed error
                    }
                    RT.Allocated hidden -> {
                      fold owners stream call ({state; convolution; cache; tokens; hidden});
                      assert pure (shapes call ({state; convolution; cache; tokens; hidden}));
                      Allocated {state; convolution; cache; tokens; hidden}
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
