module KPlay.Proof.Optimized.Caller

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
  states: RT.tensor FP.t;
  convolution: RT.tensor FP.t;
  keys: RT.tensor BF.t;
  values: RT.tensor BF.t;
  tokens: RT.tensor I64.t;
  hidden: RT.tensor BF.t;
}

inline_for_extraction noextract
let shapes (call:Model.invocation) (b:buffers call) : prop =
  RT.metadata b.states [18; 16; 128; 128] /\
  RT.metadata b.convolution [18; 6144; 4] /\
  RT.metadata b.keys [6; 2; 2048; 256] /\
  RT.metadata b.values [6; 2; 2048; 256] /\
  RT.metadata b.tokens [call.decode_steps] /\
  RT.metadata b.hidden [call.decode_steps; 1024]

inline_for_extraction noextract
let owners (stream:Stream.stream_t) (call:Model.invocation) (b:buffers call) : slprop =
  RT.owner stream b.states ** RT.owner stream b.convolution **
  RT.owner stream b.keys ** RT.owner stream b.values **
  RT.owner stream b.tokens ** RT.owner stream b.hidden

inline_for_extraction noextract
let pending_contents (call:Model.invocation) (b:buffers call) (frame:slprop) : slprop =
  RT.contents b.hidden ** (RT.contents b.tokens ** (RT.contents b.values **
    (RT.contents b.keys ** (RT.contents b.convolution ** (RT.contents b.states ** frame)))))

noextract
noeq type allocation (call:Model.invocation) =
  | Failed : error:RT.allocation_error -> allocation call
  | Allocated : value:buffers call -> allocation call

(* reduced.cu:382-388, including reverse native destruction on exceptional exits. *)
noextract
fn allocate (call:Model.invocation) (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns outcome:allocation call
  ensures (
    match outcome with
    | Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | Allocated b -> owners stream call b ** pure (shapes call b) **
        Pledge.pledge0 (Epoch.epoch_done stream epoch) (pending_contents call b frame))
{
  let state_result = RT.empty RT.Float32 stream #epoch [18; 16; 128; 128] frame ();
  match state_result {
    RT.Failed error -> { Failed error }
    RT.Allocated states -> {
      let conv_result = RT.empty RT.Float32 stream #epoch [18; 6144; 4]
        (RT.contents states ** frame) ();
      match conv_result {
        RT.Failed error -> {
          RT.release stream #epoch states frame ();
          Failed error
        }
        RT.Allocated convolution -> {
          let key_result = RT.empty RT.BFloat16 stream #epoch [6; 2; 2048; 256]
            (RT.contents convolution ** (RT.contents states ** frame)) ();
          match key_result {
            RT.Failed error -> {
              RT.release stream #epoch convolution (RT.contents states ** frame) ();
              RT.release stream #epoch states frame ();
              Failed error
            }
            RT.Allocated keys -> {
              let value_result = RT.empty RT.BFloat16 stream #epoch [6; 2; 2048; 256]
                (RT.contents keys ** (RT.contents convolution ** (RT.contents states ** frame))) ();
              match value_result {
                RT.Failed error -> {
                  RT.release stream #epoch keys (RT.contents convolution ** (RT.contents states ** frame)) ();
                  RT.release stream #epoch convolution (RT.contents states ** frame) ();
                  RT.release stream #epoch states frame ();
                  Failed error
                }
                RT.Allocated values -> {
                  let token_result = RT.empty RT.Int64 stream #epoch [call.decode_steps]
                    (RT.contents values ** (RT.contents keys **
                      (RT.contents convolution ** (RT.contents states ** frame)))) ();
                  match token_result {
                    RT.Failed error -> {
                      RT.release stream #epoch values (RT.contents keys **
                        (RT.contents convolution ** (RT.contents states ** frame))) ();
                      RT.release stream #epoch keys (RT.contents convolution ** (RT.contents states ** frame)) ();
                      RT.release stream #epoch convolution (RT.contents states ** frame) ();
                      RT.release stream #epoch states frame ();
                      Failed error
                    }
                    RT.Allocated tokens -> {
                      let hidden_result = RT.empty RT.BFloat16 stream #epoch [call.decode_steps; 1024]
                        (RT.contents tokens ** (RT.contents values ** (RT.contents keys **
                          (RT.contents convolution ** (RT.contents states ** frame))))) ();
                      match hidden_result {
                        RT.Failed error -> {
                          RT.release stream #epoch tokens (RT.contents values ** (RT.contents keys **
                            (RT.contents convolution ** (RT.contents states ** frame)))) ();
                          RT.release stream #epoch values (RT.contents keys **
                            (RT.contents convolution ** (RT.contents states ** frame))) ();
                          RT.release stream #epoch keys (RT.contents convolution ** (RT.contents states ** frame)) ();
                          RT.release stream #epoch convolution (RT.contents states ** frame) ();
                          RT.release stream #epoch states frame ();
                          Failed error
                        }
                        RT.Allocated hidden -> {
                          fold owners stream call ({states; convolution; keys; values; tokens; hidden});
                          assert pure (shapes call ({states; convolution; keys; values; tokens; hidden}));
                          Allocated {states; convolution; keys; values; tokens; hidden}
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
  }
}
