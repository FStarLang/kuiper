module KPlay.Proof.BaseCaller

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send
open Kuiper.Tensor.Runtime.Helpers

module RT = Kuiper.Tensor.Runtime
module Helpers = Kuiper.Tensor.Runtime.Helpers
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge
module Native = KPlay.Proof.Shared.Native

inline_for_extraction noextract noeq
type allocation =
  | Failed : error:RT.allocation_error -> allocation
  | Allocated : handles:Native.output_handles -> allocation

inline_for_extraction noextract
let metadata (handles:Native.output_handles) : prop =
  RT.metadata handles.demand [8] /\
  RT.metadata handles.indices [8; 32] /\
  RT.metadata handles.probabilities [8; 32] /\
  RT.metadata handles.survival [256; 8]

inline_for_extraction noextract
let contents (handles:Native.output_handles) : slprop =
  RT.contents handles.survival **
  RT.contents handles.probabilities **
  RT.contents handles.indices **
  RT.contents handles.demand

inline_for_extraction noextract
let bundle
  (d:RT.tensor FStar.Int32.t) (i:RT.tensor FStar.Int64.t)
  (p:RT.tensor Kuiper.Float32.t) (s:RT.tensor FStar.Int32.t)
  : Native.output_handles = {
  Native.demand = d;
  Native.indices = i;
  Native.probabilities = p;
  Native.survival = s
}

inline_for_extraction noextract
ghost
fn group_contents (handles:Native.output_handles) (frame:slprop) ()
  requires
    RT.contents handles.survival ** RT.contents handles.probabilities **
    RT.contents handles.indices ** RT.contents handles.demand ** frame
  ensures contents handles ** frame
{
  fold contents handles;
}

inline_for_extraction noextract
fn allocate_outputs
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns result:allocation
  ensures (
    match result with
    | Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | Allocated handles ->
      Native.output_owners stream handles ** pure (metadata handles) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (contents handles ** frame))
{
  let demand = RT.empty RT.Int32 stream #epoch [8] frame ();
  match demand {
    RT.Failed error -> { Failed error }
    RT.Allocated d -> {
      let indices = RT.empty RT.Int64 stream #epoch [8; 32]
        (RT.contents d ** frame) ();
      match indices {
        RT.Failed error -> {
          RT.release stream #epoch d frame ();
          Failed error
        }
        RT.Allocated i -> {
          let probabilities = RT.empty RT.Float32 stream #epoch [8; 32]
            (RT.contents i ** (RT.contents d ** frame)) ();
          match probabilities {
            RT.Failed error -> {
              RT.release stream #epoch i (RT.contents d ** frame) ();
              RT.release stream #epoch d frame ();
              Failed error
            }
            RT.Allocated p -> {
              let survival = RT.empty RT.Int32 stream #epoch [256; 8]
                (RT.contents p ** (RT.contents i ** (RT.contents d ** frame))) ();
              match survival {
                RT.Failed error -> {
                  RT.release stream #epoch p (RT.contents i ** (RT.contents d ** frame)) ();
                  RT.release stream #epoch i (RT.contents d ** frame) ();
                  RT.release stream #epoch d frame ();
                  Failed error
                }
                RT.Allocated s -> {
                  Pledge.rewrite_pledge
                    (RT.contents s ** RT.contents p ** RT.contents i ** RT.contents d ** frame)
                    (contents (bundle d i p s) ** frame)
                    (group_contents (bundle d i p s) frame);
                  fold Native.output_owners stream (bundle d i p s);
                  Allocated (bundle d i p s)
                }
              }
            }
          }
        }
      }
    }
  }
}

inline_for_extraction noextract
ghost
fn locate_frame_body (dependency:slprop) (host:loc_id)
  (payload frame:slprop) {| is_send payload, is_send frame |} ()
  requires in_same_process host ** payload ** frame
  ensures Pledge.pledge0 dependency (payload ** on host frame)
{
  is_send_intro_on frame host;
  drop_ (in_same_process host);
  Pledge.return_pledge dependency (payload ** on host frame) #FStar.Tactics.Typeclasses.solve;
}

inline_for_extraction noextract
ghost
fn locate_frame (dependency:slprop) (host:loc_id)
  (payload frame:slprop) {| is_send payload, is_send frame |} ()
  requires in_same_process host ** Pledge.pledge0 dependency (payload ** frame)
  ensures Pledge.pledge0 dependency (payload ** on host frame)
{
  Pledge.bind_pledge' (in_same_process host)
    (locate_frame_body dependency host payload frame);
}

inline_for_extraction noextract
ghost
fn restore_frame_body (dependency:slprop) (host:loc_id)
  (payload frame:slprop) {| is_send payload, is_send frame |} ()
  requires in_same_process host ** payload ** on host frame
  ensures Pledge.pledge0 dependency (payload ** frame)
{
  is_send_elim_on frame #FStar.Tactics.Typeclasses.solve #host;
  drop_ (in_same_process host);
  Pledge.return_pledge dependency (payload ** frame) #FStar.Tactics.Typeclasses.solve;
}

inline_for_extraction noextract
ghost
fn restore_frame (dependency:slprop) (host:loc_id)
  (payload frame:slprop) {| is_send payload, is_send frame |} ()
  requires in_same_process host ** Pledge.pledge0 dependency (payload ** on host frame)
  ensures Pledge.pledge0 dependency (payload ** frame)
{
  Pledge.bind_pledge' (in_same_process host)
    (restore_frame_body dependency host payload frame);
}
