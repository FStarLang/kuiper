module KPlay.Proof.SourceCaller
#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send
open Kuiper.Tensor.Runtime.Helpers

module RT = Kuiper.Tensor.Runtime
module Results = KPlay.Proof.SourceResults
module Outputs = KPlay.Proof.OutputAllocations
module Native = KPlay.Proof.Shared.Native
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge

noextract noeq
type buffers = {
  scratch: Results.scratch_handles;
  outputs: Native.output_handles
}

noextract noeq
type allocation =
  | Failed : error:RT.allocation_error -> allocation
  | Allocated : values:buffers -> allocation

inline_for_extraction noextract
let metadata (values:buffers) : prop =
  RT.metadata values.scratch.scores [256; 8] /\
  RT.metadata values.scratch.selected [256; 8] /\
  RT.metadata values.scratch.routes [256; 8] /\
  Outputs.metadata values.outputs

inline_for_extraction noextract
let owners (stream:Stream.stream_t) (values:buffers) : slprop =
  Native.output_owners stream values.outputs **
    Results.scratch_owners stream values.scratch

inline_for_extraction noextract
let contents (values:buffers) (frame:slprop) : slprop =
  Outputs.contents values.outputs ** Results.scratch_contents values.scratch frame

inline_for_extraction noextract
ghost
fn expose_outputs (values:buffers) (frame:slprop) ()
  requires Outputs.contents values.outputs ** Results.scratch_contents values.scratch frame
  ensures contents values frame
{
  fold contents values frame;
}

inline_for_extraction noextract
ghost
fn group_scratch (scratch:Results.scratch_handles) (frame:slprop) ()
  requires RT.contents scratch.routes **
    (RT.contents scratch.selected ** (RT.contents scratch.scores ** frame))
  ensures Results.scratch_contents scratch frame
{
  fold Results.scratch_contents scratch frame;
}

inline_for_extraction noextract
fn allocate_all
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns result:allocation
  ensures (
    match result with
    | Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | Allocated values ->
      owners stream values ** pure (metadata values) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (contents values frame))
{
  let scores = RT.empty RT.Float32 stream #epoch [256; 8] frame ();
  match scores {
    RT.Failed error -> { Failed error }
    RT.Allocated s -> {
      let selected = RT.empty RT.Int32 stream #epoch [256; 8] (RT.contents s ** frame) ();
      match selected {
        RT.Failed error -> {
          RT.release stream #epoch s frame ();
          Failed error
        }
        RT.Allocated f -> {
          let routes = RT.empty RT.Float32 stream #epoch [256; 8]
            (RT.contents f ** (RT.contents s ** frame)) ();
          match routes {
            RT.Failed error -> {
              RT.release stream #epoch f (RT.contents s ** frame) ();
              RT.release stream #epoch s frame ();
              Failed error
            }
            RT.Allocated p -> {
              let scratch:Results.scratch_handles = {
                Results.scores = s; Results.selected = f; Results.routes = p
              };
              assert (rewrites_to scratch {
                Results.scores = s; Results.selected = f; Results.routes = p
              });
              fold Results.scratch_owners stream scratch;
              Pledge.rewrite_pledge
                (RT.contents p ** (RT.contents f ** (RT.contents s ** frame)))
                (Results.scratch_contents scratch frame)
                (group_scratch scratch frame);
              let output = Outputs.allocate_outputs stream #epoch
                (Results.scratch_contents scratch frame) ();
              match output {
                Outputs.Failed error -> {
                  Results.release_scratch stream #epoch scratch frame ();
                  Failed error
                }
                Outputs.Allocated handles -> {
                  let values:buffers = {scratch = scratch; outputs = handles};
                  assert (rewrites_to values {scratch = scratch; outputs = handles});
                  fold owners stream values;
                  Pledge.rewrite_pledge
                    (Outputs.contents handles ** Results.scratch_contents scratch frame)
                    (contents values frame)
                    (expose_outputs values frame);
                  Allocated values
                }
              }
            }
          }
        }
      }
    }
  }
}
