module KPlay.Proof.SourceResults
#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send
open FStar.Ghost
open Kuiper.Tensor.Runtime.Helpers

module RT = Kuiper.Tensor.Runtime
module R = Kuiper.Tensor.Results
module Input = Kuiper.Tensor.Input
module Native = KPlay.Proof.Shared.Native
module Capacity = KPlay.Proof.Shared.Capacity
module KL = Kuiper.Locs
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module I32 = FStar.Int32
module I64 = FStar.Int64
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16

noextract type result_vector = R.vector4 I32.t I64.t F32.t I32.t
noextract type result_outcome = R.construction_result I32.t I64.t F32.t I32.t

inline_for_extraction noextract
let vector_handles (value:result_vector) : GTot Native.output_handles = {
  demand = R.first value;
  indices = R.second value;
  probabilities = R.third value;
  survival = R.fourth value
}

inline_for_extraction noextract
let matches (handles:Native.output_handles) (value:result_vector) : prop =
  R.first value == handles.demand /\ R.second value == handles.indices /\
  R.third value == handles.probabilities /\ R.fourth value == handles.survival

inline_for_extraction noextract
let output_binding_preserved
  (handles:Native.output_handles) (values:Native.output_values)
  (out:Capacity.public_outputs Capacity.Smoke) (value:result_vector)
  : Lemma
      (requires matches handles value /\ Native.output_binding handles values out)
      (ensures Native.output_binding (vector_handles value) values out)
  = ()

noextract noeq
type scratch_handles = {
  scores: RT.tensor F32.t;
  selected: RT.tensor I32.t;
  routes: RT.tensor F32.t
}

inline_for_extraction noextract
let scratch_owners (stream:Stream.stream_t) (scratch:scratch_handles) : slprop =
  RT.owner stream scratch.scores **
  RT.owner stream scratch.selected **
  RT.owner stream scratch.routes

inline_for_extraction noextract
let scratch_contents (scratch:scratch_handles) (frame:slprop) : slprop =
  RT.contents scratch.routes **
    (RT.contents scratch.selected ** (RT.contents scratch.scores ** frame))

inline_for_extraction noextract
instance send_output_cells (handles:Native.output_handles) (values:Native.output_values)
  : is_send (Native.output_cells handles values) = FStar.Tactics.Typeclasses.solve

inline_for_extraction noextract
ghost
fn forget_output_values
  (handles:Native.output_handles) (values:Native.output_values) (frame:slprop) ()
  requires Native.output_cells handles values ** frame
  ensures RT.contents handles.survival **
    (RT.contents handles.probabilities **
      (RT.contents handles.indices ** (RT.contents handles.demand ** frame)))
{
  unfold Native.output_cells handles values;
  fold RT.contents handles.survival;
  fold RT.contents handles.probabilities;
  fold RT.contents handles.indices;
  fold RT.contents handles.demand;
}

inline_for_extraction noextract
ghost
fn expose_scratch (scratch:scratch_handles) (frame:slprop) ()
  requires scratch_contents scratch frame
  ensures RT.contents scratch.routes **
    (RT.contents scratch.selected ** (RT.contents scratch.scores ** frame))
{
  unfold scratch_contents scratch frame;
}

inline_for_extraction noextract
ghost
fn frame_output_values
  (handles:Native.output_handles) (values:Native.output_values)
  (scratch:scratch_handles) (frame:slprop) ()
  requires Native.output_cells handles values ** scratch_contents scratch frame
  ensures scratch_contents scratch (Native.output_cells handles values ** frame)
{
  unfold scratch_contents scratch frame;
  fold scratch_contents scratch (Native.output_cells handles values ** frame);
}

inline_for_extraction noextract
fn release_scratch
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (scratch:scratch_handles) (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires scratch_owners stream scratch **
    Pledge.pledge0 (Epoch.epoch_done stream epoch) (scratch_contents scratch frame)
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
{
  unfold scratch_owners stream scratch;
  Pledge.rewrite_pledge (scratch_contents scratch frame)
    (RT.contents scratch.routes **
      (RT.contents scratch.selected ** (RT.contents scratch.scores ** frame)))
    (expose_scratch scratch frame);
  RT.release stream #epoch scratch.routes
    (RT.contents scratch.selected ** (RT.contents scratch.scores ** frame)) ();
  RT.release stream #epoch scratch.selected (RT.contents scratch.scores ** frame) ();
  RT.release stream #epoch scratch.scores frame ();
}

(* The post-launch suffix at optimized_s.cu:148, inside both input scopes.
   Its incoming resources must still be derived from the seven allocations
   and three launches; this is not an entry contract for the whole program. *)
inline_for_extraction noextract
fn source_return_suffix
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens weights:Input.argument BF16.t)
  (handles:Native.output_handles) (scratch:scratch_handles)
  (#values:erased Native.output_values)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch ** Input.reference tokens ** Input.reference weights
  requires Native.output_owners stream handles ** scratch_owners stream scratch **
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (Native.output_cells handles values ** scratch_contents scratch frame)
  returns outcome:result_outcome
  ensures (
    match outcome with
    | R.Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | R.Constructed value ->
      Native.output_owners stream handles ** pure (matches handles value) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch)
        (Native.output_cells handles values ** frame))
{
  unfold Native.output_owners stream handles;
  let outcome = R.return_four stream #epoch
    handles.demand handles.indices handles.probabilities handles.survival
    (Native.output_cells handles values ** scratch_contents scratch frame) ();
  match outcome {
    R.Failed _ -> {
      Pledge.rewrite_pledge
        (Native.output_cells handles values ** scratch_contents scratch frame)
        (RT.contents handles.survival **
          (RT.contents handles.probabilities **
            (RT.contents handles.indices **
              (RT.contents handles.demand ** scratch_contents scratch frame))))
        (forget_output_values handles values (scratch_contents scratch frame));
      RT.release stream #epoch handles.survival
        (RT.contents handles.probabilities **
          (RT.contents handles.indices **
            (RT.contents handles.demand ** scratch_contents scratch frame))) ();
      RT.release stream #epoch handles.probabilities
        (RT.contents handles.indices **
          (RT.contents handles.demand ** scratch_contents scratch frame)) ();
      RT.release stream #epoch handles.indices
        (RT.contents handles.demand ** scratch_contents scratch frame) ();
      RT.release stream #epoch handles.demand (scratch_contents scratch frame) ();
      release_scratch stream #epoch scratch frame ();
      outcome
    }
    R.Constructed value -> {
      fold Native.output_owners stream handles;
      Pledge.rewrite_pledge
        (Native.output_cells handles values ** scratch_contents scratch frame)
        (scratch_contents scratch (Native.output_cells handles values ** frame))
        (frame_output_values handles values scratch frame);
      release_scratch stream #epoch scratch (Native.output_cells handles values ** frame) ();
      outcome
    }
  }
}
