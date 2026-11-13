module KPlay.Proof.BaseReturn

#lang-pulse

open Kuiper
open Pulse.Lib.Send
open Kuiper.Tensor.Runtime.Helpers

module PA = Pulse.Lib.Array
module S = FStar.Seq
module RT = Kuiper.Tensor.Runtime
module Results = Kuiper.Tensor.Results
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module Pledge = Pulse.Lib.Pledge
module Native = KPlay.Proof.Shared.Native
module Caller = KPlay.Proof.BaseCaller
module SourceNative = KPlay.Proof.BaseNative

inline_for_extraction noextract
let vector = Results.vector4 i32 i64 f32 i32

inline_for_extraction noextract
let outcome = Results.construction_result i32 i64 f32 i32

inline_for_extraction noextract
let handles (value:vector) : GTot Native.output_handles = {
  Native.demand = Results.first value;
  Native.indices = Results.second value;
  Native.probabilities = Results.third value;
  Native.survival = Results.fourth value
}

inline_for_extraction noextract
let handles_are_exact (before:Native.output_handles) (value:vector)
  : Lemma
      (requires
        Results.first value == before.demand /\
        Results.second value == before.indices /\
        Results.third value == before.probabilities /\
        Results.fourth value == before.survival)
      (ensures handles value == before)
  = ()

inline_for_extraction noextract
let after_return
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (xs:SourceNative.token_values) (ws:SourceNative.weight_values)
  (fp fw:perm) (frame:slprop) (result:outcome) : slprop =
  match result with
  | Results.Failed _ ->
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (Native.input_cells tokens weights fp fw xs ws ** frame)
  | Results.Constructed value ->
    Native.output_owners stream (handles value) **
    pure (Caller.metadata (handles value)) **
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (SourceNative.payload_post tokens weights xs ws fp fw (handles value) ** frame)

inline_for_extraction noextract
ghost
fn forget_cell_values (#a:Type0) (value:RT.tensor a) (values:S.seq a)
  requires on gpu_loc (PA.pts_to (RT.data value) values)
  ensures RT.contents value
{
  Pulse.Lib.Core.intro_exists
    (fun values -> on gpu_loc (PA.pts_to (RT.data value) values)) values;
  fold RT.contents value;
}

(* Forget only the output observations, never the pending input/frame cells. *)
inline_for_extraction noextract
ghost
fn release_payload
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (xs:SourceNative.token_values) (ws:SourceNative.weight_values)
  (fp fw:perm) (outputs:Native.output_handles) (frame:slprop) ()
  requires SourceNative.payload_post tokens weights xs ws fp fw outputs ** frame
  ensures RT.contents outputs.survival **
    (RT.contents outputs.probabilities **
      (RT.contents outputs.indices **
        (RT.contents outputs.demand **
          (Native.input_cells tokens weights fp fw xs ws ** frame))))
{
  unfold SourceNative.payload_post tokens weights xs ws fp fw outputs;
  unfold SourceNative.output_post outputs xs ws;
  with values. assert Native.output_cells outputs values;
  unfold Native.output_cells outputs values;
  forget_cell_values outputs.demand values.Native.demand_values;
  forget_cell_values outputs.indices values.Native.index_values;
  forget_cell_values outputs.probabilities values.Native.probability_values;
  forget_cell_values outputs.survival values.Native.survival_values;
}

(* base_s.cu:86: the source return and exceptional output-local unwinding.
   Its ownership and payload premises are supplied by the actual launch. *)
inline_for_extraction noextract
fn finish_source_return
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:SourceNative.token_argument) (weights:SourceNative.weight_argument)
  (#xs:erased SourceNative.token_values) (#ws:erased SourceNative.weight_values)
  (#fp #fw:perm) (outputs:Native.output_handles)
  (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Native.output_owners stream outputs ** pure (Caller.metadata outputs) **
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (SourceNative.payload_post tokens weights xs ws fp fw outputs ** frame)
  returns result:outcome
  ensures after_return stream epoch tokens weights xs ws fp fw frame result
{
  unfold Native.output_owners stream outputs;
  let result = Results.return_four stream #epoch
    outputs.demand outputs.indices outputs.probabilities outputs.survival
    (SourceNative.payload_post tokens weights xs ws fp fw outputs ** frame) ();
  match result {
    Results.Failed error -> {
      Pledge.rewrite_pledge
        (SourceNative.payload_post tokens weights xs ws fp fw outputs ** frame)
        (RT.contents outputs.survival **
          (RT.contents outputs.probabilities **
            (RT.contents outputs.indices **
              (RT.contents outputs.demand **
                (Native.input_cells tokens weights fp fw xs ws ** frame)))))
        (release_payload tokens weights xs ws fp fw outputs frame);
      RT.release stream #epoch outputs.survival
        (RT.contents outputs.probabilities **
          (RT.contents outputs.indices **
            (RT.contents outputs.demand **
              (Native.input_cells tokens weights fp fw xs ws ** frame)))) ();
      RT.release stream #epoch outputs.probabilities
        (RT.contents outputs.indices **
          (RT.contents outputs.demand **
            (Native.input_cells tokens weights fp fw xs ws ** frame))) ();
      RT.release stream #epoch outputs.indices
        (RT.contents outputs.demand **
          (Native.input_cells tokens weights fp fw xs ws ** frame)) ();
      RT.release stream #epoch outputs.demand
        (Native.input_cells tokens weights fp fw xs ws ** frame) ();
      fold after_return stream epoch tokens weights xs ws fp fw frame
        (Results.Failed error);
      Results.Failed error
    }
    Results.Constructed value -> {
      handles_are_exact outputs value;
      fold Native.output_owners stream outputs;
      rewrite Native.output_owners stream outputs
        as Native.output_owners stream (handles value);
      rewrite Pledge.pledge0 (Epoch.epoch_done stream epoch)
        (SourceNative.payload_post tokens weights xs ws fp fw outputs ** frame)
        as Pledge.pledge0 (Epoch.epoch_done stream epoch)
          (SourceNative.payload_post tokens weights xs ws fp fw (handles value) ** frame);
      fold after_return stream epoch tokens weights xs ws fp fw frame
        (Results.Constructed value);
      Results.Constructed value
    }
  }
}
