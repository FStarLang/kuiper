module KPlay.Proof.SourceProgram
#lang-pulse

open Kuiper
open Pulse.Lib.Send
open Kuiper.Tensor.Runtime.Helpers

module RT = Kuiper.Tensor.Runtime
module R = Kuiper.Tensor.Results
module Input = Kuiper.Tensor.Input
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module Pledge = Pulse.Lib.Pledge
module Kernel = Kuiper.Kernel
module Native = KPlay.Proof.Shared.Native
module Shared = KPlay.SharedSpec
module Layout = KPlay.Proof.SourceLayouts
module Results = KPlay.Proof.SourceResults
module Caller = KPlay.Proof.SourceCaller
module Device = KPlay.Proof.SourceDevice
module Launch = KPlay.Proof.SourceLaunch
module Bind = KPlay.Proof.NativeBindings
module Steps = KPlay.Proof.RouteSteps

inline_for_extraction noextract
let advanced (epoch:Epoch.epoch_t) : Epoch.epoch_t =
  Epoch.epoch_next (Epoch.epoch_next (Epoch.epoch_next epoch))

inline_for_extraction noextract
fn launch_program
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:Bind.token_argument) (weights:Bind.weight_argument)
  (values:Caller.buffers{Caller.metadata values})
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (#fp #fw:perm)
  (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream **
    Input.reference tokens ** Input.reference weights ** Caller.owners stream values
  requires Epoch.epoch_live stream epoch **
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (Caller.contents values (Native.input_cells tokens weights fp fw x w ** frame)) **
    pure (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))
  ensures Epoch.epoch_live stream (advanced epoch) **
    Pledge.pledge0 (Epoch.epoch_done stream (advanced epoch))
      (Bind.output_post values.outputs x w **
        Results.scratch_contents values.scratch
          (Native.input_cells tokens weights fp fw x w ** frame))
{
  let a:erased Device.arrays = FStar.Ghost.hide (Bind.names tokens weights values);
  assert pure (Bind.array_binding a tokens weights values);
  Pledge.rewrite_pledge
    (Caller.contents values (Native.input_cells tokens weights fp fw x w ** frame))
    (on gpu_loc (Device.phase0 a x w fp fw) ** frame)
    (Bind.prepare a tokens weights values x w fp fw frame ());
  unfold Caller.owners stream values;
  unfold Native.output_owners stream values.outputs;
  unfold Results.scratch_owners stream values.scratch;

  let token_ptr = Input.data_ptr tokens;
  let weight_ptr = Input.data_ptr weights;
  let score_ptr = RT.data_ptr stream values.scratch.scores;
  assert pure (token_ptr == a.Device.tokens /\ weight_ptr == a.Device.weights /\ score_ptr == a.Device.scores);
  Launch.launch (Device.projection_kernel #a token_ptr weight_ptr score_ptr #x #w fp fw)
    stream #epoch frame ();

  let score_ptr = RT.data_ptr stream values.scratch.scores;
  let selected_ptr = RT.data_ptr stream values.scratch.selected;
  let routes_ptr = RT.data_ptr stream values.scratch.routes;
  assert pure (score_ptr == a.Device.scores /\ selected_ptr == a.Device.selected /\ routes_ptr == a.Device.routes);
  Launch.launch (Kernel.k11_as_kfull (Device.routing_kernel #a
    score_ptr selected_ptr routes_ptr #x #w fp fw ()))
    stream #(Epoch.epoch_next epoch) frame ();

  let selected_ptr = RT.data_ptr stream values.scratch.selected;
  let routes_ptr = RT.data_ptr stream values.scratch.routes;
  let demand_ptr = RT.data_ptr stream values.outputs.demand;
  let index_ptr = RT.data_ptr stream values.outputs.indices;
  let probability_ptr = RT.data_ptr stream values.outputs.probabilities;
  let survival_ptr = RT.data_ptr stream values.outputs.survival;
  assert pure (selected_ptr == a.Device.selected /\ routes_ptr == a.Device.routes /\
    demand_ptr == a.Device.demand /\ index_ptr == a.Device.indices /\
    probability_ptr == a.Device.probabilities /\ survival_ptr == a.Device.survival);
  Launch.launch (Kernel.k11_as_kfull (Device.capacity_kernel #a
    selected_ptr routes_ptr demand_ptr index_ptr probability_ptr survival_ptr #x #w fp fw ()))
    stream #(Epoch.epoch_next (Epoch.epoch_next epoch)) frame ();
  fold Results.scratch_owners stream values.scratch;
  fold Native.output_owners stream values.outputs;
  fold Caller.owners stream values;
  Pledge.rewrite_pledge
    (on gpu_loc (Device.phase3 a x w fp fw) ** frame)
    (Bind.output_post values.outputs x w **
      Results.scratch_contents values.scratch (Native.input_cells tokens weights fp fw x w ** frame))
    (Bind.finish a tokens weights values x w fp fw frame ());
}

inline_for_extraction noextract
let matching_handles (handles:Native.output_handles) (value:Results.result_vector)
  : Lemma
      (requires Results.matches handles value)
      (ensures Results.vector_handles value == handles)
  = ()

inline_for_extraction noextract
ghost
fn failed_contents (handles:Native.output_handles)
  (x:Layout.token_values) (w:Layout.weight_values) (frame:slprop) ()
  requires Bind.output_post handles x w ** frame
  ensures RT.contents handles.survival **
    (RT.contents handles.probabilities **
      (RT.contents handles.indices ** (RT.contents handles.demand ** frame)))
{
  unfold Bind.output_post handles x w;
  with state. assert Bind.output_payload handles x w state;
  unfold Bind.output_payload handles x w state;
  Results.forget_output_values handles (Bind.output_values state) frame ();
}

inline_for_extraction noextract
ghost
fn success_contents (values:Caller.buffers)
  (x:Layout.token_values) (w:Layout.weight_values)
  (value:Results.result_vector) (frame:slprop)
  (matches:squash (Results.matches values.outputs value)) ()
  requires Bind.output_post values.outputs x w ** Results.scratch_contents values.scratch frame
  ensures Results.scratch_contents values.scratch
    (Bind.output_post (Results.vector_handles value) x w ** frame)
{
  matching_handles values.outputs value;
  rewrite Bind.output_post values.outputs x w as Bind.output_post (Results.vector_handles value) x w;
  unfold Results.scratch_contents values.scratch frame;
  fold Results.scratch_contents values.scratch
    (Bind.output_post (Results.vector_handles value) x w ** frame);
}

inline_for_extraction noextract
fn return_outputs
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (values:Caller.buffers) (#x:erased Layout.token_values) (#w:erased Layout.weight_values)
  (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream ** Epoch.epoch_live stream epoch
  requires Caller.owners stream values **
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (Bind.output_post values.outputs x w ** Results.scratch_contents values.scratch frame)
  returns result:Results.result_outcome
  ensures (
    match result with
    | R.Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | R.Constructed value ->
      Native.output_owners stream (Results.vector_handles value) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch)
        (Bind.output_post (Results.vector_handles value) x w ** frame))
{
  unfold Caller.owners stream values;
  unfold Native.output_owners stream values.outputs;
  let result = R.return_four stream #epoch
    values.outputs.demand values.outputs.indices values.outputs.probabilities values.outputs.survival
    (Bind.output_post values.outputs x w ** Results.scratch_contents values.scratch frame) ();
  match result {
    R.Failed error -> {
      Pledge.rewrite_pledge
        (Bind.output_post values.outputs x w ** Results.scratch_contents values.scratch frame)
        (RT.contents values.outputs.survival **
          (RT.contents values.outputs.probabilities **
            (RT.contents values.outputs.indices **
              (RT.contents values.outputs.demand ** Results.scratch_contents values.scratch frame))))
        (failed_contents values.outputs x w (Results.scratch_contents values.scratch frame));
      RT.release stream #epoch values.outputs.survival
        (RT.contents values.outputs.probabilities **
          (RT.contents values.outputs.indices **
            (RT.contents values.outputs.demand ** Results.scratch_contents values.scratch frame))) ();
      RT.release stream #epoch values.outputs.probabilities
        (RT.contents values.outputs.indices **
          (RT.contents values.outputs.demand ** Results.scratch_contents values.scratch frame)) ();
      RT.release stream #epoch values.outputs.indices
        (RT.contents values.outputs.demand ** Results.scratch_contents values.scratch frame) ();
      RT.release stream #epoch values.outputs.demand (Results.scratch_contents values.scratch frame) ();
      Results.release_scratch stream #epoch values.scratch frame ();
      R.Failed error
    }
    R.Constructed value -> {
      matching_handles values.outputs value;
      fold Native.output_owners stream values.outputs;
      rewrite Native.output_owners stream values.outputs
        as Native.output_owners stream (Results.vector_handles value);
      Pledge.rewrite_pledge
        (Bind.output_post values.outputs x w ** Results.scratch_contents values.scratch frame)
        (Results.scratch_contents values.scratch (Bind.output_post (Results.vector_handles value) x w ** frame))
        (success_contents values x w value frame ());
      Results.release_scratch stream #epoch values.scratch
        (Bind.output_post (Results.vector_handles value) x w ** frame) ();
      R.Constructed value
    }
  }
}

noextract noeq
type outcome =
  | AllocationFailed : error:RT.allocation_error -> outcome
  | Returned : result:Results.result_outcome -> outcome

inline_for_extraction noextract
let before_call
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:Bind.token_argument) (weights:Bind.weight_argument)
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) (frame:slprop) : slprop =
  Stream.stream_live stream ** RT.runtime stream ** Epoch.epoch_live stream epoch **
  Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (Native.input_cells tokens weights fp fw x w ** frame) **
  pure (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))

inline_for_extraction noextract
let after_call
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:Bind.token_argument) (weights:Bind.weight_argument)
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) (frame:slprop)
  (result:outcome) : slprop =
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
   | AllocationFailed _ ->
     Epoch.epoch_live stream epoch **
     Pledge.pledge0 (Epoch.epoch_done stream epoch) (Native.input_cells tokens weights fp fw x w ** frame)
   | Returned result ->
     Epoch.epoch_live stream (advanced epoch) **
     (match result with
      | R.Failed _ ->
        Pledge.pledge0 (Epoch.epoch_done stream (advanced epoch))
          (Native.input_cells tokens weights fp fw x w ** frame)
      | R.Constructed value ->
        Native.output_owners stream (Results.vector_handles value) **
        Pledge.pledge0 (Epoch.epoch_done stream (advanced epoch))
          (Bind.output_post (Results.vector_handles value) x w **
            (Native.input_cells tokens weights fp fw x w ** frame))))

inline_for_extraction noextract
fn parameter_body
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:Bind.token_argument) (weights:Bind.weight_argument)
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (#fp #fw:perm)
  (frame:slprop) {| is_send frame |} ()
  requires cpu ** Input.reference weights ** Input.reference tokens **
    before_call stream epoch tokens weights x w fp fw frame
  returns result:outcome
  ensures cpu ** Input.reference weights ** Input.reference tokens **
    after_call stream epoch tokens weights x w fp fw frame result
{
  unfold before_call stream epoch tokens weights x w fp fw frame;
  let allocation = Caller.allocate_all stream #epoch
    (Native.input_cells tokens weights fp fw x w ** frame) ();
  match allocation {
    Caller.Failed error -> {
      fold after_call stream epoch tokens weights x w fp fw frame (AllocationFailed error);
      AllocationFailed error
    }
    Caller.Allocated values -> {
      launch_program stream #epoch tokens weights values #x #w #fp #fw frame ();
      let result = return_outputs stream #(advanced epoch) values #x #w
        (Native.input_cells tokens weights fp fw x w ** frame) ();
      fold after_call stream epoch tokens weights x w fp fw frame (Returned result);
      Returned result
    }
  }
}

inline_for_extraction noextract
fn weight_scope
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:Bind.token_argument) (weights:Bind.weight_argument)
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (#fp #fw:perm)
  (frame:slprop) {| is_send frame |} ()
  requires cpu ** Input.reference tokens ** before_call stream epoch tokens weights x w fp fw frame
  returns result:outcome
  ensures cpu ** Input.reference tokens ** after_call stream epoch tokens weights x w fp fw frame result
{
  Input.with_parameter weights
    (Input.reference tokens ** before_call stream epoch tokens weights x w fp fw frame)
    (fun result -> Input.reference tokens ** after_call stream epoch tokens weights x w fp fw frame result)
    (parameter_body stream #epoch tokens weights #x #w #fp #fw frame)
}

inline_for_extraction noextract
fn run
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:Bind.token_argument) (weights:Bind.weight_argument)
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (#fp #fw:perm)
  (frame:slprop) {| is_send frame |} ()
  requires cpu ** before_call stream epoch tokens weights x w fp fw frame
  returns result:outcome
  ensures cpu ** after_call stream epoch tokens weights x w fp fw frame result
{
  Input.with_parameter tokens
    (before_call stream epoch tokens weights x w fp fw frame)
    (after_call stream epoch tokens weights x w fp fw frame)
    (weight_scope stream #epoch tokens weights #x #w #fp #fw frame)
}
