module KPlay.Proof.NativeBindings
#lang-pulse

open Kuiper
open Pulse.Lib.Send
open Kuiper.Tensor.Runtime.Helpers

module PA = Pulse.Lib.Array
module S = FStar.Seq
module RT = Kuiper.Tensor.Runtime
module Input = Kuiper.Tensor.Input
module KL = Kuiper.Locs
module Native = KPlay.Proof.Shared.Native
module Layout = KPlay.Proof.SourceLayouts
module Caller = KPlay.Proof.SourceCaller
module Outputs = KPlay.Proof.OutputAllocations
module Results = KPlay.Proof.SourceResults
module Device = KPlay.Proof.SourceDevice
module Project = KPlay.Proof.SourceProjection
module Routing = KPlay.Proof.SourceRouting
module Observe = KPlay.Proof.SourceCapacity
module Steps = KPlay.Proof.RouteSteps
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let token_argument = value:Input.argument bf16{Input.dimensions value == [256; 512]}

inline_for_extraction noextract
let weight_argument = value:Input.argument bf16{Input.dimensions value == [8; 512]}

inline_for_extraction noextract
let array_binding (a:Device.arrays) (tokens:token_argument) (weights:weight_argument)
  (values:Caller.buffers) : prop =
  a.tokens == Input.data tokens /\ a.weights == Input.data weights /\
  a.scores == RT.data values.scratch.scores /\
  a.selected == RT.data values.scratch.selected /\ a.routes == RT.data values.scratch.routes /\
  a.demand == RT.data values.outputs.demand /\ a.indices == RT.data values.outputs.indices /\
  a.probabilities == RT.data values.outputs.probabilities /\ a.survival == RT.data values.outputs.survival

inline_for_extraction noextract
let names (tokens:token_argument) (weights:weight_argument)
  (values:Caller.buffers{Caller.metadata values})
  : GTot (a:Device.arrays{array_binding a tokens weights values}) =
  Input.view_metadata tokens;
  Input.view_metadata weights;
  {
    tokens = Input.data tokens;
    weights = Input.data weights;
    scores = RT.data values.scratch.scores;
    selected = RT.data values.scratch.selected;
    routes = RT.data values.scratch.routes;
    demand = RT.data values.outputs.demand;
    indices = RT.data values.outputs.indices;
    probabilities = RT.data values.outputs.probabilities;
    survival = RT.data values.outputs.survival
  }

inline_for_extraction noextract
let output_values (state:Steps.capacity_buffers) : Native.output_values = {
  demand_values = state.demand;
  index_values = state.indices;
  probability_values = state.probabilities;
  survival_values = state.survival
}

inline_for_extraction noextract
let output_payload (handles:Native.output_handles)
  (x:Layout.token_values) (w:Layout.weight_values) (state:Steps.capacity_buffers) : slprop =
  Native.output_cells handles (output_values state) **
  pure (Native.output_binding handles (output_values state) (Observe.public_output state) /\
    Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
      (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state))

inline_for_extraction noextract
let output_post (handles:Native.output_handles) (x:Layout.token_values) (w:Layout.weight_values)
  : slprop =
  exists* (state:Steps.capacity_buffers). output_payload handles x w state

inline_for_extraction noextract
ghost
fn rebind (#a:Type0) (before after:PA.array a) (#f:perm) (#values:S.seq a)
  requires on gpu_loc (PA.pts_to before #f values) ** pure (before == after)
  ensures on gpu_loc (PA.pts_to after #f values)
{
  rewrite on gpu_loc (PA.pts_to before #f values) as on gpu_loc (PA.pts_to after #f values);
}

inline_for_extraction noextract
ghost
fn contents_values (#a:Type0) (value:RT.tensor a)
  requires RT.contents value
  returns values:erased (lseq a (PA.length (RT.data value)))
  ensures on gpu_loc (PA.pts_to (RT.data value) values)
{
  unfold RT.contents value;
  with values. assert on gpu_loc (PA.pts_to (RT.data value) values);
  map_loc gpu_loc #(PA.pts_to (RT.data value) values)
    #(PA.pts_to (RT.data value) values **
      pure (PA.length (RT.data value) == S.length values))
    fn () { PA.pts_to_len (RT.data value); };
  FStar.Ghost.hide (values <: lseq a (PA.length (RT.data value)))
}

inline_for_extraction noextract
ghost
fn group_initial (a:Device.arrays) (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm)
  (ss:lseq f32 2048) (fs:Routing.flags) (qs:Routing.probabilities) (state:Steps.capacity_buffers) ()
  requires PA.pts_to a.tokens #fp x ** PA.pts_to a.weights #fw w **
    PA.pts_to a.scores ss ** PA.pts_to a.selected fs ** PA.pts_to a.routes qs **
    PA.pts_to a.demand state.demand ** PA.pts_to a.indices state.indices **
    PA.pts_to a.probabilities state.probabilities ** PA.pts_to a.survival state.survival
  ensures Device.phase0 a x w fp fw
{
  fold Project.pre a.tokens a.weights a.scores x w fp fw;
  fold Device.raw_routes a;
  fold Observe.points a.demand a.indices a.probabilities a.survival state;
  fold Device.raw_outputs a;
  fold Device.phase0 a x w fp fw;
}

inline_for_extraction noextract
ghost
fn prepare (a:Device.arrays) (tokens:token_argument) (weights:weight_argument)
  (values:Caller.buffers{Caller.metadata values})
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) (frame:slprop)
  (binding:squash (array_binding a tokens weights values)) ()
  requires Caller.contents values (Native.input_cells tokens weights fp fw x w ** frame)
  ensures on gpu_loc (Device.phase0 a x w fp fw) ** frame
{
  unfold Caller.contents values (Native.input_cells tokens weights fp fw x w ** frame);
  unfold Outputs.contents values.outputs;
  unfold Results.scratch_contents values.scratch (Native.input_cells tokens weights fp fw x w ** frame);
  unfold Native.input_cells tokens weights fp fw x w;
  let ss = contents_values values.scratch.scores;
  let fs = contents_values values.scratch.selected;
  let qs = contents_values values.scratch.routes;
  let ds = contents_values values.outputs.demand;
  let ids = contents_values values.outputs.indices;
  let ps = contents_values values.outputs.probabilities;
  let vs = contents_values values.outputs.survival;
  rebind (Input.data tokens) a.tokens #fp #x;
  rebind (Input.data weights) a.weights #fw #w;
  rebind (RT.data values.scratch.scores) a.scores #1.0R #ss;
  rebind (RT.data values.scratch.selected) a.selected #1.0R #fs;
  rebind (RT.data values.scratch.routes) a.routes #1.0R #qs;
  rebind (RT.data values.outputs.demand) a.demand #1.0R #ds;
  rebind (RT.data values.outputs.indices) a.indices #1.0R #ids;
  rebind (RT.data values.outputs.probabilities) a.probabilities #1.0R #ps;
  rebind (RT.data values.outputs.survival) a.survival #1.0R #vs;
  map_loc gpu_loc
    #(PA.pts_to a.tokens #fp x ** PA.pts_to a.weights #fw w **
      PA.pts_to a.scores ss ** PA.pts_to a.selected fs ** PA.pts_to a.routes qs **
      PA.pts_to a.demand ds ** PA.pts_to a.indices ids **
      PA.pts_to a.probabilities ps ** PA.pts_to a.survival vs)
    #(Device.phase0 a x w fp fw)
    (group_initial a x w fp fw ss fs qs {
      Steps.demand = ds; Steps.indices = ids; Steps.probabilities = ps; Steps.survival = vs});
}

inline_for_extraction noextract
let raw_final (a:Device.arrays) (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) : slprop =
  PA.pts_to a.tokens #fp x ** PA.pts_to a.weights #fw w **
  PA.pts_to a.scores (Project.scores x w) **
  (exists* (fs:Routing.flags) (qs:Routing.probabilities).
    PA.pts_to a.selected fs ** PA.pts_to a.routes qs) **
  (exists* (state:Steps.capacity_buffers).
    Observe.points a.demand a.indices a.probabilities a.survival state **
    pure (Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
      (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state)))

inline_for_extraction noextract
ghost
fn expose_final (a:Device.arrays) (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) ()
  requires Device.phase3 a x w fp fw
  ensures raw_final a x w fp fw
{
  unfold Device.phase3 a x w fp fw;
  unfold Project.post a.tokens a.weights a.scores x w fp fw;
  unfold Device.raw_routes a;
  unfold Device.result a x w;
  with state. assert Device.result_payload a x w state;
  unfold Device.result_payload a x w state;
  fold raw_final a x w fp fw;
}

inline_for_extraction noextract
ghost
fn expose_points (a:Device.arrays) (state:Steps.capacity_buffers) ()
  requires Observe.points a.demand a.indices a.probabilities a.survival state
  ensures PA.pts_to a.demand state.demand ** PA.pts_to a.indices state.indices **
    PA.pts_to a.probabilities state.probabilities ** PA.pts_to a.survival state.survival
{
  unfold Observe.points a.demand a.indices a.probabilities a.survival state;
}

inline_for_extraction noextract
ghost
fn group_output (handles:Native.output_handles{Outputs.metadata handles})
  (x:Layout.token_values) (w:Layout.weight_values) (state:Steps.capacity_buffers)
  requires on gpu_loc (
    PA.pts_to (RT.data handles.demand) state.demand **
    PA.pts_to (RT.data handles.indices) state.indices **
    PA.pts_to (RT.data handles.probabilities) state.probabilities **
    PA.pts_to (RT.data handles.survival) state.survival) **
    pure (Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
      (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state))
  ensures output_post handles x w
{
  fold Native.output_cells handles (output_values state);
  fold output_payload handles x w state;
  Pulse.Lib.Core.intro_exists (fun state -> output_payload handles x w state) state;
  fold output_post handles x w;
}

inline_for_extraction noextract
ghost
fn finish (a:Device.arrays) (tokens:token_argument) (weights:weight_argument)
  (values:Caller.buffers{Caller.metadata values})
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) (frame:slprop)
  (binding:squash (array_binding a tokens weights values)) ()
  requires on gpu_loc (Device.phase3 a x w fp fw) ** frame
  ensures output_post values.outputs x w **
    Results.scratch_contents values.scratch (Native.input_cells tokens weights fp fw x w ** frame)
{
  map_loc gpu_loc #(Device.phase3 a x w fp fw) #(raw_final a x w fp fw)
    (expose_final a x w fp fw);
  map_loc gpu_loc #(raw_final a x w fp fw)
    #(PA.pts_to a.tokens #fp x ** PA.pts_to a.weights #fw w **
      PA.pts_to a.scores (Project.scores x w) **
      (exists* (fs:Routing.flags) (qs:Routing.probabilities).
        PA.pts_to a.selected fs ** PA.pts_to a.routes qs) **
      (exists* (state:Steps.capacity_buffers).
        Observe.points a.demand a.indices a.probabilities a.survival state **
        pure (Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
          (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state))))
    fn () { unfold raw_final a x w fp fw; };
  with fs qs state.
    assert on gpu_loc (PA.pts_to a.selected fs ** PA.pts_to a.routes qs) **
      on gpu_loc (Observe.points a.demand a.indices a.probabilities a.survival state) **
      pure (Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
        (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state));
  map_loc gpu_loc
    #(Observe.points a.demand a.indices a.probabilities a.survival state)
    #(PA.pts_to a.demand state.demand ** PA.pts_to a.indices state.indices **
      PA.pts_to a.probabilities state.probabilities ** PA.pts_to a.survival state.survival)
    (expose_points a state);
  rebind a.tokens (Input.data tokens) #fp #x;
  rebind a.weights (Input.data weights) #fw #w;
  rebind a.scores (RT.data values.scratch.scores) #1.0R #(Project.scores x w);
  rebind a.selected (RT.data values.scratch.selected) #1.0R #fs;
  rebind a.routes (RT.data values.scratch.routes) #1.0R #qs;
  rebind a.demand (RT.data values.outputs.demand) #1.0R #state.demand;
  rebind a.indices (RT.data values.outputs.indices) #1.0R #state.indices;
  rebind a.probabilities (RT.data values.outputs.probabilities) #1.0R #state.probabilities;
  rebind a.survival (RT.data values.outputs.survival) #1.0R #state.survival;
  group_output values.outputs x w state;
  fold Native.input_cells tokens weights fp fw x w;
  fold RT.contents values.scratch.scores;
  fold RT.contents values.scratch.selected;
  fold RT.contents values.scratch.routes;
  fold Results.scratch_contents values.scratch (Native.input_cells tokens weights fp fw x w ** frame);
}
