module KPlay.Proof.SourceDevice
#lang-pulse

open Kuiper
open Pulse.Lib.Send
open KPlay.Proof.SourceProjection

module PA = Pulse.Lib.Array
module S = FStar.Seq
module Kernel = Kuiper.Kernel
module Layout = KPlay.Proof.SourceLayouts
module Project = KPlay.Proof.SourceProjection
module Routing = KPlay.Proof.SourceRouting
module History = KPlay.Proof.SourceHistory
module Observe = KPlay.Proof.SourceCapacity
module Steps = KPlay.Proof.RouteSteps
module Shared = KPlay.SharedSpec
module Capacity = KPlay.Proof.Shared.Capacity
module Launch = KPlay.Proof.SourceLaunch
module FamilyPacking = KPlay.Proof.FamilyPacking

noextract noeq
type arrays = {
  tokens: Project.garray bf16 131072;
  weights: Project.garray bf16 4096;
  scores: Project.garray f32 2048;
  selected: Project.garray i32 2048;
  routes: Project.garray f32 2048;
  demand: Project.garray i32 8;
  indices: Project.garray i64 256;
  probabilities: Project.garray f32 256;
  survival: Project.garray i32 2048
}

inline_for_extraction noextract
let raw_routes (a:arrays) : slprop =
  exists* (fs:Routing.flags) (qs:Routing.probabilities).
    PA.pts_to a.selected fs ** PA.pts_to a.routes qs

inline_for_extraction noextract
let raw_outputs (a:arrays) : slprop =
  exists* (state:Steps.capacity_buffers).
    Observe.points a.demand a.indices a.probabilities a.survival state

inline_for_extraction noextract
let phase0 (a:arrays) (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) : slprop =
  Project.pre a.tokens a.weights a.scores x w fp fw ** (raw_routes a ** raw_outputs a)

inline_for_extraction noextract
let phase1 (a:arrays) (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) : slprop =
  Project.post a.tokens a.weights a.scores x w fp fw ** (raw_routes a ** raw_outputs a)

inline_for_extraction noextract
let phase2 (a:arrays) (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) : slprop =
  Project.post a.tokens a.weights a.scores x w fp fw **
    Routing.loop_state 256 x w a.selected a.routes ** raw_outputs a

inline_for_extraction noextract
let result_payload (a:arrays) (x:Layout.token_values) (w:Layout.weight_values)
  (state:Steps.capacity_buffers) : slprop =
  Observe.points a.demand a.indices a.probabilities a.survival state **
  pure (Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
    (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state))

inline_for_extraction noextract
let result (a:arrays) (x:Layout.token_values) (w:Layout.weight_values) : slprop =
  exists* (state:Steps.capacity_buffers). result_payload a x w state

inline_for_extraction noextract
let phase3 (a:arrays) (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) : slprop =
  Project.post a.tokens a.weights a.scores x w fp fw ** raw_routes a ** result a x w

inline_for_extraction noextract
ghost
fn rebind (#a:Type0) (before after:PA.array a) (#f:perm) (#values:S.seq a)
  requires PA.pts_to before #f values ** pure (before == after)
  ensures PA.pts_to after #f values
{
  rewrite PA.pts_to before #f values as PA.pts_to after #f values;
}

inline_for_extraction noextract
ghost
fn projection_pre (a:arrays)
  (tokens:Project.garray bf16 131072{tokens == a.tokens})
  (weights:Project.garray bf16 4096{weights == a.weights})
  (scores:Project.garray f32 2048{scores == a.scores})
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) ()
  requires phase0 a x w fp fw
  ensures Project.pre tokens weights scores x w fp fw ** (raw_routes a ** raw_outputs a)
{
  unfold phase0 a x w fp fw;
  rewrite Project.pre a.tokens a.weights a.scores x w fp fw
    as Project.pre tokens weights scores x w fp fw;
}

inline_for_extraction noextract
ghost
fn projection_post (a:arrays)
  (tokens:Project.garray bf16 131072{tokens == a.tokens})
  (weights:Project.garray bf16 4096{weights == a.weights})
  (scores:Project.garray f32 2048{scores == a.scores})
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) ()
  requires Project.post tokens weights scores x w fp fw ** (raw_routes a ** raw_outputs a)
  ensures phase1 a x w fp fw
{
  rewrite Project.post tokens weights scores x w fp fw
    as Project.post a.tokens a.weights a.scores x w fp fw;
  fold phase1 a x w fp fw;
}

inline_for_extraction noextract
let projection_kernel (#a:erased arrays)
  (tokens:Project.garray bf16 131072{tokens == a.tokens})
  (weights:Project.garray bf16 4096{weights == a.weights})
  (scores:Project.garray f32 2048{scores == a.scores})
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (fp fw:perm)
  : Kernel.kernel_desc (phase0 a x w fp fw) (phase1 a x w fp fw) =
  Kernel.kd_weaken
    (Launch.frame_kernel (Kernel.k1n_as_kfull (Project.kernel tokens weights scores #x #w fp fw))
      (raw_routes a ** raw_outputs a))
    (projection_pre a tokens weights scores x w fp fw)
    (projection_post a tokens weights scores x w fp fw)

inline_for_extraction noextract
fn routing_thread (#a:erased arrays)
  (scores:Project.garray f32 2048{scores == a.scores})
  (selected:Project.garray i32 2048{selected == a.selected})
  (routes:Project.garray f32 2048{routes == a.routes})
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (fp fw:perm)
  (domain:squash (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))) ()
  norewrite
  requires gpu ** phase1 a x w fp fw
  ensures gpu ** phase2 a x w fp fw
{
  unfold phase1 a x w fp fw;
  unfold Project.post a.tokens a.weights a.scores x w fp fw;
  unfold raw_routes a;
  with fs qs. assert PA.pts_to a.selected fs ** PA.pts_to a.routes qs;
  rebind a.scores scores #1.0R #(Project.scores x w);
  rebind a.selected selected #1.0R #fs;
  rebind a.routes routes #1.0R #qs;
  let h = Routing.source_router scores selected routes #x #w #fs #qs ();
  with next_fs next_qs. assert PA.pts_to selected next_fs ** PA.pts_to routes next_qs;
  rebind scores a.scores #1.0R #(Project.scores x w);
  rebind selected a.selected #1.0R #next_fs;
  rebind routes a.routes #1.0R #next_qs;
  Routing.pack_loop 256 x w a.selected a.routes h next_fs next_qs;
  fold Project.post a.tokens a.weights a.scores x w fp fw;
  fold phase2 a x w fp fw;
}

inline_for_extraction noextract
let routing_kernel (#a:erased arrays)
  (scores:Project.garray f32 2048{scores == a.scores})
  (selected:Project.garray i32 2048{selected == a.selected})
  (routes:Project.garray f32 2048{routes == a.routes})
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (fp fw:perm)
  (domain:squash (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w)))
  : Kernel.kernel_desc_1_1 (phase1 a x w fp fw) (phase2 a x w fp fw) = {
  f = routing_thread #a scores selected routes #x #w fp fw domain;
  full_pre_sendable = solve;
  full_post_sendable = solve
}

inline_for_extraction noextract
let task_from_actual_history (x:Layout.token_values) (w:Layout.weight_values)
  (h:Routing.history 256)
  (actual:Shared.numerical_execution (Layout.logical_tokens x) (Layout.logical_weights w))
  (state:Steps.capacity_buffers)
  : Lemma
      (requires Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w) /\
        actual == History.observation x w h /\
        (let b = Shared.selected_boundary (Layout.logical_tokens x) (Layout.logical_weights w) actual in
          Capacity.route_post b (Observe.public_output state) /\
          Shared.packing_post b (History.normalization h) (Observe.public_output state)))
      (ensures Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
        (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state))
  =
  let b = Shared.selected_boundary (Layout.logical_tokens x) (Layout.logical_weights w) actual in
  History.normalization_valid h b;
  Shared.postcondition_instantiates_task (Layout.logical_tokens x) (Layout.logical_weights w)
    (Layout.logical_tokens x) (Layout.logical_weights w)
    actual (History.normalization h) (Observe.public_output state)

inline_for_extraction noextract
ghost
fn pack_result (a:arrays) (x:Layout.token_values) (w:Layout.weight_values)
  (state:Steps.capacity_buffers)
  requires Observe.points a.demand a.indices a.probabilities a.survival state **
    pure (Shared.task_relation (Layout.logical_tokens x) (Layout.logical_weights w)
      (Layout.logical_tokens x) (Layout.logical_weights w) (Observe.public_output state))
  ensures result a x w
{
  fold result_payload a x w state;
  Pulse.Lib.Core.intro_exists (fun state -> result_payload a x w state) state;
  fold result a x w;
}

inline_for_extraction noextract
fn capacity_thread (#a:erased arrays)
  (selected:Project.garray i32 2048{selected == a.selected})
  (routes:Project.garray f32 2048{routes == a.routes})
  (demand:Project.garray i32 8{demand == a.demand})
  (indices:Project.garray i64 256{indices == a.indices})
  (probabilities:Project.garray f32 256{probabilities == a.probabilities})
  (survival:Project.garray i32 2048{survival == a.survival})
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (fp fw:perm)
  (domain:squash (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))) ()
  norewrite
  requires gpu ** phase2 a x w fp fw
  ensures gpu ** phase3 a x w fp fw
{
  unfold phase2 a x w fp fw;
  unfold Routing.loop_state 256 x w a.selected a.routes;
  with h fs qs. assert Routing.loop_payload 256 x w a.selected a.routes h fs qs;
  unfold Routing.loop_payload 256 x w a.selected a.routes h fs qs;
  History.observation_valid x w h;
  let actual:erased (Shared.numerical_execution (Layout.logical_tokens x) (Layout.logical_weights w)) =
    FStar.Ghost.hide (History.observation x w h);
  let b:erased (Capacity.selected_boundary Capacity.Smoke) = FStar.Ghost.hide
    (Shared.selected_boundary (Layout.logical_tokens x) (Layout.logical_weights w) actual);
  unfold raw_outputs a;
  with before. assert Observe.points a.demand a.indices a.probabilities a.survival before;
  rebind a.selected selected #1.0R #fs;
  rebind a.routes routes #1.0R #qs;
  rewrite Observe.points a.demand a.indices a.probabilities a.survival before
    as Observe.points demand indices probabilities survival before;
  let state = Observe.source_router selected routes #fs #qs
    demand indices probabilities survival #h #b #before ();
  FamilyPacking.source_family_post h b state;
  rebind selected a.selected #1.0R #fs;
  rebind routes a.routes #1.0R #qs;
  rewrite Observe.points demand indices probabilities survival state
    as Observe.points a.demand a.indices a.probabilities a.survival state;
  task_from_actual_history x w h actual state;
  pack_result a x w state;
  fold raw_routes a;
  fold phase3 a x w fp fw;
}

inline_for_extraction noextract
let capacity_kernel (#a:erased arrays)
  (selected:Project.garray i32 2048{selected == a.selected})
  (routes:Project.garray f32 2048{routes == a.routes})
  (demand:Project.garray i32 8{demand == a.demand})
  (indices:Project.garray i64 256{indices == a.indices})
  (probabilities:Project.garray f32 256{probabilities == a.probabilities})
  (survival:Project.garray i32 2048{survival == a.survival})
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (fp fw:perm)
  (domain:squash (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w)))
  : Kernel.kernel_desc_1_1 (phase2 a x w fp fw) (phase3 a x w fp fw) = {
  f = capacity_thread #a selected routes demand indices probabilities survival #x #w fp fw domain;
  full_pre_sendable = solve;
  full_post_sendable = solve
}
