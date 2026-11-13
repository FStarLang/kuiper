module KPlay.Proof.Shared.Native

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send
open PulseCore.FractionalPermission

module Array = Pulse.Lib.Array
module Seq = FStar.Seq
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64
module C = Kuiper.Chest
module RT = Kuiper.Tensor.Runtime
module Input = Kuiper.Tensor.Input
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module Capacity = KPlay.Proof.Shared.Capacity

(* A separate native envelope, not the mathematical task's representation.
   References are actual native arguments; no array-to-reference or
   borrowed-reference-to-owner conversion is supplied here. *)
inline_for_extraction noextract
let input_binding
  (tokens weights:Input.argument BF16.t)
  (token_values weight_values:Seq.seq BF16.t)
  (x:Projection.token_matrix true)
  (w:Projection.weight_matrix true) : prop =
  Input.dimensions tokens == [256; 512] /\
  Input.dimensions weights == [8; 512] /\
  Seq.length token_values == 256 * 512 /\
  Seq.length weight_values == 8 * 512 /\
  (forall (t:Capacity.token_index Capacity.Smoke)
          (h:Capacity.hidden_index Capacity.Smoke).
    Seq.index token_values (t * 512 + h) == C.acc2 x t h) /\
  (forall (h:Capacity.hidden_index Capacity.Smoke)
          (e:Capacity.expert_index Capacity.Smoke).
    Seq.index weight_values (e * 512 + h) == C.acc2 w h e)

inline_for_extraction noextract
let input_cells
  (tokens weights:Input.argument BF16.t)
  (token_permission weight_permission:perm)
  (token_values weight_values:Seq.seq BF16.t) : slprop =
  on KL.gpu_loc
    (Array.pts_to (Input.data tokens) #token_permission token_values **
     Array.pts_to (Input.data weights) #weight_permission weight_values)

inline_for_extraction noextract noeq
type output_handles = {
  demand: RT.tensor I32.t;
  indices: RT.tensor I64.t;
  probabilities: RT.tensor F32.t;
  survival: RT.tensor I32.t
}

inline_for_extraction noextract noeq
type output_values = {
  demand_values: Seq.seq I32.t;
  index_values: Seq.seq I64.t;
  probability_values: Seq.seq F32.t;
  survival_values: Seq.seq I32.t
}

inline_for_extraction noextract
let output_binding (handles:output_handles) (values:output_values)
  (out:Capacity.public_outputs Capacity.Smoke) : prop =
  RT.metadata handles.demand [8] /\
  RT.metadata handles.indices [8; 32] /\
  RT.metadata handles.probabilities [8; 32] /\
  RT.metadata handles.survival [256; 8] /\
  Seq.length values.demand_values == 8 /\
  Seq.length values.index_values == 8 * 32 /\
  Seq.length values.probability_values == 8 * 32 /\
  Seq.length values.survival_values == 256 * 8 /\
  (forall e. Seq.index values.demand_values e == out.expert_demand e) /\
  (forall e c.
    Seq.index values.index_values (e * 32 + c) == out.token_indices e c /\
    Seq.index values.probability_values (e * 32 + c) ==
      out.probabilities e c) /\
  (forall t e.
    Seq.index values.survival_values (t * 8 + e) == out.survival t e)

inline_for_extraction noextract
let output_owners (stream:Stream.stream_t) (handles:output_handles) : slprop =
  RT.owner stream handles.demand **
  RT.owner stream handles.indices **
  RT.owner stream handles.probabilities **
  RT.owner stream handles.survival

inline_for_extraction noextract
let output_cells (handles:output_handles) (values:output_values) : slprop =
  on KL.gpu_loc
    (Array.pts_to (RT.data handles.demand) values.demand_values **
     Array.pts_to (RT.data handles.indices) values.index_values **
     Array.pts_to (RT.data handles.probabilities) values.probability_values **
     Array.pts_to (RT.data handles.survival) values.survival_values)

inline_for_extraction noextract
let call_resources
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument BF16.t)
  (token_permission weight_permission:perm)
  (token_values weight_values:Seq.seq BF16.t)
  (frame:slprop) {| is_send frame |} : slprop =
  KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
  Epoch.epoch_live stream epoch **
  Input.reference tokens ** Input.reference weights **
  Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (input_cells tokens weights token_permission weight_permission
       token_values weight_values ** frame)

(* These predicates do not mint an epoch, complete a stream, initialize an
   empty allocation, or discharge the source's native parameter lifetimes.
   A role must preserve the same frame and input values, retain actual
   references/owners through their pending uses, and prove its native calls. *)
inline_for_extraction noextract
let return_resources
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument BF16.t)
  (token_permission weight_permission:perm)
  (token_values weight_values:Seq.seq BF16.t)
  (handles:output_handles) (values:output_values)
  (frame:slprop) {| is_send frame |} : slprop =
  KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
  Epoch.epoch_live stream epoch **
  Input.reference tokens ** Input.reference weights **
  output_owners stream handles **
  Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (input_cells tokens weights token_permission weight_permission
       token_values weight_values **
     output_cells handles values ** frame)

inline_for_extraction noextract
let signed_int32_dtype : RT.dtype I32.t = RT.Int32

inline_for_extraction noextract
let native_allocation_bounds ()
  : Lemma (
      RT.numel [256; 512] <= 2147483647 /\
      RT.numel [8; 512] <= 2147483647 /\
      RT.numel [256; 8] <= 2147483647 /\
      RT.numel [8; 32] <= 2147483647 /\
      RT.numel [8] <= 2147483647)
  = ()

inline_for_extraction noextract
let input_view_extent (value:Input.argument BF16.t)
  : Lemma (
      Array.length (Input.data value) == RT.numel (Input.dimensions value) /\
      Array.base_of (Input.data value) == Array.base_of (Input.backing value) /\
      Array.offset_of (Input.data value) ==
        Array.offset_of (Input.backing value) + Input.offset value)
  = Input.view_metadata value
