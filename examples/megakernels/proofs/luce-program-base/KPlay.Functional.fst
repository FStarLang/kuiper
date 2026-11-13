module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Send

module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module Model = KPlay.Proof.Shared.Model
module O = KPlay.Proof.Shared.Observations
module Transfer = KPlay.Proof.Base.Transfer
module Order = KPlay.Proof.Base.Order
module Cache = KPlay.Proof.Base.Cache
module NativeLayout = KPlay.Proof.Base.NativeLayout
module Native = KPlay.Proof.Base.NativeProjection
module MMA = Kuiper.TensorCore.MMA
module Views = Kuiper.TensorCore.MMA.Views
module SZ = Kuiper.SizeT
module Scratch = KPlay.Proof.Base.Scratch
module Vocabulary = KPlay.Proof.Base.Vocabulary
module Rotary = KPlay.Proof.Base.Rotary
module RoPESource = KPlay.Proof.RoPE.Source
module I64 = FStar.Int64
module Array = Pulse.Lib.Array
module Approx = Kuiper.Approximates.Base
module Initialization = KPlay.Proof.Base.Initialization
module Allocator = KPlay.Proof.Base.Allocator
module RT = Kuiper.Tensor.Runtime
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module Pledge = Pulse.Lib.Pledge
module Caller = KPlay.Proof.Base.Caller
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module Inputs = KPlay.Proof.Base.Inputs
module Embedding = KPlay.Proof.Base.Embedding
module Storage = KPlay.Proof.Shared.Storage
module Selection = KPlay.Proof.Shared.Selection
module Entry = KPlay.Proof.Base.Entry
module Norm = KPlay.Proof.Base.Norm
module NormNative = KPlay.Proof.Base.NormNative
module Head = KPlay.Proof.Base.Head
module PackedOwnership = KPlay.Proof.Base.PackedOwnership
module InputOwnership = KPlay.Proof.Base.InputOwnership
module Shared = KPlay.SharedSpec
module Schedule = KPlay.Proof.Base.Schedule
module Affine = KPlay.Proof.Base.AffineMMA
module SourceProjection = KPlay.Proof.Base.SourceProjection
module MatvecWrites = KPlay.Proof.Base.MatvecWrites
module SIMTMath = KPlay.Proof.Base.SIMTMath
module SIMT = KPlay.Proof.Base.SIMT
module CacheNative = KPlay.Proof.Base.CacheNative
module CacheProjection = KPlay.Proof.Base.CacheProjection
module Program = KPlay.Proof.Base.Program

noextract
fn program_refines
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (steps:I64.t{I64.v steps == call.Model.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#weights:erased Transfer.packed)
  (#physical_prompt:erased (lseq I64.t call.Model.prompt_length))
  (#prompt:erased (Selection.vec (Selection.token 248320) call.Model.prompt_length)) (#permission:perm)
  preserves cpu
  requires Program.pre call inputs stream epoch weights physical_prompt prompt permission
  returns result:Program.outcome
  ensures Program.post call inputs stream epoch weights physical_prompt prompt permission result
{
  Program.apply call inputs steps stream #epoch #weights #physical_prompt #prompt #permission
}

let source_native_layout_obstruction k layout =
  NativeLayout.no_source_mma_layout k layout

inline_for_extraction noextract
fn source_native_projection_block
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (weight:array2 BF.t Native.weight_layout)
  (hidden:array2 BF.t Native.hidden_layout)
  (accum part:MMA.fragment)
  (#vw:C.chest2 BF.t 752393024 1) (#vh:C.chest2 BF.t 16000 1)
  (#initial #vc:C.chest2 FP.t 16 8)
  preserves gpu
  preserves weight |-> Frac (1.0R /. 32) vw
  preserves hidden |-> Frac (1.0R /. 32) vh
  requires accum |-> initial ** part |-> vc
  ensures accum |-> Views.filled 5 initial (zero #FP.t)
  ensures part |-> Transfer.block_accumulator
    (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64
{
  Native.final_projection_block row weight hidden accum part;
}

inline_for_extraction noextract
fn source_native_score_store
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (hidden:array2 BF.t Native.hidden_layout) (accum part:MMA.fragment)
  (#vw:C.chest2 BF.t 752393024 1) (#vh:C.chest2 BF.t 16000 1)
  (#initial:C.chest2 FP.t 16 8) (#old:C.chest2 BF.t 16 1)
  preserves gpu
  preserves part |-> Transfer.block_accumulator
    (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64
  requires accum |-> Views.filled 5 initial (zero #FP.t)
  requires Native.score_view hidden |-> Frac (1.0R /. 32) old
  ensures accum |-> Views.added 5 (Views.filled 5 initial (zero #FP.t))
    (Transfer.block_accumulator
      (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64)
  ensures Native.score_view hidden |-> Frac (1.0R /. 32)
    (Native.stored_column (Transfer.block_accumulator
      (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64))
{
  Native.publish_block hidden accum part;
}

let projection_selection_safety weights hidden row before =
  let produced = Transfer.block_accumulator weights hidden row 64 in
  Order.scan_dominates (Transfer.consumed_scores produced) row before 16;
  Order.scan_special_values (Transfer.consumed_scores produced) row before

let projection_consumer_probe weights hidden normalized row before =
  Transfer.block_handoff weights hidden normalized row;
  let produced = Transfer.block_accumulator weights hidden row 64 in
  Order.scan_dominates (Transfer.consumed_scores produced) row before 16;
  Order.scan_special_values (Transfer.consumed_scores produced) row before

let native_score_handoff vw vh normalized row before =
  Native.consumer_binding
    (Transfer.block_accumulator (Native.packed_values vw) (Native.hidden_values vh) row 64);
  projection_consumer_probe (Native.packed_values vw) (Native.hidden_values vh)
    normalized row before;
  projection_selection_safety (Native.packed_values vw) (Native.hidden_values vh) row before

let source_bound_state_transition call p layer before physical hidden =
  Cache.source_structure before hidden layer p;
  Cache.append_represents physical hidden layer p (before.O.kv layer);
  let frame_cell (i:P.index 12582912)
    : Lemma (
          ~(Cache.key_base layer p <= i /\ i < Cache.key_base layer p + 512) /\
          ~(Cache.value_base layer p <= i /\ i < Cache.value_base layer p + 512) ==>
          Cache.append_loop physical hidden layer p 512 i == physical i) =
    if (i < Cache.key_base layer p || Cache.key_base layer p + 512 <= i) &&
       (i < Cache.value_base layer p || Cache.value_base layer p + 512 <= i)
    then Cache.append_frames physical hidden layer p i
  in
  FStar.Classical.forall_intro frame_cell

inline_for_extraction noextract
fn source_owned_projection_block
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (weight:array2 BF.t Native.weight_layout)
  (hidden:array2 BF.t Native.hidden_layout)
  (accum part:MMA.fragment)
  (#vw:C.chest2 BF.t 752393024 1) (#vh:C.chest2 BF.t 16000 1)
  (#initial #vc:C.chest2 FP.t 16 8)
  preserves gpu ** weight |-> Frac (1.0R /. 32) vw
  requires hidden |-> Frac (1.0R /. 32) vh
  requires accum |-> initial ** part |-> vc
  ensures part |-> Transfer.block_accumulator
    (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64
  ensures accum |-> Views.added 5 (Views.filled 5 initial (zero #FP.t))
    (Transfer.block_accumulator
      (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64)
  ensures hidden |-> Frac (1.0R /. 32) (Scratch.replace_window vh 12500 16
    (Native.stored_column (Transfer.block_accumulator
      (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64)))
{
  Scratch.owned_projection_block row weight hidden accum part;
}

let complete_source_score_selection weights hidden =
  Vocabulary.complete_selection weights hidden

let complete_source_score_numerical weights hidden normalized =
  Vocabulary.complete_numerical_scores weights hidden normalized

let source_rotary_connection call p before =
  Rotary.source_position_value call p;
  Rotary.source_connection before (Rotary.source_position call p)

inline_for_extraction noextract
fn source_rotary_producer
  (hidden:larray BF.t 16000) (p:RoPESource.position) (#before:Rotary.snapshot)
  preserves gpu
  requires Array.pts_to_mask hidden before Rotary.rotary_mask
  ensures Array.pts_to_mask hidden (Rotary.source_writes before p 32) Rotary.rotary_mask
  ensures pure (
    RoPESource.rotary_post p (Rotary.produced_cells before p) /\
    Rotary.rope_represents (I64.v p) (Rotary.produced_cells before p))
{
  Rotary.position_embedding hidden p;
}

inline_for_extraction noextract
fn source_rotary_consumer
  (hidden:larray BF.t 16000) (p:RoPESource.position) (j:sz{SZ.v j < 32})
  (#before:Rotary.snapshot) (#fraction:perm)
  preserves gpu
  preserves Array.pts_to_mask hidden #fraction
    (Rotary.source_writes before p 32) Rotary.rotary_mask
  returns result:(FP.t & FP.t)
  ensures pure (
    result == Rotary.consumed_pair before p (SZ.v j) /\
    (let cosine, sine = result in
      Approx.v_approximates cosine (A.rope_cos (I64.v p) (SZ.v j)) /\
      Approx.v_approximates sine (A.rope_sin (I64.v p) (SZ.v j))))
{
  let result = Rotary.read_source_pair hidden p j;
  result
}

inline_for_extraction noextract
fn source_cache_initialization
  (state:larray FP.t 4718592) (conv:larray BF.t 442368)
  (#old_state:lseq (option FP.t) 4718592)
  (#old_conv:lseq (option BF.t) 442368)
  preserves gpu
  requires Array.pts_to_mask state old_state Initialization.all_cells
  requires Array.pts_to_mask conv old_conv Initialization.all_cells
  ensures state |-> Initialization.state_values
  ensures conv |-> Initialization.convolution_values
{
  Initialization.initialize_caches state conv;
}

let source_first_use_inputs metadata layer =
  Initialization.first_use_inputs metadata layer

noextract
fn source_state_allocation
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns outcome:RT.allocation_result FP.t
  ensures (
    match outcome with
    | RT.Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | RT.Allocated value ->
      RT.owner stream value ** pure (RT.metadata value [4718592]) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (RT.contents value ** frame))
{
  Allocator.source_state_extent ();
  Allocator.allocate_source_state stream #epoch frame ()
}

noextract
fn source_state_pointer
  (stream:Stream.stream_t)
  (value:RT.tensor FP.t{RT.metadata value [4718592]})
  preserves cpu ** RT.owner stream value
  returns state:larray FP.t 4718592
  ensures pure (state == RT.data value /\ is_global_array state)
{
  let state = Allocator.typed_pointer stream value;
  state
}

inline_for_extraction noextract
fn source_allocated_cache_initialization
  (#allocation:RT.tensor FP.t{RT.metadata allocation [4718592]})
  (state:larray FP.t 4718592{state == RT.data allocation})
  (conv:larray BF.t 442368)
  (metadata:O.cache_configuration)
  (#old_state:lseq (option FP.t) 4718592)
  (#old_conv:lseq (option BF.t) 442368)
  preserves gpu
  requires Array.pts_to_mask state old_state Initialization.all_cells
  requires Array.pts_to_mask conv old_conv Initialization.all_cells
  ensures state |-> Initialization.state_values
  ensures conv |-> Initialization.convolution_values
  ensures pure (
    forall (layer:Model.linear_layer).
      O.recurrent_input (O.empty_cache #O.cache_configuration #BF.t #FP.t #BF.t metadata)
        layer (Initialization.state_layer Initialization.state_values layer) /\
      O.convolution_input (O.empty_cache #O.cache_configuration #BF.t #FP.t #BF.t metadata)
        layer (Initialization.convolution_layer Initialization.convolution_values layer))
{
  Initialization.initialize_caches state conv;
  Initialization.all_first_use_inputs metadata;
}

noextract
fn source_caller_allocations
  (call:Model.invocation) (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns outcome:Caller.allocation call
  ensures (
    match outcome with
    | Caller.Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | Caller.Allocated b ->
      Caller.owners stream call b ** pure (Caller.shapes call b) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (Caller.pending_contents call b frame))
{
  Caller.allocate call stream #epoch frame ()
}

noextract
fn source_native_input_queries
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length) (frame:slprop)
  preserves cpu ** frame
  returns addresses:(Array.array BF.t & Array.array I64.t & I64.t)
  ensures pure (NativeInput.pointer_result inputs addresses)
{
  NativeInput.source_pointer_queries inputs frame
}

let source_native_embedding_address call inputs token i =
  Inputs.embedding_cell_binding inputs token i

inline_for_extraction noextract
fn source_native_embedding_step
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.prompt_length})
  (weights:larray BF.t 752393024{weights == Input.data inputs.weights})
  (prompt:larray I64.t call.prompt_length{prompt == Input.data inputs.prompt})
  (next:I64.t) (hidden:larray BF.t 16000)
  (#values:Transfer.packed) (#prompt_values:lseq I64.t call.prompt_length)
  (#logical:Selection.vec (Selection.token 248320) call.prompt_length)
  (#before:Rotary.snapshot) (#fraction:perm)
  preserves gpu
  preserves Storage.inputs_owns call I64.v fraction (Inputs.parameter_values values)
    (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs prompt_values)
  requires pure (SZ.v position < call.prompt_length \/
    (0 <= I64.v next /\ I64.v next < 248320))
  requires Array.pts_to_mask hidden before Embedding.embedding_mask
  returns token:Inputs.token_value
  ensures Array.pts_to_mask hidden
    (Embedding.writes before values (I64.v token) 1024) Embedding.embedding_mask
  ensures pure (
    token == Embedding.source_token call (SZ.v position) prompt_values next /\
    KPlay.Proof.Shared.Embedding.embedding_post
      (Inputs.parameter_values values).O.embedding (I64.v token)
      (Embedding.row values (I64.v token)))
{
  Inputs.embedding_step call inputs position prompt_length weights prompt next hidden
}

noextract
fn source_caller_arguments
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (stream:Stream.stream_t) (b:Caller.buffers call)
  (steps:I64.t{I64.v steps == call.decode_steps})
  preserves cpu ** Input.reference inputs.weights ** Input.reference inputs.prompt
  preserves Caller.owners stream call b
  requires pure (Caller.shapes call b)
  returns args:Entry.arguments call
  ensures pure (Entry.bound call inputs b args)
{
  Entry.query_arguments call inputs stream b steps
}

noextract
fn source_caller_private_release
  (call:Model.invocation) (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (b:Caller.buffers call) (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires Entry.private_owners stream call b
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) (Entry.private_contents call b frame)
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
{
  Entry.release_private call stream #epoch b frame ()
}

let source_rms_model_connection before weights residual =
  Norm.final_connection before weights residual

inline_for_extraction noextract
fn source_final_normalization
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch)
  (#before:erased NormNative.partials) (#fraction:perm)
  preserves gpu ** Array.pts_to weights #fraction vw
  requires Array.pts_to hidden (FStar.Ghost.reveal vh) **
    Array.pts_to partial (FStar.Ghost.reveal before)
  ensures exists* (after:NormNative.partials).
    Array.pts_to hidden (Norm.final_writes vh vw) ** Array.pts_to partial after
{
  NormNative.source_final_norm weights hidden partial
}

inline_for_extraction noextract
fn source_full_projection
  (weight:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (#vw:C.chest2 BF.t 752393024 1) (#vh:C.chest2 BF.t 16000 1)
  (#weight_fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weight |-> Frac weight_fraction vw
  requires hidden |-> Frac (1.0R /. 32) vh
  requires Head.selection_at best next Vocabulary.seed (SZ.v tid)
  ensures hidden |-> Frac (1.0R /. 32)
    (Vocabulary.scratch_after (Native.packed_values vw) vh 15520)
  ensures Head.selection_at best next
    (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values vh) 15520) (SZ.v tid)
{
  Head.final_projection weight hidden best next tid
}

ghost
fn source_input_arrays
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (weights:Transfer.packed) (prompt:lseq I64.t call.prompt_length)
  (logical:Selection.vec (Selection.token 248320) call.prompt_length) (fraction:perm)
  requires Shared.caller_precondition call I64.v fraction (Inputs.parameter_values weights)
    (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs prompt)
  ensures Array.pts_to (Input.data inputs.weights) #fraction weights **
    Array.pts_to (Input.data inputs.prompt) #fraction prompt
  ensures pure (forall (i:P.index call.prompt_length). I64.v (FStar.Seq.index prompt i) == FStar.Seq.index logical i)
{
  unfold Shared.caller_precondition call I64.v fraction (Inputs.parameter_values weights)
    (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs prompt);
  unfold Storage.inputs_owns call I64.v fraction (Inputs.parameter_values weights)
    (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs prompt);
  PackedOwnership.parameters_to_array inputs weights fraction;
  InputOwnership.prompt_to_array inputs prompt logical fraction;
}

ghost
fn source_restore_inputs
  (call:Model.invocation) (inputs:NativeInput.arguments call.prompt_length)
  (weights:Transfer.packed) (prompt:lseq I64.t call.prompt_length)
  (logical:Selection.vec (Selection.token 248320) call.prompt_length) (fraction:perm)
  requires Array.pts_to (Input.data inputs.weights) #fraction weights **
    Array.pts_to (Input.data inputs.prompt) #fraction prompt
  requires pure (forall (i:P.index call.prompt_length). I64.v (FStar.Seq.index prompt i) == FStar.Seq.index logical i)
  ensures Shared.caller_precondition call I64.v fraction (Inputs.parameter_values weights)
    (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs prompt)
{
  PackedOwnership.array_to_parameters inputs weights fraction;
  InputOwnership.array_to_prompt inputs prompt logical fraction;
  fold Storage.inputs_owns call I64.v fraction (Inputs.parameter_values weights)
    (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs prompt);
  fold Shared.caller_precondition call I64.v fraction (Inputs.parameter_values weights)
    (Inputs.parameter_cells inputs) logical (Inputs.prompt_storage inputs prompt);
}

inline_for_extraction noextract
fn source_decoder_weight_projection
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24})
  (phase:Schedule.phase layer{not (Schedule.attention layer phase)})
  (weight:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (#vw:C.chest2 BF.t 752393024 1) (#vh:C.chest2 BF.t 16000 1) (#weight_fraction:perm)
  preserves gpu ** weight |-> Frac weight_fraction vw
  requires hidden |-> Frac (1.0R /. 32) vh
  ensures hidden |-> Frac (1.0R /. 32)
    (MatvecWrites.after (SourceProjection.weight_geometry call position layer phase) vw vh
      (Schedule.projection_at call position layer phase).Schedule.rows)
{
  SourceProjection.weight_projection call position layer phase weight hidden;
}

inline_for_extraction noextract
fn source_attention_mma_projection
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer)
  (phase:Schedule.phase layer{
    Schedule.attention layer phase /\ Schedule.mma_path layer phase position})
  (cache:larray BF.t 12582912) (hidden:array2 BF.t Native.hidden_layout)
  (#vk:erased CacheNative.snapshot) (#vh:C.chest2 BF.t 16000 1) (#cache_fraction:perm)
  (ready:squash (CacheNative.initialized vk layer (position + 1)))
  preserves gpu ** Array.pts_to_mask cache #cache_fraction vk Initialization.all_cells
  requires hidden |-> Frac (1.0R /. 32) vh
  ensures hidden |-> Frac (1.0R /. 32)
    (MatvecWrites.after (CacheProjection.geometry call position layer phase)
      (CacheProjection.tensor_contents vk call position layer phase ready) vh
      (Schedule.projection_at call position layer phase).Schedule.rows)
{
  CacheProjection.mma call position layer phase cache hidden #vk #vh #cache_fraction ready;
}

inline_for_extraction noextract
fn source_attention_simt_projection
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer)
  (phase:Schedule.phase layer{
    Schedule.attention layer phase /\ not (Schedule.mma_path layer phase position)})
  (cache:larray BF.t 12582912) (hidden:larray BF.t 16000)
  (#fraction:perm)
  (#sw:erased CacheNative.snapshot) (#sx:erased (lseq BF.t 16000))
  (#wr:erased (A.matrix (Schedule.projection_at call position layer phase).Schedule.rows
    (Schedule.projection_at call position layer phase).Schedule.columns))
  (#xr:erased (A.vector (Schedule.projection_at call position layer phase).Schedule.columns)) (#scope:prop)
  (ready:squash (CacheNative.initialized sw layer (position + 1)))
  preserves gpu ** Array.pts_to_mask cache #fraction sw Initialization.all_cells
  requires Array.pts_to hidden (FStar.Ghost.reveal sx)
  requires pure (scope ==>
    SIMTMath.matrix_rep (CacheProjection.geometry call position layer phase)
      (CacheProjection.contents sw call position layer phase ready) wr /\
    SIMTMath.input_rep (CacheProjection.geometry call position layer phase) sx xr)
  ensures exists* (result:lseq BF.t 16000).
    Array.pts_to hidden result **
    pure (SIMT.frame (CacheProjection.geometry call position layer phase) sx result
        (Schedule.projection_at call position layer phase).Schedule.rows /\
      (scope ==> SIMT.produced (CacheProjection.geometry call position layer phase) result wr xr
        (Schedule.projection_at call position layer phase).Schedule.rows))
{
  CacheProjection.simt call position layer phase cache hidden #sw #sx #wr #xr #fraction #scope ready;
}
