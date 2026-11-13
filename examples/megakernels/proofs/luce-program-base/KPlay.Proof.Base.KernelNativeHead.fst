module KPlay.Proof.Base.KernelNativeHead

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module C = Kuiper.Chest
module T = Kuiper.Tensor
module GR = Pulse.Lib.GhostReference
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Math = KPlay.Proof.Base.KernelHead
module Result = KPlay.Proof.Base.KernelHeadResult
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Book = KPlay.Proof.Base.WarpBook
module Head = KPlay.Proof.Base.Head
module Native = KPlay.Proof.Base.NativeProjection
module Arrays = KPlay.Proof.Base.SourceArrays
module Transfer = KPlay.Proof.Base.Transfer
module Linear = KPlay.Proof.Base.LinearBook
module Init = KPlay.Proof.Base.Initialization
module History = KPlay.Proof.Base.HistoryNative
module RMS = KPlay.Proof.Base.RMS
module Config = KPlay.Proof.Base.RMSConfigs
module Selection = KPlay.Proof.Base.NativeSelection
module Vocabulary = KPlay.Proof.Base.Vocabulary

inline_for_extraction noextract
fn capture
  (call:Model.invocation) (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (hidden:larray BF.t 16000) (output:larray BF.t (call.Model.decode_steps * 1024))
  (#vh:erased Transfer.scratch)
  (#before:erased (lseq (option BF.t) (call.Model.decode_steps * 1024)))
  preserves gpu ** A.pts_to hidden vh
  requires A.pts_to_mask output before Init.all_cells
  ensures A.pts_to_mask output
    (if SZ.v position < call.Model.prompt_length - 1 then before
      else History.hidden_writes call before vh (History.retained_row call (SZ.v position)) 1024) Init.all_cells
{
  if (position <^ prompt_length -^ 1sz) {
  } else {
    History.capture call position prompt_length hidden output;
  }
}

inline_for_extraction noextract
fn prelude
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1)
  (position:sz{SZ.v position < Model.calls call}) (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm)
  preserves gpu ** weights |-> Frac fraction vw
  requires Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call (Native.packed_values vw) prompt (SZ.v position) 24)
  ensures Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.head_ready call (Native.packed_values vw) prompt (SZ.v position))
{
  unfold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call (Native.packed_values vw) prompt (SZ.v position) 24);
  with m before kv. assert GR.pts_to refs.Owned.global m ** hidden |-> before ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  K.layer_facts call (Native.packed_values vw) prompt (SZ.v position) 24 m before kv;
  unfold Owned.local_owners call refs args best next m;
  unfold Owned.memory call args best next (reveal #(K.metadata call) m).K.memory;
  unfold Owned.other_memory call args best next (reveal #(K.metadata call) m).K.memory;
  Arrays.weight_to_array weights;
  Head.hidden_to_array hidden #before #1.0R;
  capture call position prompt_length (T.core hidden) args.Entry.hidden;
  RMS.source_apply (Config.model_norm 24 false) (T.core weights) (T.core hidden);
  Selection.initialize best next;
  unfold Selection.owns best next Vocabulary.seed;
  with word. assert next |-> Seq.create 1 word;
  assert pure (word == 0L);
  Math.start_valid call (Native.packed_values vw) prompt (SZ.v position) m before kv;
  let fresh : erased (K.metadata call) = hide #(K.metadata call)
    (Math.start call (Native.packed_values vw) m (K.decoding call m) (Native.hidden_values before));
  GR.write refs.Owned.global fresh;
  Linear.tensor_values (Math.normalized (Native.packed_values vw) (Native.hidden_values before));
  Head.array_to_hidden hidden #(Linear.tensor (Math.normalized (Native.packed_values vw) (Native.hidden_values before))) #1.0R;
  Arrays.array_to_weight weights;
  fold Owned.other_memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.local_owners call refs args best next fresh;
  fold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.head_ready call (Native.packed_values vw) prompt (SZ.v position));
}

inline_for_extraction noextract
let without_selector (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (m:K.metadata call) : slprop =
  args.Entry.state |-> m.K.memory.K.state ** A.pts_to args.Entry.convolution m.K.memory.K.convolution **
  A.pts_to_mask args.Entry.tokens m.K.memory.K.tokens Init.all_cells **
  A.pts_to_mask args.Entry.hidden m.K.memory.K.hidden Init.all_cells **
  Owned.parked refs.Owned.linear ** Owned.parked refs.Owned.attention

inline_for_extraction noextract
let selected_owners (call:Model.invocation) (weights:Transfer.packed)
  (best:larray FP.t 1) (next:larray I64.t 1) (m:K.metadata call) : slprop =
  match m.K.phase with
  | K.Normalized _ values -> Selection.owns best next (Vocabulary.scan_blocks weights values 15520)
  | _ -> pure False

inline_for_extraction noextract
let projected_owners (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:Transfer.packed) (best:larray FP.t 1) (next:larray I64.t 1) (m:K.metadata call) : slprop =
  without_selector call refs args m ** selected_owners call weights best next m

inline_for_extraction noextract
let observed_selection (best:larray FP.t 1) (next:larray I64.t 1)
  (expected:Selection.bounded_selection) : slprop =
  exists* (actual:Selection.bounded_selection).
    Selection.owns best next actual ** pure (actual == expected)

inline_for_extraction noextract
fn projection
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (position:nat{position < Model.calls call})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.head_ready call (Native.packed_values vw) prompt position))
  ensures Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache
      (projected_owners call refs args (Native.packed_values vw) best next)
      (Result.projected call (Native.packed_values vw) prompt position))
{
  Book.scatter_wait refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.head_ready call (Native.packed_values vw) prompt position) #(SZ.v tid);
  unfold Book.shared refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.head_ready call (Native.packed_values vw) prompt position) (SZ.v tid);
  with m before kv. assert Book.chunk refs.Owned.global hidden args.Entry.cache m before kv;
  Math.ready_facts call (Native.packed_values vw) prompt position m before kv;
  unfold Book.chunk refs.Owned.global hidden args.Entry.cache m before kv;
  if (FStar.SizeT.eq tid 0sz) {
    rewrite Head.lane_only (SZ.v tid) (Owned.local_owners call refs args best next m)
      as Owned.local_owners call refs args best next m;
    unfold Owned.local_owners call refs args best next m;
    unfold Owned.memory call args best next (reveal #(K.metadata call) m).K.memory;
    unfold Owned.other_memory call args best next (reveal #(K.metadata call) m).K.memory;
    fold without_selector call refs args m;
    fold Selection.owns best next Vocabulary.seed;
    rewrite without_selector call refs args m as Head.lane_only (SZ.v tid) (without_selector call refs args m);
    rewrite Selection.owns best next Vocabulary.seed as Head.selection_at best next Vocabulary.seed (SZ.v tid);
  } else {
    rewrite Head.lane_only (SZ.v tid) (Owned.local_owners call refs args best next m)
      as Head.lane_only (SZ.v tid) (without_selector call refs args m);
    rewrite emp as Head.selection_at best next Vocabulary.seed (SZ.v tid);
  };
  Head.final_projection weights hidden best next tid #vw #before #fraction;
  Result.projection_valid call (Native.packed_values vw) prompt position m before kv;
  if (FStar.SizeT.eq tid 0sz) {
    rewrite Head.lane_only (SZ.v tid) (without_selector call refs args m) as without_selector call refs args m;
    rewrite Head.selection_at best next (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values before) 15520) (SZ.v tid)
      as Selection.owns best next (Vocabulary.scan_blocks (Native.packed_values vw) (Math.values call m) 15520);
    rewrite Selection.owns best next (Vocabulary.scan_blocks (Native.packed_values vw) (Math.values call m) 15520)
      as selected_owners call (Native.packed_values vw) best next m;
    fold projected_owners call refs args (Native.packed_values vw) best next m;
    rewrite projected_owners call refs args (Native.packed_values vw) best next m
      as Head.lane_only (SZ.v tid) (projected_owners call refs args (Native.packed_values vw) best next m);
  } else {
    rewrite Head.selection_at best next (Vocabulary.scan_blocks (Native.packed_values vw) (Native.hidden_values before) 15520) (SZ.v tid) as emp;
    rewrite Head.lane_only (SZ.v tid) (without_selector call refs args m)
      as Head.lane_only (SZ.v tid) (projected_owners call refs args (Native.packed_values vw) best next m);
  };
  fold Book.chunk refs.Owned.global hidden args.Entry.cache m (Vocabulary.scratch_after (Native.packed_values vw) before 15520) kv;
  fold Book.shared refs.Owned.global hidden args.Entry.cache
    (projected_owners call refs args (Native.packed_values vw) best next)
    (Result.projected call (Native.packed_values vw) prompt position) (SZ.v tid);
  Book.gather_wait refs.Owned.global hidden args.Entry.cache
    (projected_owners call refs args (Native.packed_values vw) best next)
    (Result.projected call (Native.packed_values vw) prompt position) #(SZ.v tid);
}

ghost
fn finish
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (weights:Transfer.packed) (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (position:nat{position < Model.calls call}) ()
  requires Book.full refs.Owned.global hidden args.Entry.cache (projected_owners call refs args weights best next)
    (Result.projected call weights prompt position)
  ensures Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.emitting call weights prompt position)
{
  unfold Book.full refs.Owned.global hidden args.Entry.cache (projected_owners call refs args weights best next)
    (Result.projected call weights prompt position);
  with m h kv. assert GR.pts_to refs.Owned.global m ** hidden |-> h ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  Result.projected_facts call weights prompt position m h kv;
  unfold projected_owners call refs args weights best next m;
  rewrite selected_owners call weights best next m
    as Selection.owns best next (Vocabulary.scan_blocks weights (Math.values call m) 15520);
  Head.prefix_bound weights (Math.values call m) 15520;
  fold observed_selection best next (Vocabulary.scan_blocks weights (Math.values call m) 15520);
  unfold observed_selection best next (Vocabulary.scan_blocks weights (Math.values call m) 15520);
  with winner. assert Selection.owns best next winner;
  unfold Selection.owns best next winner;
  with word. assert next |-> Seq.create 1 word;
  Result.finish_valid call weights prompt position m h kv winner word;
  let fresh : erased (K.metadata call) = hide #(K.metadata call) (Result.finish call weights m winner word);
  Result.finish_memory call weights m winner word;
  GR.write refs.Owned.global fresh;
  unfold without_selector call refs args m;
  fold Owned.other_memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.local_owners call refs args best next fresh;
  fold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.emitting call weights prompt position);
}

inline_for_extraction noextract
fn apply
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (position:sz{SZ.v position < Model.calls call}) (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt (SZ.v position) 24))
  ensures Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.emitting call (Native.packed_values vw) prompt (SZ.v position)))
{
  if (FStar.SizeT.eq tid 0sz) {
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt (SZ.v position) 24))
      as Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt (SZ.v position) 24);
    prelude call refs args weights hidden best next position prompt_length #prompt #vw #fraction;
    rewrite Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.head_ready call (Native.packed_values vw) prompt (SZ.v position))
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.head_ready call (Native.packed_values vw) prompt (SZ.v position)));
  } else {
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt (SZ.v position) 24))
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.head_ready call (Native.packed_values vw) prompt (SZ.v position)));
  };
  projection call refs args weights hidden best next tid (SZ.v position) #prompt #vw #fraction;
  Book.map_lane_zero (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache
      (projected_owners call refs args (Native.packed_values vw) best next)
      (Result.projected call (Native.packed_values vw) prompt (SZ.v position)))
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.emitting call (Native.packed_values vw) prompt (SZ.v position)))
    (finish call refs args hidden best next (Native.packed_values vw) prompt (SZ.v position));
  Book.keep_wait (fun i -> Head.lane_only i
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.emitting call (Native.packed_values vw) prompt (SZ.v position)))) #(SZ.v tid);
}
