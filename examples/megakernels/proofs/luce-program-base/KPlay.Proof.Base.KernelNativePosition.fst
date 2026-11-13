module KPlay.Proof.Base.KernelNativePosition

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
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Math = KPlay.Proof.Base.KernelPosition
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Book = KPlay.Proof.Base.WarpBook
module Head = KPlay.Proof.Base.Head
module Native = KPlay.Proof.Base.NativeProjection
module Arrays = KPlay.Proof.Base.SourceArrays
module Transfer = KPlay.Proof.Base.Transfer
module Linear = KPlay.Proof.Base.LinearBook
module Init = KPlay.Proof.Base.Initialization
module Embedding = KPlay.Proof.Base.KernelEmbedding
module History = KPlay.Proof.Base.HistoryNative
module Layers = KPlay.Proof.Base.KernelNativeLayers
module Final = KPlay.Proof.Base.KernelNativeHead

inline_for_extraction noextract
fn enter
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1)
  (position:sz{SZ.v position < Model.calls call}) (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#sp:erased (lseq I64.t call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#wf #pf:perm)
  preserves gpu ** weights |-> Frac wf vw ** A.pts_to args.Entry.prompt #pf sp
  requires pure (forall (i:P.index call.Model.prompt_length). I64.v (Seq.index sp i) == Seq.index prompt i)
  requires Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call (Native.packed_values vw) prompt (SZ.v position))
  ensures Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call (Native.packed_values vw) prompt (SZ.v position) 0)
{
  unfold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call (Native.packed_values vw) prompt (SZ.v position));
  with m before kv. assert GR.pts_to refs.Owned.global m ** hidden |-> before ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  Math.ready_facts call (Native.packed_values vw) prompt (SZ.v position) m before kv;
  Math.token_binding call (Native.packed_values vw) prompt sp (SZ.v position) m before kv;
  unfold Owned.local_owners call refs args best next m;
  unfold Owned.memory call args best next (reveal #(K.metadata call) m).K.memory;
  unfold Owned.other_memory call args best next (reveal #(K.metadata call) m).K.memory;
  Arrays.weight_to_array weights;
  Head.hidden_to_array hidden;
  let word = Embedding.apply call position prompt_length (T.core weights) args.Entry.prompt (T.core hidden) next;
  with after. assert A.pts_to (T.core hidden) after;
  assert pure (I64.v word == (K.caller call (Native.packed_values vw) prompt m).D.next_token);
  Math.enter_valid call (Native.packed_values vw) prompt (SZ.v position) m before kv (I64.v word) after;
  let fresh : erased (K.metadata call) = hide #(K.metadata call)
    (Math.enter call (Native.packed_values vw) prompt m (I64.v word) after);
  GR.write refs.Owned.global fresh;
  Linear.tensor_values after;
  Head.array_to_hidden hidden #(Linear.tensor after);
  Arrays.array_to_weight weights;
  fold Owned.other_memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.local_owners call refs args best next fresh;
  fold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call (Native.packed_values vw) prompt (SZ.v position) 0);
}

inline_for_extraction noextract
fn emit_token
  (call:Model.invocation) (position:sz{SZ.v position < Model.calls call})
  (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (next:larray I64.t 1) (tokens:larray I64.t call.Model.decode_steps)
  (#word:erased I64.t) (#before:erased (lseq (option I64.t) call.Model.decode_steps))
  preserves gpu ** next |-> Seq.create #I64.t 1 word
  requires A.pts_to_mask tokens before Init.all_cells
  ensures A.pts_to_mask tokens
    (if SZ.v position < call.Model.prompt_length - 1 then before
      else Seq.upd before (History.retained_row call (SZ.v position)) (Some #I64.t word)) Init.all_cells
{
  if (position <^ prompt_length -^ 1sz) {
  } else {
    History.emit call position prompt_length next tokens;
  }
}

inline_for_extraction noextract
fn leave
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (position:sz{SZ.v position < Model.calls call}) (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (#weights:erased Transfer.packed) (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  preserves gpu
  requires Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.emitting call weights prompt (SZ.v position))
  ensures Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call weights prompt (SZ.v position + 1))
{
  unfold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.emitting call weights prompt (SZ.v position));
  with m h kv. assert GR.pts_to refs.Owned.global m ** hidden |-> h ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  Math.emitting_facts call weights prompt (SZ.v position) m h kv;
  unfold Owned.local_owners call refs args best next m;
  unfold Owned.memory call args best next (reveal #(K.metadata call) m).K.memory;
  unfold Owned.other_memory call args best next (reveal #(K.metadata call) m).K.memory;
  emit_token call position prompt_length next args.Entry.tokens;
  Math.advance_valid call weights prompt (SZ.v position) m h kv;
  let fresh : erased (K.metadata call) = hide #(K.metadata call) (Math.advance call m);
  GR.write refs.Owned.global fresh;
  fold Owned.other_memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.memory call args best next (reveal #(K.metadata call) fresh).K.memory;
  fold Owned.local_owners call refs args best next fresh;
  fold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.ready call weights prompt (SZ.v position + 1));
}

inline_for_extraction noextract
fn apply
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (position:sz{SZ.v position < Model.calls call}) (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#sp:erased (lseq I64.t call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#wf #pf:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac wf vw ** A.pts_to args.Entry.prompt #pf sp
  requires pure (forall (i:P.index call.Model.prompt_length). I64.v (Seq.index sp i) == Seq.index prompt i)
  requires Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt (SZ.v position)))
  ensures Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt (SZ.v position + 1)))
{
  if (FStar.SizeT.eq tid 0sz) {
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.ready call (Native.packed_values vw) prompt (SZ.v position)))
      as Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.ready call (Native.packed_values vw) prompt (SZ.v position));
    enter call refs args weights hidden best next position prompt_length #prompt #sp #vw #wf #pf;
    rewrite Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt (SZ.v position) 0)
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.layer call (Native.packed_values vw) prompt (SZ.v position) 0));
  } else {
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.ready call (Native.packed_values vw) prompt (SZ.v position)))
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.layer call (Native.packed_values vw) prompt (SZ.v position) 0));
  };
  Book.keep_wait (fun i -> Head.lane_only i
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt (SZ.v position) 0))) #(SZ.v tid);
  Layers.apply_all call refs args weights hidden best next tid (SZ.v position) #prompt #vw #wf;
  Final.apply call refs args weights hidden best next tid position prompt_length #prompt #vw #wf;
  if (FStar.SizeT.eq tid 0sz) {
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.emitting call (Native.packed_values vw) prompt (SZ.v position)))
      as Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.emitting call (Native.packed_values vw) prompt (SZ.v position));
    leave call refs args hidden best next position prompt_length #(Native.packed_values vw) #prompt;
    rewrite Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt (SZ.v position + 1))
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.ready call (Native.packed_values vw) prompt (SZ.v position + 1)));
  } else {
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.emitting call (Native.packed_values vw) prompt (SZ.v position)))
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.ready call (Native.packed_values vw) prompt (SZ.v position + 1)));
  };
  Book.keep_wait (fun i -> Head.lane_only i
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt (SZ.v position + 1)))) #(SZ.v tid);
}

inline_for_extraction noextract
fn apply_all
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (prompt_length:sz{SZ.v prompt_length == call.Model.prompt_length})
  (count:sz{SZ.v count == Model.calls call})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#sp:erased (lseq I64.t call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#wf #pf:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac wf vw ** A.pts_to args.Entry.prompt #pf sp
  requires pure (forall (i:P.index call.Model.prompt_length). I64.v (Seq.index sp i) == Seq.index prompt i)
  requires Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt 0))
  ensures Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt (Model.calls call)))
{
  let mut position = 0sz;
  while (!position <^ count)
    invariant exists* (p:sz{SZ.v p <= Model.calls call}).
      position |-> p ** Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.ready call (Native.packed_values vw) prompt (SZ.v p)))
    invariant gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac wf vw ** A.pts_to args.Entry.prompt #pf sp
    decreases (Model.calls call - SZ.v !position)
  {
    let p = !position;
    apply call refs args weights hidden best next tid p prompt_length #prompt #sp #vw #wf #pf;
    let following = SZ.add p 1sz;
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.ready call (Native.packed_values vw) prompt (SZ.v p + 1)))
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.ready call (Native.packed_values vw) prompt (SZ.v following)));
    position := following;
  };
  let completed = !position;
  assert pure (SZ.v completed == Model.calls call);
  rewrite Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt (SZ.v completed)))
    as Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.ready call (Native.packed_values vw) prompt (Model.calls call)));
}
