module KPlay.Proof.Base.KernelNative

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module Cast = FStar.Int.Cast
module SZ = Kuiper.SizeT
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Entry = KPlay.Proof.Base.Entry
module Owned = KPlay.Proof.Base.KernelOwnership
module Book = KPlay.Proof.Base.WarpBook
module Head = KPlay.Proof.Base.Head
module Native = KPlay.Proof.Base.NativeProjection
module Start = KPlay.Proof.Base.KernelStart
module Positions = KPlay.Proof.Base.KernelNativePosition

inline_for_extraction noextract
fn apply
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#sp:erased (lseq I64.t call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#wf #pf:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac wf vw ** A.pts_to args.Entry.prompt #pf sp
  requires pure (forall (i:P.index call.Model.prompt_length). I64.v (Seq.index sp i) == Seq.index prompt i)
  requires Head.lane_only (SZ.v tid) (Start.pending call refs args hidden best next)
  ensures Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt (Model.calls call)))
{
  if (FStar.SizeT.eq tid 0sz) {
    rewrite Head.lane_only (SZ.v tid) (Start.pending call refs args hidden best next)
      as Start.pending call refs args hidden best next;
    Start.initialize call refs args hidden best next #(Native.packed_values vw) #prompt;
    rewrite Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt 0)
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.ready call (Native.packed_values vw) prompt 0));
  } else {
    rewrite Head.lane_only (SZ.v tid) (Start.pending call refs args hidden best next)
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.ready call (Native.packed_values vw) prompt 0));
  };
  Book.keep_wait (fun i -> Head.lane_only i
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.ready call (Native.packed_values vw) prompt 0))) #(SZ.v tid);
  let prompt_length = SZ.uint32_to_sizet (Cast.int64_to_uint32 args.Entry.prompt_length);
  let steps = SZ.uint32_to_sizet (Cast.int64_to_uint32 args.Entry.steps);
  let count = SZ.add prompt_length steps -^ 1sz;
  Positions.apply_all call refs args weights hidden best next tid prompt_length count #prompt #sp #vw #wf #pf;
}
