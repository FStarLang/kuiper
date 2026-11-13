module KPlay.Proof.Base.KernelOwnership

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Array

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module GR = Pulse.Lib.GhostReference
module FP = Kuiper.Float32
module I64 = FStar.Int64
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Entry = KPlay.Proof.Base.Entry
module Init = KPlay.Proof.Base.Initialization
module Linear = KPlay.Proof.Base.LinearBook
module Attention = KPlay.Proof.Base.AttentionBook

[@@erasable]
noeq type controls (call:Model.invocation) = {
  global:GR.ref (K.metadata call);
  linear:GR.ref (Linear.metadata (K.metadata call));
  attention:GR.ref (Attention.metadata (K.metadata call));
}

inline_for_extraction noextract
let parked (#a:Type0) (reference:GR.ref a) : slprop =
  exists* (value:a). GR.pts_to reference value

inline_for_extraction noextract
let other_memory (call:Model.invocation) (args:Entry.arguments call)
  (best:larray FP.t 1) (next:larray I64.t 1) (m:K.memory call) : slprop =
  A.pts_to_mask args.Entry.tokens m.K.tokens Init.all_cells **
  A.pts_to_mask args.Entry.hidden m.K.hidden Init.all_cells **
  best |-> Seq.create 1 m.K.best ** next |-> Seq.create 1 m.K.next

inline_for_extraction noextract
let memory (call:Model.invocation) (args:Entry.arguments call)
  (best:larray FP.t 1) (next:larray I64.t 1) (m:K.memory call) : slprop =
  args.Entry.state |-> m.K.state ** A.pts_to args.Entry.convolution m.K.convolution **
  other_memory call args best next m

inline_for_extraction noextract
let local_owners (call:Model.invocation) (refs:controls call) (args:Entry.arguments call)
  (best:larray FP.t 1) (next:larray I64.t 1) (m:K.metadata call) : slprop =
  memory call args best next m.K.memory ** parked refs.linear ** parked refs.attention

inline_for_extraction noextract
let linear_extra (call:Model.invocation) (refs:controls call) (args:Entry.arguments call)
  (best:larray FP.t 1) (next:larray I64.t 1) (m:K.metadata call) : slprop =
  GR.pts_to refs.global m ** other_memory call args best next m.K.memory ** parked refs.attention

inline_for_extraction noextract
let attention_extra (call:Model.invocation) (refs:controls call) (args:Entry.arguments call)
  (best:larray FP.t 1) (next:larray I64.t 1) (m:K.metadata call) : slprop =
  GR.pts_to refs.global m ** memory call args best next m.K.memory ** parked refs.linear
