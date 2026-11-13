module KPlay.Proof.Base.KernelControls

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Lib.Send

module Seq = FStar.Seq
module GR = Pulse.Lib.GhostReference
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Owned = KPlay.Proof.Base.KernelOwnership
module Start = KPlay.Proof.Base.KernelStart
module Layers = KPlay.Proof.Base.KernelLayers
module Linear = KPlay.Proof.Base.LinearBook
module Attention = KPlay.Proof.Base.AttentionBook
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let parked (call:Model.invocation) (refs:Owned.controls call) : slprop =
  Owned.parked refs.Owned.global ** Owned.parked refs.Owned.linear ** Owned.parked refs.Owned.attention

ghost
fn allocate
  (call:Model.invocation)
  (#weights:erased Transfer.packed) (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  returns refs:erased (Owned.controls call)
  ensures parked call refs
{
  let seed : erased (K.metadata call) = hide #(K.metadata call)
    (Start.initial call (Seq.create #(option I64.t) call.Model.decode_steps None)
      (Seq.create #(option BF.t) (call.Model.decode_steps * 1024) None) (zero #FP.t));
  let hidden : erased Transfer.scratch = hide #Transfer.scratch (Seq.create 16000 (zero #BF.t));
  let cache : erased KPlay.Proof.Base.CacheNative.snapshot =
    hide #KPlay.Proof.Base.CacheNative.snapshot (Seq.create #(option BF.t) 12582912 None);
  let global = GR.alloc #(K.metadata call) seed;
  let linear_seed : erased (Linear.metadata (K.metadata call)) =
    hide #(Linear.metadata (K.metadata call)) (Layers.linear_start call weights prompt 0 0 seed (Linear.tensor hidden) cache);
  let linear = GR.alloc #(Linear.metadata (K.metadata call)) linear_seed;
  let attention_seed : erased (Attention.metadata (K.metadata call)) =
    hide #(Attention.metadata (K.metadata call)) (Layers.attention_start call weights prompt 0 3 seed (Linear.tensor hidden) cache);
  let attention = GR.alloc #(Attention.metadata (K.metadata call)) attention_seed;
  fold Owned.parked global;
  fold Owned.parked linear;
  fold Owned.parked attention;
  fold parked call ({Owned.global = global; Owned.linear = linear; Owned.attention = attention});
  hide #(Owned.controls call) ({Owned.global = global; Owned.linear = linear; Owned.attention = attention})
}

ghost
fn free (call:Model.invocation) (refs:Owned.controls call)
  requires parked call refs
{
  unfold parked call refs;
  unfold Owned.parked refs.Owned.attention;
  with attention. assert GR.pts_to refs.Owned.attention attention;
  GR.free refs.Owned.attention;
  unfold Owned.parked refs.Owned.linear;
  with linear. assert GR.pts_to refs.Owned.linear linear;
  GR.free refs.Owned.linear;
  unfold Owned.parked refs.Owned.global;
  with global. assert GR.pts_to refs.Owned.global global;
  GR.free refs.Owned.global;
}
