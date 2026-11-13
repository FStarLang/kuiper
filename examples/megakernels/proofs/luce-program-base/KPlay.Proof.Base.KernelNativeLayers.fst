module KPlay.Proof.Base.KernelNativeLayers

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module C = Kuiper.Chest
module GR = Pulse.Lib.GhostReference
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module D = KPlay.Proof.Shared.Selection
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Entry = KPlay.Proof.Base.Entry
module K = KPlay.Proof.Base.KernelState
module Layers = KPlay.Proof.Base.KernelLayers
module LTrace = KPlay.Proof.Base.LayerTrace
module Owned = KPlay.Proof.Base.KernelOwnership
module Book = KPlay.Proof.Base.WarpBook
module Head = KPlay.Proof.Base.Head
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Init = KPlay.Proof.Base.Initialization
module Global = KPlay.Proof.Base.GlobalCache
module Cache = KPlay.Proof.Base.Cache
module Linear = KPlay.Proof.Base.LinearBook
module LinearLayer = KPlay.Proof.Base.LinearLayer
module Attention = KPlay.Proof.Base.AttentionBook
module AttentionLayer = KPlay.Proof.Base.AttentionLayer

ghost
fn enter_linear
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (weights:Transfer.packed) (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (position:nat{position < Model.calls call}) (layer:Model.linear_layer) ()
  requires Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position layer)
  ensures Book.full refs.Owned.linear hidden args.Entry.cache
    (LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next))
    (Linear.valid call position layer weights True (Layers.linear_entry call weights prompt position layer) 0)
{
  unfold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position layer);
  with m h kv. assert GR.pts_to refs.Owned.global m ** hidden |-> h ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  K.layer_facts call weights prompt position layer m h kv;
  unfold Owned.local_owners call refs args best next m;
  unfold Owned.memory call args best next (reveal #(K.metadata call) m).K.memory;
  unfold Owned.parked refs.Owned.linear;
  with old. assert GR.pts_to refs.Owned.linear old;
  Layers.linear_start_valid call weights prompt position layer m h kv;
  let value : erased (Linear.metadata (K.metadata call)) =
    hide #(Linear.metadata (K.metadata call)) (Layers.linear_start call weights prompt position layer m h kv);
  GR.write refs.Owned.linear value;
  fold Owned.linear_extra call refs args best next m;
  rewrite Owned.linear_extra call refs args best next m
    as Owned.linear_extra call refs args best next (reveal #(Linear.metadata (K.metadata call)) value).Linear.root.Linear.payload;
  fold LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next) value;
  fold Book.full refs.Owned.linear hidden args.Entry.cache
    (LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next))
    (Linear.valid call position layer weights True (Layers.linear_entry call weights prompt position layer) 0);
}

ghost
fn leave_linear
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (weights:Transfer.packed) (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (position:nat{position < Model.calls call}) (layer:Model.linear_layer) ()
  requires Book.full refs.Owned.linear hidden args.Entry.cache
    (LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next))
    (Linear.valid call position layer weights True (Layers.linear_entry call weights prompt position layer) 13)
  ensures Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position (layer + 1))
{
  unfold Book.full refs.Owned.linear hidden args.Entry.cache
    (LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next))
    (Linear.valid call position layer weights True (Layers.linear_entry call weights prompt position layer) 13);
  with m h kv. assert GR.pts_to refs.Owned.linear m ** hidden |-> h ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (Linear.valid call position layer weights True (Layers.linear_entry call weights prompt position layer) 13 m h kv);
  let old : erased (K.metadata call) = hide #(K.metadata call) (reveal #(Linear.metadata (K.metadata call)) m).Linear.root.Linear.payload;
  norm_spec [delta_once ["KPlay.Proof.Base.KernelLayers.linear_entry"]]
    (Layers.linear_entry call weights prompt position layer (reveal #(Linear.metadata (K.metadata call)) m).Linear.root);
  K.layer_facts call weights prompt position layer old
    (Linear.tensor (reveal #(Linear.metadata (K.metadata call)) m).Linear.root.Linear.hidden)
    (reveal #(Linear.metadata (K.metadata call)) m).Linear.root.Linear.cache;
  let current : erased (K.current call) = hide #(K.current call) (K.decoding call old);
  Layers.linear_end_valid call weights prompt position layer old current m h kv;
  let value : erased (K.metadata call) = hide #(K.metadata call)
    (Layers.advance call old current h
      (Global.linear_after (LTrace.last (reveal #(K.current call) current).K.layers).O.cache layer
        (reveal #(Linear.metadata (K.metadata call)) m).Linear.state (reveal #(Linear.metadata (K.metadata call)) m).Linear.convolution)
      (Global.linear_update layer
        (reveal #(Linear.metadata (K.metadata call)) m).Linear.root.Linear.state
        (reveal #(Linear.metadata (K.metadata call)) m).Linear.state
        (reveal #(Linear.metadata (K.metadata call)) m).Linear.root.Linear.convolution
        (reveal #(Linear.metadata (K.metadata call)) m).Linear.mixer_input)
      (reveal #(Linear.metadata (K.metadata call)) m).Linear.state (reveal #(Linear.metadata (K.metadata call)) m).Linear.convolution);
  unfold LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next) m;
  rewrite Owned.linear_extra call refs args best next (reveal #(Linear.metadata (K.metadata call)) m).Linear.root.Linear.payload
    as Owned.linear_extra call refs args best next old;
  unfold Owned.linear_extra call refs args best next old;
  GR.write refs.Owned.global value;
  fold Owned.parked refs.Owned.linear;
  unfold Owned.other_memory call args best next (reveal #(K.metadata call) old).K.memory;
  fold Owned.other_memory call args best next (reveal #(K.metadata call) value).K.memory;
  fold Owned.memory call args best next (reveal #(K.metadata call) value).K.memory;
  fold Owned.local_owners call refs args best next value;
  fold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position (layer + 1));
}

ghost
fn enter_attention
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (weights:Transfer.packed) (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (position:nat{position < Model.calls call}) (layer:KPlay.Proof.Shared.Attention.full_attention_layer) ()
  requires Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position layer)
  ensures Book.full refs.Owned.attention hidden args.Entry.cache
    (AttentionLayer.local_owners (Owned.attention_extra call refs args best next))
    (Attention.valid call position layer weights True (Layers.attention_entry call weights prompt position layer) 0)
{
  unfold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position layer);
  with m h kv. assert GR.pts_to refs.Owned.global m ** hidden |-> h ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  K.layer_facts call weights prompt position layer m h kv;
  unfold Owned.local_owners call refs args best next m;
  unfold Owned.parked refs.Owned.attention;
  with old. assert GR.pts_to refs.Owned.attention old;
  Layers.attention_start_valid call weights prompt position layer m h kv;
  let value : erased (Attention.metadata (K.metadata call)) =
    hide #(Attention.metadata (K.metadata call)) (Layers.attention_start call weights prompt position layer m h kv);
  GR.write refs.Owned.attention value;
  fold Owned.attention_extra call refs args best next m;
  rewrite Owned.attention_extra call refs args best next m
    as Owned.attention_extra call refs args best next (reveal #(Attention.metadata (K.metadata call)) value).Attention.root.Attention.payload;
  fold AttentionLayer.local_owners (Owned.attention_extra call refs args best next) value;
  fold Book.full refs.Owned.attention hidden args.Entry.cache
    (AttentionLayer.local_owners (Owned.attention_extra call refs args best next))
    (Attention.valid call position layer weights True (Layers.attention_entry call weights prompt position layer) 0);
}

ghost
fn leave_attention
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (hidden:array2 BF.t Native.hidden_layout) (best:larray FP.t 1) (next:larray I64.t 1)
  (weights:Transfer.packed) (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (position:nat{position < Model.calls call}) (layer:KPlay.Proof.Shared.Attention.full_attention_layer) ()
  requires Book.full refs.Owned.attention hidden args.Entry.cache
    (AttentionLayer.local_owners (Owned.attention_extra call refs args best next))
    (Attention.valid call position layer weights True (Layers.attention_entry call weights prompt position layer) 37)
  ensures Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position (layer + 1))
{
  unfold Book.full refs.Owned.attention hidden args.Entry.cache
    (AttentionLayer.local_owners (Owned.attention_extra call refs args best next))
    (Attention.valid call position layer weights True (Layers.attention_entry call weights prompt position layer) 37);
  with m h kv. assert GR.pts_to refs.Owned.attention m ** hidden |-> h ** A.pts_to_mask args.Entry.cache kv Init.all_cells;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (Attention.valid call position layer weights True (Layers.attention_entry call weights prompt position layer) 37 m h kv);
  let old : erased (K.metadata call) = hide #(K.metadata call) (reveal #(Attention.metadata (K.metadata call)) m).Attention.root.Attention.payload;
  norm_spec [delta_once ["KPlay.Proof.Base.KernelLayers.attention_entry"]]
    (Layers.attention_entry call weights prompt position layer (reveal #(Attention.metadata (K.metadata call)) m).Attention.root);
  K.layer_facts call weights prompt position layer old
    (Linear.tensor (reveal #(Attention.metadata (K.metadata call)) m).Attention.root.Attention.hidden)
    (reveal #(Attention.metadata (K.metadata call)) m).Attention.root.Attention.cache;
  let current : erased (K.current call) = hide #(K.current call) (K.decoding call old);
  LTrace.shape (Inputs.parameter_values weights) position (reveal #(K.current call) current).K.token
    (K.caller call weights prompt old).D.model_cache (reveal #(K.current call) current).K.layers;
  let after : erased Cache.native_cache = hide #Cache.native_cache
    (Cache.source_after (LTrace.last (reveal #(K.current call) current).K.layers).O.cache
      (reveal #(Attention.metadata (K.metadata call)) m).Attention.prepared layer position);
  Layers.attention_end_valid call weights prompt position layer old current m h kv after;
  let value : erased (K.metadata call) = hide #(K.metadata call)
    (Layers.advance call old current h after
      (O.AttentionUpdate
        (Cache.produced_key (reveal #(Attention.metadata (K.metadata call)) m).Attention.prepared)
        (Cache.produced_value (reveal #(Attention.metadata (K.metadata call)) m).Attention.prepared))
      (reveal #(K.metadata call) old).K.memory.K.state (reveal #(K.metadata call) old).K.memory.K.convolution);
  unfold AttentionLayer.local_owners (Owned.attention_extra call refs args best next) m;
  rewrite Owned.attention_extra call refs args best next (reveal #(Attention.metadata (K.metadata call)) m).Attention.root.Attention.payload
    as Owned.attention_extra call refs args best next old;
  unfold Owned.attention_extra call refs args best next old;
  GR.write refs.Owned.global value;
  fold Owned.parked refs.Owned.attention;
  rewrite Owned.memory call args best next (reveal #(K.metadata call) old).K.memory
    as Owned.memory call args best next (reveal #(K.metadata call) value).K.memory;
  fold Owned.local_owners call refs args best next value;
  fold Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
    (K.layer call weights prompt position (layer + 1));
}

inline_for_extraction noextract
fn apply_layer
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (position:nat{position < Model.calls call}) (layer:sz{SZ.v layer < 24})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt position (SZ.v layer)))
  ensures Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt position (SZ.v layer + 1)))
{
  if (layer %^ 4sz <^ 3sz) {
    Book.map_lane_zero (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt position (SZ.v layer)))
      (Book.full refs.Owned.linear hidden args.Entry.cache
        (LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next))
        (Linear.valid call position (SZ.v layer) (Native.packed_values vw) True
          (Layers.linear_entry call (Native.packed_values vw) prompt position (SZ.v layer)) 0))
      (enter_linear call refs args hidden best next (Native.packed_values vw) prompt position (SZ.v layer));
    LinearLayer.apply refs.Owned.linear call position (SZ.v layer) weights hidden args.Entry.cache args.Entry.state args.Entry.convolution
      tid (Owned.linear_extra call refs args best next) (Layers.linear_entry call (Native.packed_values vw) prompt position (SZ.v layer))
      #vw #fraction #True;
    Book.map_lane_zero (SZ.v tid)
      (Book.full refs.Owned.linear hidden args.Entry.cache
        (LinearLayer.local_owners args.Entry.state args.Entry.convolution (Owned.linear_extra call refs args best next))
        (Linear.valid call position (SZ.v layer) (Native.packed_values vw) True
          (Layers.linear_entry call (Native.packed_values vw) prompt position (SZ.v layer)) 13))
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt position (SZ.v layer + 1)))
      (leave_linear call refs args hidden best next (Native.packed_values vw) prompt position (SZ.v layer));
  } else {
    Book.map_lane_zero (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt position (SZ.v layer)))
      (Book.full refs.Owned.attention hidden args.Entry.cache
        (AttentionLayer.local_owners (Owned.attention_extra call refs args best next))
        (Attention.valid call position (SZ.v layer) (Native.packed_values vw) True
          (Layers.attention_entry call (Native.packed_values vw) prompt position (SZ.v layer)) 0))
      (enter_attention call refs args hidden best next (Native.packed_values vw) prompt position (SZ.v layer));
    AttentionLayer.apply refs.Owned.attention call position (SZ.v layer) weights hidden args.Entry.cache tid
      (Owned.attention_extra call refs args best next) (Layers.attention_entry call (Native.packed_values vw) prompt position (SZ.v layer))
      #vw #fraction #True;
    Book.map_lane_zero (SZ.v tid)
      (Book.full refs.Owned.attention hidden args.Entry.cache
        (AttentionLayer.local_owners (Owned.attention_extra call refs args best next))
        (Attention.valid call position (SZ.v layer) (Native.packed_values vw) True
          (Layers.attention_entry call (Native.packed_values vw) prompt position (SZ.v layer)) 37))
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt position (SZ.v layer + 1)))
      (leave_attention call refs args hidden best next (Native.packed_values vw) prompt position (SZ.v layer));
  }
}

inline_for_extraction noextract
fn apply_all
  (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (best:larray FP.t 1) (next:larray I64.t 1) (tid:sz{SZ.v tid < 32})
  (position:nat{position < Model.calls call})
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length))
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt position 0))
  ensures Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt position 24))
{
  let mut layer = 0sz;
  while (!layer <^ 24sz)
    invariant exists* (l:sz{SZ.v l <= 24}).
      layer |-> l ** Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.layer call (Native.packed_values vw) prompt position (SZ.v l)))
    invariant gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
    decreases (24 - SZ.v !layer)
  {
    let l = !layer;
    apply_layer call refs args weights hidden best next tid position l #prompt #vw #fraction;
    let following = SZ.add l 1sz;
    rewrite Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt position (SZ.v l + 1)))
      as Head.lane_only (SZ.v tid)
        (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
          (K.layer call (Native.packed_values vw) prompt position (SZ.v following)));
    layer := following;
  };
  let completed = !layer;
  assert pure (SZ.v completed == 24);
  rewrite Head.lane_only (SZ.v tid)
    (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
      (K.layer call (Native.packed_values vw) prompt position (SZ.v completed)))
    as Head.lane_only (SZ.v tid)
      (Book.full refs.Owned.global hidden args.Entry.cache (Owned.local_owners call refs args best next)
        (K.layer call (Native.packed_values vw) prompt position 24));
}
