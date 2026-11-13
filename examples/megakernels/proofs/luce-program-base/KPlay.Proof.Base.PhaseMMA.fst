module KPlay.Proof.Base.PhaseMMA

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open FStar.Ghost

module A = Pulse.Lib.Array
module GR = Pulse.Lib.GhostReference
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module SZ = Kuiper.SizeT
module Attention = KPlay.Proof.Shared.Attention
module Model = KPlay.Proof.Shared.Model
module Schedule = KPlay.Proof.Base.Schedule
module Source = KPlay.Proof.Base.SourceProjection
module Cache = KPlay.Proof.Base.CacheNative
module CacheProjection = KPlay.Proof.Base.CacheProjection
module Writes = KPlay.Proof.Base.MatvecWrites
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Book = KPlay.Proof.Base.WarpBook
module Head = KPlay.Proof.Base.Head
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let weight_post (#a:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24}) (phase:Schedule.phase layer{not (Schedule.attention layer phase)})
  (weights:C.chest2 BF.t 752393024 1)
  (pre:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (state:a) (hidden:Hidden.snapshot) (cache:Cache.snapshot) : prop =
  exists (before:Hidden.snapshot).
    pre state before cache /\
    hidden == Writes.after (Source.weight_geometry call position layer phase) weights before
      (Schedule.projection_at call position layer phase).Schedule.rows

inline_for_extraction noextract
let weight_step (#a:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24}) (phase:Schedule.phase layer{not (Schedule.attention layer phase)})
  (weights:C.chest2 BF.t 752393024 1)
  (pre:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (state:a) (before:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires pre state before cache)
      (ensures weight_post call position layer phase weights pre state
        (Writes.after (Source.weight_geometry call position layer phase) weights before
          (Schedule.projection_at call position layer phase).Schedule.rows) cache) = ()

inline_for_extraction noextract
fn weight
  (#a:Type0) (book:GR.ref a)
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor{layer < 24}) (phase:Schedule.phase layer{not (Schedule.attention layer phase)})
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912) (tid:sz{SZ.v tid < 32})
  (local_owners:a -> slprop)
  (pre:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid) (Book.full book hidden cache local_owners pre)
  ensures Head.lane_only (SZ.v tid)
    (Book.full book hidden cache local_owners (weight_post call position layer phase vw pre))
{
  Book.scatter_wait book hidden cache local_owners pre #(SZ.v tid);
  unfold Book.shared book hidden cache local_owners pre (SZ.v tid);
  with state before keys. assert Book.chunk book hidden cache state before keys;
  unfold Book.chunk book hidden cache state before keys;
  Source.weight_projection call position layer phase weights hidden;
  weight_step call position layer phase vw pre state before keys;
  fold Book.chunk book hidden cache state
    (Writes.after (Source.weight_geometry call position layer phase) vw before
      (Schedule.projection_at call position layer phase).Schedule.rows) keys;
  fold Book.shared book hidden cache local_owners (weight_post call position layer phase vw pre) (SZ.v tid);
  Book.gather_wait book hidden cache local_owners (weight_post call position layer phase vw pre) #(SZ.v tid);
}

inline_for_extraction noextract
let cache_post (#a:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Attention.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (pre:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (state:a) (hidden:Hidden.snapshot) (cache:Cache.snapshot) : prop =
  exists (before:Hidden.snapshot) (ready:squash (Cache.initialized cache layer (position + 1))).
    pre state before cache /\
    hidden == Writes.after (CacheProjection.geometry call position layer phase)
      (CacheProjection.tensor_contents cache call position layer phase ready) before
      (Schedule.projection_at call position layer phase).Schedule.rows

inline_for_extraction noextract
let cache_step (#a:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Attention.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (pre:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (state:a) (before:Hidden.snapshot) (cache:Cache.snapshot)
  (ready:squash (Cache.initialized cache layer (position + 1)))
  : Lemma
      (requires pre state before cache)
      (ensures cache_post call position layer phase pre state
        (Writes.after (CacheProjection.geometry call position layer phase)
          (CacheProjection.tensor_contents cache call position layer phase ready) before
          (Schedule.projection_at call position layer phase).Schedule.rows) cache) = ()

inline_for_extraction noextract
fn cache
  (#a:Type0) (book:GR.ref a)
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Attention.full_attention_layer)
  (phase:Schedule.phase layer{Schedule.attention layer phase /\ Schedule.mma_path layer phase position})
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912) (tid:sz{SZ.v tid < 32})
  (local_owners:a -> slprop)
  (pre:a -> Hidden.snapshot -> Cache.snapshot -> prop)
  (initialized:squash (forall (state:a) (h:Hidden.snapshot) (k:Cache.snapshot).
    pre state h k ==> Cache.initialized k layer (position + 1)))
  preserves gpu ** thread_id 32 (SZ.v tid)
  requires Head.lane_only (SZ.v tid) (Book.full book hidden cache local_owners pre)
  ensures Head.lane_only (SZ.v tid) (Book.full book hidden cache local_owners (cache_post call position layer phase pre))
{
  Book.scatter_wait book hidden cache local_owners pre #(SZ.v tid);
  unfold Book.shared book hidden cache local_owners pre (SZ.v tid);
  with state before keys. assert Book.chunk book hidden cache state before keys;
  unfold Book.chunk book hidden cache state before keys;
  CacheProjection.mma call position layer phase cache hidden #keys #before #(1.0R /. 32) ();
  cache_step call position layer phase pre state before keys ();
  fold Book.chunk book hidden cache state
    (Writes.after (CacheProjection.geometry call position layer phase)
      (CacheProjection.tensor_contents keys call position layer phase ()) before
      (Schedule.projection_at call position layer phase).Schedule.rows) keys;
  fold Book.shared book hidden cache local_owners (cache_post call position layer phase pre) (SZ.v tid);
  Book.gather_wait book hidden cache local_owners (cache_post call position layer phase pre) #(SZ.v tid);
}
