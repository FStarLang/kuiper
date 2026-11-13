module KPlay.Proof.Base.CacheProjection

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open FStar.Ghost

module A = Pulse.Lib.Array
module T = Kuiper.Tensor
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module S = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Model = KPlay.Proof.Shared.Model
module Layout = KPlay.Proof.Base.MatvecLayout
module Affine = KPlay.Proof.Base.AffineMMA
module Matvec = KPlay.Proof.Base.Matvec
module Writes = KPlay.Proof.Base.MatvecWrites
module Native = KPlay.Proof.Base.NativeProjection
module Schedule = KPlay.Proof.Base.Schedule
module Source = KPlay.Proof.Base.SourceProjection
module Cache = KPlay.Proof.Base.CacheNative
module Init = KPlay.Proof.Base.Initialization
module SIMT = KPlay.Proof.Base.SIMT
module SM = KPlay.Proof.Base.SIMTMath
module Local = KPlay.Proof.Base.SIMTLocal

inline_for_extraction noextract
let value_plane (phase:nat) : bool = phase % 2 = 0

inline_for_extraction noextract
let geometry (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  : SM.geometry ((position + 1) * 512) =
  Schedule.bounds call position layer phase;
  Source.decoder_bounds call position layer phase;
  Layout.source_nonoverlap call position layer phase;
  let d = Schedule.projection_at call position layer phase in
  let score = phase % 2 = 1 in
  let offset = (((phase - 3) / 2) / 4) * 256 in
  let row_stride = if score then 512 else 1 in
  let col_stride = if score then 1 else 512 in
  {
    Layout.rows = SZ.uint_to_t d.Schedule.rows;
    Layout.columns = SZ.uint_to_t d.Schedule.columns;
    Layout.offset = SZ.uint_to_t offset;
    Layout.row_stride = SZ.uint_to_t row_stride;
    Layout.col_stride = SZ.uint_to_t col_stride;
    Layout.input_offset = SZ.uint_to_t d.Schedule.input_offset;
    Layout.output_offset = SZ.uint_to_t d.Schedule.output_offset;
    Layout.splits = 1sz;
    Layout.bounds = ();
    Layout.separated = ();
  }

inline_for_extraction noextract
let source_address (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (row:P.index (Schedule.projection_at call position layer phase).Schedule.rows)
  (col:P.index (Schedule.projection_at call position layer phase).Schedule.columns)
  : Lemma (
      Cache.plane_offset layer (value_plane phase) +
        Layout.tile_offset (geometry call position layer phase) row col ==
      Schedule.source_cache_address layer ((phase - 3) / 2) (position + 1) (phase % 2 = 1) row col) =
  Schedule.bounds call position layer phase

inline_for_extraction noextract
let actual_address (cache:larray BF.t 12582912)
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (row:P.index (Schedule.projection_at call position layer phase).Schedule.rows)
  (col:P.index (Schedule.projection_at call position layer phase).Schedule.columns)
  : Lemma (
      A.base_of (Cache.prefix_view cache layer (position + 1) (value_plane phase)) == A.base_of cache /\
      A.offset_of (Cache.prefix_view cache layer (position + 1) (value_plane phase)) +
        Layout.tile_offset (geometry call position layer phase) row col ==
      A.offset_of cache +
        Schedule.source_cache_address layer ((phase - 3) / 2) (position + 1) (phase % 2 = 1) row col) =
  source_address call position layer phase row col;
  A.base_of_gsub cache (Cache.plane_offset layer (value_plane phase))
    (Cache.plane_offset layer (value_plane phase) + (position + 1) * 512);
  A.offset_of_gsub cache (Cache.plane_offset layer (value_plane phase))
    (Cache.plane_offset layer (value_plane phase) + (position + 1) * 512)

inline_for_extraction noextract
let contents (before:Cache.snapshot) (call:Model.invocation)
  (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (ready:squash (Cache.initialized before layer (position + 1)))
  : GTot (lseq BF.t ((position + 1) * 512)) =
  Cache.prefix_contents before layer (position + 1) (value_plane phase) ready

inline_for_extraction noextract
let tensor_contents (before:Cache.snapshot) (call:Model.invocation)
  (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer) (phase:Schedule.phase layer{Schedule.attention layer phase})
  (ready:squash (Cache.initialized before layer (position + 1)))
  : GTot (chest2 BF.t ((position + 1) * 512) 1) =
  T.from_seq (Affine.parent_layout ((position + 1) * 512))
    (contents before call position layer phase ready)

inline_for_extraction noextract
fn mma
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer)
  (phase:Schedule.phase layer{Schedule.attention layer phase /\ Schedule.mma_path layer phase position})
  (cache:larray BF.t 12582912) (hidden:array2 BF.t Native.hidden_layout)
  (#before:erased Cache.snapshot) (#vh:chest2 BF.t 16000 1) (#fraction:perm)
  (ready:squash (Cache.initialized before layer (position + 1)))
  preserves gpu
  preserves A.pts_to_mask cache #fraction before Init.all_cells
  requires hidden |-> Frac (1.0R /. 32) vh
  ensures hidden |-> Frac (1.0R /. 32)
    (Writes.after (geometry call position layer phase) (tensor_contents before call position layer phase ready) vh
      (Schedule.projection_at call position layer phase).Schedule.rows)
{
  let prefix = Cache.borrow_prefix cache layer (position + 1) (value_plane phase);
  with values. assert A.pts_to prefix #fraction values;
  Cache.prefix_unique before layer (position + 1) (value_plane phase) ready values;
  rewrite A.pts_to prefix #fraction values
    as A.pts_to prefix #fraction (contents before call position layer phase ready);
  T.tensor_abs' (Affine.parent_layout ((position + 1) * 512)) prefix;
  let parent = T.from_array (Affine.parent_layout ((position + 1) * 512)) prefix;
  assert rewrites_to parent (T.from_array (Affine.parent_layout ((position + 1) * 512)) prefix);
  let g = geometry call position layer phase;
  Matvec.projection g parent hidden;
  T.tensor_concr parent;
  T.to_from (Affine.parent_layout ((position + 1) * 512)) (contents before call position layer phase ready);
  T.lem_from_array_core #BF.t #2 #(((position + 1) * 512) @| 1 @| INil)
    #(Affine.parent_layout ((position + 1) * 512)) prefix;
  rewrite A.pts_to (T.core parent) #fraction
    (T.to_seq (Affine.parent_layout ((position + 1) * 512)) (tensor_contents before call position layer phase ready))
    as A.pts_to (Cache.prefix_view cache layer (position + 1) (value_plane phase)) #fraction
      (contents before call position layer phase ready);
  Cache.restore_prefix cache layer (position + 1) (value_plane phase);
}

inline_for_extraction noextract
fn simt
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer)
  (phase:Schedule.phase layer{Schedule.attention layer phase /\ not (Schedule.mma_path layer phase position)})
  (cache:larray BF.t 12582912) (hidden:larray BF.t 16000)
  (#before:erased Cache.snapshot) (#sx:erased (lseq BF.t 16000))
  (#wr:erased (S.matrix (Schedule.projection_at call position layer phase).Schedule.rows
    (Schedule.projection_at call position layer phase).Schedule.columns))
  (#xr:erased (S.vector (Schedule.projection_at call position layer phase).Schedule.columns))
  (#fraction:perm) (#scope:prop)
  (ready:squash (Cache.initialized before layer (position + 1)))
  preserves gpu ** A.pts_to_mask cache #fraction before Init.all_cells
  requires A.pts_to hidden (reveal sx)
  requires pure (scope ==>
    SM.matrix_rep (geometry call position layer phase) (contents before call position layer phase ready) wr /\
    SM.input_rep (geometry call position layer phase) sx xr)
  ensures exists* (result:lseq BF.t 16000).
    A.pts_to hidden result **
    pure (SIMT.frame (geometry call position layer phase) sx result
      (Schedule.projection_at call position layer phase).Schedule.rows /\
      (scope ==> SIMT.produced (geometry call position layer phase) result wr xr
        (Schedule.projection_at call position layer phase).Schedule.rows))
{
  let prefix = Cache.borrow_prefix cache layer (position + 1) (value_plane phase);
  with values. assert A.pts_to prefix #fraction values;
  Cache.prefix_unique before layer (position + 1) (value_plane phase) ready values;
  rewrite A.pts_to prefix #fraction values
    as A.pts_to prefix #fraction (contents before call position layer phase ready);
  let g = geometry call position layer phase;
  let score = phase % 2 = 1;
  SIMT.source_bounds (position + 1) score;
  let groups = SZ.uint_to_t (Schedule.simt_groups score (position + 1));
  let chunk = SZ.uint_to_t (Schedule.simt_chunk score (position + 1));
  let linear = Schedule.simt_linear_finish score (position + 1);
  Local.projection g prefix hidden groups chunk linear
    #fraction #(contents before call position layer phase ready) #sx #wr #xr #scope;
  rewrite A.pts_to prefix #fraction (contents before call position layer phase ready)
    as A.pts_to (Cache.prefix_view cache layer (position + 1) (value_plane phase)) #fraction
      (contents before call position layer phase ready);
  Cache.restore_prefix cache layer (position + 1) (value_plane phase);
}
