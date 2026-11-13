module KPlay.Proof.Base.Initialization

#lang-pulse

open Kuiper
open FStar.Ghost

module S = FStar.Seq
module Array = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module P = KPlay.Proof.Shared.Projection
module Rotary = KPlay.Proof.Base.Rotary
module SH = Kuiper.SHMem

inline_for_extraction noextract
let all_cells (_:nat) : prop = True

inline_for_extraction noextract
let prefix (#a:Type0) (#n:nat) (before:lseq (option a) n)
  (value:a) (count:nat{count <= n}) : GTot (lseq (option a) n) =
  S.init_ghost n (fun i -> if i < count then Some value else S.index before i)

inline_for_extraction noextract
let prefix_zero (#a:Type0) (#n:nat) (before:lseq (option a) n) (value:a)
  : Lemma (prefix before value 0 == before) =
  assert (S.equal (prefix before value 0) before);
  S.lemma_eq_elim (prefix before value 0) before

inline_for_extraction noextract
let prefix_step (#a:Type0) (#n:nat) (before:lseq (option a) n) (value:a)
  (count:nat{count < n})
  : Lemma (
      S.upd (prefix before value count) count (Some value) ==
      prefix before value (count + 1)) =
  assert (S.equal
    (S.upd (prefix before value count) count (Some value))
    (prefix before value (count + 1)));
  S.lemma_eq_elim
    (S.upd (prefix before value count) count (Some value))
    (prefix before value (count + 1))

inline_for_extraction noextract
fn fill_source_order (#a:Type0) (size:szp) (buffer:larray a (SZ.v size)) (value:a)
  (#before:erased (lseq (option a) (SZ.v size)))
  preserves gpu
  requires Array.pts_to_mask buffer before all_cells
  ensures buffer |-> S.create (SZ.v size) value
{
  prefix_zero before value;
  let mut index = 0sz;
  while (!index <^ size)
    invariant exists* (current:sz{SZ.v current <= SZ.v size}).
      index |-> current **
      Array.pts_to_mask buffer (prefix before value (SZ.v current)) all_cells
    invariant gpu
    decreases (SZ.v size - SZ.v !index)
  {
    let current = !index;
    prefix_step before value (SZ.v current);
    Array.mask_write buffer current value;
    index := SZ.add current 1sz;
  };
  Array.from_mask buffer;
  with actual. assert buffer |-> actual;
  assert pure (S.equal actual (S.create (SZ.v size) value));
  S.lemma_eq_elim actual (S.create (SZ.v size) value);
}

inline_for_extraction noextract
let state_values : lseq FP.t 4718592 = S.create 4718592 (zero #FP.t)

inline_for_extraction noextract
let convolution_values : lseq BF.t 442368 = S.create 442368 (zero #BF.t)

(* kernel.cu:222-223: all FP32 stores precede all BF16 stores. *)
inline_for_extraction noextract
fn initialize_caches
  (state:larray FP.t 4718592) (conv:larray BF.t 442368)
  (#old_state:erased (lseq (option FP.t) 4718592))
  (#old_conv:erased (lseq (option BF.t) 442368))
  preserves gpu
  requires Array.pts_to_mask state old_state all_cells
  requires Array.pts_to_mask conv old_conv all_cells
  ensures state |-> state_values
  ensures conv |-> convolution_values
{
  fill_source_order (SZ.uint32_to_sizet 4718592ul) state (zero #FP.t);
  fill_source_order (SZ.uint32_to_sizet 442368ul) conv (zero #BF.t);
}

inline_for_extraction noextract
let state_layer (values:lseq FP.t 4718592) (layer:Model.linear_layer)
  : O.tensor FP.t 16 128 128 =
  fun head key value ->
    S.index values ((layer - layer / 4) * 262144 + (head * 128 + key) * 128 + value)

inline_for_extraction noextract
let convolution_layer (values:lseq BF.t 442368) (layer:Model.linear_layer)
  : O.matrix BF.t 6144 4 =
  fun channel tap ->
    S.index values ((layer - layer / 4) * 24576 + channel * 4 + tap)

inline_for_extraction noextract
let first_use_inputs (metadata:O.cache_configuration) (layer:Model.linear_layer)
  : Lemma (
      O.recurrent_input (O.empty_cache #O.cache_configuration #BF.t #FP.t #BF.t metadata)
        layer (state_layer state_values layer) /\
      O.convolution_input (O.empty_cache #O.cache_configuration #BF.t #FP.t #BF.t metadata)
        layer (convolution_layer convolution_values layer)) = ()

inline_for_extraction noextract
let all_first_use_inputs (metadata:O.cache_configuration)
  : Lemma (
      forall (layer:Model.linear_layer).
        O.recurrent_input (O.empty_cache #O.cache_configuration #BF.t #FP.t #BF.t metadata)
          layer (state_layer state_values layer) /\
        O.convolution_input (O.empty_cache #O.cache_configuration #BF.t #FP.t #BF.t metadata)
          layer (convolution_layer convolution_values layer)) =
  FStar.Classical.forall_intro (first_use_inputs metadata)

inline_for_extraction noextract
let state_address (layer:Model.linear_layer) (head:P.index 16)
  (key value:P.index 128)
  : P.index 4718592 =
  (layer - layer / 4) * 262144 + (head * 128 + key) * 128 + value

inline_for_extraction noextract
let state_layer_frame (layer other:Model.linear_layer{layer <> other})
  (head:P.index 16) (key value:P.index 128)
  : Lemma (
      state_address layer head key value < (other - other / 4) * 262144 \/
      (other - other / 4 + 1) * 262144 <= state_address layer head key value) = ()

inline_for_extraction noextract
let outside_rotary (i:nat) : prop = ~(Rotary.rotary_mask i)

ghost
fn shared_scratch_mask (hidden:SH.c_shmem (SH.SHArray BF.t 16000sz))
  requires SH.live_c_shmem #(SH.SHArray BF.t 16000sz) hidden
  ensures exists* (before:Rotary.snapshot).
    Array.pts_to_mask hidden before all_cells
{
  SH.unfold_live_c_shmem hidden;
  with values. assert hidden |-> values;
  Array.pts_to_len hidden;
  Array.to_mask hidden;
}

ghost
fn isolate_rotary_window (hidden:larray BF.t 16000)
  (#before:Rotary.snapshot) (#fraction:perm)
  requires Array.pts_to_mask hidden #fraction before all_cells
  ensures Array.pts_to_mask hidden #fraction before Rotary.rotary_mask
  ensures Array.pts_to_mask hidden #fraction before outside_rotary
{
  Array.split_mask hidden Rotary.rotary_mask;
  Array.mask_mext hidden #fraction #before
    #(Array.mask_isect all_cells Rotary.rotary_mask) Rotary.rotary_mask;
  Array.mask_mext hidden #fraction #before
    #(Array.mask_diff all_cells Rotary.rotary_mask) outside_rotary;
}

inline_for_extraction noextract
fn initialize_rotary_window (hidden:larray BF.t 16000)
  (position:KPlay.Proof.RoPE.Source.position) (#before:Rotary.snapshot)
  preserves gpu
  requires Array.pts_to_mask hidden before all_cells
  ensures Array.pts_to_mask hidden (Rotary.source_writes before position 32) Rotary.rotary_mask
  ensures Array.pts_to_mask hidden before outside_rotary
  ensures pure (
    KPlay.Proof.RoPE.Source.rotary_post position (Rotary.produced_cells before position) /\
    Rotary.rope_represents (FStar.Int64.v position) (Rotary.produced_cells before position))
{
  isolate_rotary_window hidden;
  Rotary.position_embedding hidden position;
}
