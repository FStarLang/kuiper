module KPlay.Proof.Base.RecurrentMixer

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module H = KPlay.Proof.Base.RecurrentHeadModel
module Heads = KPlay.Proof.Base.RecurrentHeads
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Bridge = KPlay.Proof.Base.RecurrentPipelineBridge
module Match = KPlay.Proof.Base.RecurrentModelMatch
module Conv = KPlay.Proof.Base.Convolution
module Convolution = KPlay.Proof.Base.ConvolutionLoop
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let head_models (stages:Prior.decoder_stages) : Heads.models =
  fun head -> Bridge.inputs (stages.Prior.head_plans head)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let inputs (call:Prior.logical_call) (stages:Prior.decoder_stages)
  (weights:Transfer.packed) (hidden:Transfer.scratch) (state:lseq FP.t 4718592) (conv:Conv.cells)
  (base:nat{base + 19458208 <= 752393024}) (slot:nat{slot < 18}) : prop =
  stages == Prior.build_stages call /\
  Convolution.input_rep conv hidden weights slot base
    (Prior.old_convolution call.Prior.call call.Prior.before)
    call.Prior.weight.R.conv_kernel stages.Prior.raw /\
  (forall (i:P.index 2048).
    Approx.v_approximates (Seq.index hidden (8192 + i)) (stages.Prior.projected_z i)) /\
  (forall (head:P.index 16).
    Approx.v_approximates (Seq.index hidden (10240 + head)) (stages.Prior.projected_b head) /\
    Approx.v_approximates (Seq.index hidden (10256 + head)) (stages.Prior.projected_a head) /\
    Approx.v_approximates (Seq.index weights (base + 19458048 + head)) (call.Prior.weight.R.a_log head) /\
    Approx.v_approximates (Seq.index weights (base + 19458064 + head)) (call.Prior.weight.R.dt_bias head) /\
    (forall (i j:P.index 128).
      Approx.v_approximates (Seq.index state (Heads.offset slot head + i * 128 + j))
        (Prior.old_recurrent call.Prior.call call.Prior.before head i j))) /\
  (forall (i:P.index 128).
    Approx.v_approximates (Seq.index weights (base + 19458080 + i)) (call.Prior.weight.R.gated_norm i))

inline_for_extraction noextract
let head_ready (call:Prior.logical_call) (stages:Prior.decoder_stages)
  (weights:Transfer.packed) (before current:Transfer.scratch) (state:lseq FP.t 4718592) (conv:Conv.cells)
  (base:nat{base + 19458208 <= 752393024}) (slot:nat{slot < 18}) (head:P.index 16)
  : Lemma
      (requires inputs call stages weights before state conv base slot /\
        Convolution.hidden_frame before current 6144 /\
        Convolution.output_rep current
          (Prior.old_convolution call.Prior.call call.Prior.before)
          call.Prior.weight.R.conv_kernel stages.Prior.raw 6144)
      (ensures H.represented weights current state base (Heads.offset slot head) head (head_models stages head)) =
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentMixer.inputs"]]
    (inputs call stages weights before state conv base slot);
  let canonical = Prior.build_stages call in
  Prior.stage_head_view_lemma call head;
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.stage_head_view"]] (Prior.stage_head_view call head);
  let at_index (i:P.index 6144)
    : Lemma (Convolution.convolution_value
        (Prior.old_convolution call.Prior.call call.Prior.before) call.Prior.weight.R.conv_kernel
        stages.Prior.raw i == stages.Prior.convolved i) =
    norm_spec [delta_once ["KPlay.Proof.Base.ConvolutionLoop.convolution_value"]]
      (Convolution.convolution_value (Prior.old_convolution call.Prior.call call.Prior.before)
        call.Prior.weight.R.conv_kernel stages.Prior.raw i)
  in
  FStar.Classical.forall_intro at_index;
  let unchanged (i:P.index 128)
    : Lemma (Seq.index current (8192 + head * 128 + i) == Seq.index before (8192 + head * 128 + i)) = ()
  in
  FStar.Classical.forall_intro unchanged;
  assert (Seq.index current (10240 + head) == Seq.index before (10240 + head));
  assert (Seq.index current (10256 + head) == Seq.index before (10256 + head));
  norm_spec [delta_only [
    "KPlay.Proof.Base.RecurrentHeadModel.represented";
    "KPlay.Proof.Base.RecurrentMixer.head_models";
    "KPlay.Proof.Base.RecurrentPipelineBridge.inputs"]]
    (H.represented weights current state base (Heads.offset slot head) head (head_models stages head))

inline_for_extraction noextract
let conditional_head_ready (scope:prop) (call:Prior.logical_call) (stages:Prior.decoder_stages)
  (weights:Transfer.packed) (before current:Transfer.scratch) (state:lseq FP.t 4718592) (conv:Conv.cells)
  (base:nat{base + 19458208 <= 752393024}) (slot:nat{slot < 18}) (head:P.index 16)
  : Lemma
      (requires Convolution.hidden_frame before current 6144 /\
        (scope ==> inputs call stages weights before state conv base slot /\
          Convolution.output_rep current
            (Prior.old_convolution call.Prior.call call.Prior.before)
            call.Prior.weight.R.conv_kernel stages.Prior.raw 6144))
      (ensures scope ==> H.represented weights current state base (Heads.offset slot head) head (head_models stages head)) =
  let under (_:scope)
    : Lemma (H.represented weights current state base (Heads.offset slot head) head (head_models stages head)) =
    head_ready call stages weights before current state conv base slot head
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let output (call:Prior.logical_call) (stages:Prior.decoder_stages)
  (state:lseq FP.t 4718592) (hidden:Transfer.scratch) (slot:nat{slot < 18})
  : Lemma
      (requires stages == Prior.build_stages call /\ Heads.completed state hidden slot (head_models stages) 16)
      (ensures
        (forall (head:P.index 16) (i j:P.index 128).
          Approx.v_approximates (Seq.index state (Heads.offset slot head + i * 128 + j))
            ((R.decoder_values call.Prior.call call.Prior.weight call.Prior.input call.Prior.before).R.recurrent_value head i j)) /\
        (forall (i:P.index 2048).
          Approx.v_approximates (Seq.index hidden (10400 + i)) (stages.Prior.gated_values i))) =
  let state_cell (head:P.index 16) (i j:P.index 128)
    : Lemma (Approx.v_approximates (Seq.index state (Heads.offset slot head + i * 128 + j))
        ((R.decoder_values call.Prior.call call.Prior.weight call.Prior.input call.Prior.before).R.recurrent_value head i j)) =
    Match.state_matches call stages head i j
  in
  FStar.Classical.forall_intro_3 state_cell;
  let hidden_cell (i:P.index 2048)
    : Lemma (Approx.v_approximates (Seq.index hidden (10400 + i)) (stages.Prior.gated_values i)) =
    Match.gated_matches call stages (i / 128) (i % 128)
  in
  FStar.Classical.forall_intro hidden_cell

inline_for_extraction noextract
let outside (before after:Transfer.scratch) : prop =
  forall (i:P.index 16000).
    ~(2048 <= i /\ i < 8192) /\ ~(10400 <= i /\ i < 12448) ==>
    Seq.index after i == Seq.index before i

inline_for_extraction noextract
fn apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (state:larray FP.t 4718592) (conv:larray BF.t 442368)
  (base:sz{SZ.v base + 19458208 <= 752393024}) (slot:sz{SZ.v slot < 18})
  (#call:erased Prior.logical_call) (#stages:erased Prior.decoder_stages)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch)
  (#ss:erased (lseq FP.t 4718592)) (#cv:erased Conv.cells)
  (#fraction:perm) (#scope:prop)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** state |-> ss ** A.pts_to conv cv
  requires pure (scope ==> inputs call stages vw vh ss cv (SZ.v base) (SZ.v slot))
  ensures exists* (result:Transfer.scratch) (updated:lseq FP.t 4718592).
    A.pts_to hidden result ** state |-> updated ** A.pts_to conv (Convolution.after cv vh (SZ.v slot) 6144) **
    pure (Heads.state_frame ss updated (SZ.v slot) 16 /\ outside vh result /\
      (scope ==> Heads.completed updated result (SZ.v slot) (head_models stages) 16))
{
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentMixer.inputs"]]
    (inputs call stages vw vh ss cv (SZ.v base) (SZ.v slot));
  Convolution.apply weights hidden conv base slot #vw #vh #cv #fraction
    #(Prior.old_convolution (reveal #Prior.logical_call call).Prior.call (reveal #Prior.logical_call call).Prior.before)
    #(reveal #Prior.logical_call call).Prior.weight.R.conv_kernel #(reveal #Prior.decoder_stages stages).Prior.raw #scope;
  with convolved. assert A.pts_to hidden convolved;
  FStar.Classical.forall_intro (conditional_head_ready scope call stages vw vh convolved ss cv (SZ.v base) (SZ.v slot));
  Heads.apply weights hidden state base slot
    #vw #convolved #ss #fraction #(head_models stages) #scope;
}
