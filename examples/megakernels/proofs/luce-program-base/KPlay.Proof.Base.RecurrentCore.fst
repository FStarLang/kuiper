module KPlay.Proof.Base.RecurrentCore

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost
open FStar.Real

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Native = KPlay.Proof.Base.RecurrentNative
module Scalar = KPlay.Proof.Base.RecurrentScalars
module H = KPlay.Proof.Base.RecurrentHeadModel
module Math = KPlay.Proof.Base.RecurrentStateMath
module Dot = KPlay.Proof.Base.RecurrentDot
module DotMath = KPlay.Proof.Base.RecurrentDotMath
module Sub = KPlay.Proof.Base.SoftmaxStripes
module Conditional = KPlay.Proof.Base.ConditionalNumerics
module SIMTMath = KPlay.Proof.Base.SIMTMath
module Local = KPlay.Proof.Base.LocalArray
module Init = KPlay.Proof.Base.Initialization
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
fn apply
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (state:larray FP.t 4718592)
  (query key:larray FP.t 128)
  (base:sz{SZ.v base + 19458208 <= 752393024})
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (head:sz{SZ.v head < 16})
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#ss:erased (lseq FP.t 4718592))
  (#vq #vk:erased (Local.snapshot 128))
  (#fraction:perm) (#x:erased H.inputs) (#scope:prop)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh ** state |-> ss **
    A.pts_to_mask query vq Init.all_cells ** A.pts_to_mask key vk Init.all_cells
  requires pure (scope ==> H.represented vw vh ss (SZ.v base) (SZ.v offset) (SZ.v head) x)
  ensures exists* (result:Transfer.scratch) (updated:lseq FP.t 4718592).
    A.pts_to hidden result ** state |-> updated **
    query |-> Native.query_values vh (SZ.v head) **
    key |-> Native.key_values vh (SZ.v head) **
    pure (Math.state_frame ss updated (SZ.v offset) /\ Math.hidden_frame vh result (SZ.v head) 128 /\
      (scope ==> Math.columns updated (SZ.v offset) x 128 /\ Math.core_prefix result (SZ.v head) x 128))
{
  Native.source_prepare_qk hidden query key head;
  Math.qk_approx scope vw vh ss (SZ.v base) (SZ.v offset) (SZ.v head) x;
  let control = Native.controls weights hidden base head;
  let beta = control._1;
  let gain = control._2;
  Math.controls_approx scope vw vh ss (SZ.v base) (SZ.v offset) (SZ.v head) x;
  Native.decay_state state offset gain;
  Math.decay_columns scope ss (SZ.v offset) gain x;
  let input_offset = SZ.add 6144sz (SZ.mul head 128sz);
  let output_offset = SZ.add 10400sz (SZ.mul head 128sz);
  let mut j = 0sz;
  while (!j <^ 128sz)
    invariant exists* (column:sz{SZ.v column <= 128}) (current_state:lseq FP.t 4718592)
      (current_hidden:Transfer.scratch).
      j |-> column ** state |-> current_state ** A.pts_to hidden current_hidden **
      pure (Math.state_frame ss current_state (SZ.v offset) /\
        Math.hidden_frame vh current_hidden (SZ.v head) (SZ.v column) /\
        (scope ==> Math.columns current_state (SZ.v offset) x (SZ.v column) /\
          Math.core_prefix current_hidden (SZ.v head) x (SZ.v column)))
    invariant gpu ** A.pts_to weights #fraction vw **
      query |-> Native.query_values vh (SZ.v head) **
      key |-> Native.key_values vh (SZ.v head)
    decreases (128 - SZ.v !j)
  {
    let column = !j;
    with current_state current_hidden. assert state |-> current_state ** A.pts_to hidden current_hidden;
    let memory = Dot.source_dot state key offset column #1.0R #1.0R;
    DotMath.conditional_value_approx scope current_state (Native.key_values vh (SZ.v head))
      (SZ.v offset) (SZ.v column) (H.decayed_column x (SZ.v column)) (H.key x);
    Math.raw_value_frame vh current_hidden (SZ.v head) (SZ.v column) (SZ.v column);
    let raw = hidden.(SZ.add input_offset column);
    Conditional.widen scope raw (H.raw_value_at x (SZ.v column));
    let difference = Sub.subtract (Cast.cast_bf16_to_f32 raw) memory
      #(H.raw_value_at x (SZ.v column)) #(H.memory x (SZ.v column)) #scope;
    let delta = F.mul difference beta;
    Conditional.multiply scope difference beta
      (H.raw_value_at x (SZ.v column) -. H.memory x (SZ.v column)) (H.beta x);
    Native.update_column state key offset column delta;
    Math.column_step scope ss current_state (Native.key_values vh (SZ.v head))
      (SZ.v offset) (SZ.v column) delta x;
    with updated. assert state |-> updated;
    let core = Dot.source_dot state query offset column #1.0R #1.0R;
    DotMath.conditional_value_approx scope updated (Native.query_values vh (SZ.v head))
      (SZ.v offset) (SZ.v column) (H.next_column x (SZ.v column)) (H.query x);
    SIMTMath.conditional_round scope core (H.core x (SZ.v column));
    let stored = Cast.cast_f32_to_bf16 core;
    Math.core_store scope vh current_hidden (SZ.v head) x (SZ.v column) stored;
    hidden.(SZ.add output_offset column) <- stored;
    j := SZ.add column 1sz;
  }
}
