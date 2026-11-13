module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module R = FStar.Real
module Sqrt = FStar.Math.Sqrt
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module Fast = Kuiper.Float32.FastMath
module Approx = Kuiper.Approximates.Base
module Cast = Kuiper.Float.Casts
module C = FStar.Classical
module SC = Kuiper.Seq.Common
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
module Sh = KPlay.SharedSpec
module RoPE = Luce.RoPE
module A = Pulse.Lib.Array
module Q = Kuiper.Seq.Common
module Exp = FStar.Math.Exp
module Proof = FStar.Tactics.V2

#set-options "--fuel 2 --ifuel 1 --z3rlimit 30"

(* Only the suffix is opaque; its contracts preserve the produced scores. *)
[@@CPrologue "__device__"]
fn attend_all (position : i64{valid_position (I64.v position)})
  (scores : larray f32 (score_elements (I64.v position))) (values : larray bf16 cache_elements)
  (gates : larray f32 query_gate_width) (output : larray f32 activation_width)
  (#sv : erased (lseq f32 (score_elements (I64.v position))))
  (#vv : erased (lseq bf16 cache_elements)) (#gv : erased (lseq f32 query_gate_width))
  (#old : erased (lseq f32 activation_width)) (#fs #fv #fg : perm)
  preserves gpu ** scores |-> Frac fs sv ** values |-> Frac fv vv ** gates |-> Frac fg gv
  requires output |-> old
  ensures exists* (after : lseq f32 activation_width). output |-> after **
    pure (forall (model : gated_interpretation (I64.v position)). gated_inputs sv vv gv model ==>
      after %~ gated_attention (I64.v position) sv model)
{
  let mut local_output = [| zero #f32 #_; 256sz |];
  let mut head = 0sz;
  while (!head <^ 8sz)
    invariant exists* (h : sz{h <= query_heads}) (current : lseq f32 activation_width).
      head |-> h ** output |-> current **
      pure (forall (model : gated_interpretation (I64.v position)). gated_inputs sv vv gv model ==>
        (forall (i : nat{i < SZ.v h * head_width}).
          S.index current i %~ S.index (gated_attention (I64.v position) sv model) i))
    invariant exists* (local_values : lseq f32 head_width). local_output |-> local_values
    decreases (query_heads - !head)
  {
    let h = !head;
    attend_head position scores values h local_output;
    with attended. assert local_output |-> attended;
    attended_lift (I64.v position) sv vv gv (SZ.v h) attended;
    gate_attention_head position gates local_output h #sv #vv;
    with gated. assert local_output |-> gated;
    with current. assert output |-> current;
    C.forall_intro (gated_head_slice (I64.v position) sv (SZ.v h));
    lift_replace_prefix #(gated_interpretation (I64.v position)) #activation_width #head_width
      (gated_inputs #(I64.v position) sv vv gv) (gated_attention (I64.v position) sv)
      current (SZ.v h * head_width) gated;
    write_f32_slice output local_output (h *^ 256sz) 256sz;
    head := h +^ 1sz;
  }
}

[@@CPrologue "__device__"]
fn finish_mlp (weights : larray bf16 weight_elements) (hidden : larray bf16 hidden_width)
  (#wv : erased (lseq bf16 weight_elements)) (#before : erased (lseq bf16 hidden_width)) (#fw : perm)
  preserves gpu ** weights |-> Frac fw wv
  requires hidden |-> before
  ensures exists* (after : lseq bf16 hidden_width). hidden |-> after **
    pure (forall (model : matrix_interpretation hidden_width). normalization_inputs wv before model ==>
      after %~ mlp_residual (fst model) (snd model))
{
  let mut normalized = [| zero #f32 #_; 1024sz |];
  let mut gate = [| zero #f32 #_; 3584sz |];
  let mut up = [| zero #f32 #_; 3584sz |];
  let mut projection = [| zero #f32 #_; 1024sz |];
  normalize_hidden weights hidden normalized 1024sz;
  with nv. assert normalized |-> nv;
  lift_normalization #(matrix_interpretation hidden_width) (normalization_inputs wv before)
    fst snd post_norm_offset wv before nv;
  matrix_project #hidden_width #mlp_width weights normalized gate 1024sz 3584sz 2048sz true;
  with gv. assert gate |-> gv;
  lift_matrix #(matrix_interpretation hidden_width) #hidden_width #mlp_width
    (normalization_inputs wv before) fst tail_normalized mlp_gate_offset wv nv gv;
  matrix_project #hidden_width #mlp_width weights normalized up
    1024sz 3584sz (SZ.uint32_to_sizet 3672064ul) true;
  with uv. assert up |-> uv;
  lift_matrix #(matrix_interpretation hidden_width) #hidden_width #mlp_width
    (normalization_inputs wv before) fst tail_normalized mlp_up_offset wv nv uv;
  activate_mlp gate up;
  with av. assert gate |-> av;
  lift_mlp #(matrix_interpretation hidden_width) (normalization_inputs wv before) tail_gate tail_up gv uv av;
  matrix_project #mlp_width #hidden_width weights gate projection
    3584sz 1024sz (SZ.uint32_to_sizet 7342080ul) false;
  with pv. assert projection |-> pv;
  lift_matrix #(matrix_interpretation hidden_width) #mlp_width #hidden_width
    (normalization_inputs wv before) fst tail_activation mlp_down_offset wv av pv;
  add_residual hidden projection;
  with result. assert hidden |-> result;
  lift_residual #(matrix_interpretation hidden_width) (normalization_inputs wv before)
    snd tail_down before pv result;
  assert pure (forall (model : matrix_interpretation hidden_width).
    normalization_inputs wv before model ==> result %~ residual_vector (snd model) (tail_down model));
  tail_post_equiv wv before result;
  rewrite pure (forall (model : matrix_interpretation hidden_width).
    normalization_inputs wv before model ==> result %~ residual_vector (snd model) (tail_down model)) as
    pure (forall (model : matrix_interpretation hidden_width).
      normalization_inputs wv before model ==> result %~ mlp_residual (fst model) (snd model));
}
