module KPlay.Proof.Optimized.WeightFields

#lang-pulse

#set-options "--fuel 4 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Core

module S = FStar.Seq
module BF = Kuiper.BFloat16
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module Inputs = KPlay.Proof.Optimized.Layout
module Transfer = KPlay.Proof.Optimized.Layout
module O = KPlay.Proof.Shared.Observations
module Storage = KPlay.Proof.Shared.Storage
module Mem = Kuiper.Array.Core
module Model = KPlay.Proof.Shared.Model
module Attention = KPlay.Proof.Shared.Attention
module Adapters = KPlay.Proof.Optimized.StorageAdapters

ghost
fn linear
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (layer:Model.linear_layer) (fraction:perm)
  requires Storage.linear_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.linear layer)
    ((Inputs.parameter_values values).O.linear layer)
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360)
    (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360))
{
  Input.view_metadata inputs.weights;
  Inputs.layer_extent layer;
  let start = Inputs.layer_base layer;
  unfold Storage.linear_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.linear layer)
    ((Inputs.parameter_values values).O.linear layer);
  Adapters.vector_to_slice #BF.t #1024 (Input.data inputs.weights) start values
    ((Inputs.parameter_cells inputs).O.linear layer).O.input_norm
    ((Inputs.parameter_values values).O.linear layer).O.input_norm fraction;
  Adapters.vector_to_slice #BF.t #1024 (Input.data inputs.weights) (start + 1024) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.post_attention_norm
    ((Inputs.parameter_values values).O.linear layer).O.post_attention_norm fraction;
  Adapters.matrix_to_slice #BF.t #3584 #1024 (Input.data inputs.weights) (start + 2048) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.mlp_gate
    ((Inputs.parameter_values values).O.linear layer).O.mlp_gate fraction;
  Adapters.matrix_to_slice #BF.t #3584 #1024 (Input.data inputs.weights) (start + 3672064) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.mlp_up
    ((Inputs.parameter_values values).O.linear layer).O.mlp_up fraction;
  Adapters.matrix_to_slice #BF.t #1024 #3584 (Input.data inputs.weights) (start + 7342080) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.mlp_down
    ((Inputs.parameter_values values).O.linear layer).O.mlp_down fraction;
  Adapters.matrix_to_slice #BF.t #6144 #1024 (Input.data inputs.weights) (start + 11012096) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_qkv
    ((Inputs.parameter_values values).O.linear layer).O.in_qkv fraction;
  Adapters.matrix_to_slice #BF.t #2048 #1024 (Input.data inputs.weights) (start + 17303552) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_z
    ((Inputs.parameter_values values).O.linear layer).O.in_z fraction;
  Adapters.matrix_to_slice #BF.t #16 #1024 (Input.data inputs.weights) (start + 19400704) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_beta
    ((Inputs.parameter_values values).O.linear layer).O.in_beta fraction;
  Adapters.matrix_to_slice #BF.t #16 #1024 (Input.data inputs.weights) (start + 19417088) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_a
    ((Inputs.parameter_values values).O.linear layer).O.in_a fraction;
  Adapters.matrix_to_slice #BF.t #6144 #4 (Input.data inputs.weights) (start + 19433472) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.conv_kernel
    ((Inputs.parameter_values values).O.linear layer).O.conv_kernel fraction;
  Adapters.vector_to_slice #BF.t #16 (Input.data inputs.weights) (start + 19458048) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.a_log
    ((Inputs.parameter_values values).O.linear layer).O.a_log fraction;
  Adapters.vector_to_slice #BF.t #16 (Input.data inputs.weights) (start + 19458064) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.dt_bias
    ((Inputs.parameter_values values).O.linear layer).O.dt_bias fraction;
  Adapters.vector_to_slice #BF.t #128 (Input.data inputs.weights) (start + 19458080) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.gated_norm
    ((Inputs.parameter_values values).O.linear layer).O.gated_norm fraction;
  Adapters.matrix_to_slice #BF.t #1024 #2048 (Input.data inputs.weights) (start + 19458208) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.out_projection
    ((Inputs.parameter_values values).O.linear layer).O.out_projection fraction;
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 1024) (start + 2048);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 2048) (start + 3672064);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 3672064) (start + 7342080);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 7342080) (start + 11012096);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 11012096) (start + 17303552);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 17303552) (start + 19400704);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 19400704) (start + 19417088);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 19417088) (start + 19433472);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 19433472) (start + 19458048);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 19458048) (start + 19458064);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 19458064) (start + 19458080);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 19458080) (start + 19458208);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 19458208) (start + 21555360);
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction start (start + 21555360)
    (S.slice values start (start + 21555360))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction
      (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360)
      (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360));
}

ghost
fn restore_linear
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (layer:Model.linear_layer) (fraction:perm)
  requires Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360)
    (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360))
  ensures Storage.linear_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.linear layer)
    ((Inputs.parameter_values values).O.linear layer)
{
  Input.view_metadata inputs.weights;
  Inputs.layer_extent layer;
  let start = Inputs.layer_base layer;
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360)
    (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 21555360))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction start (start + 21555360)
      (S.slice values start (start + 21555360));
  Adapters.split_slices (Input.data inputs.weights) values fraction
    start (start + 1024) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 1024) (start + 2048) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 2048) (start + 3672064) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 3672064) (start + 7342080) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 7342080) (start + 11012096) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 11012096) (start + 17303552) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 17303552) (start + 19400704) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 19400704) (start + 19417088) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 19417088) (start + 19433472) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 19433472) (start + 19458048) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 19458048) (start + 19458064) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 19458064) (start + 19458080) (start + 21555360);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 19458080) (start + 19458208) (start + 21555360);
  Adapters.slice_to_vector #BF.t #1024 (Input.data inputs.weights) start values
    ((Inputs.parameter_cells inputs).O.linear layer).O.input_norm
    ((Inputs.parameter_values values).O.linear layer).O.input_norm fraction;
  Adapters.slice_to_vector #BF.t #1024 (Input.data inputs.weights) (start + 1024) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.post_attention_norm
    ((Inputs.parameter_values values).O.linear layer).O.post_attention_norm fraction;
  Adapters.slice_to_matrix #BF.t #3584 #1024 (Input.data inputs.weights) (start + 2048) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.mlp_gate
    ((Inputs.parameter_values values).O.linear layer).O.mlp_gate fraction;
  Adapters.slice_to_matrix #BF.t #3584 #1024 (Input.data inputs.weights) (start + 3672064) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.mlp_up
    ((Inputs.parameter_values values).O.linear layer).O.mlp_up fraction;
  Adapters.slice_to_matrix #BF.t #1024 #3584 (Input.data inputs.weights) (start + 7342080) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.mlp_down
    ((Inputs.parameter_values values).O.linear layer).O.mlp_down fraction;
  Adapters.slice_to_matrix #BF.t #6144 #1024 (Input.data inputs.weights) (start + 11012096) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_qkv
    ((Inputs.parameter_values values).O.linear layer).O.in_qkv fraction;
  Adapters.slice_to_matrix #BF.t #2048 #1024 (Input.data inputs.weights) (start + 17303552) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_z
    ((Inputs.parameter_values values).O.linear layer).O.in_z fraction;
  Adapters.slice_to_matrix #BF.t #16 #1024 (Input.data inputs.weights) (start + 19400704) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_beta
    ((Inputs.parameter_values values).O.linear layer).O.in_beta fraction;
  Adapters.slice_to_matrix #BF.t #16 #1024 (Input.data inputs.weights) (start + 19417088) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.in_a
    ((Inputs.parameter_values values).O.linear layer).O.in_a fraction;
  Adapters.slice_to_matrix #BF.t #6144 #4 (Input.data inputs.weights) (start + 19433472) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.conv_kernel
    ((Inputs.parameter_values values).O.linear layer).O.conv_kernel fraction;
  Adapters.slice_to_vector #BF.t #16 (Input.data inputs.weights) (start + 19458048) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.a_log
    ((Inputs.parameter_values values).O.linear layer).O.a_log fraction;
  Adapters.slice_to_vector #BF.t #16 (Input.data inputs.weights) (start + 19458064) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.dt_bias
    ((Inputs.parameter_values values).O.linear layer).O.dt_bias fraction;
  Adapters.slice_to_vector #BF.t #128 (Input.data inputs.weights) (start + 19458080) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.gated_norm
    ((Inputs.parameter_values values).O.linear layer).O.gated_norm fraction;
  Adapters.slice_to_matrix #BF.t #1024 #2048 (Input.data inputs.weights) (start + 19458208) values
    ((Inputs.parameter_cells inputs).O.linear layer).O.out_projection
    ((Inputs.parameter_values values).O.linear layer).O.out_projection fraction;
  fold Storage.linear_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.linear layer)
    ((Inputs.parameter_values values).O.linear layer);
}

ghost
fn full
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (layer:Attention.full_attention_layer) (fraction:perm)
  requires Storage.full_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.full layer)
    ((Inputs.parameter_values values).O.full layer)
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640)
    (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640))
{
  Input.view_metadata inputs.weights;
  Inputs.layer_extent layer;
  let start = Inputs.layer_base layer;
  unfold Storage.full_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.full layer)
    ((Inputs.parameter_values values).O.full layer);
  Adapters.vector_to_slice #BF.t #1024 (Input.data inputs.weights) start values
    ((Inputs.parameter_cells inputs).O.full layer).O.input_norm
    ((Inputs.parameter_values values).O.full layer).O.input_norm fraction;
  Adapters.vector_to_slice #BF.t #1024 (Input.data inputs.weights) (start + 1024) values
    ((Inputs.parameter_cells inputs).O.full layer).O.post_attention_norm
    ((Inputs.parameter_values values).O.full layer).O.post_attention_norm fraction;
  Adapters.matrix_to_slice #BF.t #3584 #1024 (Input.data inputs.weights) (start + 2048) values
    ((Inputs.parameter_cells inputs).O.full layer).O.mlp_gate_proj
    ((Inputs.parameter_values values).O.full layer).O.mlp_gate_proj fraction;
  Adapters.matrix_to_slice #BF.t #3584 #1024 (Input.data inputs.weights) (start + 3672064) values
    ((Inputs.parameter_cells inputs).O.full layer).O.mlp_up_proj
    ((Inputs.parameter_values values).O.full layer).O.mlp_up_proj fraction;
  Adapters.matrix_to_slice #BF.t #1024 #3584 (Input.data inputs.weights) (start + 7342080) values
    ((Inputs.parameter_cells inputs).O.full layer).O.mlp_down_proj
    ((Inputs.parameter_values values).O.full layer).O.mlp_down_proj fraction;
  Adapters.matrix_to_slice #BF.t #4096 #1024 (Input.data inputs.weights) (start + 11012096) values
    ((Inputs.parameter_cells inputs).O.full layer).O.query_gate_proj
    ((Inputs.parameter_values values).O.full layer).O.query_gate_proj fraction;
  Adapters.matrix_to_slice #BF.t #512 #1024 (Input.data inputs.weights) (start + 15206400) values
    ((Inputs.parameter_cells inputs).O.full layer).O.key_proj
    ((Inputs.parameter_values values).O.full layer).O.key_proj fraction;
  Adapters.matrix_to_slice #BF.t #512 #1024 (Input.data inputs.weights) (start + 15730688) values
    ((Inputs.parameter_cells inputs).O.full layer).O.value_proj
    ((Inputs.parameter_values values).O.full layer).O.value_proj fraction;
  Adapters.vector_to_slice #BF.t #256 (Input.data inputs.weights) (start + 16254976) values
    ((Inputs.parameter_cells inputs).O.full layer).O.query_norm
    ((Inputs.parameter_values values).O.full layer).O.query_norm fraction;
  Adapters.vector_to_slice #BF.t #256 (Input.data inputs.weights) (start + 16255232) values
    ((Inputs.parameter_cells inputs).O.full layer).O.key_norm
    ((Inputs.parameter_values values).O.full layer).O.key_norm fraction;
  Adapters.matrix_to_slice #BF.t #1024 #2048 (Input.data inputs.weights) (start + 16255488) values
    ((Inputs.parameter_cells inputs).O.full layer).O.output_proj
    ((Inputs.parameter_values values).O.full layer).O.output_proj fraction;
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 1024) (start + 2048);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 2048) (start + 3672064);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 3672064) (start + 7342080);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 7342080) (start + 11012096);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 11012096) (start + 15206400);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 15206400) (start + 15730688);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 15730688) (start + 16254976);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 16254976) (start + 16255232);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 16255232) (start + 16255488);
  Adapters.join_slices (Input.data inputs.weights) values fraction
    start (start + 16255488) (start + 18352640);
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction start (start + 18352640)
    (S.slice values start (start + 18352640))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction
      (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640)
      (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640));
}

ghost
fn restore_full
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (layer:Attention.full_attention_layer) (fraction:perm)
  requires Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640)
    (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640))
  ensures Storage.full_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.full layer)
    ((Inputs.parameter_values values).O.full layer)
{
  Input.view_metadata inputs.weights;
  Inputs.layer_extent layer;
  let start = Inputs.layer_base layer;
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640)
    (S.slice values (Inputs.layer_base layer) (Inputs.layer_base layer + 18352640))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction start (start + 18352640)
      (S.slice values start (start + 18352640));
  Adapters.split_slices (Input.data inputs.weights) values fraction
    start (start + 1024) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 1024) (start + 2048) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 2048) (start + 3672064) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 3672064) (start + 7342080) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 7342080) (start + 11012096) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 11012096) (start + 15206400) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 15206400) (start + 15730688) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 15730688) (start + 16254976) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 16254976) (start + 16255232) (start + 18352640);
  Adapters.split_slices (Input.data inputs.weights) values fraction
    (start + 16255232) (start + 16255488) (start + 18352640);
  Adapters.slice_to_vector #BF.t #1024 (Input.data inputs.weights) start values
    ((Inputs.parameter_cells inputs).O.full layer).O.input_norm
    ((Inputs.parameter_values values).O.full layer).O.input_norm fraction;
  Adapters.slice_to_vector #BF.t #1024 (Input.data inputs.weights) (start + 1024) values
    ((Inputs.parameter_cells inputs).O.full layer).O.post_attention_norm
    ((Inputs.parameter_values values).O.full layer).O.post_attention_norm fraction;
  Adapters.slice_to_matrix #BF.t #3584 #1024 (Input.data inputs.weights) (start + 2048) values
    ((Inputs.parameter_cells inputs).O.full layer).O.mlp_gate_proj
    ((Inputs.parameter_values values).O.full layer).O.mlp_gate_proj fraction;
  Adapters.slice_to_matrix #BF.t #3584 #1024 (Input.data inputs.weights) (start + 3672064) values
    ((Inputs.parameter_cells inputs).O.full layer).O.mlp_up_proj
    ((Inputs.parameter_values values).O.full layer).O.mlp_up_proj fraction;
  Adapters.slice_to_matrix #BF.t #1024 #3584 (Input.data inputs.weights) (start + 7342080) values
    ((Inputs.parameter_cells inputs).O.full layer).O.mlp_down_proj
    ((Inputs.parameter_values values).O.full layer).O.mlp_down_proj fraction;
  Adapters.slice_to_matrix #BF.t #4096 #1024 (Input.data inputs.weights) (start + 11012096) values
    ((Inputs.parameter_cells inputs).O.full layer).O.query_gate_proj
    ((Inputs.parameter_values values).O.full layer).O.query_gate_proj fraction;
  Adapters.slice_to_matrix #BF.t #512 #1024 (Input.data inputs.weights) (start + 15206400) values
    ((Inputs.parameter_cells inputs).O.full layer).O.key_proj
    ((Inputs.parameter_values values).O.full layer).O.key_proj fraction;
  Adapters.slice_to_matrix #BF.t #512 #1024 (Input.data inputs.weights) (start + 15730688) values
    ((Inputs.parameter_cells inputs).O.full layer).O.value_proj
    ((Inputs.parameter_values values).O.full layer).O.value_proj fraction;
  Adapters.slice_to_vector #BF.t #256 (Input.data inputs.weights) (start + 16254976) values
    ((Inputs.parameter_cells inputs).O.full layer).O.query_norm
    ((Inputs.parameter_values values).O.full layer).O.query_norm fraction;
  Adapters.slice_to_vector #BF.t #256 (Input.data inputs.weights) (start + 16255232) values
    ((Inputs.parameter_cells inputs).O.full layer).O.key_norm
    ((Inputs.parameter_values values).O.full layer).O.key_norm fraction;
  Adapters.slice_to_matrix #BF.t #1024 #2048 (Input.data inputs.weights) (start + 16255488) values
    ((Inputs.parameter_cells inputs).O.full layer).O.output_proj
    ((Inputs.parameter_values values).O.full layer).O.output_proj fraction;
  fold Storage.full_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.full layer)
    ((Inputs.parameter_values values).O.full layer);
}
