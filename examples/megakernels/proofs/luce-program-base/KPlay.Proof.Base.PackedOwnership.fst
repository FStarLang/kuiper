module KPlay.Proof.Base.PackedOwnership

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Core
open Kuiper.ForEvery

module S = FStar.Seq
module A = Pulse.Lib.Array
module BF = Kuiper.BFloat16
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module Inputs = KPlay.Proof.Base.Inputs
module Views = KPlay.Proof.Base.InputOwnership
module Fields = KPlay.Proof.Base.WeightFields
module Adapters = KPlay.Proof.Base.StorageAdapters
module Transfer = KPlay.Proof.Base.Transfer
module O = KPlay.Proof.Shared.Observations
module Storage = KPlay.Proof.Shared.Storage
module Model = KPlay.Proof.Shared.Model
module Attention = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Every = Kuiper.ForEvery
module Mem = Kuiper.Array.Core

inline_for_extraction noextract
let cursor_offset (cursor:Model.cursor) : nat =
  if cursor = 24 then 752393024 else Inputs.layer_base cursor

inline_for_extraction noextract
let cursor_step (cursor:Attention.layer_index)
  : Lemma (
      cursor_offset (cursor + 1) ==
      Inputs.layer_base cursor + (if cursor % 4 < 3 then 21555360 else 18352640)) = ()

inline_for_extraction noextract
let linear_at (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (layer:Model.linear_layer) : slprop =
  Storage.linear_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.linear layer)
    ((Inputs.parameter_values values).O.linear layer)

inline_for_extraction noextract
let full_at (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (layer:Attention.full_attention_layer) : slprop =
  Storage.full_weights_owns fraction
    ((Inputs.parameter_cells inputs).O.full layer)
    ((Inputs.parameter_values values).O.full layer)

inline_for_extraction noextract
let linear_suffix (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Model.cursor) : slprop =
  Every.forevery (layer:Model.linear_layer{cursor <= layer})
    (fun layer -> linear_at inputs values fraction layer)

inline_for_extraction noextract
let full_suffix (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Model.cursor) : slprop =
  Every.forevery (layer:Attention.full_attention_layer{cursor <= layer})
    (fun layer -> full_at inputs values fraction layer)

ghost
fn collect_linear
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Model.linear_layer)
  requires linear_suffix inputs values fraction cursor ** full_suffix inputs values fraction cursor
  ensures linear_suffix inputs values fraction (cursor + 1) ** full_suffix inputs values fraction (cursor + 1)
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (cursor_offset cursor) (cursor_offset (cursor + 1))
    (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)))
{
  Input.view_metadata inputs.weights;
  cursor_step cursor;
  unfold linear_suffix inputs values fraction cursor;
  unfold full_suffix inputs values fraction cursor;
  Every.forevery_remove' #Model.linear_layer
    (fun layer -> cursor <= layer) (fun layer -> linear_at inputs values fraction layer) cursor;
  unfold linear_at inputs values fraction cursor;
  Fields.linear inputs values cursor fraction;
  Every.forevery_refine_ext #Model.linear_layer
    #(fun layer -> cursor <= layer /\ ~(Prims.eq2 #Model.linear_layer layer cursor))
    (fun layer -> cursor + 1 <= layer) (fun layer -> linear_at inputs values fraction layer);
  Every.forevery_refine_ext #Attention.full_attention_layer
    #(fun layer -> cursor <= layer)
    (fun layer -> cursor + 1 <= layer) (fun layer -> full_at inputs values fraction layer);
  fold linear_suffix inputs values fraction (cursor + 1);
  fold full_suffix inputs values fraction (cursor + 1);
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base cursor) (Inputs.layer_base cursor + 21555360)
    (S.slice values (Inputs.layer_base cursor) (Inputs.layer_base cursor + 21555360))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction
      (cursor_offset cursor) (cursor_offset (cursor + 1))
      (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)));
}

ghost
fn collect_full
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Attention.full_attention_layer)
  requires linear_suffix inputs values fraction cursor ** full_suffix inputs values fraction cursor
  ensures linear_suffix inputs values fraction (cursor + 1) ** full_suffix inputs values fraction (cursor + 1)
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (cursor_offset cursor) (cursor_offset (cursor + 1))
    (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)))
{
  Input.view_metadata inputs.weights;
  cursor_step cursor;
  unfold linear_suffix inputs values fraction cursor;
  unfold full_suffix inputs values fraction cursor;
  Every.forevery_remove' #Attention.full_attention_layer
    (fun layer -> cursor <= layer) (fun layer -> full_at inputs values fraction layer) cursor;
  unfold full_at inputs values fraction cursor;
  Fields.full inputs values cursor fraction;
  Every.forevery_refine_ext #Attention.full_attention_layer
    #(fun layer -> cursor <= layer /\ ~(Prims.eq2 #Attention.full_attention_layer layer cursor))
    (fun layer -> cursor + 1 <= layer) (fun layer -> full_at inputs values fraction layer);
  Every.forevery_refine_ext #Model.linear_layer
    #(fun layer -> cursor <= layer)
    (fun layer -> cursor + 1 <= layer) (fun layer -> linear_at inputs values fraction layer);
  fold linear_suffix inputs values fraction (cursor + 1);
  fold full_suffix inputs values fraction (cursor + 1);
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (Inputs.layer_base cursor) (Inputs.layer_base cursor + 18352640)
    (S.slice values (Inputs.layer_base cursor) (Inputs.layer_base cursor + 18352640))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction
      (cursor_offset cursor) (cursor_offset (cursor + 1))
      (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)));
}

ghost
fn rec collect_layers
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Model.cursor)
  requires linear_suffix inputs values fraction cursor ** full_suffix inputs values fraction cursor
  requires Mem.pts_to_slice (Input.data inputs.weights) #fraction 0 (cursor_offset cursor)
    (S.slice values 0 (cursor_offset cursor))
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction 0 752393024 values
  decreases (24 - cursor)
{
  Input.view_metadata inputs.weights;
  if (cursor < 24) {
    if (cursor % 4 < 3) {
      collect_linear inputs values fraction cursor;
      Adapters.join_slices (Input.data inputs.weights) values fraction
        0 (cursor_offset cursor) (cursor_offset (cursor + 1));
      collect_layers inputs values fraction (cursor + 1);
    } else {
      collect_full inputs values fraction cursor;
      Adapters.join_slices (Input.data inputs.weights) values fraction
        0 (cursor_offset cursor) (cursor_offset (cursor + 1));
      collect_layers inputs values fraction (cursor + 1);
    }
  } else {
    unfold linear_suffix inputs values fraction cursor;
    unfold full_suffix inputs values fraction cursor;
    Every.forevery_elim_empty #(layer:Model.linear_layer{cursor <= layer})
      (fun layer -> linear_at inputs values fraction layer);
    Every.forevery_elim_empty #(layer:Attention.full_attention_layer{cursor <= layer})
      (fun layer -> full_at inputs values fraction layer);
    S.slice_length values;
    rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction 0 (cursor_offset cursor)
      (S.slice values 0 (cursor_offset cursor))
      as Mem.pts_to_slice (Input.data inputs.weights) #fraction 0 752393024 values;
  }
}

ghost
fn parameters_to_array
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed) (fraction:perm)
  requires Storage.parameters_owns fraction (Inputs.parameter_cells inputs) (Inputs.parameter_values values)
  ensures A.pts_to (Input.data inputs.weights) #fraction values
{
  Input.view_metadata inputs.weights;
  unfold Storage.parameters_owns fraction (Inputs.parameter_cells inputs) (Inputs.parameter_values values);
  Views.embedding inputs values fraction;
  Adapters.vector_to_slice #BF.t #1024 (Input.data inputs.weights) 254279680 values
    (Inputs.parameter_cells inputs).O.final_norm (Inputs.parameter_values values).O.final_norm fraction;
  Adapters.join_slices (Input.data inputs.weights) values fraction 0 254279680 254280704;
  Every.forevery_ext #Model.linear_layer
    (fun layer -> Storage.linear_weights_owns fraction
      ((Inputs.parameter_cells inputs).O.linear layer)
      ((Inputs.parameter_values values).O.linear layer))
    (fun layer -> linear_at inputs values fraction layer);
  Every.forevery_ext #Attention.full_attention_layer
    (fun layer -> Storage.full_weights_owns fraction
      ((Inputs.parameter_cells inputs).O.full layer)
      ((Inputs.parameter_values values).O.full layer))
    (fun layer -> full_at inputs values fraction layer);
  Every.forevery_ext #Model.linear_layer
    (fun layer -> linear_at inputs values fraction layer)
    (fun layer -> Every.when_ (0 <= layer) (linear_at inputs values fraction layer));
  Every.forevery_refine_pred #Model.linear_layer
    (fun layer -> linear_at inputs values fraction layer) (fun layer -> 0 <= layer);
  Every.forevery_ext #Attention.full_attention_layer
    (fun layer -> full_at inputs values fraction layer)
    (fun layer -> Every.when_ (0 <= layer) (full_at inputs values fraction layer));
  Every.forevery_refine_pred #Attention.full_attention_layer
    (fun layer -> full_at inputs values fraction layer) (fun layer -> 0 <= layer);
  fold linear_suffix inputs values fraction 0;
  fold full_suffix inputs values fraction 0;
  collect_layers inputs values fraction 0;
  Mem.slice_to_array_full #BF.t (Input.data inputs.weights) #fraction #values #752393024;
}

ghost
fn restore_linear
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Model.linear_layer)
  requires linear_suffix inputs values fraction (cursor + 1) ** full_suffix inputs values fraction (cursor + 1)
  requires Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (cursor_offset cursor) (cursor_offset (cursor + 1))
    (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)))
  ensures linear_suffix inputs values fraction cursor ** full_suffix inputs values fraction cursor
{
  Input.view_metadata inputs.weights;
  cursor_step cursor;
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (cursor_offset cursor) (cursor_offset (cursor + 1))
    (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction
      (Inputs.layer_base cursor) (Inputs.layer_base cursor + 21555360)
      (S.slice values (Inputs.layer_base cursor) (Inputs.layer_base cursor + 21555360));
  Fields.restore_linear inputs values cursor fraction;
  fold linear_at inputs values fraction cursor;
  unfold linear_suffix inputs values fraction (cursor + 1);
  unfold full_suffix inputs values fraction (cursor + 1);
  Every.forevery_insert #Model.linear_layer #(fun layer -> cursor + 1 <= layer)
    (fun layer -> linear_at inputs values fraction layer) cursor;
  Every.forevery_refine_ext #Model.linear_layer
    #(fun layer -> cursor + 1 <= layer \/ Prims.eq2 #Model.linear_layer cursor layer)
    (fun layer -> cursor <= layer) (fun layer -> linear_at inputs values fraction layer);
  Every.forevery_refine_ext #Attention.full_attention_layer
    #(fun layer -> cursor + 1 <= layer)
    (fun layer -> cursor <= layer) (fun layer -> full_at inputs values fraction layer);
  fold linear_suffix inputs values fraction cursor;
  fold full_suffix inputs values fraction cursor;
}

ghost
fn restore_full
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Attention.full_attention_layer)
  requires linear_suffix inputs values fraction (cursor + 1) ** full_suffix inputs values fraction (cursor + 1)
  requires Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (cursor_offset cursor) (cursor_offset (cursor + 1))
    (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)))
  ensures linear_suffix inputs values fraction cursor ** full_suffix inputs values fraction cursor
{
  Input.view_metadata inputs.weights;
  cursor_step cursor;
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction
    (cursor_offset cursor) (cursor_offset (cursor + 1))
    (S.slice values (cursor_offset cursor) (cursor_offset (cursor + 1)))
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction
      (Inputs.layer_base cursor) (Inputs.layer_base cursor + 18352640)
      (S.slice values (Inputs.layer_base cursor) (Inputs.layer_base cursor + 18352640));
  Fields.restore_full inputs values cursor fraction;
  fold full_at inputs values fraction cursor;
  unfold linear_suffix inputs values fraction (cursor + 1);
  unfold full_suffix inputs values fraction (cursor + 1);
  Every.forevery_insert #Attention.full_attention_layer #(fun layer -> cursor + 1 <= layer)
    (fun layer -> full_at inputs values fraction layer) cursor;
  Every.forevery_refine_ext #Attention.full_attention_layer
    #(fun layer -> cursor + 1 <= layer \/ Prims.eq2 #Attention.full_attention_layer cursor layer)
    (fun layer -> cursor <= layer) (fun layer -> full_at inputs values fraction layer);
  Every.forevery_refine_ext #Model.linear_layer
    #(fun layer -> cursor + 1 <= layer)
    (fun layer -> cursor <= layer) (fun layer -> linear_at inputs values fraction layer);
  fold linear_suffix inputs values fraction cursor;
  fold full_suffix inputs values fraction cursor;
}

ghost
fn rec restore_layers
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (fraction:perm) (cursor:Model.cursor)
  requires Mem.pts_to_slice (Input.data inputs.weights) #fraction (cursor_offset cursor) 752393024
    (S.slice values (cursor_offset cursor) 752393024)
  ensures linear_suffix inputs values fraction cursor ** full_suffix inputs values fraction cursor
  decreases (24 - cursor)
{
  Input.view_metadata inputs.weights;
  if (cursor < 24) {
    cursor_step cursor;
    Adapters.split_slices (Input.data inputs.weights) values fraction
      (cursor_offset cursor) (cursor_offset (cursor + 1)) 752393024;
    restore_layers inputs values fraction (cursor + 1);
    if (cursor % 4 < 3) {
      restore_linear inputs values fraction cursor;
    } else {
      restore_full inputs values fraction cursor;
    }
  } else {
    drop_ (Mem.pts_to_slice (Input.data inputs.weights) #fraction (cursor_offset cursor) 752393024
      (S.slice values (cursor_offset cursor) 752393024));
    Every.forevery_intro_empty #(layer:Model.linear_layer{cursor <= layer})
      (fun layer -> linear_at inputs values fraction layer);
    Every.forevery_intro_empty #(layer:Attention.full_attention_layer{cursor <= layer})
      (fun layer -> full_at inputs values fraction layer);
    fold linear_suffix inputs values fraction cursor;
    fold full_suffix inputs values fraction cursor;
  }
}

ghost
fn array_to_parameters
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed) (fraction:perm)
  requires A.pts_to (Input.data inputs.weights) #fraction values
  ensures Storage.parameters_owns fraction (Inputs.parameter_cells inputs) (Inputs.parameter_values values)
{
  Input.view_metadata inputs.weights;
  Mem.array_to_slice (Input.data inputs.weights) #fraction #values;
  drop_ (Mem.is_full_slice (Input.data inputs.weights) 752393024);
  S.slice_length values;
  rewrite Mem.pts_to_slice (Input.data inputs.weights) #fraction 0 752393024 values
    as Mem.pts_to_slice (Input.data inputs.weights) #fraction 0 752393024 (S.slice values 0 752393024);
  Adapters.split_slices (Input.data inputs.weights) values fraction 0 254279680 752393024;
  Adapters.split_slices (Input.data inputs.weights) values fraction 254279680 254280704 752393024;
  restore_layers inputs values fraction 0;
  Adapters.slice_to_vector #BF.t #1024 (Input.data inputs.weights) 254279680 values
    (Inputs.parameter_cells inputs).O.final_norm (Inputs.parameter_values values).O.final_norm fraction;
  Adapters.slice_to_matrix #BF.t #248320 #1024 (Input.data inputs.weights) 0 values
    (fun token i -> S.index (S.index (Inputs.parameter_cells inputs).O.embedding token) i)
    (fun token i -> S.index (S.index (Inputs.parameter_values values).O.embedding token) i) fraction;
  unfold Storage.matrix_owns #BF.t #248320 #1024 fraction
    (fun token i -> S.index (S.index (Inputs.parameter_cells inputs).O.embedding token) i)
    (fun token i -> S.index (S.index (Inputs.parameter_values values).O.embedding token) i);
  Every.forevery_rw_type (P.index 248320) (KPlay.Proof.Shared.Selection.token 248320)
    (fun token -> Storage.vector_owns #BF.t #1024 fraction
      (fun i -> S.index (S.index (Inputs.parameter_cells inputs).O.embedding token) i)
      (fun i -> S.index (S.index (Inputs.parameter_values values).O.embedding token) i));
  unfold linear_suffix inputs values fraction 0;
  unfold full_suffix inputs values fraction 0;
  Every.forevery_unrefine #Model.linear_layer #(fun layer -> 0 <= layer)
    (fun layer -> linear_at inputs values fraction layer);
  Every.forevery_unrefine #Attention.full_attention_layer #(fun layer -> 0 <= layer)
    (fun layer -> full_at inputs values fraction layer);
  Every.forevery_map #Model.linear_layer
    (fun layer -> linear_at inputs values fraction layer)
    (fun layer -> Storage.linear_weights_owns fraction
      ((Inputs.parameter_cells inputs).O.linear layer) ((Inputs.parameter_values values).O.linear layer))
    fn layer { unfold linear_at inputs values fraction layer; };
  Every.forevery_map #Attention.full_attention_layer
    (fun layer -> full_at inputs values fraction layer)
    (fun layer -> Storage.full_weights_owns fraction
      ((Inputs.parameter_cells inputs).O.full layer) ((Inputs.parameter_values values).O.full layer))
    fn layer { unfold full_at inputs values fraction layer; };
  fold Storage.parameters_owns fraction (Inputs.parameter_cells inputs) (Inputs.parameter_values values);
}
