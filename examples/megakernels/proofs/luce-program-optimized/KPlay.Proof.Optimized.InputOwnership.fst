module KPlay.Proof.Optimized.InputOwnership

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Core

module S = FStar.Seq
module A = Pulse.Lib.Array
module Mem = Kuiper.Array.Core
module Every = Kuiper.ForEvery
module BF = Kuiper.BFloat16
module Input = Kuiper.Tensor.Input
module NativeInput = KPlay.Proof.NativeInput
module Inputs = KPlay.Proof.Optimized.Layout
module Transfer = KPlay.Proof.Optimized.Layout
module Storage = KPlay.Proof.Shared.Storage
module O = KPlay.Proof.Shared.Observations
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Attention = KPlay.Proof.Shared.Attention
module Adapters = KPlay.Proof.Optimized.StorageAdapters
module I64 = FStar.Int64

ghost
fn native_vector
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (offset:nat) (count:pos{offset + count <= 752393024}) (fraction:perm)
  requires Storage.vector_owns #BF.t #count fraction
    (fun i -> NativeInput.cell inputs.weights (offset + i))
    (fun i -> S.index values (offset + i))
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction offset (offset + count)
    (S.slice values offset (offset + count))
{
  Input.view_metadata inputs.weights;
  Adapters.vector_to_slice #BF.t #count (Input.data inputs.weights) offset values
    (fun i -> NativeInput.cell inputs.weights (offset + i))
    (fun i -> S.index values (offset + i)) fraction;
}

ghost
fn native_matrix
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed)
  (offset:nat) (rows columns:pos{offset + rows * columns <= 752393024}) (fraction:perm)
  requires Storage.matrix_owns #BF.t #rows #columns fraction
    (fun r c -> NativeInput.cell inputs.weights (offset + r * columns + c))
    (fun r c -> S.index values (offset + r * columns + c))
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction offset (offset + rows * columns)
    (S.slice values offset (offset + rows * columns))
{
  Input.view_metadata inputs.weights;
  Adapters.matrix_to_slice #BF.t #rows #columns (Input.data inputs.weights) offset values
    (fun r c -> NativeInput.cell inputs.weights (offset + r * columns + c))
    (fun r c -> S.index values (offset + r * columns + c)) fraction;
}

ghost
fn embedding
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:Transfer.packed) (fraction:perm)
  requires Every.forevery (D.token 248320) (fun token ->
    Storage.vector_owns #BF.t #1024 fraction
      (fun i -> S.index (S.index (Inputs.parameter_cells inputs).O.embedding token) i)
      (fun i -> S.index (S.index (Inputs.parameter_values values).O.embedding token) i))
  ensures Mem.pts_to_slice (Input.data inputs.weights) #fraction 0 254279680
    (S.slice values 0 254279680)
{
  Every.forevery_rw_type (D.token 248320) (P.index 248320)
    (fun token -> Storage.vector_owns #BF.t #1024 fraction
      (fun i -> S.index (S.index (Inputs.parameter_cells inputs).O.embedding token) i)
      (fun i -> S.index (S.index (Inputs.parameter_values values).O.embedding token) i));
  fold Storage.matrix_owns #BF.t #248320 #1024 fraction
    (fun token i -> S.index (S.index (Inputs.parameter_cells inputs).O.embedding token) i)
    (fun token i -> S.index (S.index (Inputs.parameter_values values).O.embedding token) i);
  Input.view_metadata inputs.weights;
  Adapters.matrix_to_slice #BF.t #248320 #1024 (Input.data inputs.weights) 0 values
    (fun token i -> S.index (S.index (Inputs.parameter_cells inputs).O.embedding token) i)
    (fun token i -> S.index (S.index (Inputs.parameter_values values).O.embedding token) i)
    fraction;
}

ghost
fn prompt_to_array
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:lseq I64.t p)
  (logical:D.vec (D.token 248320) p) (fraction:perm)
  requires Storage.tokens_owns I64.v fraction (Inputs.prompt_storage inputs values) logical
  ensures A.pts_to (Input.data inputs.prompt) #fraction values
  ensures pure (forall (i:P.index p). I64.v (S.index values i) == S.index logical i)
{
  Input.view_metadata inputs.prompt;
  unfold Storage.tokens_owns I64.v fraction (Inputs.prompt_storage inputs values) logical;
  Adapters.vector_to_slice #I64.t #p (Input.data inputs.prompt) 0 values
    (Inputs.prompt_storage inputs values).Storage.addresses
    (fun i -> S.index (Inputs.prompt_storage inputs values).Storage.contents i) fraction;
  S.slice_length values;
  rewrite Mem.pts_to_slice (Input.data inputs.prompt) #fraction 0 p (S.slice values 0 p)
    as Mem.pts_to_slice (Input.data inputs.prompt) #fraction 0 p values;
  Mem.slice_to_array_full #I64.t (Input.data inputs.prompt) #fraction #values #p;
}

ghost
fn array_to_prompt
  (#p:pos{p <= 520}) (inputs:NativeInput.arguments p) (values:lseq I64.t p)
  (logical:D.vec (D.token 248320) p) (fraction:perm)
  requires A.pts_to (Input.data inputs.prompt) #fraction values
  requires pure (forall (i:P.index p). I64.v (S.index values i) == S.index logical i)
  ensures Storage.tokens_owns I64.v fraction (Inputs.prompt_storage inputs values) logical
{
  Input.view_metadata inputs.prompt;
  Mem.array_to_slice (Input.data inputs.prompt) #fraction #values;
  drop_ (Mem.is_full_slice (Input.data inputs.prompt) p);
  S.slice_length values;
  rewrite Mem.pts_to_slice (Input.data inputs.prompt) #fraction 0 p values
    as Mem.pts_to_slice (Input.data inputs.prompt) #fraction 0 p (S.slice values 0 p);
  Adapters.slice_to_vector #I64.t #p (Input.data inputs.prompt) 0 values
    (Inputs.prompt_storage inputs values).Storage.addresses
    (fun i -> S.index (Inputs.prompt_storage inputs values).Storage.contents i) fraction;
  fold Storage.tokens_owns I64.v fraction (Inputs.prompt_storage inputs values) logical;
}
