module KPlay.Proof.Optimized.Final

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 50"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module Norm = KPlay.Proof.Optimized.Norm
module NormMath = KPlay.Proof.Optimized.NormMath
module Dot = KPlay.Proof.Optimized.Dot
module DotMath = KPlay.Proof.Optimized.DotMath
module Memory = KPlay.Proof.Optimized.Memory

(* reduced.cu:343-345 produces FP32, unlike the decoder's BF16 normalization. *)
noextract
fn normalize (weights:larray bf16 752393024) (hidden:larray bf16 1024)
  (activation:larray f32 2048)
  (#wv:erased (lseq bf16 752393024)) (#hv:erased (lseq bf16 1024))
  (#av:erased (lseq (option f32) 2048)) (#fw #fh:perm)
  preserves gpu ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires A.pts_to_mask activation av Memory.all_cells
  ensures exists* (after:lseq (option f32) 2048).
    A.pts_to_mask activation after Memory.all_cells **
    pure (Memory.ready after 0 1024 /\
      (forall (i:nat{1024 <= i /\ i < 2048}). S.index after i == S.index av i) /\
      (forall (m:Norm.interpretation 752393024). Norm.inputs wv hv m ==>
        (forall (i:nat{i < 1024}). Dot.option_models (S.index after i)
          (S.index (Norm.normalized m 254279680) i))))
{
  let rstd = Norm.hidden_rstd hidden;
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i:sz{i <= 1024}) (pv:lseq (option f32) 2048).
      index |-> i ** A.pts_to_mask activation pv Memory.all_cells **
      pure (Memory.ready pv 0 (SZ.v i)) **
      pure (forall (j:nat{SZ.v i <= j /\ j < 2048}). S.index pv j == S.index av j) **
      pure (forall (m:Norm.interpretation 752393024). Norm.inputs wv hv m ==>
        (forall (j:nat{j < SZ.v i}). Dot.option_models (S.index pv j)
          (S.index (Norm.normalized m 254279680) j)))
    decreases (1024 - !index)
  {
    let i = !index;
    let x = Cast.cast_bf16_to_f32 (hidden.(i));
    let w = Cast.cast_bf16_to_f32 (weights.(SZ.uint32_to_sizet 254279680ul +^ i));
    let scaled = FP.mul_rn_ftz x rstd;
    lift_mul #(Norm.interpretation 752393024) (Norm.inputs wv hv)
      (fun m -> S.index (snd m) (SZ.v i))
      (fun m -> (NormMath.scale (snd m) <: real)) x rstd scaled;
    let gain = FP.add_rn_ftz (one #f32) w;
    lift_add #(Norm.interpretation 752393024) (Norm.inputs wv hv)
      (fun _ -> 1.0R) (fun m -> S.index (fst m) (254279680 + SZ.v i))
      (one #f32) w gain;
    let value = FP.mul_rn_ftz scaled gain;
    lift_mul #(Norm.interpretation 752393024) (Norm.inputs wv hv)
      (fun m -> S.index (snd m) (SZ.v i) *. NormMath.scale (snd m))
      (fun m -> 1.0R +. S.index (fst m) (254279680 + SZ.v i))
      scaled gain value;
    A.mask_write activation i value;
    index := i +^ 1sz;
  }
}

(* A proof-only view of the same activation pointer consumed at reduced.cu:354. *)
noextract
fn score (weights:larray bf16 752393024) (activation:larray f32 2048)
  (row:sz{row < 248320})
  (#wv:erased (lseq bf16 752393024))
  (#av:erased (lseq (option f32) 2048)) (#fw #fa:perm)
  (initialized:squash (Memory.ready av 0 1024))
  preserves gpu ** weights |-> Frac fw wv
  requires A.pts_to_mask activation #fa av Memory.all_cells
  returns result:f32
  ensures A.pts_to_mask activation #fa av Memory.all_cells
  ensures pure (forall (m:DotMath.interpretation 752393024 1024).
    DotMath.float_inputs wv (Memory.values av 0 1024 initialized) m ==>
      result %~ rsum (S.init_ghost 1024 (DotMath.term m (SZ.v row * 1024))))
{
  let values = Memory.borrow activation 0sz 1024sz #av #fa initialized;
  let result = Dot.dot_float weights values 1024sz (row *^ 1024sz);
  rewrite A.pts_to values #fa (Memory.values av 0 1024 initialized)
    as A.pts_to (Memory.view activation 0 1024) #fa (Memory.values av 0 1024 initialized);
  Memory.restore activation 0 1024 #av #(Memory.values av 0 1024 initialized) #fa;
  Memory.restore_readonly av 0 1024 initialized;
  rewrite A.pts_to_mask activation #fa
    (Memory.replace av 0 (Memory.values av 0 1024 initialized)) Memory.all_cells
    as A.pts_to_mask activation #fa av Memory.all_cells;
  result
}
