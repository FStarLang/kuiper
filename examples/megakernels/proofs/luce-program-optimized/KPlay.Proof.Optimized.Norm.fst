module KPlay.Proof.Optimized.Norm

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 50"

open Kuiper
open KPlay.Proof.Optimized.Sum
open KPlay.Proof.Optimized.NormMath
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module C = FStar.Classical
module Masks = KPlay.Proof.Optimized.Dot

noextract
fn square_warp (hidden:larray bf16 1024) (lanes:larray f32 32) (warp:sz{warp < 16})
  (#before:erased (lseq bf16 1024)) (#old:erased (lseq (option f32) 32)) (#fraction:perm)
  preserves gpu ** hidden |-> Frac fraction before
  requires A.pts_to_mask lanes old Masks.all_cells
  returns result:f32
  ensures (exists* (after:lseq (option f32) 32).
    A.pts_to_mask lanes after Masks.all_cells) **
    pure (forall (model:lseq real 1024). before %~ model ==>
      result %~ warp_square_sum model (SZ.v warp))
{
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i:sz{i <= 32}) (pv:lseq (option f32) 32).
      lane |-> i ** A.pts_to_mask lanes pv Masks.all_cells **
      pure (Masks.initialized pv (SZ.v i)) **
      pure (forall (model:lseq real 1024). before %~ model ==>
        (forall (j:nat{j < SZ.v i}).
          Masks.option_models (S.index pv j) (square_pair model (SZ.v warp) j)))
    decreases (32 - !lane)
  {
    let i = !lane;
    let a = Cast.cast_bf16_to_f32 (hidden.(warp *^ 32sz +^ i));
    let b = Cast.cast_bf16_to_f32 (hidden.(warp *^ 32sz +^ i +^ 512sz));
    let a_squared = FP.fma_rn_ftz a a (zero #f32);
    lift_fma (fun (model:lseq real 1024) -> before %~ model)
      (fun model -> S.index model (SZ.v warp * 32 + SZ.v i))
      (fun model -> S.index model (SZ.v warp * 32 + SZ.v i))
      (fun _ -> 0.0R) a a (zero #f32) a_squared;
    let pair = FP.fma_rn_ftz b b a_squared;
    lift_fma (fun (model:lseq real 1024) -> before %~ model)
      (fun model -> S.index model (SZ.v warp * 32 + SZ.v i + 512))
      (fun model -> S.index model (SZ.v warp * 32 + SZ.v i + 512))
      (fun model -> square_cell model (SZ.v warp * 32 + SZ.v i))
      b b a_squared pair;
    assert pure (forall (model:lseq real 1024).
      before %~ model ==> pair %~ square_pair model (SZ.v warp) (SZ.v i));
    A.mask_write lanes i pair;
    lane := i +^ 1sz;
  };
  A.from_mask lanes;
  with values. assert lanes |-> values;
  let result = sum32 lanes;
  lift_sum32 (fun (model:lseq real 1024) -> before %~ model)
    (fun model -> square_pair model (SZ.v warp)) values result;
  A.to_mask lanes;
  result
}

noextract
fn hidden_rstd (hidden:larray bf16 1024)
  (#before:erased (lseq bf16 1024)) (#fraction:perm)
  preserves gpu ** hidden |-> Frac fraction before
  returns result:f32
  ensures pure (forall (model:lseq real 1024).
    before %~ model ==> result %~ (scale model <: real))
{
  let mut warps:A.array f32 = [| 32sz |];
  let mut lanes:A.array f32 = [| 32sz |];
  let mut warp = 0sz;
  while (!warp <^ 16sz)
    invariant exists* (w:sz{w <= 16}) (pv:lseq (option f32) 32).
      warp |-> w ** A.pts_to_mask warps pv Masks.all_cells **
      pure (Masks.initialized pv (SZ.v w)) **
      pure (forall (model:lseq real 1024). before %~ model ==>
        (forall (j:nat{j < SZ.v w}).
          Masks.option_models (S.index pv j) (warp_square_sum model j)))
    invariant exists* (lv:lseq (option f32) 32). A.pts_to_mask lanes lv Masks.all_cells
    decreases (16 - !warp)
  {
    let w = !warp;
    let value = square_warp hidden lanes w;
    A.mask_write warps w value;
    warp := w +^ 1sz;
  };
  while (!warp <^ 32sz)
    invariant exists* (w:sz{16 <= w /\ w <= 32}) (pv:lseq (option f32) 32).
      warp |-> w ** A.pts_to_mask warps pv Masks.all_cells **
      pure (Masks.initialized pv (SZ.v w)) **
      pure (forall (model:lseq real 1024). before %~ model ==>
        (forall (j:nat{j < SZ.v w}).
          Masks.option_models (S.index pv j) (warp_square_sum model j)))
    decreases (32 - !warp)
  {
    let w = !warp;
    A.mask_write warps w (zero #f32);
    warp := w +^ 1sz;
  };
  A.from_mask warps;
  with values. assert warps |-> values;
  let squared_total = sum32 warps;
  lift_sum32 (fun (model:lseq real 1024) -> before %~ model)
    warp_square_sum values squared_total;
  C.forall_intro all_squares;
  inverse_width_approx 1024;
  epsilon_approx ();
  let argument = FP.fma_rn_ftz squared_total (inverse_width 1024) epsilon;
  lift_fma (fun (model:lseq real 1024) -> before %~ model)
    (fun model -> (sum_squares model 1024 <: real))
    (fun _ -> 1.0R /. 1024.0R) (fun _ -> 0.000001R)
    squared_total (inverse_width 1024) epsilon argument;
  let result = FP.rsqrt_approx_ftz argument;
  rms_finish before argument result;
  A.to_mask warps;
  result
}

inline_for_extraction noextract
let interpretation (nw:nat) = lseq real nw & lseq real 1024

inline_for_extraction noextract
let inputs (#nw:nat) (weights:lseq bf16 nw) (hidden:lseq bf16 1024)
  (model:interpretation nw) : prop =
  weights %~ fst model /\ hidden %~ snd model

inline_for_extraction noextract
let normalized (#nw:nat) (model:interpretation nw)
  (offset:nat{offset + 1024 <= nw}) : GTot (lseq real 1024) =
  S.init_ghost 1024 (fun i ->
    (S.index (snd model) i *. scale (snd model)) *. (1.0R +. S.index (fst model) (offset + i)))

noextract
fn normalize_hidden (#nw:erased pos{nw <= 752393024})
  (weights:larray bf16 nw) (hidden:larray bf16 1024) (output:larray bf16 1024)
  (offset:sz{SZ.v offset + 1024 <= nw})
  (#wv:erased (lseq bf16 nw)) (#hv:erased (lseq bf16 1024))
  (#ov:erased (lseq (option bf16) 1024)) (#fw #fh:perm)
  preserves gpu ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires A.pts_to_mask output ov Masks.all_cells
  ensures exists* (after:lseq bf16 1024).
    output |-> after **
    pure (forall (model:interpretation nw). inputs wv hv model ==>
      after %~ normalized model (SZ.v offset))
{
  let rstd = hidden_rstd hidden;
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i:sz{i <= 1024}) (pv:lseq (option bf16) 1024).
      index |-> i ** A.pts_to_mask output pv Masks.all_cells **
      pure (forall (j:nat{j < SZ.v i}). Some? (S.index pv j)) **
      pure (forall (model:interpretation nw). inputs wv hv model ==>
        (forall (j:nat{j < SZ.v i}).
          match S.index pv j with
          | None -> False
          | Some v -> v %~ S.index (normalized model (SZ.v offset)) j))
    decreases (1024 - !index)
  {
    let i = !index;
    let x = Cast.cast_bf16_to_f32 (hidden.(i));
    let w = Cast.cast_bf16_to_f32 (weights.(offset +^ i));
    let scaled = FP.mul_rn_ftz x rstd;
    lift_mul #(interpretation nw) (inputs wv hv)
      (fun model -> S.index (snd model) (SZ.v i))
      (fun model -> (scale (snd model) <: real)) x rstd scaled;
    let gain = FP.add_rn_ftz (one #f32) w;
    lift_add #(interpretation nw) (inputs wv hv)
      (fun _ -> 1.0R) (fun model -> S.index (fst model) (SZ.v offset + SZ.v i))
      (one #f32) w gain;
    let result = FP.mul_rn_ftz scaled gain;
    lift_mul #(interpretation nw) (inputs wv hv)
      (fun model -> S.index (snd model) (SZ.v i) *. scale (snd model))
      (fun model -> 1.0R +. S.index (fst model) (SZ.v offset + SZ.v i))
      scaled gain result;
    let stored = Cast.cast_f32_to_bf16 result;
    assert pure (forall (model:interpretation nw). inputs wv hv model ==>
      stored %~ S.index (normalized model (SZ.v offset)) (SZ.v i));
    A.mask_write output i stored;
    index := i +^ 1sz;
  };
  A.from_mask output;
}
