module KPlay.Support.Luce.MMA

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as row_major }

module M = Kuiper.TensorCore.MMA
module V = Kuiper.TensorCore.MMA.Views
module RO = Kuiper.TensorRO
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
module Cast = Kuiper.Float.Casts.Base

(* Local bindings for kernel.cu:285-372, not a whole-program postcondition.
   The first recurrent projection uses base=254280704, weight_offset=
   base+11012096, input_offset=1024 and output_offset=2048. *)

let weight_layout = row_major 752393024 1
let hidden_layout = row_major 16000 1
let depth = k:nat{k <= 1008 /\ k % 16 == 0}

inline_for_extraction noextract
let weight_index (k:depth) (idx:abs (16 @| 16 @| INil))
  : abs (752393024 @| 1 @| INil) =
  let i, (j, ()) = idx in (265292800 + i * 1024 + k + j, (0, ()))

inline_for_extraction noextract
let hidden_index (k:depth) (idx:abs (16 @| 8 @| INil))
  : abs (16000 @| 1 @| INil) =
  let i, (_, ()) = idx in (1024 + k + i, (0, ()))

let weight_tile (k:depth) (v:chest2 bf16 752393024 1) : chest2 bf16 16 16 =
  mk2 #bf16 #16 #16 (fun i j -> acc2 v (265292800 + i * 1024 + k + j) 0)

let hidden_tile (k:depth) (v:chest2 bf16 16000 1) : chest2 bf16 16 8 =
  mk2 #bf16 #16 #8 (fun i _ -> acc2 v (1024 + k + i) 0)

let source_lane_loads (k:depth)
  (weight:chest2 bf16 752393024 1) (hidden:chest2 bf16 16000 1)
  (lane:natlt 32) (areg:natlt 4) (breg:natlt 2) (half:natlt 2)
  : Lemma (
    acc2 (weight_tile k weight) (lane / 4 + (areg % 2) * 8)
      ((lane % 4) * 2 + (areg / 2) * 8 + half) ==
    acc2 weight (265292800 + (lane / 4 + (areg % 2) * 8) * 1024 +
      k + (lane % 4) * 2 + (areg / 2) * 8 + half) 0 /\
    acc2 (hidden_tile k hidden) ((lane % 4) * 2 + breg * 8 + half) (lane / 4) ==
    acc2 hidden (1024 + k + (lane % 4) * 2 + breg * 8 + half) 0) = ()

inline_for_extraction noextract
fn projection_step (k:sz{SZ.v k <= 1008 /\ SZ.v k % 16 == 0})
  (weight:array2 bf16 weight_layout) (hidden:array2 bf16 hidden_layout)
  (fr:M.fragment)
  (#vw:chest2 bf16 752393024 1) (#vh:chest2 bf16 16000 1)
  (#vc:chest2 float 16 8)
  preserves gpu
  preserves weight |-> Frac (1.0R /. 32) vw
  preserves hidden |-> Frac (1.0R /. 32) vh
  requires fr |-> vc
  ensures fr |-> M.emma vc (weight_tile (SZ.v k) vw) (hidden_tile (SZ.v k) vh)
{
  let wr = RO.tensor_to_rotensor weight;
  let hr = RO.tensor_to_rotensor hidden;
  RO.tensor_borrow_reindex wr (16 @| 16 @| INil) (weight_index (SZ.v k));
  RO.tensor_borrow_reindex hr (16 @| 8 @| INil) (hidden_index (SZ.v k));
  let a = RO.from_array
    (RO.vtlayout_reindex (16 @| 16 @| INil)
      (RO.vtlayout_of_tlayout weight_layout) (weight_index (SZ.v k))) (RO.core wr);
  let b = RO.from_array
    (RO.vtlayout_reindex (16 @| 8 @| INil)
      (RO.vtlayout_of_tlayout hidden_layout) (hidden_index (SZ.v k))) (RO.core hr);
  assert rewrites_to a (RO.from_array
    (RO.vtlayout_reindex (16 @| 16 @| INil)
      (RO.vtlayout_of_tlayout weight_layout) (weight_index (SZ.v k))) (RO.core wr));
  assert rewrites_to b (RO.from_array
    (RO.vtlayout_reindex (16 @| 8 @| INil)
      (RO.vtlayout_of_tlayout hidden_layout) (hidden_index (SZ.v k))) (RO.core hr));
  Kuiper.Chest.ext
    (mk (16 @| 16 @| INil) (fun i -> acc vw (weight_index (SZ.v k) i)))
    (weight_tile (SZ.v k) vw);
  Kuiper.Chest.ext
    (mk (16 @| 8 @| INil) (fun i -> acc vh (hidden_index (SZ.v k) i)))
    (hidden_tile (SZ.v k) vh);
  V.mma_sync 16sz 16sz 8sz a (SZ.add (SZ.uint_to_t 265292800) k) 1024sz 1sz
    b (SZ.add 1024sz k) 1sz 0sz fr ();
  Kuiper.Chest.ext (V.pad_a (weight_tile (SZ.v k) vw)) (weight_tile (SZ.v k) vw);
  Kuiper.Chest.ext (V.pad_b (hidden_tile (SZ.v k) vh)) (hidden_tile (SZ.v k) vh);
  Trade.elim_trade
    (a |-> Frac (1.0R /. 64) (weight_tile (SZ.v k) vw))
    (wr |-> Frac (1.0R /. 32) vw);
  Trade.elim_trade
    (b |-> Frac (1.0R /. 64) (hidden_tile (SZ.v k) vh))
    (hr |-> Frac (1.0R /. 32) vh);
  RO.rotensor_to_tensor weight wr;
  RO.rotensor_to_tensor hidden hr;
}

let output_layout : layout2 16 1 = {
  ulen = 16000;
  imap = Kuiper.Injection.mk_injection #(abs (16 @| 1 @| INil)) #(natlt 16000)
    (fun (idx:abs (16 @| 1 @| INil)) -> let i, (_, ()) = idx in 2048 + i)
    (fun x y -> ());
}

inline_for_extraction noextract
instance output_column : strided_col_major output_layout = {
  offset = 2048sz;
  stride = 16sz;
  pf = (fun i j -> ());
}

inline_for_extraction noextract
fn finish_part (accum part:M.fragment) (hidden:array2 bf16 output_layout)
  (#va #vp:chest2 float 16 8) (#vh:chest2 bf16 16 1)
  preserves gpu ** part |-> vp
  requires accum |-> va ** hidden |-> Frac (1.0R /. 32) vh
  ensures accum |-> V.added 5 va vp
  ensures hidden |-> Frac (1.0R /. 32) (V.rounded_window 0 0 16 1 (V.added 5 va vp))
{
  V.add_masked accum part 5sz;
  V.store_bf16 accum 0sz 0sz 16sz 1sz hidden;
}

let source_store_values (accum part:chest2 float 16 8) (group:natlt 8)
  : Lemma (
    cell_of_pos output_layout group 0 == 2048 + group /\
    cell_of_pos output_layout (group + 8) 0 == 2048 + group + 8 /\
    acc2 (V.rounded_window 0 0 16 1 (V.added 5 accum part)) group 0 ==
      Cast.cast_f32_to_bf16 (Kuiper.Scalars.Base.add #float
        (acc2 accum group 0) (acc2 part group 0)) /\
    acc2 (V.rounded_window 0 0 16 1 (V.added 5 accum part)) (group + 8) 0 ==
      Cast.cast_f32_to_bf16 (Kuiper.Scalars.Base.add #float
        (acc2 accum (group + 8) 0) (acc2 part (group + 8) 0))) = ()

(* These are the two affine cache orientations in the same source. *)
let cache_score_address (layer:natlt 24) (head:natlt 8)
  (row:natlt 2048) (k:natlt 256) (i:nat{row+i < 2048}) (j:nat{k+j < 256})
  : Lemma (
    ((layer / 4) * 4096 + row + i) * 512 + (head / 4) * 256 + k + j ==
      ((layer / 4) * 4096 + row) * 512 + (head / 4) * 256 + k + i*512 + j) = ()

let cache_value_address (layer:natlt 24) (head:natlt 8)
  (row:natlt 256) (k:natlt 2048) (i:nat{row+i < 256}) (j:nat{k+j < 2048})
  : Lemma (
    ((layer / 4) * 4096 + 2048 + k + j) * 512 + (head / 4) * 256 + row + i ==
      ((layer / 4) * 4096 + 2048 + k) * 512 + (head / 4) * 256 + row + i + j*512) = ()
