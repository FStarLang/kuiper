module KPlay.Proof.Base.NativeProjection

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided

module C = Kuiper.Chest
module S = FStar.Seq
module SZ = Kuiper.SizeT
module RO = Kuiper.TensorRO
module M = Kuiper.TensorCore.MMA
module V = Kuiper.TensorCore.MMA.Views
module Trade = Pulse.Lib.Trade
module Support = KPlay.Support.Luce.MMA
module Transfer = KPlay.Proof.Base.Transfer
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Cast = Kuiper.Float.Casts.Base

let weight_layout = Support.weight_layout
let hidden_layout = Support.hidden_layout

inline_for_extraction noextract
let packed_values (v:chest2 bf16 752393024 1) : GTot Transfer.packed =
  S.init_ghost 752393024 (fun i -> acc2 v i 0)

inline_for_extraction noextract
let hidden_values (v:chest2 bf16 16000 1) : GTot Transfer.scratch =
  S.init_ghost 16000 (fun i -> acc2 v i 0)

let observation_cells
  (vw:chest2 bf16 752393024 1) (vh:chest2 bf16 16000 1)
  (i:P.index 752393024) (j:P.index 16000)
  : Lemma (
    S.index (packed_values vw) i == acc2 vw i 0 /\
    S.index (hidden_values vh) j == acc2 vh j 0) = ()

inline_for_extraction noextract
let weight_index (row:Transfer.row_block) (k:Transfer.depth_block)
  (idx:abs (16 @| 16 @| INil)) : abs (752393024 @| 1 @| INil) =
  let i, (j, ()) = idx in ((row + i) * 1024 + k + j, (0, ()))

inline_for_extraction noextract
let hidden_index (k:Transfer.depth_block) (idx:abs (16 @| 8 @| INil))
  : abs (16000 @| 1 @| INil) =
  let i, (_, ()) = idx in (1024 + k + i, (0, ()))

(* kernel.cu:326-364 with final=true. Both recovery trades retain the
   physical cells outside the selected tile; B has column stride zero. *)
inline_for_extraction noextract
fn final_projection_step
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (k:sz{SZ.v k % 16 == 0 /\ SZ.v k + 16 <= 1024})
  (weight:array2 bf16 weight_layout) (hidden:array2 bf16 hidden_layout)
  (part:M.fragment)
  (#vw:chest2 bf16 752393024 1) (#vh:chest2 bf16 16000 1)
  (#vc:chest2 float 16 8)
  (#weight_fraction:perm)
  preserves gpu
  preserves weight |-> Frac weight_fraction vw
  preserves hidden |-> Frac (1.0R /. 32) vh
  requires part |-> vc
  ensures part |-> Transfer.instruction (packed_values vw) (hidden_values vh)
    (SZ.v row) (SZ.v k) vc
{
  let wr = RO.tensor_to_rotensor weight;
  let hr = RO.tensor_to_rotensor hidden;
  RO.tensor_borrow_reindex wr (16 @| 16 @| INil) (weight_index (SZ.v row) (SZ.v k));
  RO.tensor_borrow_reindex hr (16 @| 8 @| INil) (hidden_index (SZ.v k));
  let a = RO.from_array
    (RO.vtlayout_reindex (16 @| 16 @| INil)
      (RO.vtlayout_of_tlayout weight_layout) (weight_index (SZ.v row) (SZ.v k)))
    (RO.core wr);
  let b = RO.from_array
    (RO.vtlayout_reindex (16 @| 8 @| INil)
      (RO.vtlayout_of_tlayout hidden_layout) (hidden_index (SZ.v k))) (RO.core hr);
  assert rewrites_to a (RO.from_array
    (RO.vtlayout_reindex (16 @| 16 @| INil)
      (RO.vtlayout_of_tlayout weight_layout) (weight_index (SZ.v row) (SZ.v k)))
    (RO.core wr));
  assert rewrites_to b (RO.from_array
    (RO.vtlayout_reindex (16 @| 8 @| INil)
      (RO.vtlayout_of_tlayout hidden_layout) (hidden_index (SZ.v k))) (RO.core hr));
  C.ext (mk (16 @| 16 @| INil)
      (fun i -> acc vw (weight_index (SZ.v row) (SZ.v k) i)))
    (Transfer.weight_fragment (packed_values vw) (SZ.v row) (SZ.v k));
  C.ext (mk (16 @| 8 @| INil) (fun i -> acc vh (hidden_index (SZ.v k) i)))
    (Transfer.input_fragment (hidden_values vh) (SZ.v k));
  V.mma_sync 16sz 16sz 8sz a (SZ.add (SZ.mul row 1024sz) k) 1024sz 1sz
    b (SZ.add 1024sz k) 1sz 0sz part ();
  C.ext (V.pad_a (Transfer.weight_fragment (packed_values vw) (SZ.v row) (SZ.v k)))
    (Transfer.weight_fragment (packed_values vw) (SZ.v row) (SZ.v k));
  C.ext (V.pad_b (Transfer.input_fragment (hidden_values vh) (SZ.v k)))
    (Transfer.input_fragment (hidden_values vh) (SZ.v k));
  Trade.elim_trade
    (a |-> Frac (weight_fraction /. 2)
      (Transfer.weight_fragment (packed_values vw) (SZ.v row) (SZ.v k)))
    (wr |-> Frac weight_fraction vw);
  Trade.elim_trade
    (b |-> Frac (1.0R /. 64) (Transfer.input_fragment (hidden_values vh) (SZ.v k)))
    (hr |-> Frac (1.0R /. 32) vh);
  RO.rotensor_to_tensor weight wr;
  RO.rotensor_to_tensor hidden hr;
}

inline_for_extraction noextract
fn final_projection_block
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (weight:array2 bf16 weight_layout) (hidden:array2 bf16 hidden_layout)
  (accum part:M.fragment)
  (#vw:chest2 bf16 752393024 1) (#vh:chest2 bf16 16000 1)
  (#initial #vc:chest2 float 16 8)
  (#weight_fraction:perm)
  preserves gpu
  preserves weight |-> Frac weight_fraction vw
  preserves hidden |-> Frac (1.0R /. 32) vh
  requires accum |-> initial ** part |-> vc
  ensures accum |-> V.filled 5 initial (zero #float)
  ensures part |-> Transfer.block_accumulator
    (packed_values vw) (hidden_values vh) (SZ.v row) 64
{
  V.fill_masked accum 5sz (zero #float);
  M.fill part (zero #float);
  C.ext (const (16 @| 8 @| INil) (zero #float))
    (Transfer.block_accumulator (packed_values vw) (hidden_values vh) (SZ.v row) 0);
  let mut k = 0sz;
  while (!k <^ 1024sz)
    invariant exists* (current:sz{SZ.v current <= 1024 /\ SZ.v current % 16 == 0}).
      k |-> current **
      part |-> Transfer.block_accumulator
        (packed_values vw) (hidden_values vh) (SZ.v row) (SZ.v current / 16)
    invariant gpu **
      weight |-> Frac weight_fraction vw **
      hidden |-> Frac (1.0R /. 32) vh
    decreases (1024 - SZ.v !k)
  {
    let current = !k;
    final_projection_step row current weight hidden part;
    assert pure (
      Transfer.instruction (packed_values vw) (hidden_values vh) (SZ.v row)
        (SZ.v current)
        (Transfer.block_accumulator (packed_values vw) (hidden_values vh)
          (SZ.v row) (SZ.v current / 16)) ==
      Transfer.block_accumulator (packed_values vw) (hidden_values vh)
        (SZ.v row) ((SZ.v current + 16) / 16));
    k := SZ.add current 16sz;
  }
}

let score_layout : layout2 16 1 = {
  ulen = 16000;
  imap = Kuiper.Injection.mk_injection #(abs (16 @| 1 @| INil)) #(natlt 16000)
    (fun (idx:abs (16 @| 1 @| INil)) -> let i, (_, ()) = idx in 12500 + i)
    (fun x y -> ());
}

inline_for_extraction noextract
instance score_column : strided_col_major score_layout = {
  offset = 12500sz;
  stride = 16sz;
  pf = (fun i j -> ());
}

inline_for_extraction noextract
let score_view (hidden:array2 bf16 hidden_layout) : array2 bf16 score_layout =
  from_array score_layout (core hidden)

inline_for_extraction noextract
let stored_column (produced:chest2 float 16 8) : chest2 bf16 16 1 =
  mk2 (fun i _ -> S.index (Transfer.stored_scores produced) i)

inline_for_extraction noextract
let consumer_values (stored:chest2 bf16 16 1) : GTot (D.vec float 16) =
  S.init_ghost 16 (fun i -> Cast.cast_bf16_to_f32 (acc2 stored i 0))

let consumer_binding (produced:chest2 float 16 8)
  : Lemma (
    consumer_values (stored_column produced) == Transfer.consumed_scores produced) =
  assert (S.equal (consumer_values (stored_column produced)) (Transfer.consumed_scores produced));
  S.lemma_eq_elim (consumer_values (stored_column produced)) (Transfer.consumed_scores produced)

let source_score_addresses (hidden:array2 bf16 hidden_layout) (i:P.index 16)
  : Lemma (
    core (score_view hidden) == core hidden /\
    cell_of_pos score_layout i 0 == 12500 + i /\
    V.selected 5 i 0) = ()

let masked_store_matches (initial produced:chest2 float 16 8)
  : Lemma (
    V.rounded_window 0 0 16 1
      (V.added 5 (V.filled 5 initial (zero #float)) produced) ==
    stored_column produced) =
  C.ext
    (V.rounded_window 0 0 16 1
      (V.added 5 (V.filled 5 initial (zero #float)) produced))
    (stored_column produced)

(* The output resource is the source's own sixteen-cell window, not an
   additional allocation. Its extraction from live scratch is a caller goal. *)
inline_for_extraction noextract
fn publish_block
  (hidden:array2 bf16 hidden_layout) (accum part:M.fragment)
  (#initial #produced:chest2 float 16 8) (#old:chest2 bf16 16 1)
  preserves gpu ** part |-> produced
  requires accum |-> V.filled 5 initial (zero #float)
  requires score_view hidden |-> Frac (1.0R /. 32) old
  ensures accum |-> V.added 5 (V.filled 5 initial (zero #float)) produced
  ensures score_view hidden |-> Frac (1.0R /. 32) (stored_column produced)
{
  masked_store_matches initial produced;
  assert pure (
    V.rounded_window 0 0 16 1
      (V.added 5 (V.filled 5 initial (zero #float)) produced) ==
    stored_column produced);
  V.add_masked accum part 5sz;
  V.store_bf16 accum 0sz 0sz 16sz 1sz (score_view hidden);
}
