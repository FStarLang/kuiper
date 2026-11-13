module KPlay.Proof.Base.AffineMMA

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided

module C = Kuiper.Chest
module RO = Kuiper.TensorRO
module L = Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT
module M = Kuiper.TensorCore.MMA
module V = Kuiper.TensorCore.MMA.Views
module Trade = Pulse.Lib.Trade
module Native = KPlay.Proof.Base.NativeProjection

inline_for_extraction noextract
let parent_layout (length:pos) = L.l2_row_major length 1

inline_for_extraction noextract
let fits (length rows columns:pos) (offset row_stride col_stride:nat) : prop =
  forall (i:nat{i < rows}) (j:nat{j < columns}).
    offset + i * row_stride + j * col_stride < length

inline_for_extraction noextract
let a_index (#length #rows #columns:pos) (offset row_stride col_stride:nat)
  (bound:squash (fits length rows columns offset row_stride col_stride))
  (index:abs (rows @| columns @| INil)) : abs (length @| 1 @| INil) =
  let i, (j, ()) = index in
  (offset + i * row_stride + j * col_stride, (0, ()))

inline_for_extraction noextract
let h_index (#columns:pos) (offset:nat{offset + columns <= 16000})
  (index:abs (columns @| 8 @| INil)) : abs (16000 @| 1 @| INil) =
  let i, (_, ()) = index in
  (offset + i, (0, ()))

inline_for_extraction noextract
let a_values (#length:pos) (rows columns:pos) (values:chest2 bf16 length 1)
  (offset row_stride col_stride:nat)
  (bound:squash (fits length rows columns offset row_stride col_stride))
  : chest2 bf16 rows columns =
  mk2 (fun i j -> acc2 values (offset + i * row_stride + j * col_stride) 0)

inline_for_extraction noextract
let h_values (columns:pos) (values:chest2 bf16 16000 1)
  (offset:nat{offset + columns <= 16000}) : chest2 bf16 columns 8 =
  mk2 (fun i _ -> acc2 values (offset + i) 0)

inline_for_extraction noextract
fn instruction
  (#length:pos)
  (rows columns:sz{0 < SZ.v rows /\ SZ.v rows <= 16 /\ 0 < SZ.v columns /\ SZ.v columns <= 16})
  (parent:array2 bf16 (parent_layout length)) (hidden:array2 bf16 Native.hidden_layout)
  (offset row_stride col_stride:sz)
  (input_offset:sz{SZ.v input_offset + SZ.v columns <= 16000})
  (fragment:M.fragment)
  (bound:squash (fits length (SZ.v rows) (SZ.v columns)
    (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride)))
  (#vp:chest2 bf16 length 1) (#vh:chest2 bf16 16000 1) (#vc:chest2 float 16 8)
  (#parent_fraction #hidden_fraction:perm)
  preserves gpu ** parent |-> Frac parent_fraction vp ** hidden |-> Frac hidden_fraction vh
  requires fragment |-> vc
  ensures fragment |-> M.emma vc
    (V.pad_a (a_values (SZ.v rows) (SZ.v columns) vp
      (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride) bound))
    (V.pad_b (h_values (SZ.v columns) vh (SZ.v input_offset)))
{
  let pr = RO.tensor_to_rotensor parent;
  let hr = RO.tensor_to_rotensor hidden;
  RO.tensor_borrow_reindex pr (SZ.v rows @| SZ.v columns @| INil)
    (a_index #length #(SZ.v rows) #(SZ.v columns)
      (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride) bound);
  RO.tensor_borrow_reindex hr (SZ.v columns @| 8 @| INil)
    (h_index #(SZ.v columns) (SZ.v input_offset));
  let a = RO.from_array
    (RO.vtlayout_reindex (SZ.v rows @| SZ.v columns @| INil)
      (RO.vtlayout_of_tlayout (parent_layout length))
      (a_index #length #(SZ.v rows) #(SZ.v columns)
        (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride) bound)) (RO.core pr);
  let b = RO.from_array
    (RO.vtlayout_reindex (SZ.v columns @| 8 @| INil)
      (RO.vtlayout_of_tlayout Native.hidden_layout)
      (h_index #(SZ.v columns) (SZ.v input_offset))) (RO.core hr);
  assert rewrites_to a (RO.from_array
    (RO.vtlayout_reindex (SZ.v rows @| SZ.v columns @| INil)
      (RO.vtlayout_of_tlayout (parent_layout length))
      (a_index #length #(SZ.v rows) #(SZ.v columns)
        (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride) bound)) (RO.core pr));
  assert rewrites_to b (RO.from_array
    (RO.vtlayout_reindex (SZ.v columns @| 8 @| INil)
      (RO.vtlayout_of_tlayout Native.hidden_layout)
      (h_index #(SZ.v columns) (SZ.v input_offset))) (RO.core hr));
  C.ext (mk (SZ.v rows @| SZ.v columns @| INil)
      (fun index -> acc vp (a_index #length #(SZ.v rows) #(SZ.v columns)
        (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride) bound index)))
    (a_values (SZ.v rows) (SZ.v columns) vp (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride) bound);
  C.ext (mk (SZ.v columns @| 8 @| INil)
      (fun index -> acc vh (h_index #(SZ.v columns) (SZ.v input_offset) index)))
    (h_values (SZ.v columns) vh (SZ.v input_offset));
  V.mma_sync rows columns 8sz a offset row_stride col_stride
    b input_offset 1sz 0sz fragment ();
  Trade.elim_trade
    (a |-> Frac (parent_fraction /. 2)
      (a_values (SZ.v rows) (SZ.v columns) vp (SZ.v offset) (SZ.v row_stride) (SZ.v col_stride) bound))
    (pr |-> Frac parent_fraction vp);
  Trade.elim_trade
    (b |-> Frac (hidden_fraction /. 2) (h_values (SZ.v columns) vh (SZ.v input_offset)))
    (hr |-> Frac hidden_fraction vh);
  RO.rotensor_to_tensor parent pr;
  RO.rotensor_to_tensor hidden hr;
}
