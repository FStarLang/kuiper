module KPlay.Proof.Base.NativeLayout

module P = KPlay.Proof.Shared.Projection
module Shape = Kuiper.Shape
module Layout = Kuiper.Tensor.Layout
module Stride = Kuiper.Array2.Strided
module RO = Kuiper.TensorRO
module SZ = Kuiper.SizeT
module C = Kuiper.Chest
module Seq = FStar.Seq
module Transfer = KPlay.Proof.Base.Transfer

(* kernel.cu:354-359 loads B without a group/column offset. *)
inline_for_extraction noextract
let source_b_address (k:Transfer.depth_block) (row:P.index 16) (_col:P.index 8)
  : P.index 16000 =
  1024 + k + row

inline_for_extraction noextract
let source_b_readonly (k:Transfer.depth_block) : RO.vlayout2 16 8 = {
  RO.ulen = 16000;
  RO.imap = (fun (index:Shape.abs
    (Shape.ICons 16 (Shape.ICons 8 Shape.INil))) ->
    let row, (col, ()) = index in source_b_address k row col);
}

inline_for_extraction noextract
let source_b_readonly_exact (k:Transfer.depth_block) (row:P.index 16) (col:P.index 8)
  : Lemma (
      (source_b_readonly k).RO.imap (row, (col, ())) ==
      source_b_address k row col) = ()

inline_for_extraction noextract
let source_lane_load (k:Transfer.depth_block) (lane:P.index 32)
  (reg:P.index 2) (half:P.index 2)
  : Lemma (
      source_b_address k ((lane % 4) * 2 + reg * 8 + half) (lane / 4) ==
      1024 + k + (lane % 4) * 2 + reg * 8 + half) = ()

inline_for_extraction noextract
let checked_probe_uses_source_cells
  (hidden:Transfer.scratch) (k:Transfer.depth_block) (row:P.index 16) (col:P.index 8)
  : Lemma (
      C.acc2 (Transfer.input_fragment hidden k) row col ==
      Seq.index hidden ((source_b_readonly k).RO.imap (row, (col, ())))) = ()

inline_for_extraction noextract
let source_b_not_injective (k:Transfer.depth_block)
  : Lemma (
      source_b_address k 0 0 == source_b_address k 0 1 /\
      (0, (0, ())) <> (0, (1, ()))) = ()

inline_for_extraction noextract
let layout_columns_distinct (#cols:nat{cols >= 2}) (layout:Layout.layout2 16 cols)
  : Lemma (
      Stride.cell_of_pos layout 0 0 <> Stride.cell_of_pos layout 0 1) =
  if Stride.cell_of_pos layout 0 0 = Stride.cell_of_pos layout 0 1 then
    layout.Layout.imap.is_inj (0, (0, ())) (0, (1, ()))

inline_for_extraction noextract
let no_source_mma_layout (k:Transfer.depth_block) (layout:Layout.layout2 16 8)
  : Lemma (
      ~(forall (row:P.index 16) (col:P.index 8).
        Stride.cell_of_pos layout row col == source_b_address k row col)) =
  layout_columns_distinct layout;
  source_b_not_injective k

inline_for_extraction noextract
let no_source_mma7_layout (k:Transfer.depth_block) (layout:Layout.layout2 16 7)
  : Lemma (
      ~(forall (row:P.index 16) (col:P.index 7).
        Stride.cell_of_pos layout row col == source_b_address k row col)) =
  layout_columns_distinct layout;
  source_b_not_injective k

inline_for_extraction noextract
let positive_stride_separates_columns
  (layout:Layout.layout2 16 8) (view:Stride.strided_col_major layout)
  : Lemma (
      Stride.cell_of_pos layout 0 1 ==
        Stride.cell_of_pos layout 0 0 + SZ.v view.Stride.stride /\
      SZ.v view.Stride.stride > 0) =
  let _ = view.Stride.pf 0 0 in
  let _ = view.Stride.pf 0 1 in
  ()

inline_for_extraction noextract
let readonly_borrow_does_not_make_source_injective
  (k:Transfer.depth_block) (layout:Layout.layout2 16 8)
  : Lemma (
      ~(forall (row:P.index 16) (col:P.index 8).
        (RO.vtlayout_of_tlayout layout).RO.imap (row, (col, ())) ==
        (source_b_readonly k).RO.imap (row, (col, ())))) =
  no_source_mma_layout k layout

inline_for_extraction noextract
let native_store7_stride
  (layout:Layout.layout2 16 7) (view:Stride.strided_col_major layout)
  : Lemma (SZ.v view.Stride.stride >= 16) =
  if SZ.v view.Stride.stride < 16 then (
    let _ = view.Stride.pf 0 1 in
    let _ = view.Stride.pf (SZ.v view.Stride.stride) 0 in
    assert (Stride.cell_of_pos layout 0 1 ==
      Stride.cell_of_pos layout (SZ.v view.Stride.stride) 0);
    layout.Layout.imap.is_inj (0, (1, ())) (SZ.v view.Stride.stride, (0, ()))
  )

(* store7_bf16 writes row0/column1; kernel.cu:369-372 writes only column0. *)
inline_for_extraction noextract
let native_store7_changes_source_frame
  (layout:Layout.layout2 16 7) (view:Stride.strided_col_major layout)
  : Lemma (
      Stride.cell_of_pos layout 0 1 >= SZ.v view.Stride.offset + 16) =
  native_store7_stride layout view;
  let _ = view.Stride.pf 0 1 in
  ()
