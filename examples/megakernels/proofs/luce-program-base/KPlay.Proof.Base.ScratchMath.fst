module KPlay.Proof.Base.ScratchMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module C = Kuiper.Chest
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Transfer = KPlay.Proof.Base.Transfer
module Hidden = KPlay.Proof.Base.WarpHidden
module Native = KPlay.Proof.Base.NativeProjection
module Layout = KPlay.Proof.Base.MatvecLayout
module Matmul = KPlay.Proof.Base.Matmul
module Writes = KPlay.Proof.Base.MatvecWrites
module RMS = KPlay.Proof.Base.RMS
module RMSMath = KPlay.Proof.Base.RMSMath
module Scalar = KPlay.Proof.Base.Scalar

inline_for_extraction noextract
let represents (#n:pos) (values:Transfer.scratch) (offset:nat{offset + n <= 16000})
  (logical:P.vector real n) : prop =
  forall (i:P.index n). Approx.v_approximates (Seq.index values (offset + i)) (logical i)

inline_for_extraction noextract
let outside (before after:Transfer.scratch) (offset count:nat{offset + count <= 16000}) : prop =
  forall (i:P.index 16000).
    ~(offset <= i /\ i < offset + count) ==> Seq.index after i == Seq.index before i

inline_for_extraction noextract
let frame_vector (#n:pos) (scope:prop) (before after:Transfer.scratch)
  (written count:nat{written + count <= 16000}) (offset:nat{offset + n <= 16000})
  (logical:P.vector real n)
  : Lemma
      (requires outside before after written count /\
        (offset + n <= written \/ written + count <= offset) /\
        (scope ==> represents before offset logical))
      (ensures scope ==> represents after offset logical) = ()

inline_for_extraction noextract
let mma_frame (#length:pos{length <= 752393024}) (g:Layout.geometry length)
  (weights:C.chest2 BF.t length 1) (before:Hidden.snapshot)
  : Lemma (outside (Native.hidden_values before)
      (Native.hidden_values (Writes.after g weights before (SZ.v g.Layout.rows)))
      (SZ.v g.Layout.output_offset) (SZ.v g.Layout.rows)) =
  FStar.Classical.forall_intro (Writes.result_cell g weights before)

inline_for_extraction noextract
let mma_approx (#length:pos{length <= 752393024}) (scope:prop) (g:Layout.geometry length)
  (weights:C.chest2 BF.t length 1) (before:Hidden.snapshot)
  (wr:P.matrix real (SZ.v g.Layout.rows) (SZ.v g.Layout.columns))
  (xr:P.vector real (SZ.v g.Layout.columns))
  : Lemma
      (requires scope ==> O.matrix_approximates (Layout.matrix g weights) wr /\
        represents (Native.hidden_values before) (SZ.v g.Layout.input_offset) xr)
      (ensures scope ==> represents
        (Native.hidden_values (Writes.after g weights before (SZ.v g.Layout.rows)))
        (SZ.v g.Layout.output_offset) (Matmul.project wr xr)) =
  let under (_:scope)
    : Lemma (represents
        (Native.hidden_values (Writes.after g weights before (SZ.v g.Layout.rows)))
        (SZ.v g.Layout.output_offset) (Matmul.project wr xr)) =
    let at_index (i:P.index (SZ.v g.Layout.rows))
      : Lemma (Approx.v_approximates
          (Seq.index (Native.hidden_values (Writes.after g weights before (SZ.v g.Layout.rows)))
            (SZ.v g.Layout.output_offset + i))
          (Matmul.project wr xr i)) =
      Writes.result_cell g weights before (SZ.v g.Layout.output_offset + i);
      Matmul.output_approx (Layout.matrix g weights) (Layout.vector g before) wr xr (SZ.v g.Layout.splits) i
    in
    FStar.Classical.forall_intro at_index
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let rms_frame (g:RMS.layout) (weights:Transfer.packed) (before:Transfer.scratch)
  : Lemma (outside before (RMS.after g before weights (SZ.v g.RMS.rows))
      (SZ.v g.RMS.destination) (SZ.v g.RMS.rows * SZ.v g.RMS.width)) =
  let at_index (i:P.index 16000)
    : Lemma (~(SZ.v g.RMS.destination <= i /\
        i < SZ.v g.RMS.destination + SZ.v g.RMS.rows * SZ.v g.RMS.width) ==>
        Seq.index (RMS.after g before weights (SZ.v g.RMS.rows)) i == Seq.index before i) =
    if i < SZ.v g.RMS.destination || SZ.v g.RMS.destination + SZ.v g.RMS.rows * SZ.v g.RMS.width <= i
    then RMSMath.outside_frame g before weights (SZ.v g.RMS.rows) i
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let rms_approx (scope:prop) (g:RMS.layout) (weights:Transfer.packed) (before:Transfer.scratch)
  (row:P.index (SZ.v g.RMS.rows))
  (xr wr:P.vector real (SZ.v g.RMS.width))
  : Lemma
      (requires scope ==>
        represents before (SZ.v g.RMS.source + row * SZ.v g.RMS.stride) xr /\
        O.vector_approximates
          (RMSMath.weight_values (SZ.v g.RMS.width) weights (SZ.v g.RMS.weight_offset)) wr)
      (ensures scope ==> represents
        (RMS.after g before weights (SZ.v g.RMS.rows))
        (SZ.v g.RMS.destination + row * SZ.v g.RMS.width)
        (P.normalized_for_logits P.reference_epsilon xr wr)) =
  let under (_:scope)
    : Lemma (represents (RMS.after g before weights (SZ.v g.RMS.rows))
        (SZ.v g.RMS.destination + row * SZ.v g.RMS.width)
        (P.normalized_for_logits P.reference_epsilon xr wr)) =
    FStar.Classical.forall_intro (RMSMath.complete_row_approx g before weights row xr wr)
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let scalar_frame (op:Scalar.operation) (before:Transfer.scratch)
  : Lemma (outside before (Scalar.writes op before (Scalar.width op))
      (Scalar.destination op) (Scalar.width op)) =
  Scalar.bounds op;
  FStar.Classical.forall_intro (Scalar.outside_frame op before)

inline_for_extraction noextract
let scalar_approx (scope:prop) (op:Scalar.operation) (before:Transfer.scratch)
  (left right:P.vector real (Scalar.width op))
  : Lemma
      (requires scope ==>
        represents before (Scalar.left_offset op) left /\
        represents before (Scalar.right_offset op) right)
      (ensures scope ==> represents (Scalar.writes op before (Scalar.width op)) (Scalar.destination op)
        (fun i -> Scalar.logical op (left i) (right i))) =
  Scalar.bounds op;
  let under (_:scope)
    : Lemma (represents (Scalar.writes op before (Scalar.width op)) (Scalar.destination op)
        (fun i -> Scalar.logical op (left i) (right i))) =
    let at_index (i:P.index (Scalar.width op))
      : Lemma (Approx.v_approximates
          (Seq.index (Scalar.writes op before (Scalar.width op)) (Scalar.destination op + i))
          (Scalar.logical op (left i) (right i))) =
      Scalar.value_approx op (Seq.index before (Scalar.left_offset op + i))
        (Seq.index before (Scalar.right_offset op + i)) (left i) (right i)
    in
    FStar.Classical.forall_intro at_index
  in
  FStar.Classical.impl_intro under
