module KPlay.Proof.Base.RMSMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Root = FStar.Math.Sqrt
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module Norm = KPlay.Proof.Base.Norm
module Native = KPlay.Proof.Base.RMSNative
module RMS = KPlay.Proof.Base.RMS
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let weight_values (width:Norm.width) (weights:Transfer.packed)
  (offset:nat{offset + width <= 752393024}) : P.vector BF.t width =
  fun i -> Seq.index weights (offset + i)

inline_for_extraction noextract
let row_value_approx (width:Norm.width) (before:Transfer.scratch) (weights:Transfer.packed)
  (source:nat{source + width <= 16000}) (weight_offset:nat{weight_offset + width <= 752393024})
  (xr wr:P.vector real width) (i:P.index width)
  : Lemma
      (requires O.vector_approximates (Native.input width before source) xr /\
        O.vector_approximates (weight_values width weights weight_offset) wr)
      (ensures Approx.v_approximates (Native.row_value width before weights source weight_offset i)
        (P.normalized_for_logits #width P.reference_epsilon xr wr i)) =
  Norm.source_scale_approx width (Native.input width before source) xr;
  let x = Seq.index before (source + i) in
  let w = Seq.index weights (weight_offset + i) in
  Cast.cast_bf16_to_f32_ok x (xr i);
  Cast.cast_bf16_to_f32_ok w (wr i);
  let scale = Norm.source_scale width (Native.input width before source) in
  let real_scale = Root.rsqrt (P.rms_argument #width P.reference_epsilon xr) in
  Approx.a_mul (Cast.cast_bf16_to_f32 x) scale (xr i) real_scale;
  Approx.a_add (F.one #FP.t) (Cast.cast_bf16_to_f32 w) 1.0R (wr i);
  Approx.a_mul (F.mul (Cast.cast_bf16_to_f32 x) scale)
    (F.add (F.one #FP.t) (Cast.cast_bf16_to_f32 w))
    (xr i *. real_scale) (1.0R +. wr i);
  Cast.cast_f32_to_bf16_ok
    (F.mul (F.mul (Cast.cast_bf16_to_f32 x) scale)
      (F.add (F.one #FP.t) (Cast.cast_bf16_to_f32 w)))
    ((xr i *. real_scale) *. (1.0R +. wr i))

inline_for_extraction noextract
let source_scale_frame (g:RMS.layout) (before:Transfer.scratch) (weights:Transfer.packed)
  (row:P.index (SZ.v g.RMS.rows))
  : Lemma (
      Norm.source_scale (SZ.v g.RMS.width)
        (Native.input (SZ.v g.RMS.width) (RMS.after g before weights row)
          (SZ.v g.RMS.source + row * SZ.v g.RMS.stride)) ==
      Norm.source_scale (SZ.v g.RMS.width)
        (Native.input (SZ.v g.RMS.width) before (SZ.v g.RMS.source + row * SZ.v g.RMS.stride))) =
  RMS.row_bounds g row;
  FStar.Classical.forall_intro (RMS.unread_frame g before weights row row);
  RMS.scale_equal (SZ.v g.RMS.width)
    (Native.input (SZ.v g.RMS.width) (RMS.after g before weights row)
      (SZ.v g.RMS.source + row * SZ.v g.RMS.stride))
    (Native.input (SZ.v g.RMS.width) before (SZ.v g.RMS.source + row * SZ.v g.RMS.stride))

inline_for_extraction noextract
let rec output_cell (g:RMS.layout) (before:Transfer.scratch) (weights:Transfer.packed)
  (count:pos{count <= SZ.v g.RMS.rows}) (row:nat{row < count}) (i:P.index (SZ.v g.RMS.width))
  : Lemma (
      Seq.index (RMS.after g before weights count) (SZ.v g.RMS.destination + row * SZ.v g.RMS.width + i) ==
      Native.row_value (SZ.v g.RMS.width) before weights
        (SZ.v g.RMS.source + row * SZ.v g.RMS.stride) (SZ.v g.RMS.weight_offset) i)
      (decreases count) =
  RMS.row_bounds g row;
  RMS.row_bounds g (count - 1);
  norm_spec [delta_once ["KPlay.Proof.Base.RMS.after"]] (RMS.after g before weights count);
  if row = count - 1 then (
    source_scale_frame g before weights row;
    RMS.unread_frame g before weights row row i
  ) else output_cell g before weights (count - 1) row i

inline_for_extraction noextract
let complete_row_approx (g:RMS.layout) (before:Transfer.scratch) (weights:Transfer.packed)
  (row:P.index (SZ.v g.RMS.rows))
  (xr wr:P.vector real (SZ.v g.RMS.width)) (i:P.index (SZ.v g.RMS.width))
  : Lemma
      (requires
        O.vector_approximates (Native.input (SZ.v g.RMS.width) before
          (SZ.v g.RMS.source + row * SZ.v g.RMS.stride)) xr /\
        O.vector_approximates (weight_values (SZ.v g.RMS.width) weights (SZ.v g.RMS.weight_offset)) wr)
      (ensures Approx.v_approximates
        (Seq.index (RMS.after g before weights (SZ.v g.RMS.rows))
          (SZ.v g.RMS.destination + row * SZ.v g.RMS.width + i))
        (P.normalized_for_logits #(SZ.v g.RMS.width) P.reference_epsilon xr wr i)) =
  RMS.row_bounds g row;
  output_cell g before weights (SZ.v g.RMS.rows) row i;
  row_value_approx (SZ.v g.RMS.width) before weights
    (SZ.v g.RMS.source + row * SZ.v g.RMS.stride) (SZ.v g.RMS.weight_offset) xr wr i

inline_for_extraction noextract
let rec outside_frame (g:RMS.layout) (before:Transfer.scratch) (weights:Transfer.packed)
  (count:nat{count <= SZ.v g.RMS.rows}) (i:P.index 16000)
  : Lemma
      (requires ~(SZ.v g.RMS.destination <= i /\
        i < SZ.v g.RMS.destination + count * SZ.v g.RMS.width))
      (ensures Seq.index (RMS.after g before weights count) i == Seq.index before i)
      (decreases count) =
  if count > 0 then (
    outside_frame g before weights (count - 1) i;
    norm_spec [delta_once ["KPlay.Proof.Base.RMS.after"]] (RMS.after g before weights count)
  )
