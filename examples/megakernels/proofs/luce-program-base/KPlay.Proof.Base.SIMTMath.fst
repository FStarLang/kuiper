module KPlay.Proof.Base.SIMTMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
open Kuiper
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module S = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Layout = KPlay.Proof.Base.MatvecLayout
module Math = KPlay.Proof.Base.AttentionAlgebra
module Sum = KPlay.Proof.Base.ReductionMath
module Reduction = KPlay.Proof.Base.ReductionNative
module Norm = KPlay.Proof.Base.Norm

type geometry (length:pos{length <= 12582912}) = g:Layout.geometry length{
  SZ.v g.Layout.rows <= 647 /\ SZ.v g.Layout.columns <= 647}

inline_for_extraction noextract
let matrix_at (#length:pos{length <= 12582912}) (g:geometry length)
  (logical:FStar.Ghost.erased (S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)))
  (row:S.index (SZ.v g.Layout.rows)) (col:S.index (SZ.v g.Layout.columns)) : GTot real =
  (FStar.Ghost.reveal #(S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)) logical) row col

inline_for_extraction noextract
let vector_at (#length:pos{length <= 12582912}) (g:geometry length)
  (logical:FStar.Ghost.erased (S.vector (SZ.v g.Layout.columns)))
  (col:S.index (SZ.v g.Layout.columns)) : GTot real =
  (FStar.Ghost.reveal #(S.vector (SZ.v g.Layout.columns)) logical) col

inline_for_extraction noextract
let address_bounds (#length:pos{length <= 12582912}) (g:geometry length)
  (row:S.index (SZ.v g.Layout.rows)) (col:S.index (SZ.v g.Layout.columns))
  : Lemma (
      row * SZ.v g.Layout.row_stride < length /\
      col * SZ.v g.Layout.col_stride < length /\
      SZ.v g.Layout.offset + row * SZ.v g.Layout.row_stride < length /\
      SZ.v g.Layout.offset + row * SZ.v g.Layout.row_stride + col * SZ.v g.Layout.col_stride < length) =
  let address = Layout.tile_offset g row col in
  assert (address == SZ.v g.Layout.offset + row * SZ.v g.Layout.row_stride + col * SZ.v g.Layout.col_stride)

inline_for_extraction noextract
let matrix_rep (#length:pos{length <= 12582912}) (g:geometry length) (actual:lseq BF.t length)
  (logical:S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)) : prop =
  forall (row:S.index (SZ.v g.Layout.rows)) (col:S.index (SZ.v g.Layout.columns)).
    Approx.v_approximates
      (Seq.index actual (SZ.v g.Layout.offset + row * SZ.v g.Layout.row_stride + col * SZ.v g.Layout.col_stride))
      (logical row col)

inline_for_extraction noextract
let input_rep (#length:pos{length <= 12582912}) (g:geometry length) (actual:lseq BF.t 16000)
  (logical:S.vector (SZ.v g.Layout.columns)) : prop =
  forall (col:S.index (SZ.v g.Layout.columns)).
    Approx.v_approximates (Seq.index actual (SZ.v g.Layout.input_offset + col)) (logical col)

inline_for_extraction noextract
let partial_rep (count:nat{count <= 32}) (actual:lseq FP.t 32) (logical:S.vector 256) : prop =
  forall (i:S.index count). Approx.v_approximates (Seq.index actual i) (logical i)

inline_for_extraction noextract
let chunk_origin (groups chunk:pos) (group:nat{group < groups}) (block:nat)
  : Lemma (Math.chunk_index groups chunk group (block * chunk) ==
      block * (groups * chunk) + group * chunk) = ()

inline_for_extraction noextract
let work_bound (n groups chunk:pos) (start:nat{start < n})
  : Lemma (
      start / (groups * chunk) < Math.slots n (groups * chunk) /\
      (start / (groups * chunk) + 1) * chunk <= Math.chunk_work n groups chunk) =
  Math.slots_bounds n (groups * chunk)

inline_for_extraction noextract
let chunk_step (n groups chunk:pos) (group:nat{group < groups}) (terms:S.vector n)
  (start:nat{start < n /\ start % (groups * chunk) == group * chunk})
  (index:nat{start <= index /\ index < start + chunk /\ index < n})
  : Lemma (
      Math.bounded_sum (start / (groups * chunk) * chunk + (index + 1) - start)
        (Math.chunk_terms n groups chunk group terms) ==
      Math.bounded_sum (start / (groups * chunk) * chunk + index - start)
        (Math.chunk_terms n groups chunk group terms) +. terms index) =
  work_bound n groups chunk start;
  Math.chunk_coordinate (Math.slots n (groups * chunk)) groups chunk group
    (start / (groups * chunk)) (index - start);
  Math.bounded_sum_step (start / (groups * chunk) * chunk + index - start)
    (Math.chunk_terms n groups chunk group terms)

inline_for_extraction noextract
let conditional_chunk_step (scope:prop) (n groups chunk:pos)
  (group:nat{group < groups}) (terms:S.vector n)
  (start:nat{start < n /\ start % (groups * chunk) == group * chunk})
  (index:nat{start <= index /\ index < start + chunk /\ index < n}) (result:FP.t)
  : Lemma
      (requires (start / (groups * chunk) + 1) * chunk <= Math.chunk_work n groups chunk /\
        (scope ==> Approx.v_approximates result
        (terms index +. Math.bounded_sum (start / (groups * chunk) * chunk + index - start)
          (Math.chunk_terms n groups chunk group terms))))
      (ensures scope ==> Approx.v_approximates result
        (Math.bounded_sum (start / (groups * chunk) * chunk + (index + 1) - start)
          (Math.chunk_terms n groups chunk group terms))) =
  chunk_step n groups chunk group terms start index

inline_for_extraction noextract
let chunk_monotonic (groups chunk:pos) (group:nat{group < groups})
  (left:nat) (right:nat{left <= right})
  : Lemma (Math.chunk_index groups chunk group left <= Math.chunk_index groups chunk group right) =
  assert (left / chunk <= right / chunk);
  if left / chunk = right / chunk then
    assert (left % chunk <= right % chunk)
  else (
    assert (left / chunk + 1 <= right / chunk);
    assert (chunk <= groups * chunk)
  )

inline_for_extraction noextract
let rec zero_interval (#n:nat) (terms:S.vector n)
  (first:nat{first <= n}) (finish:nat{first <= finish /\ finish <= n})
  : Lemma
      (requires forall (i:nat{first <= i /\ i < finish}). terms i == 0.0R)
      (ensures Math.bounded_sum finish terms == Math.bounded_sum first terms)
      (decreases finish - first) =
  if first < finish then (
    zero_interval terms first (finish - 1);
    Math.bounded_sum_step (finish - 1) terms
  )

inline_for_extraction noextract
let skipped_suffix (n groups chunk:pos) (group:nat{group < groups}) (terms:S.vector n)
  (first:nat{first <= Math.chunk_work n groups chunk})
  (finish:nat{first <= finish /\ finish <= Math.chunk_work n groups chunk})
  : Lemma
      (requires n <= Math.chunk_index groups chunk group first)
      (ensures Math.bounded_sum finish (Math.chunk_terms n groups chunk group terms) ==
        Math.bounded_sum first (Math.chunk_terms n groups chunk group terms)) =
  let at_index (i:nat{first <= i /\ i < finish})
    : Lemma (Math.chunk_terms n groups chunk group terms i == 0.0R) =
    chunk_monotonic groups chunk group first i
  in
  FStar.Classical.forall_intro at_index;
  zero_interval (Math.chunk_terms n groups chunk group terms) first finish

inline_for_extraction noextract
let group_finished (n groups chunk:pos) (group:nat{group < groups}) (terms:S.vector n)
  (block:nat{block <= Math.slots n (groups * chunk)})
  : Lemma
      (requires n <= block * (groups * chunk) + group * chunk)
      (ensures Math.bounded_sum (block * chunk) (Math.chunk_terms n groups chunk group terms) ==
        S.sum (Math.chunk_work n groups chunk) (Math.chunk_terms n groups chunk group terms)) =
  chunk_origin groups chunk group block;
  skipped_suffix n groups chunk group terms (block * chunk) (Math.chunk_work n groups chunk);
  Math.bounded_sum_full (Math.chunk_terms n groups chunk group terms)

inline_for_extraction noextract
let chunk_finished (n groups chunk:pos) (group:nat{group < groups}) (terms:S.vector n)
  (start:nat{start < n /\ start % (groups * chunk) = group * chunk})
  (finish:nat{start <= finish /\ finish <= start + chunk /\ finish <= n /\
    (finish = start + chunk \/ finish = n)})
  : Lemma (
      let block = start / (groups * chunk) in
      block < Math.slots n (groups * chunk) /\
      Math.bounded_sum ((block + 1) * chunk) (Math.chunk_terms n groups chunk group terms) ==
      Math.bounded_sum (block * chunk + finish - start) (Math.chunk_terms n groups chunk group terms)) =
  Math.slots_bounds n (groups * chunk);
  let block = start / (groups * chunk) in
  assert (block < Math.slots n (groups * chunk));
  if finish < start + chunk then (
    Math.chunk_coordinate (Math.slots n (groups * chunk)) groups chunk group block (finish - start);
    skipped_suffix n groups chunk group terms (block * chunk + finish - start) ((block + 1) * chunk)
  )

inline_for_extraction noextract
let conditional_fma (scope:prop) (left right:BF.t) (previous result:FP.t) (a b c:real)
  : Lemma
      (requires
        (scope ==> Approx.v_approximates left a /\ Approx.v_approximates right b /\
          Approx.v_approximates previous c) /\
        (forall (xr yr zr:real).
          Approx.v_approximates (Cast.cast_bf16_to_f32 left) xr /\
          Approx.v_approximates (Cast.cast_bf16_to_f32 right) yr /\
          Approx.v_approximates previous zr ==>
          Approx.v_approximates result (xr *. yr +. zr)))
      (ensures scope ==> Approx.v_approximates result (a *. b +. c)) =
  let under (_:scope)
    : Lemma (Approx.v_approximates result (a *. b +. c)) =
    Cast.cast_bf16_to_f32_ok left a;
    Cast.cast_bf16_to_f32_ok right b;
    assert (Approx.v_approximates result (a *. b +. c))
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let conditional_reduction (scope:prop) (actual:lseq FP.t 32) (logical:S.vector 256)
  (groups:pos{groups = 16 \/ groups = 32}) (linear:bool) (result:FP.t)
  : Lemma
      (requires
        (scope ==> partial_rep groups actual logical) /\
        result == (if linear then Reduction.linear_value actual groups
          else Norm.tree groups (Reduction.values actual)))
      (ensures scope ==> Approx.v_approximates result (Math.partial_sum groups logical)) =
  let under (_:scope)
    : Lemma (Approx.v_approximates result (Math.partial_sum groups logical)) =
    if linear then Reduction.linear_approx actual logical groups
    else Norm.tree_approx groups (Reduction.values actual) logical
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let conditional_round (scope:prop) (value:FP.t) (logical:real)
  : Lemma
      (requires scope ==> Approx.v_approximates value logical)
      (ensures scope ==> Approx.v_approximates (Cast.cast_f32_to_bf16 value) logical) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (Cast.cast_f32_to_bf16 value) logical) =
    Cast.cast_f32_to_bf16_ok value logical
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let input_write_frame (#length:pos{length <= 12582912}) (g:geometry length) (before:lseq BF.t 16000)
  (row:S.index (SZ.v g.Layout.rows)) (value:BF.t) (logical:S.vector (SZ.v g.Layout.columns))
  : Lemma
      (requires input_rep g before logical)
      (ensures input_rep g (Seq.upd before (SZ.v g.Layout.output_offset + row) value) logical) =
  let at_index (i:S.index (SZ.v g.Layout.columns))
    : Lemma (
        Seq.index (Seq.upd before (SZ.v g.Layout.output_offset + row) value)
          (SZ.v g.Layout.input_offset + i) ==
        Seq.index before (SZ.v g.Layout.input_offset + i)) = ()
  in
  FStar.Classical.forall_intro at_index
