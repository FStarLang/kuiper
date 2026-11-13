module KPlay.Proof.Base.RecurrentStateMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open FStar.Real
module Seq = FStar.Seq
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module F = Kuiper.Floating.Base
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module H = KPlay.Proof.Base.RecurrentHeadModel
module Native = KPlay.Proof.Base.RecurrentNative
module Scalar = KPlay.Proof.Base.RecurrentScalars
module Dot = KPlay.Proof.Base.RecurrentDotMath
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let state_frame (before current:lseq FP.t 4718592) (offset:nat{offset + 16384 <= 4718592}) : prop =
  forall (i:P.index 4718592).
    ~(offset <= i /\ i < offset + 16384) ==> Seq.index current i == Seq.index before i

inline_for_extraction noextract
let hidden_frame (before current:Transfer.scratch) (head:P.index 16) (count:nat{count <= 128}) : prop =
  forall (i:P.index 16000).
    ~(10400 + head * 128 <= i /\ i < 10400 + head * 128 + count) ==>
      Seq.index current i == Seq.index before i

inline_for_extraction noextract
let columns (state:lseq FP.t 4718592) (offset:nat{offset + 16384 <= 4718592})
  (x:H.inputs) (count:nat{count <= 128}) : prop =
  forall (column:P.index 128).
    Dot.column_rep state offset column
      (if column < count then H.next_column x column else H.decayed_column x column)

inline_for_extraction noextract
let core_prefix (hidden:Transfer.scratch) (head:P.index 16) (x:H.inputs) (count:nat{count <= 128}) : prop =
  forall (j:nat{j < count}).
    Approx.v_approximates (Seq.index hidden (10400 + head * 128 + j)) (H.core x j)

inline_for_extraction noextract
let qk_approx (scope:prop) (weights:Transfer.packed) (hidden:Transfer.scratch) (state:lseq FP.t 4718592)
  (base:nat{base + 19458208 <= 752393024}) (offset:nat{offset + 16384 <= 4718592})
  (head:P.index 16) (x:H.inputs)
  : Lemma
      (requires scope ==> H.represented weights hidden state base offset head x)
      (ensures scope ==>
        Dot.vector_rep (Native.query_values hidden head) (H.query x) /\
        Dot.vector_rep (Native.key_values hidden head) (H.key x)) =
  let under (_:scope)
    : Lemma (
        Dot.vector_rep (Native.query_values hidden head) (H.query x) /\
        Dot.vector_rep (Native.key_values hidden head) (H.key x)) =
    let at_index (i:P.index 128)
      : Lemma (
          Approx.v_approximates (Seq.index (Native.query_values hidden head) i) (H.query x i) /\
          Approx.v_approximates (Seq.index (Native.key_values hidden head) i) (H.key x i)) =
      Scalar.query_approx hidden head x.H.raw_query i;
      Scalar.key_approx hidden head x.H.raw_key i
    in
    FStar.Classical.forall_intro at_index
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let controls_approx (scope:prop) (weights:Transfer.packed) (hidden:Transfer.scratch) (state:lseq FP.t 4718592)
  (base:nat{base + 19458208 <= 752393024}) (offset:nat{offset + 16384 <= 4718592})
  (head:P.index 16) (x:H.inputs)
  : Lemma
      (requires scope ==> H.represented weights hidden state base offset head x)
      (ensures scope ==> Approx.v_approximates (Scalar.beta (Seq.index hidden (10240 + head))) (H.beta x) /\
        Approx.v_approximates
          (Scalar.decay (Seq.index hidden (10256 + head))
            (Seq.index weights (base + 19458064 + head)) (Seq.index weights (base + 19458048 + head)))
          (H.gain x)) =
  let under (_:scope)
    : Lemma (
        Approx.v_approximates (Scalar.beta (Seq.index hidden (10240 + head))) (H.beta x) /\
        Approx.v_approximates
          (Scalar.decay (Seq.index hidden (10256 + head))
            (Seq.index weights (base + 19458064 + head)) (Seq.index weights (base + 19458048 + head)))
          (H.gain x)) =
    Scalar.beta_approx (Seq.index hidden (10240 + head)) x.H.beta_input;
    Scalar.decay_approx (Seq.index hidden (10256 + head))
      (Seq.index weights (base + 19458064 + head)) (Seq.index weights (base + 19458048 + head))
      x.H.decay_input x.H.decay_bias x.H.decay_log
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let decay_columns (scope:prop) (before:lseq FP.t 4718592) (offset:nat{offset + 16384 <= 4718592})
  (gain:FP.t) (x:H.inputs)
  : Lemma
      (requires scope ==> Approx.v_approximates gain (H.gain x) /\
        (forall (i j:P.index 128).
          Approx.v_approximates (Seq.index before (offset + i * 128 + j)) (x.H.state i j)))
      (ensures state_frame before (Native.decayed before offset gain 16384) offset /\
        (scope ==> columns (Native.decayed before offset gain 16384) offset x 0)) =
  let under (_:scope)
    : Lemma (columns (Native.decayed before offset gain 16384) offset x 0) =
    let at_index (i j:P.index 128)
      : Lemma (
          Approx.v_approximates (Seq.index (Native.decayed before offset gain 16384) (offset + i * 128 + j))
            (H.decayed_column x j i)) =
      Approx.a_mul (Seq.index before (offset + i * 128 + j)) gain (x.H.state i j) (H.gain x)
    in
    FStar.Classical.forall_intro_2 at_index
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let other_column (before:lseq FP.t 4718592) (keys:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column other:P.index 128{column <> other})
  (delta:FP.t) (i:P.index 128)
  : Lemma (
      Seq.index (Native.updated_column before keys offset column delta 128) (offset + i * 128 + other) ==
      Seq.index before (offset + i * 128 + other)) = ()

inline_for_extraction noextract
let update_approx (before:lseq FP.t 4718592) (keys:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (delta:FP.t) (x:H.inputs)
  : Lemma
      (requires Dot.column_rep before offset column (H.decayed_column x column) /\
        Dot.vector_rep keys (H.key x) /\ Approx.v_approximates delta (H.delta x column))
      (ensures Dot.column_rep (Native.updated_column before keys offset column delta 128)
        offset column (H.next_column x column)) =
  let at_index (i:P.index 128)
    : Lemma (
        Approx.v_approximates
          (Seq.index (Native.updated_column before keys offset column delta 128) (offset + i * 128 + column))
          (H.next_column x column i)) =
    Approx.a_mul (Seq.index keys i) delta (H.key x i) (H.delta x column);
    Approx.a_add (Seq.index before (offset + i * 128 + column)) (F.mul (Seq.index keys i) delta)
      (H.decayed_column x column i) (H.key x i *. H.delta x column)
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let column_step (scope:prop) (initial before:lseq FP.t 4718592) (keys:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (delta:FP.t) (x:H.inputs)
  : Lemma
      (requires state_frame initial before offset /\ (scope ==> columns before offset x column /\
        Dot.vector_rep keys (H.key x) /\ Approx.v_approximates delta (H.delta x column)))
      (ensures state_frame initial (Native.updated_column before keys offset column delta 128) offset /\
        (scope ==> columns (Native.updated_column before keys offset column delta 128) offset x (column + 1))) =
  let under (_:scope)
    : Lemma (columns (Native.updated_column before keys offset column delta 128) offset x (column + 1)) =
    update_approx before keys offset column delta x;
    let at_column (other:P.index 128)
      : Lemma (
          Dot.column_rep (Native.updated_column before keys offset column delta 128) offset other
            (if other < column + 1 then H.next_column x other else H.decayed_column x other)) =
      if other <> column then (
        FStar.Classical.forall_intro (other_column before keys offset column other delta);
        assert (other < column + 1 <==> other < column)
      )
    in
    FStar.Classical.forall_intro at_column
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let raw_value_frame (before current:Transfer.scratch) (head:P.index 16) (count:nat{count <= 128})
  (i:P.index 128)
  : Lemma
      (requires hidden_frame before current head count)
      (ensures Seq.index current (6144 + head * 128 + i) == Seq.index before (6144 + head * 128 + i)) = ()

inline_for_extraction noextract
let core_store (scope:prop) (initial current:Transfer.scratch) (head:P.index 16) (x:H.inputs)
  (column:P.index 128) (value:BF.t)
  : Lemma
      (requires hidden_frame initial current head column /\ (scope ==> core_prefix current head x column /\
        Approx.v_approximates value (H.core x column)))
      (ensures hidden_frame initial (Seq.upd current (10400 + head * 128 + column) value) head (column + 1) /\
        (scope ==> core_prefix (Seq.upd current (10400 + head * 128 + column) value) head x (column + 1))) = ()
