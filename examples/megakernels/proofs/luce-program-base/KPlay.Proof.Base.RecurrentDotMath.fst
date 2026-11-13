module KPlay.Proof.Base.RecurrentDotMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
open Kuiper
module Seq = FStar.Seq
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Approx = Kuiper.Approximates.Base
module KSum = Kuiper.Sum
module S = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Native = KPlay.Proof.Base.RecurrentDot
module Math = KPlay.Proof.Base.AttentionAlgebra
module Sum = KPlay.Proof.Base.ReductionMath

inline_for_extraction noextract
let column_rep (state:lseq FP.t 4718592)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (logical:S.vector 128) : prop =
  forall (i:P.index 128).
    Approx.v_approximates (Seq.index state (offset + i * 128 + column)) (logical i)

inline_for_extraction noextract
let vector_rep (vector:lseq FP.t 128) (logical:S.vector 128) : prop =
  forall (i:P.index 128). Approx.v_approximates (Seq.index vector i) (logical i)

inline_for_extraction noextract
let products (left right:S.vector 128) : S.vector 128 =
  fun i -> left i *. right i

inline_for_extraction noextract
let rec component_approx (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128)
  (wr xr:S.vector 128) (group:P.index 16) (count:nat{count <= 8})
  : Lemma
      (requires column_rep state offset column wr /\ vector_rep vector xr)
      (ensures Approx.v_approximates (Native.component state vector offset column group count)
        (Math.bounded_sum #8 count (Math.stripe_terms 128 16 group (products wr xr))))
      (decreases count) =
  if count > 0 then (
    component_approx state vector offset column wr xr group (count - 1);
    let i = group + 16 * (count - 1) in
    let a = Seq.index state (offset + i * 128 + column) in
    let b = Seq.index vector i in
    Approx.a_mul a b (wr i) (xr i);
    Math.bounded_sum_step #8 (count - 1) (Math.stripe_terms 128 16 group (products wr xr));
    Approx.a_add (Native.component state vector offset column group (count - 1)) (F.mul a b)
      (Math.bounded_sum #8 (count - 1) (Math.stripe_terms 128 16 group (products wr xr)))
      (wr i *. xr i)
  )

inline_for_extraction noextract
let group_values (wr xr:S.vector 128) : S.vector 256 =
  Math.group_sums 128 16 (products wr xr)

inline_for_extraction noextract
let group_approx (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128)
  (wr xr:S.vector 128) (group:P.index 16)
  : Lemma
      (requires column_rep state offset column wr /\ vector_rep vector xr)
      (ensures Approx.v_approximates (Native.component state vector offset column group 8)
        (group_values wr xr group)) =
  component_approx state vector offset column wr xr group 8;
  Math.bounded_sum_full #8 (Math.stripe_terms 128 16 group (products wr xr))

inline_for_extraction noextract
let row_value (groups:S.vector 256) (y:P.index 4) : real =
  ((groups y +. groups (y + 4)) +. groups (y + 8)) +. groups (y + 12)

inline_for_extraction noextract
let partial_approx (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128)
  (wr xr:S.vector 128) (y:P.index 4)
  : Lemma
      (requires column_rep state offset column wr /\ vector_rep vector xr)
      (ensures Approx.v_approximates (Native.partial_value state vector offset column y)
        (row_value (group_values wr xr) y)) =
  group_approx state vector offset column wr xr y;
  group_approx state vector offset column wr xr (y + 4);
  group_approx state vector offset column wr xr (y + 8);
  group_approx state vector offset column wr xr (y + 12);
  let groups = group_values wr xr in
  let a = Native.component state vector offset column y 8 in
  let b = Native.component state vector offset column (y + 4) 8 in
  let c = Native.component state vector offset column (y + 8) 8 in
  let d = Native.component state vector offset column (y + 12) 8 in
  Approx.a_add a b (groups y) (groups (y + 4));
  Approx.a_add (F.add a b) c (groups y +. groups (y + 4)) (groups (y + 8));
  Approx.a_add (F.add (F.add a b) c) d ((groups y +. groups (y + 4)) +. groups (y + 8)) (groups (y + 12))

inline_for_extraction noextract
let sum_four (f:S.vector 4)
  : Lemma (S.sum 4 f == ((f 0 +. f 1) +. f 2) +. f 3) =
  Sum.sum_step 4 f;
  Sum.sum_step 3 (fun i -> f i);
  Sum.sum_step 2 (fun i -> f i);
  Sum.sum_step 1 (fun i -> f i);
  Sum.sum_zero (fun i -> f i)

inline_for_extraction noextract
let row_expanded (groups:S.vector 256) (y:P.index 4)
  : Lemma (row_value groups y == S.sum 4 (fun r -> groups (y + 4 * r))) =
  sum_four (fun r -> groups (y + 4 * r))

inline_for_extraction noextract
let rows_grid (groups:S.vector 256)
  : Lemma (S.sum 4 (fun y -> row_value groups y) == Math.partial_sum 16 groups) =
  FStar.Classical.forall_intro (row_expanded groups);
  Sum.sum_ext 4 (fun y -> row_value groups y) (fun y -> S.sum 4 (fun r -> groups (y + 4 * r)));
  Math.sum_swap 4 4 (fun y r -> groups (y + 4 * r));
  let at_row (r:P.index 4)
    : Lemma (S.sum 4 (fun y -> groups (y + 4 * r)) == S.sum 4 (fun y -> groups (r * 4 + y))) =
    Sum.sum_ext 4 (fun y -> groups (y + 4 * r)) (fun y -> groups (r * 4 + y))
  in
  FStar.Classical.forall_intro at_row;
  Sum.sum_ext 4 (fun r -> S.sum 4 (fun y -> groups (y + 4 * r)))
    (fun r -> S.sum 4 (fun y -> groups (r * 4 + y)));
  Math.sum_grid 4 4 (fun i -> groups i);
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.partial_sum"]] (Math.partial_sum 16 groups)

inline_for_extraction noextract
let rows_sum (wr xr:S.vector 128)
  : Lemma (
      let groups = group_values wr xr in
      (row_value groups 0 +. row_value groups 2) +.
        (row_value groups 1 +. row_value groups 3) ==
      KSum.sum #real #KSum.real_add_semigroup 0 128 (products wr xr)) =
  let groups = group_values wr xr in
  sum_four (fun y -> row_value groups y);
  rows_grid groups;
  Math.stripes_sum 128 16 (products wr xr);
  Sum.sum_kuiper 128 (products wr xr)

inline_for_extraction noextract
let value_approx (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (wr xr:S.vector 128)
  : Lemma
      (requires column_rep state offset column wr /\ vector_rep vector xr)
      (ensures Approx.v_approximates (Native.value state vector offset column)
        (KSum.sum #real #KSum.real_add_semigroup 0 128 (products wr xr))) =
  FStar.Classical.forall_intro (partial_approx state vector offset column wr xr);
  let groups = group_values wr xr in
  let a = Native.partial_value state vector offset column 0 in
  let b = Native.partial_value state vector offset column 2 in
  let c = Native.partial_value state vector offset column 1 in
  let d = Native.partial_value state vector offset column 3 in
  Approx.a_add a b (row_value groups 0) (row_value groups 2);
  Approx.a_add c d (row_value groups 1) (row_value groups 3);
  Approx.a_add (F.add a b) (F.add c d)
    (row_value groups 0 +. row_value groups 2) (row_value groups 1 +. row_value groups 3);
  rows_sum wr xr

inline_for_extraction noextract
let conditional_value_approx (scope:prop) (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (wr xr:S.vector 128)
  : Lemma
      (requires scope ==> column_rep state offset column wr /\ vector_rep vector xr)
      (ensures scope ==> Approx.v_approximates (Native.value state vector offset column)
        (KSum.sum #real #KSum.real_add_semigroup 0 128 (products wr xr))) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (Native.value state vector offset column)
        (KSum.sum #real #KSum.real_add_semigroup 0 128 (products wr xr))) =
    value_approx state vector offset column wr xr
  in
  FStar.Classical.impl_intro under
