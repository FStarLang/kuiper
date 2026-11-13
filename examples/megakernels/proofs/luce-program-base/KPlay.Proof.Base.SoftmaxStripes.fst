module KPlay.Proof.Base.SoftmaxStripes

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost
open FStar.Real

module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Fast = Kuiper.Float32.FastMath
module Approx = Kuiper.Approximates.Base
module Exp = FStar.Math.Exp
module S = KPlay.Proof.Shared.Attention
module Math = KPlay.Proof.Base.AttentionAlgebra
module Scale = KPlay.Proof.Base.SoftmaxScale
module Reduction = KPlay.Proof.Base.ReductionNative
module Norm = KPlay.Proof.Base.Norm
module Sum = KPlay.Proof.Base.ReductionMath
module Conditional = KPlay.Proof.Base.ConditionalNumerics

inline_for_extraction noextract
let score_at (#n:pos) (scores:erased (S.vector n)) (i:S.index n) : GTot real =
  (reveal #(S.vector n) scores) i

inline_for_extraction noextract
let exponential_values (n:pos) (scores:S.vector n) (shift:real) : S.vector n =
  fun i -> Exp.exp (scores i -. shift)

inline_for_extraction noextract
let exponential_binding (n:pos) (scores:S.vector n) (shift:real)
  : Lemma (S.sum n (exponential_values n scores shift) == Math.exponential_sum n scores shift) =
  Sum.sum_ext n (exponential_values n scores shift) (Math.exponentials n scores shift);
  Math.exponential_sum_bridge n scores shift

inline_for_extraction noextract
let stage (n:pos{n <= 647}) (width:pos{width <= 32}) (count:nat{count <= width})
  (actual:lseq FP.t 2048) (scores:S.vector n) (shift:real) : prop =
  forall (i:S.index n).
    Approx.v_approximates (Seq.index actual i)
      (if i % width < count then Exp.exp (scores i -. shift) else scores i)

inline_for_extraction noextract
let mixed (n:pos{n <= 647}) (width:pos{width <= 32}) (group:nat{group < width})
  (cursor:nat) (actual:lseq FP.t 2048) (scores:S.vector n) (shift:real) : prop =
  forall (i:S.index n).
    Approx.v_approximates (Seq.index actual i)
      (if i % width < group || (i % width = group && i < cursor)
       then Exp.exp (scores i -. shift) else scores i)

inline_for_extraction noextract
let mixed_start (scope:prop) (n:pos{n <= 647}) (width:pos{width <= 32}) (group:nat{group < width})
  (actual:lseq FP.t 2048) (scores:S.vector n) (shift:real)
  : Lemma
      (requires scope ==> stage n width group actual scores shift)
      (ensures scope ==> mixed n width group group actual scores shift) = ()

inline_for_extraction noextract
let mixed_step (scope:prop) (n:pos{n <= 647}) (width:pos{width <= 32}) (group:nat{group < width})
  (cursor:nat{cursor < n /\ cursor % width = group})
  (actual:lseq FP.t 2048) (scores:S.vector n) (shift:real) (value:FP.t)
  : Lemma
      (requires scope ==> mixed n width group cursor actual scores shift /\
        Approx.v_approximates value (Exp.exp (scores cursor -. shift)))
      (ensures scope ==> mixed n width group (cursor + width) (Seq.upd actual cursor value) scores shift) =
  let at_index (i:S.index n)
    : Lemma (scope ==>
        Approx.v_approximates (Seq.index (Seq.upd actual cursor value) i)
          (if i % width < group || (i % width = group && i < cursor + width)
           then Exp.exp (scores i -. shift) else scores i)) =
    if i % width = group && cursor <= i && i < cursor + width then assert (i == cursor)
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let mixed_finish (scope:prop) (n:pos{n <= 647}) (width:pos{width <= 32}) (group:nat{group < width})
  (cursor:nat{n <= cursor}) (actual:lseq FP.t 2048) (scores:S.vector n) (shift:real)
  : Lemma
      (requires scope ==> mixed n width group cursor actual scores shift)
      (ensures scope ==> stage n width (group + 1) actual scores shift) = ()

inline_for_extraction noextract
let partial_frame (before current:lseq FP.t 32) (group:nat{group < 32}) : prop =
  forall (i:nat{i < 32 /\ i <> group}). Seq.index current i == Seq.index before i

inline_for_extraction noextract
let sum_prefix (n width:pos) (group:nat{group < width}) (scores:S.vector n) (shift:real)
  (count:nat{count <= Math.slots n width}) : real =
  Math.bounded_sum #(Math.slots n width) count
    (Math.stripe_terms n width group (exponential_values n scores shift))

inline_for_extraction noextract
let cursor_step (n width:pos) (group:nat{group < width})
  (i:nat{i < n /\ i % width = group})
  : Lemma (
      group <= i + width /\ i + width < n + width /\ (i + width) % width == group /\
      (i + width - group) / width == (i - group) / width + 1 /\
      (i + width - group) / width <= Math.slots n width) =
  Math.slots_bounds n width;
  assert (i == (i / width) * width + group)

inline_for_extraction noextract
let stripe_sum_step (n:pos{n <= 647}) (width:pos{width <= 32}) (group:nat{group < width})
  (i:nat{i < n /\ i % width = group}) (scores:S.vector n) (shift:real)
  : Lemma (
      sum_prefix n width group scores shift ((i - group) / width + 1) ==
      sum_prefix n width group scores shift ((i - group) / width) +.
      Exp.exp (scores i -. shift)) =
  Math.slots_bounds n width;
  assert (((i - group) / width) * width + group == i);
  Math.bounded_sum_step #(Math.slots n width) ((i - group) / width)
    (Math.stripe_terms n width group (exponential_values n scores shift))

inline_for_extraction noextract
let advanced_sum (scope:prop) (n:pos{n <= 647}) (width:pos{width <= 32}) (group:nat{group < width})
  (i:nat{i < n /\ i % width = group}) (scores:S.vector n) (shift:real) (old value:FP.t)
  : Lemma
      (requires scope ==>
        Approx.v_approximates old (sum_prefix n width group scores shift ((i - group) / width)) /\
        Approx.v_approximates value (Exp.exp (scores i -. shift)))
      (ensures scope ==>
        Approx.v_approximates (F.add old value)
          (sum_prefix n width group scores shift ((i - group) / width + 1))) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (F.add old value)
        (sum_prefix n width group scores shift ((i - group) / width + 1))) =
    stripe_sum_step n width group i scores shift;
    Approx.a_add old value
      (sum_prefix n width group scores shift ((i - group) / width))
      (Exp.exp (scores i -. shift))
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
fn subtract (x y:FP.t) (#xr #yr:erased real) (#scope:prop)
  preserves gpu
  requires pure (scope ==> Approx.v_approximates x (reveal #real xr) /\ Approx.v_approximates y (reveal #real yr))
  returns result:FP.t
  ensures pure (scope ==> Approx.v_approximates result (reveal #real xr -. reveal #real yr))
{
  let result = Fast.sub_rn x y;
  let first : squash (forall (r:real).
    Approx.v_approximates x (reveal #real xr) /\ Approx.v_approximates y r ==>
      Approx.v_approximates result (reveal #real xr -. r)) =
    FStar.Classical.Sugar.forall_elim (reveal #real xr) ();
  let second : squash (
    Approx.v_approximates x (reveal #real xr) /\ Approx.v_approximates y (reveal #real yr) ==>
      Approx.v_approximates result (reveal #real xr -. reveal #real yr)) =
    FStar.Classical.Sugar.forall_elim (reveal #real yr) first;
  result
}

inline_for_extraction noextract
fn stripe
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647})
  (width:sz{0 < SZ.v width /\ SZ.v width <= 32})
  (group:sz{SZ.v group < SZ.v width})
  (values:larray FP.t 2048) (partial:larray FP.t 32) (maximum:FP.t)
  (#sv:erased (lseq FP.t 2048)) (#sp:erased (lseq FP.t 32))
  (#scores:erased (S.vector (SZ.v n))) (#shift:erased real) (#scope:prop)
  preserves gpu
  requires values |-> sv ** partial |-> sp
  requires pure (scope ==> stage (SZ.v n) (SZ.v width) (SZ.v group) sv scores (reveal #real shift) /\
    Approx.v_approximates maximum (reveal #real shift))
  ensures exists* (next:lseq FP.t 2048) (temporary:lseq FP.t 32).
    values |-> next ** partial |-> temporary **
    pure (partial_frame sp temporary (SZ.v group) /\
      (scope ==> stage (SZ.v n) (SZ.v width) (SZ.v group + 1) next scores (reveal #real shift) /\
      Approx.v_approximates (Seq.index temporary (SZ.v group))
        (S.sum (Math.slots (SZ.v n) (SZ.v width))
          (Math.stripe_terms (SZ.v n) (SZ.v width) (SZ.v group)
            (exponential_values (SZ.v n) scores (reveal #real shift))))))
{
  Math.slots_bounds (SZ.v n) (SZ.v width);
  mixed_start scope (SZ.v n) (SZ.v width) (SZ.v group) sv scores (reveal #real shift);
  partial.(group) <- F.zero #FP.t;
  let mut j = group;
  while (!j <^ n)
    invariant exists* (i:sz{SZ.v group <= SZ.v i /\ SZ.v i < SZ.v n + SZ.v width /\
      SZ.v i % SZ.v width = SZ.v group /\
      (SZ.v i - SZ.v group) / SZ.v width <= Math.slots (SZ.v n) (SZ.v width)})
      (current:lseq FP.t 2048) (temporary:lseq FP.t 32).
      j |-> i ** values |-> current ** partial |-> temporary **
      pure (partial_frame sp temporary (SZ.v group) /\
        (scope ==> mixed (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) current scores (reveal #real shift) /\
        Approx.v_approximates (Seq.index temporary (SZ.v group))
          (sum_prefix (SZ.v n) (SZ.v width) (SZ.v group) scores (reveal #real shift)
            ((SZ.v i - SZ.v group) / SZ.v width))))
    invariant gpu
    decreases (SZ.v n + SZ.v width - SZ.v !j)
  {
    let i = !j;
    with current temporary. assert values |-> current ** partial |-> temporary;
    let input = values.(i);
    let difference = subtract input maximum #(score_at #(SZ.v n) scores (SZ.v i)) #shift #scope;
    let value = F.fexp difference;
    Conditional.exponential scope difference (score_at #(SZ.v n) scores (SZ.v i) -. reveal #real shift);
    mixed_step scope (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) current scores (reveal #real shift) value;
    values.(i) <- value;
    let old = partial.(group);
    advanced_sum scope (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) scores (reveal #real shift) old value;
    let combined = F.add old value;
    partial.(group) <- combined;
    cursor_step (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i);
    let next = SZ.add i width;
    assert pure ((SZ.v next - SZ.v group) / SZ.v width ==
      (SZ.v i - SZ.v group) / SZ.v width + 1);
    with next_partial. assert partial |-> next_partial;
    assert pure (Seq.index next_partial (SZ.v group) == combined);
    assert pure (scope ==> Approx.v_approximates (Seq.index next_partial (SZ.v group))
      (sum_prefix (SZ.v n) (SZ.v width) (SZ.v group) scores (reveal #real shift)
        ((SZ.v next - SZ.v group) / SZ.v width)));
    j := next;
  };
  let i = !j;
  with current temporary. assert values |-> current ** partial |-> temporary;
  mixed_finish scope (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) current scores (reveal #real shift);
  Math.stripe_tail_zero (SZ.v n) (SZ.v width) (SZ.v group)
    ((SZ.v i - SZ.v group) / SZ.v width) (exponential_values (SZ.v n) scores (reveal #real shift));
  Math.zero_suffix_sum (Math.slots (SZ.v n) (SZ.v width))
    ((SZ.v i - SZ.v group) / SZ.v width)
    (Math.stripe_terms (SZ.v n) (SZ.v width) (SZ.v group)
      (exponential_values (SZ.v n) scores (reveal #real shift)));
  ()
}

inline_for_extraction noextract
let denominator_approx (scope:prop) (n:pos{n <= 647})
  (width:pos{width <= 32 /\ Norm.tree_size width}) (before:lseq FP.t 32)
  (scores:S.vector n) (shift:real)
  : Lemma
      (requires scope ==> forall (i:nat{i < width}).
        Approx.v_approximates (Seq.index before i)
          (Math.group_sums n width (exponential_values n scores shift) i))
      (ensures scope ==> Approx.v_approximates (Norm.tree width (Reduction.values before))
        (Math.exponential_sum n scores shift)) =
  let under (_:scope)
    : Lemma (Approx.v_approximates (Norm.tree width (Reduction.values before))
        (Math.exponential_sum n scores shift)) =
    Norm.tree_approx width (Reduction.values before)
      (Math.group_sums n width (exponential_values n scores shift));
    Math.stripes_sum n width (exponential_values n scores shift);
    exponential_binding n scores shift
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
fn exponentiate_and_reduce
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647})
  (width:sz{0 < SZ.v width /\ SZ.v width <= 32 /\ Norm.tree_size (SZ.v width)})
  (values:larray FP.t 2048) (partial:larray FP.t 32) (maximum:FP.t)
  (#sv:erased (lseq FP.t 2048)) (#sp:erased (lseq FP.t 32))
  (#scores:erased (S.vector (SZ.v n))) (#shift:erased real) (#scope:prop)
  preserves gpu
  requires values |-> sv ** partial |-> sp
  requires pure (scope ==> Scale.values_rep (SZ.v n) sv scores /\ Approx.v_approximates maximum (reveal #real shift))
  ensures exists* (next:lseq FP.t 2048) (temporary:lseq FP.t 32).
    values |-> next ** partial |-> temporary **
    pure (scope ==> Scale.values_rep (SZ.v n) next (exponential_values (SZ.v n) scores (reveal #real shift)) /\
      Approx.v_approximates (Seq.index temporary 0) (Math.exponential_sum (SZ.v n) scores (reveal #real shift)))
{
  let mut group = 0sz;
  while (!group <^ width)
    invariant exists* (g:sz{SZ.v g <= SZ.v width})
      (current:lseq FP.t 2048) (temporary:lseq FP.t 32).
      group |-> g ** values |-> current ** partial |-> temporary **
      pure (scope ==> stage (SZ.v n) (SZ.v width) (SZ.v g) current scores (reveal #real shift) /\
        (forall (i:nat{i < SZ.v g}).
          Approx.v_approximates (Seq.index temporary i)
            (Math.group_sums (SZ.v n) (SZ.v width) (exponential_values (SZ.v n) scores (reveal #real shift)) i)))
    invariant gpu
    decreases (SZ.v width - SZ.v !group)
  {
    let g = !group;
    stripe n width g values partial maximum #_ #_ #scores #shift #scope;
    group := SZ.add g 1sz;
  };
  with current temporary. assert values |-> current ** partial |-> temporary;
  let denominator = Reduction.tree partial width;
  denominator_approx scope (SZ.v n) (SZ.v width) temporary scores (reveal #real shift);
  ()
}
