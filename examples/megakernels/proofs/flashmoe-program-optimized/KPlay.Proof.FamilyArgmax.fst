module KPlay.Proof.FamilyArgmax

#lang-pulse

open Kuiper
module Shared = KPlay.Proof.Shared.Argmax
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module A = Kuiper.Approximates.Base
module I32 = FStar.Int32
module U64 = FStar.UInt64
module SZ = Kuiper.SizeT
module S = FStar.Seq
module PA = Pulse.Lib.Array

inline_for_extraction noextract
let negative_infinity : f32 = F.neg F.infinity

inline_for_extraction noextract
let admissible_dimensions (tokens experts:nat) : prop =
  0 < tokens /\ 0 < experts /\ experts * tokens <= 2147483647

inline_for_extraction noextract
unfold let probability_values (n:nat) : Type0 =
  values:lseq f32 n {
    forall (j:Shared.index n).
      F.kind #f32 (S.index values j) == F.Finite /\
      F.lte #f32 F.zero (S.index values j) /\
      F.lte #f32 (S.index values j) F.one
  }

inline_for_extraction noextract
unfold let selected_values (tokens:nat) (experts:pos) : Type0 =
  values:lseq i32 tokens {
    forall (t:Shared.index tokens).
      0 <= I32.v (S.index values t) /\ I32.v (S.index values t) < experts
  }

inline_for_extraction noextract
let row_input (#e:pos) (row:probability_values e) : Shared.row e =
  fun j -> S.index row j

inline_for_extraction noextract
let row_at (tokens:nat) (experts:pos)
  (rows:probability_values (experts * tokens)) (token:Shared.index tokens)
  : probability_values experts =
  let row = S.slice rows (experts * token) (experts * token + experts) in
  let cell (j:Shared.index experts)
    : Lemma (
        F.kind #f32 (S.index row j) == F.Finite /\
        F.lte #f32 F.zero (S.index row j) /\
        F.lte #f32 (S.index row j) F.one)
    = S.lemma_index_slice rows (experts * token)
        (experts * token + experts) j
  in
  FStar.Classical.forall_intro cell;
  row

inline_for_extraction noextract
let logical_input (#s:nat) (#e:pos)
  (rows:probability_values (e * s)) : Shared.probabilities s e =
  fun t j -> S.index rows (e * t + j)

inline_for_extraction noextract
let selected_index (#s:nat) (#e:pos)
  (selected:selected_values s e) (t:Shared.index s) : Shared.index e =
  I32.v (S.index selected t)

inline_for_extraction noextract
let logical_output (#s:nat) (#e:pos)
  (selected:selected_values s e) (best:probability_values s)
  : Shared.outputs s e = {
  selected_expert = (fun t -> selected_index selected t);
  selected_probability = (fun t -> A.to_real #f32 (S.index best t))
}

inline_for_extraction noextract
let semantic_post (#s:nat) (#e:pos)
  (before after:probability_values (e * s))
  (selected:selected_values s e) (best:probability_values s) : prop =
  after == before /\
  (forall (t:Shared.index s).
    S.index best t == S.index before (e * t + selected_index selected t)) /\
  Shared.post (logical_input before) (logical_input after)
    (logical_output selected best) /\
  (forall (t:Shared.index s).
    A.v_approximates #f32 (S.index best t)
      ((logical_output selected best).selected_probability t))

(* Retained scan invariant: representation identity is stronger than a tie. *)
inline_for_extraction noextract
let candidate (x:f32) : prop =
  not (F.is_nan x) /\ F.lt negative_infinity x

inline_for_extraction noextract
let row_result (#e:pos) (row:lseq f32 e) (selected:i32) (best:f32) : prop =
  (I32.v selected == 0 /\
   F.bit_eq best negative_infinity /\
   (forall (j:Shared.index e). not (candidate (S.index row j))))
  \/
  (exists (chosen:Shared.index e).
    I32.v selected == chosen /\
    candidate (S.index row chosen) /\
    F.bit_eq best (S.index row chosen) /\
    (forall (j:Shared.index e).
      candidate (S.index row j) ==>
      (not (F.lt (S.index row chosen) (S.index row j)) /\
       (F.ieee_eq (S.index row j) (S.index row chosen) ==> chosen <= j))))

inline_for_extraction noextract
let argmax_post (tokens:nat) (experts:pos)
  (rows:probability_values (experts * tokens))
  (selected:lseq i32 tokens) (best:lseq f32 tokens) : prop =
  forall (t:Shared.index tokens).
    row_result (row_at tokens experts rows t) (S.index selected t) (S.index best t)

inline_for_extraction noextract
let stable_maximum_unique (#e:pos) (p:Shared.row e) (chosen:Shared.index e)
  : Lemma
      (requires Shared.stable_maximum p e chosen)
      (ensures chosen == Shared.top1_index p)
  = Shared.best_prefix_correct p e;
    Shared.comparison (p chosen) (p (Shared.top1_index p))

inline_for_extraction noextract
let ordered_total (x y : f32)
  : Lemma
      (requires not (F.is_nan x) /\ not (F.is_nan y))
      (ensures F.lt x y \/ F.ieee_eq x y \/ F.lt y x)
  = F.negate_lt_is_lte x y;
    F.lte_is_lt_or_eq y x;
    F.eq_spec x y;
    F.eq_spec y x

inline_for_extraction noextract
let strict_order (x y : f32)
  : Lemma
      (requires F.lt x y)
      (ensures not (F.lt y x) /\ not (F.ieee_eq y x) /\
               not (F.ieee_eq x y))
  = F32.lt_ordered x y;
    F.negate_lt_is_lte x y;
    F.lte_is_lt_or_eq y x;
    F.eq_spec x y;
    F.eq_spec y x

inline_for_extraction noextract
let no_self_lt (x : f32)
  : Lemma (not (F.lt x x))
  = if F.lt x x then strict_order x x

inline_for_extraction noextract
let better_than_prior (old fresh prior : f32)
  : Lemma
      (requires
        not (F.is_nan prior) /\
        F.lt old fresh /\ not (F.lt old prior))
      (ensures
        not (F.lt fresh prior) /\ not (F.ieee_eq prior fresh))
  = F32.lt_ordered old fresh;
    ordered_total prior old;
    strict_order old fresh;
    if F.lt fresh prior then F32.lt_transitive old fresh prior;
    if F.lt prior old then (
      F32.lt_transitive prior old fresh;
      strict_order prior fresh
    ) else (
      F.eq_spec prior old;
      F.eq_spec prior fresh;
      F.eq_spec old fresh
    )

(* Exact representation equality is maintained separately from numerical ties. *)
inline_for_extraction noextract
let prefix_result
  (#e : pos)
  (row : lseq f32 e)
  (count : nat{count <= e})
  (chosen : nat{chosen < e})
  (best : f32)
  : prop =
  (chosen == 0 /\ best == negative_infinity /\
   (forall (j : nat{j < count}). not (candidate (S.index row j))))
  \/
  (chosen < count /\
   candidate (S.index row chosen) /\
   best == S.index row chosen /\
   (forall (j : nat{j < count}).
      candidate (S.index row j) ==>
      (not (F.lt best (S.index row j)) /\
       (F.ieee_eq (S.index row j) best ==> chosen <= j))))

inline_for_extraction noextract
let prefix_initial (#e:pos) (row : lseq f32 e)
  : Lemma (prefix_result row 0 0 negative_infinity)
  = ()

inline_for_extraction noextract
let prefix_update
  (#e : pos)
  (row : lseq f32 e)
  (count : nat{count < e})
  (chosen : nat{chosen < e})
  (best : f32)
  : Lemma
      (requires
        prefix_result row count chosen best /\
        F.lt best (S.index row count))
      (ensures
        prefix_result row (count + 1) count (S.index row count))
  = let fresh = S.index row count in
    F32.lt_ordered best fresh;
    F.bit_eq_spec best negative_infinity;
    if F.bit_eq best negative_infinity then ()
    else F32.lt_transitive negative_infinity best fresh;
    assert (candidate fresh);
    let prior_bound (j : nat{j < count + 1})
      : Lemma
          (candidate (S.index row j) ==>
           (not (F.lt fresh (S.index row j)) /\
            (F.ieee_eq (S.index row j) fresh ==> count <= j)))
      = if j < count then (
          let prior = S.index row j in
          if F.lt negative_infinity prior then (
            F32.lt_ordered negative_infinity prior;
            better_than_prior best fresh prior
          )
        ) else no_self_lt fresh
    in
    FStar.Classical.forall_intro prior_bound

inline_for_extraction noextract
let prefix_skip
  (#e : pos)
  (row : lseq f32 e)
  (count : nat{count < e})
  (chosen : nat{chosen < e})
  (best : f32)
  : Lemma
      (requires
        prefix_result row count chosen best /\
        not (F.lt best (S.index row count)))
      (ensures prefix_result row (count + 1) chosen best)
  = ()

inline_for_extraction noextract
let prefix_complete
  (#e : pos)
  (row : lseq f32 e)
  (chosen : nat{chosen < e})
  (selected : i32{I32.v selected == chosen})
  (best : f32)
  : Lemma
      (requires prefix_result row e chosen best)
      (ensures row_result row selected best)
  = F.bit_eq_spec best negative_infinity;
    F.bit_eq_spec best (S.index row chosen);
    assert (
      (I32.v selected == 0 /\
       F.bit_eq best negative_infinity /\
       (forall (j : nat{j < e}). not (candidate (S.index row j))))
      \/
      (chosen < e /\
       I32.v selected == chosen /\
       candidate (S.index row chosen) /\
       F.bit_eq best (S.index row chosen) /\
       (forall (j : nat{j < e}).
         candidate (S.index row j) ==>
         (not (F.lt (S.index row chosen) (S.index row j)) /\
          (F.ieee_eq (S.index row j) (S.index row chosen) ==> chosen <= j)))))

inline_for_extraction noextract
let finite_is_candidate (x:Shared.finite32)
  : Lemma (candidate x)
  = F.neg_kind #f32 F.infinity;
    F.neg_kind #f32 negative_infinity;
    F.neg_neg #f32 F.infinity;
    if F.is_zero #f32 F.infinity then F.is_zero_spec #f32 F.infinity;
    F.eq_spec #f32 (F.neg negative_infinity) F.infinity;
    assert (F.neg negative_infinity == F.infinity);
    F.neg_kind #f32 x;
    F.infinity_val_spec #f32 (F.neg x);
    F.lte_is_lt_or_eq #f32 (F.neg x) F.infinity;
    F.eq_spec #f32 (F.neg x) F.infinity;
    F.lt_neg_flip #f32 negative_infinity x

inline_for_extraction noextract
let row_refines (#e:pos) (row:probability_values e) (selected:i32) (best:f32)
  : Lemma
      (requires row_result row selected best)
      (ensures
        0 <= I32.v selected /\ I32.v selected < e /\
        I32.v selected == Shared.top1_index (row_input row) /\
        best == S.index row (Shared.top1_index (row_input row)))
  = finite_is_candidate (S.index row 0);
    assert (0 <= I32.v selected /\ I32.v selected < e);
    let chosen:Shared.index e = I32.v selected in
    F.bit_eq_spec #f32 best (S.index row chosen);
    let bound (j:Shared.index e)
      : Lemma (
          F.lte #f32 (S.index row j) (S.index row chosen) /\
          (j < chosen ==> F.lt #f32 (S.index row j) (S.index row chosen)))
      = finite_is_candidate (S.index row j);
        Shared.comparison (S.index row j) (S.index row chosen)
    in
    FStar.Classical.forall_intro bound;
    stable_maximum_unique (row_input row) chosen

