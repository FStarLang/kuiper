module KPlay.Functional

#lang-pulse

open Kuiper
module Shared = KPlay.SharedSpec
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module A = Kuiper.Approximates.Base
module I32 = FStar.Int32
module U64 = FStar.UInt64
module SZ = Kuiper.SizeT
module S = FStar.Seq
module PA = Pulse.Lib.Array

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
let expert_to_i32 (expert : sz{SZ.v expert < 2147483647})
  : selected:i32{I32.v selected == SZ.v expert}
  = FStar.Int.Cast.uint32_to_int32 (SZ.sizet_to_u32 expert)

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn scan_row
  (tokens : sz{0 < SZ.v tokens /\ SZ.v tokens <= 2147483647})
  (experts : sz{admissible_dimensions (SZ.v tokens) (SZ.v experts)})
  (normalized_rows : larray f32 (SZ.v experts * SZ.v tokens))
  (token : sz{SZ.v token < SZ.v tokens})
  (#rows : erased (probability_values (SZ.v experts * SZ.v tokens)))
  preserves
    gpu **
    pts_to_slice normalized_rows 0 (SZ.v experts * SZ.v tokens) rows
  returns result : (i32 & f32)
  ensures pure (
    row_result (row_at (SZ.v tokens) (SZ.v experts) rows (SZ.v token))
      (fst result) (snd result))
{
  let mut expert = 0sz;
  let mut chosen = 0sz;
  let mut best : f32 = negative_infinity;
  prefix_initial (row_at (SZ.v tokens) (SZ.v experts) rows (SZ.v token));
  while (!expert <^ experts)
    invariant
      live expert ** live chosen ** live best **
      pure (SZ.v !expert <= SZ.v experts /\ SZ.v !chosen < SZ.v experts /\
        prefix_result (row_at (SZ.v tokens) (SZ.v experts) rows (SZ.v token))
          (SZ.v !expert) (SZ.v !chosen) !best)
    decreases (SZ.v experts - SZ.v !expert)
  {
    let e = !expert;
    let c = !chosen;
    let b = !best;
    let x = slice_read normalized_rows (token *^ experts +^ e);
    S.lemma_index_slice rows (SZ.v experts * SZ.v token)
      (SZ.v experts * SZ.v token + SZ.v experts) (SZ.v e);
    if F.lt b x {
      prefix_update (row_at (SZ.v tokens) (SZ.v experts) rows (SZ.v token))
        (SZ.v e) (SZ.v c) b;
      best := x;
      chosen := e;
    } else {
      prefix_skip (row_at (SZ.v tokens) (SZ.v experts) rows (SZ.v token))
        (SZ.v e) (SZ.v c) b;
      ();
    };
    expert := !expert +^ 1sz;
  };
  let c = !chosen;
  let b = !best;
  assert pure (SZ.v c < SZ.v experts /\ SZ.v c < 2147483647);
  let selected = expert_to_i32 c;
  assert pure (I32.v selected == SZ.v c);
  prefix_complete (row_at (SZ.v tokens) (SZ.v experts) rows (SZ.v token))
    (SZ.v c) selected b;
  (selected, b)
}
#pop-options

inline_for_extraction noextract
let completed_rows
  (tokens : nat)
  (experts : pos)
  (rows : probability_values (experts * tokens))
  (count : nat{count <= tokens})
  (selected : lseq i32 tokens)
  (best : lseq f32 tokens)
  : prop =
  forall (token : nat{token < count}).
    row_result (row_at tokens experts rows token)
      (S.index selected token) (S.index best token)

inline_for_extraction noextract
let complete_one_row
  (tokens : nat)
  (experts : pos)
  (rows : probability_values (experts * tokens))
  (token : nat{token < tokens})
  (selected : lseq i32 tokens)
  (best : lseq f32 tokens)
  (index : i32)
  (value : f32)
  : Lemma
      (requires
        completed_rows tokens experts rows token selected best /\
        row_result (row_at tokens experts rows token) index value)
      (ensures
        completed_rows tokens experts rows (token + 1)
          (S.upd selected token index) (S.upd best token value))
  = ()

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

inline_for_extraction noextract
let outputs_in_domain (tokens:nat) (experts:pos)
  (rows:probability_values (experts * tokens))
  (selected:lseq i32 tokens) (best:lseq f32 tokens)
  : Lemma
      (requires argmax_post tokens experts rows selected best)
      (ensures
        (forall (t:Shared.index tokens).
          0 <= I32.v (S.index selected t) /\ I32.v (S.index selected t) < experts) /\
        (forall (t:Shared.index tokens).
          F.kind #f32 (S.index best t) == F.Finite /\
          F.lte #f32 F.zero (S.index best t) /\
          F.lte #f32 (S.index best t) F.one))
  = let cell (t:Shared.index tokens)
      : Lemma (
          0 <= I32.v (S.index selected t) /\ I32.v (S.index selected t) < experts /\
          F.kind #f32 (S.index best t) == F.Finite /\
          F.lte #f32 F.zero (S.index best t) /\
          F.lte #f32 (S.index best t) F.one)
      = row_refines (row_at tokens experts rows t) (S.index selected t) (S.index best t)
    in
    FStar.Classical.forall_intro cell

inline_for_extraction noextract
let direct_refines (tokens:nat) (experts:pos)
  (rows:probability_values (experts * tokens))
  (selected:selected_values tokens experts) (best:probability_values tokens)
  : Lemma
      (requires argmax_post tokens experts rows selected best)
      (ensures semantic_post #tokens #experts rows rows selected best)
  = let cell (t:Shared.index tokens)
      : Lemma (
          selected_index #tokens #experts selected t ==
            Shared.top1_index (logical_input #tokens #experts rows t) /\
          S.index best t == logical_input #tokens #experts rows t
            (selected_index #tokens #experts selected t) /\
          A.v_approximates #f32 (S.index best t) (A.to_real #f32 (S.index best t)))
      = let row = row_at tokens experts rows t in
        row_refines row (S.index selected t) (S.index best t);
        let same (j:Shared.index experts)
          : Lemma (row_input row j == logical_input #tokens #experts rows t j)
          = S.lemma_index_slice rows (experts * t) (experts * t + experts) j
        in
        FStar.Classical.forall_intro same;
        prefix_extensional (row_input row) (logical_input #tokens #experts rows t) experts;
        A.to_real_ok #f32 (S.index best t)
    in
    FStar.Classical.forall_intro cell

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn argmax_kernel
  (tokens : sz{0 < SZ.v tokens /\ SZ.v tokens <= 2147483647})
  (experts : sz{admissible_dimensions (SZ.v tokens) (SZ.v experts)})
  (normalized_rows : larray f32 (SZ.v experts * SZ.v tokens))
  (selected_experts : larray i32 (SZ.v tokens))
  (best_values : larray f32 (SZ.v tokens))
  (#rows : erased (probability_values (SZ.v experts * SZ.v tokens)))
  (#selected_before : erased (seq i32))
  (#best_before : erased (seq f32))
  preserves
    gpu **
    PA.pts_to normalized_rows rows
  requires
    PA.pts_to selected_experts selected_before **
    PA.pts_to best_values best_before
  ensures
    exists* (selected : selected_values (SZ.v tokens) (SZ.v experts))
            (best : probability_values (SZ.v tokens)).
      PA.pts_to selected_experts selected **
      PA.pts_to best_values best **
      pure (semantic_post #(SZ.v tokens) #(SZ.v experts) rows rows selected best)
{
  array_to_slice normalized_rows;
  array_to_slice selected_experts;
  array_to_slice best_values;
  let mut token = 0sz;
  while (!token <^ tokens)
    invariant live token
    invariant
      exists* (selected : lseq i32 (SZ.v tokens))
              (best : lseq f32 (SZ.v tokens)).
        pts_to_slice selected_experts 0 (SZ.v tokens) selected **
        pts_to_slice best_values 0 (SZ.v tokens) best **
        pure (SZ.v !token <= SZ.v tokens /\
          completed_rows (SZ.v tokens) (SZ.v experts) rows (SZ.v !token) selected best)
    decreases (SZ.v tokens - SZ.v !token)
  {
    let t = !token;
    with selected. assert pts_to_slice selected_experts 0 (SZ.v tokens) selected;
    with best. assert pts_to_slice best_values 0 (SZ.v tokens) best;
    let result = scan_row tokens experts normalized_rows t #rows;
    slice_write selected_experts t result._1;
    slice_write best_values t result._2;
    complete_one_row (SZ.v tokens) (SZ.v experts) rows (SZ.v t)
      selected best result._1 result._2;
    token := !token +^ 1sz;
  };
  slice_to_array normalized_rows;
  slice_to_array selected_experts;
  slice_to_array best_values;
  with selected. assert PA.pts_to selected_experts selected;
  with best. assert PA.pts_to best_values best;
  assert pure (argmax_post (SZ.v tokens) (SZ.v experts) rows selected best);
  outputs_in_domain (SZ.v tokens) (SZ.v experts) rows selected best;
  direct_refines (SZ.v tokens) (SZ.v experts) rows selected best;
  assert pure (semantic_post #(SZ.v tokens) #(SZ.v experts) rows rows selected best);
}
#pop-options

inline_for_extraction noextract
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global_array arr})
  (#f : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (tokens:u64{0 < U64.v tokens /\ U64.v tokens <= 2147483647})
  (experts:u64{admissible_dimensions (U64.v tokens) (U64.v experts)})
  (#rows:erased (probability_values (U64.v experts * U64.v tokens)))
  (normalized_rows:device_array f32 (U64.v experts * U64.v tokens))
  (selected_experts:device_array i32 (U64.v tokens))
  (best_values:device_array f32 (U64.v tokens))
  requires cpu ** on gpu_loc (
    PA.pts_to normalized_rows rows **
    PA.pts_to selected_experts 'selected_before **
    PA.pts_to best_values 'best_before)
  ensures exists*
    (selected:selected_values (U64.v tokens) (U64.v experts))
    (best:probability_values (U64.v tokens)).
    cpu ** on gpu_loc (
      PA.pts_to normalized_rows rows **
      PA.pts_to selected_experts selected **
      PA.pts_to best_values best) **
    pure (semantic_post rows rows selected best)
{
  SZ.fits_ok (U64.v tokens);
  SZ.fits_ok (U64.v experts);
  let height = FStar.SizeT.uint64_to_sizet tokens;
  let width = FStar.SizeT.uint64_to_sizet experts;
  launch_kernel_1 (fun () ->
    argmax_kernel height width normalized_rows selected_experts best_values
      #(hide (reveal rows <: probability_values (SZ.v width * SZ.v height)))
      #'selected_before #'best_before);
}
