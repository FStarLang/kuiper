module KPlay.Functional

#lang-pulse

open Kuiper
open FStar.Real
module S = FStar.Seq.Base
module C = FStar.Classical
module I64 = FStar.Int64
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module F = Kuiper.Floating
module Cast = Kuiper.Float.Casts.Base
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let prompt_length : pos = Shared.prompt_length

inline_for_extraction noextract
let vocab_size : pos = Shared.vocab_size

inline_for_extraction noextract
let sequence_length : pos = Shared.sequence_length

inline_for_extraction noextract
let physical_width : pos = 153600

inline_for_extraction noextract
let logits_size : pos = prompt_length * physical_width

inline_for_extraction noextract
type row_index = r:nat{r < prompt_length}

inline_for_extraction noextract
type token_index = j:nat{j < vocab_size}

inline_for_extraction noextract
type physical_logits = lseq BF16.t logits_size

inline_for_extraction noextract
type physical_prompt = lseq I64.t prompt_length

inline_for_extraction noextract
type physical_output = lseq I64.t sequence_length

inline_for_extraction noextract
type scores (n:pos) = lseq F32.t n

inline_for_extraction noextract
let score_at (logits:physical_logits) (row:row_index) (j:token_index)
  : Tot F32.t =
  Cast.cast_bf16_to_f32 (S.index logits (row * physical_width + j))

inline_for_extraction noextract
let row_scores (logits:physical_logits) (row:row_index)
  : Tot (scores vocab_size) =
  S.init vocab_size (score_at logits row)

inline_for_extraction noextract
let ordered_scores (#n:pos) (s:scores n) : prop =
  forall (j:nat{j < n}). not (F.is_nan (S.index s j))

inline_for_extraction noextract
let finite_scores (#n:pos) (s:scores n) : prop =
  forall (j:nat{j < n}). F.is_finite (S.index s j)

inline_for_extraction noextract
let strict_irreflexive (x:F32.t)
  : Lemma (not (lt x x)) =
  if lt x x then (
    F32.lt_ordered x x;
    F.eq_spec x x;
    F.lte_is_lt_or_eq x x;
    F.negate_lt_is_lte x x)

inline_for_extraction noextract
let ordered_trichotomy (x y:F32.t)
  : Lemma
      (requires not (F.is_nan x) /\ not (F.is_nan y))
      (ensures
        (lt x y \/ F.ieee_eq x y \/ lt y x) /\
        (F.ieee_eq x y <==> (not (lt x y) /\ not (lt y x)))) =
  F.eq_spec x y;
  F.eq_spec y x;
  F.lte_is_lt_or_eq x y;
  F.lte_is_lt_or_eq y x;
  F.negate_lt_is_lte x y;
  F.negate_lt_is_lte y x

inline_for_extraction noextract
let lt_right_congruent (z x y:F32.t)
  : Lemma
      (requires
        not (F.is_nan z) /\ not (F.is_nan x) /\ not (F.is_nan y) /\
        F.ieee_eq x y)
      (ensures lt z x <==> lt z y) =
  ordered_trichotomy x y;
  ordered_trichotomy z x;
  ordered_trichotomy z y;
  if lt z x && not (lt z y) then (
    if lt y z then F32.lt_transitive y z x
    else (F.eq_spec z y; F.eq_spec x y; F.eq_spec z x))
  else if lt z y && not (lt z x) then (
    if lt x z then F32.lt_transitive x z y
    else (F.eq_spec z x; F.eq_spec x y; F.eq_spec z y))

inline_for_extraction noextract
let strict_asymmetric (x y:F32.t)
  : Lemma (requires lt x y) (ensures not (lt y x)) =
  if lt y x then (
    F32.lt_transitive x y x;
    strict_irreflexive x)

inline_for_extraction noextract
let negative_infinity : F32.t = F.neg F.infinity

(* The native infinity and negation laws suffice; no finite sentinel bound. *)
inline_for_extraction noextract
let negative_infinity_below_finite (x:F32.t)
  : Lemma
      (requires F.is_finite x)
      (ensures lt negative_infinity x) =
  let inf : F32.t = F.infinity in
  let nx = F.neg x in
  F.neg_kind x;
  F.neg_kind inf;
  F.neg_kind nx;
  F.infinity_val_spec nx;
  F.lte_is_lt_or_eq nx inf;
  F.eq_spec nx inf;
  if F.is_zero inf then F.is_zero_spec inf;
  assert (lt nx inf);
  F.lt_neg_flip nx inf;
  F.neg_neg x;
  lt_right_congruent negative_infinity (F.neg nx) x

(* This ordinal embedding is independent of the reduction and its result. *)
inline_for_extraction noextract
let rec count_less (#n:pos) (s:scores n) (x:F32.t)
  (count:nat{count <= n})
  : GTot (r:nat{r <= count}) (decreases count) =
  if count = 0 then 0
  else count_less s x (count - 1) +
       (if lt (S.index s (count - 1)) x then 1 else 0)

inline_for_extraction noextract
let rec count_less_monotone (#n:pos) (s:scores n) (x y:F32.t)
  (count:nat{count <= n})
  : Lemma
      (requires forall (k:nat{k < n}).
        lt (S.index s k) x ==> lt (S.index s k) y)
      (ensures count_less s x count <= count_less s y count)
      (decreases count) =
  if count = 0 then ()
  else count_less_monotone s x y (count - 1)

inline_for_extraction noextract
let rec count_less_strict (#n:pos) (s:scores n) (x y:F32.t)
  (count:pos{count <= n}) (witness:nat{witness < count})
  : Lemma
      (requires
        (forall (k:nat{k < n}).
          lt (S.index s k) x ==> lt (S.index s k) y) /\
        not (lt (S.index s witness) x) /\ lt (S.index s witness) y)
      (ensures count_less s x count < count_less s y count)
      (decreases count) =
  if witness = count - 1 then
    count_less_monotone s x y (count - 1)
  else count_less_strict s x y (count - 1) witness

inline_for_extraction noextract
let rank_strict (#n:pos) (s:scores n) (i:nat{i < n}) (j:nat{j < n})
  : Lemma
      (requires lt (S.index s i) (S.index s j))
      (ensures
        count_less s (S.index s i) n < count_less s (S.index s j) n) =
  let x = S.index s i in
  let y = S.index s j in
  let implication (k:nat{k < n})
    : Lemma (lt (S.index s k) x ==> lt (S.index s k) y) =
    if lt (S.index s k) x then F32.lt_transitive (S.index s k) x y in
  C.forall_intro implication;
  strict_irreflexive x;
  count_less_strict s x y n i

inline_for_extraction noextract
let rank_equal (#n:pos) (s:scores n) (i:nat{i < n}) (j:nat{j < n})
  : Lemma
      (requires ordered_scores s /\ F.ieee_eq (S.index s i) (S.index s j))
      (ensures
        count_less s (S.index s i) n == count_less s (S.index s j) n) =
  let x = S.index s i in
  let y = S.index s j in
  ordered_trichotomy x y;
  let equivalence (k:nat{k < n})
    : Lemma (lt (S.index s k) x <==> lt (S.index s k) y) =
    lt_right_congruent (S.index s k) x y in
  C.forall_intro equivalence;
  count_less_monotone s x y n;
  count_less_monotone s y x n

inline_for_extraction noextract
let real_scores (#n:pos) (s:scores n) : GTot (lseq real n) =
  S.init_ghost n (fun j -> FStar.Real.of_int (count_less s (S.index s j) n))

inline_for_extraction noextract
let rank_preserves_comparisons (#n:pos) (s:scores n)
  (i:nat{i < n}) (j:nat{j < n})
  : Lemma
      (requires ordered_scores s)
      (ensures
        (lt (S.index s i) (S.index s j) <==>
          S.index (real_scores s) i <. S.index (real_scores s) j) /\
        (F.ieee_eq (S.index s i) (S.index s j) <==>
          S.index (real_scores s) i == S.index (real_scores s) j)) =
  ordered_trichotomy (S.index s i) (S.index s j);
  if lt (S.index s i) (S.index s j) then rank_strict s i j
  else if lt (S.index s j) (S.index s i) then rank_strict s j i
  else rank_equal s i j

(* These keys record the strict tree's actual tie order, not index order. *)
inline_for_extraction noextract
let visit_r2 (x:nat) : nat = 2 * (x % 2) + (x / 2) % 2

inline_for_extraction noextract
let visit_r3 (x:nat) : nat =
  4 * (x % 2) + 2 * ((x / 2) % 2) + (x / 4) % 2

inline_for_extraction noextract
let visit_r5 (x:nat) : nat =
  16 * (x % 2) + 8 * ((x / 2) % 2) + 4 * ((x / 4) % 2) +
  2 * ((x / 8) % 2) + (x / 16) % 2

inline_for_extraction noextract
let visit_cr (c:nat) : nat = 32 * visit_r3 (c / 32) + visit_r5 (c % 32)

inline_for_extraction noextract
let visit_tr (t:nat) : nat = 32 * visit_r2 (t / 32) + visit_r5 (t % 32)

inline_for_extraction noextract
let visit_key (j:nat) : nat =
  2048 * visit_cr (j / 1600) + 16 * visit_tr ((j % 1600) % 128) +
  (j % 1600) / 128

inline_for_extraction noextract
let native_winner (s:scores vocab_size) (winner:int) : GTot prop =
  if winner = -1 then
    forall (j:token_index). not (lt negative_infinity (S.index s j))
  else if 0 <= winner && winner < vocab_size then
    lt negative_infinity (S.index s winner) /\
    (forall (j:token_index).
      lt negative_infinity (S.index s j) ==>
      (lt (S.index s j) (S.index s winner) \/
       (F.ieee_eq (S.index s j) (S.index s winner) /\
        visit_key winner <= visit_key j)))
  else False

inline_for_extraction noextract
let maximal_winner (s:scores vocab_size) (winner:int) : GTot prop =
  if 0 <= winner && winner < vocab_size then
    forall (j:token_index). not (lt (S.index s winner) (S.index s j))
  else False

inline_for_extraction noextract
let finite_native_winner (s:scores vocab_size) (winner:int)
  : Lemma
      (requires finite_scores s /\ native_winner s winner)
      (ensures maximal_winner s winner) =
  if winner = -1 then negative_infinity_below_finite (S.index s 0)
  else if 0 <= winner && winner < vocab_size then (
    let maximal (j:token_index)
      : Lemma (not (lt (S.index s winner) (S.index s j))) =
      negative_infinity_below_finite (S.index s j);
      ordered_trichotomy (S.index s j) (S.index s winner);
      if lt (S.index s j) (S.index s winner) then
        strict_asymmetric (S.index s j) (S.index s winner) in
    C.forall_intro maximal)

inline_for_extraction noextract
let prompt_valid (prefix:physical_prompt) : prop =
  forall (j:nat{j < prompt_length}).
    0 <= I64.v (S.index prefix j) /\ I64.v (S.index prefix j) < vocab_size

inline_for_extraction noextract
let logical_prompt (prefix:physical_prompt{prompt_valid prefix})
  : Tot Shared.prompt_tokens =
  S.init #Shared.token_id prompt_length (fun j -> I64.v (S.index prefix j))

inline_for_extraction noextract
let physical_row (row:row_index) (logical_row:row_index) : Tot row_index =
  if logical_row = prompt_length - 1 then row
  else if logical_row = row then prompt_length - 1
  else logical_row

inline_for_extraction noextract
let physical_row_involution (row:row_index) (position:row_index)
  : Lemma
      (physical_row row (physical_row row position) == position /\
       physical_row row (prompt_length - 1) == row /\
       (row == prompt_length - 1 ==> physical_row row position == position)) =
  ()

inline_for_extraction noextract
let logical_logits (logits:physical_logits) (row:row_index)
  : GTot Shared.prompt_logits =
  S.init_ghost #Shared.vocabulary_logits prompt_length
    (fun position -> real_scores (row_scores logits (physical_row row position)))

inline_for_extraction noextract
let logical_final_row (logits:physical_logits) (row:row_index)
  : Lemma
      (Shared.final_logits (logical_logits logits row) ==
       real_scores (row_scores logits row)) =
  ()

inline_for_extraction noextract
let represents_prompt (prefix:physical_prompt) (prompt:Shared.prompt_tokens)
  : prop =
  forall (j:nat{j < prompt_length}).
    I64.v (S.index prefix j) == S.index prompt j

(* All faithful real interpretations are covered, not just the ordinal one. *)
inline_for_extraction noextract
let represents_scores (s:scores vocab_size) (logical:Shared.vocabulary_logits)
  : GTot prop =
  ordered_scores s /\
  (forall (i:token_index) (j:token_index).
    (lt (S.index s i) (S.index s j) <==>
      S.index logical i <. S.index logical j) /\
    (F.ieee_eq (S.index s i) (S.index s j) <==>
      S.index logical i == S.index logical j))

inline_for_extraction noextract
let represents_inputs
  (logits:physical_logits) (prefix:physical_prompt) (row:row_index)
  (prompt:Shared.prompt_tokens) (logical:Shared.prompt_logits)
  : GTot prop =
  represents_prompt prefix prompt /\
  finite_scores (row_scores logits row) /\
  represents_scores (row_scores logits row) (Shared.final_logits logical)

inline_for_extraction noextract
let canonical_input_mapping
  (logits:physical_logits) (prefix:physical_prompt{prompt_valid prefix})
  (row:row_index)
  : Lemma
      (requires finite_scores (row_scores logits row))
      (ensures represents_inputs logits prefix row
        (logical_prompt prefix) (logical_logits logits row)) =
  let s = row_scores logits row in
  let comparison (i:token_index) (j:token_index)
    : Lemma
        ((lt (S.index s i) (S.index s j) <==>
           S.index (real_scores s) i <. S.index (real_scores s) j) /\
         (F.ieee_eq (S.index s i) (S.index s j) <==>
           S.index (real_scores s) i == S.index (real_scores s) j)) =
    rank_preserves_comparisons s i j in
  C.forall_intro_2 comparison;
  logical_final_row logits row

(* Keep enclosing predicates folded during runtime-signature erasure. *)
noextract
let physical_postcondition
  (logits:physical_logits) (prefix:physical_prompt) (row:row_index)
  (output:physical_output)
  : GTot prop =
  (forall (j:nat{j < prompt_length}).
    S.index output j == S.index prefix j) /\
  native_winner (row_scores logits row) (I64.v (S.index output prompt_length))

inline_for_extraction noextract
let represents_output (output:physical_output) (logical:Shared.result) : prop =
  I64.v (S.index output prompt_length) == logical._1 /\
  (forall (j:nat{j < sequence_length}).
    I64.v (S.index output j) == S.index logical._2 j)

inline_for_extraction noextract
let winner_refines_real
  (s:scores vocab_size) (logical:Shared.vocabulary_logits) (winner:token_index)
  : Lemma
      (requires represents_scores s logical /\ maximal_winner s winner)
      (ensures Shared.maximal_index logical winner) =
  let maximal (j:token_index)
    : Lemma (S.index logical j <=. S.index logical winner) =
    assert (
      lt (S.index s winner) (S.index s j) <==>
      S.index logical winner <. S.index logical j) in
  C.forall_intro maximal

noextract
let shared_refinement
  (logits:physical_logits) (prefix:physical_prompt) (row:row_index)
  (output:physical_output)
  : GTot prop =
  forall (prompt:Shared.prompt_tokens) (logical:Shared.prompt_logits).
    represents_inputs logits prefix row prompt logical ==>
    (exists (result:Shared.result).
      represents_output output result /\
      Shared.postcondition prompt logical 1 result)

inline_for_extraction noextract
let physical_refines_shared
  (logits:physical_logits) (prefix:physical_prompt) (row:row_index)
  (output:physical_output)
  : Lemma
      (requires physical_postcondition logits prefix row output)
      (ensures shared_refinement logits prefix row output) =
  let refinement (prompt:Shared.prompt_tokens) (logical:Shared.prompt_logits)
    : Lemma
        (represents_inputs logits prefix row prompt logical ==>
         (exists (result:Shared.result).
           represents_output output result /\
           Shared.postcondition prompt logical 1 result)) =
    if t2b (represents_inputs logits prefix row prompt logical) then (
      finite_native_winner (row_scores logits row)
        (I64.v (S.index output prompt_length));
      let winner : token_index = I64.v (S.index output prompt_length) in
      winner_refines_real (row_scores logits row) (Shared.final_logits logical) winner;
      let result : Shared.result = (winner, Shared.append_token prompt winner) in
      Shared.maximal_index_append_allowed prompt logical 1 winner;
      Shared.postcondition_outputs prompt logical 1 result;
      assert (represents_output output result);
      C.exists_intro
        (fun result ->
          represents_output output result /\
          Shared.postcondition prompt logical 1 result)
        result) in
  C.forall_intro_2 refinement

noextract
let postcondition
  (logits:physical_logits) (prefix:physical_prompt) (row:row_index)
  (output:physical_output)
  : GTot prop =
  physical_postcondition logits prefix row output /\
  shared_refinement logits prefix row output

inline_for_extraction noextract
let establish_postcondition
  (logits:physical_logits) (prefix:physical_prompt) (row:row_index)
  (output:physical_output)
  : Lemma
      (requires physical_postcondition logits prefix row output)
      (ensures postcondition logits prefix row output) =
  physical_refines_shared logits prefix row output

inline_for_extraction noextract
let canonical_shared_result
  (logits:physical_logits) (prefix:physical_prompt{prompt_valid prefix})
  (row:row_index) (output:physical_output)
  : Lemma
      (requires
        finite_scores (row_scores logits row) /\
        physical_postcondition logits prefix row output)
      (ensures
        exists (result:Shared.result).
          represents_output output result /\
          Shared.postcondition
            (logical_prompt prefix) (logical_logits logits row) 1 result) =
  canonical_input_mapping logits prefix row;
  physical_refines_shared logits prefix row output

fn kplay_run
  (logits:larray BF16.t logits_size{is_global_array logits})
  (prefix:larray I64.t prompt_length{is_global_array prefix})
  (score_row:I64.t{0 <= I64.v score_row /\ I64.v score_row < prompt_length})
  (output:larray I64.t sequence_length{is_global_array output})
  (#input_logits:erased physical_logits)
  (#input_prefix:erased physical_prompt)
  preserves cpu **
    on gpu_loc (logits |-> input_logits) **
    on gpu_loc (prefix |-> input_prefix)
  requires exists* old. on gpu_loc (output |-> old)
  ensures exists* (result:physical_output).
    on gpu_loc (output |-> result) **
    pure (postcondition input_logits input_prefix (I64.v score_row) result)
