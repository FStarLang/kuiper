module KPlay.Functional

#lang-pulse

open Kuiper
open FStar.Real
module S = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let vocabulary : Shared.size = 248320

inline_for_extraction noextract
let position_t = x:i64{0 <= I64.v x /\ I64.v x < 2048}

inline_for_extraction noextract
let prompt_length_t = x:i64{1 <= I64.v x /\ I64.v x <= 2048}

inline_for_extraction noextract
let position_index (x:position_t) : nat = I64.v x

inline_for_extraction noextract
let prompt_size (x:prompt_length_t) : Shared.size = I64.v x

inline_for_extraction noextract
let previous_count (position:position_t) (prompt_length:prompt_length_t) : nat =
  Shared.retained_count (prompt_size prompt_length) (position_index position)

inline_for_extraction noextract
let output_count (position:position_t) (prompt_length:prompt_length_t) : nat =
  Shared.retained_count (prompt_size prompt_length) (position_index position + 1)

inline_for_extraction noextract
let widen (x:bf16) : f32 = Cast.cast_bf16_to_f32 x

inline_for_extraction noextract
let widened_scores (#v:Shared.size) (xs:Shared.vec bf16 v) : Shared.vec f32 v =
  S.init v (fun i -> widen (S.index xs i))

inline_for_extraction noextract
let ordered (x:f32) : GTot bool = not (F.is_nan x)

inline_for_extraction noextract
let positive_infinity : f32 = F.infinity

inline_for_extraction noextract
let negative_infinity : f32 = F.neg positive_infinity

inline_for_extraction noextract
let priority_le (x y:f32) : GTot bool =
  if F.is_nan x then F.is_nan y
  else F.is_nan y || not (F.lt y x)

inline_for_extraction noextract
let priority_equal (x y:f32) : GTot bool =
  priority_le x y && priority_le y x

inline_for_extraction noextract
let native_order (x y:f32)
  : Lemma (
      (F.lt x y ==> ordered x /\ ordered y) /\
      (F.lt y x ==> ordered x /\ ordered y) /\
      (F.ieee_eq x y ==> ordered x /\ ordered y) /\
      (F.ieee_eq x y <==> F.ieee_eq y x) /\
      (ordered x /\ ordered y ==>
        (F.lt x y <==> ~(F.lt y x \/ F.ieee_eq x y)) /\
        (F.ieee_eq x y <==> ~(F.lt x y \/ F.lt y x)))) =
  F.eq_spec x y;
  F.eq_spec y x;
  if F.lt x y then F32.lt_ordered x y;
  if F.lt y x then F32.lt_ordered y x;
  if ordered x && ordered y then (
    F.lte_is_lt_or_eq x y;
    F.lte_is_lt_or_eq y x;
    F.negate_lt_is_lte x y;
    F.negate_lt_is_lte y x
  )

inline_for_extraction noextract
let native_irreflexive (x:f32) : Lemma (~(F.lt x x)) =
  native_order x x;
  F.eq_spec x x

inline_for_extraction noextract
let ieee_equal_transitive (x y z:f32)
  : Lemma
      (requires F.ieee_eq x y /\ F.ieee_eq y z)
      (ensures F.ieee_eq x z) =
  F.eq_spec x y;
  F.eq_spec y z;
  F.eq_spec x z

// Numerical equality is used only for comparisons, never as bit equality.
inline_for_extraction noextract
let ieee_equal_comparisons (x y z:f32)
  : Lemma
      (requires F.ieee_eq x y)
      (ensures
        F.lt z x == F.lt z y /\
        F.lt x z == F.lt y z) =
  native_order x y;
  native_order x z;
  native_order y z;
  if F.lt x z && F.lt z y then F32.lt_transitive x z y;
  if F.lt y z && F.lt z x then F32.lt_transitive y z x;
  if F.ieee_eq y z then ieee_equal_transitive x y z;
  if F.ieee_eq x z then ieee_equal_transitive y x z

inline_for_extraction noextract
let upper_bound (x:f32)
  : Lemma
      (requires ordered x)
      (ensures ~(F.lt positive_infinity x)) =
  F.infinity_val_spec x;
  F.negate_lt_is_lte positive_infinity x

inline_for_extraction noextract
let lower_bound (x:f32)
  : Lemma
      (requires ordered x)
      (ensures ~(F.lt x negative_infinity)) =
  F.neg_kind positive_infinity;
  F.neg_kind x;
  F.neg_kind (F.neg x);
  upper_bound (F.neg x);
  F.lt_neg_flip positive_infinity (F.neg x);
  F.neg_neg x;
  ieee_equal_comparisons (F.neg (F.neg x)) x negative_infinity

inline_for_extraction noextract
let infinity_bounds ()
  : Lemma (
      ordered positive_infinity /\ ordered negative_infinity /\
      F.lt negative_infinity (F.zero #f32) /\
      F.lt (F.zero #f32) positive_infinity /\
      F.lt negative_infinity positive_infinity) =
  F.neg_kind positive_infinity;
  if F.is_zero positive_infinity then F.is_zero_spec positive_infinity;
  if F.is_zero negative_infinity then F.is_zero_spec negative_infinity;
  F.eq_spec (F.zero #f32) positive_infinity;
  F.eq_spec negative_infinity (F.zero #f32);
  upper_bound (F.zero #f32);
  lower_bound (F.zero #f32);
  native_order (F.zero #f32) positive_infinity;
  native_order negative_infinity (F.zero #f32);
  F32.lt_transitive negative_infinity (F.zero #f32) positive_infinity

inline_for_extraction noextract
let rec count_less
  (#v:Shared.size) (xs:Shared.vec f32 v) (n:nat{n <= v}) (x:f32)
  : Tot (k:nat{k <= n}) (decreases n) =
  if n = 0 then 0
  else count_less xs (n - 1) x +
    (if F.lt (S.index xs (n - 1)) x then 1 else 0)

inline_for_extraction noextract
let rec count_less_monotone
  (#v:Shared.size) (xs:Shared.vec f32 v) (n:nat{n <= v}) (x y:f32)
  : Lemma
      (requires F.lt x y \/ F.ieee_eq x y)
      (ensures count_less xs n x <= count_less xs n y)
      (decreases n) =
  if n = 0 then ()
  else (
    count_less_monotone xs (n - 1) x y;
    let z = S.index xs (n - 1) in
    if F.lt x y && F.lt z x then F32.lt_transitive z x y;
    if F.ieee_eq x y then ieee_equal_comparisons x y z
  )

inline_for_extraction noextract
let rec count_less_strict
  (#v:Shared.size) (xs:Shared.vec f32 v) (n:Shared.size{n <= v})
  (i:nat{i < n}) (y:f32)
  : Lemma
      (requires F.lt (S.index xs i) y)
      (ensures count_less xs n (S.index xs i) < count_less xs n y)
      (decreases n) =
  let x = S.index xs i in
  if i = n - 1 then (
    count_less_monotone xs (n - 1) x y;
    native_irreflexive x
  )
  else (
    count_less_strict xs (n - 1) i y;
    let z = S.index xs (n - 1) in
    if F.lt z x then F32.lt_transitive z x y
  )

// A signed ordinal embedding of this entire row, anchored at numerical zero.
// It is permutation-invariant and preserves order/ties, not score magnitudes.
inline_for_extraction noextract
let finite_rank (#v:Shared.size) (xs:Shared.vec f32 v) (x:f32) : int =
  count_less xs v x - count_less xs v (F.zero #f32) +
  (if F.lt (F.zero #f32) x then 1 else 0)

inline_for_extraction noextract
let finite_rank_equal
  (#v:Shared.size) (xs:Shared.vec f32 v) (x y:f32)
  : Lemma
      (requires F.ieee_eq x y)
      (ensures finite_rank xs x == finite_rank xs y) =
  native_order x y;
  count_less_monotone xs v x y;
  count_less_monotone xs v y x;
  ieee_equal_comparisons x y (F.zero #f32)

inline_for_extraction noextract
let finite_rank_strict
  (#v:Shared.size) (xs:Shared.vec f32 v) (i:nat{i < v}) (y:f32)
  : Lemma
      (requires F.lt (S.index xs i) y)
      (ensures finite_rank xs (S.index xs i) < finite_rank xs y) =
  let x = S.index xs i in
  count_less_strict xs v i y;
  if F.lt (F.zero #f32) x then F32.lt_transitive (F.zero #f32) x y

inline_for_extraction noextract
let finite_rank_zero
  (#v:Shared.size) (xs:Shared.vec f32 v) (x:f32)
  : Lemma
      (requires F.is_zero x)
      (ensures finite_rank xs x == 0) =
  F.is_zero_spec x;
  F.eq_spec x (F.zero #f32);
  native_irreflexive (F.zero #f32);
  finite_rank_equal xs x (F.zero #f32)

inline_for_extraction noextract
let score_class (x:f32) : GTot nat =
  if F.is_nan x then 3
  else if F.ieee_eq x negative_infinity then 0
  else if F.ieee_eq x positive_infinity then 2
  else 1

inline_for_extraction noextract
let score_key (#v:Shared.size) (xs:Shared.vec f32 v) (x:f32) : GTot Shared.score =
  if F.is_nan x then Shared.NotANumber
  else if F.ieee_eq x negative_infinity then Shared.NegativeInfinity
  else if F.ieee_eq x positive_infinity then Shared.PositiveInfinity
  else Shared.FiniteScore (FStar.Real.of_int (finite_rank xs x))

inline_for_extraction noextract
let score_class_bounds (x:f32)
  : Lemma (
      (score_class x == 3 <==> F.is_nan x) /\
      (score_class x == 0 ==> F.ieee_eq x negative_infinity) /\
      (score_class x == 2 ==> F.ieee_eq x positive_infinity) /\
      (score_class x == 1 ==>
        F.lt negative_infinity x /\ F.lt x positive_infinity)) =
  infinity_bounds ();
  if ordered x then (
    upper_bound x;
    lower_bound x;
    native_order negative_infinity x;
    native_order x positive_infinity
  )

inline_for_extraction noextract
let score_key_order
  (#v:Shared.size) (xs:Shared.vec f32 v) (i:nat{i < v}) (j:nat{j < v})
  : Lemma (
      Shared.score_le (score_key xs (S.index xs i)) (score_key xs (S.index xs j))
      <==> priority_le (S.index xs i) (S.index xs j)) =
  let x = S.index xs i in
  let y = S.index xs j in
  score_class_bounds x;
  score_class_bounds y;
  native_order x y;
  if F.ieee_eq x negative_infinity then
    ieee_equal_comparisons x negative_infinity y;
  if F.ieee_eq y negative_infinity then
    ieee_equal_comparisons y negative_infinity x;
  if F.ieee_eq x positive_infinity then
    ieee_equal_comparisons x positive_infinity y;
  if F.ieee_eq y positive_infinity then
    ieee_equal_comparisons y positive_infinity x;
  if F.ieee_eq x y then finite_rank_equal xs x y;
  if F.lt x y then finite_rank_strict xs i y;
  if F.lt y x then finite_rank_strict xs j x

inline_for_extraction noextract
let logical_scores
  (#v:Shared.size) (xs:Shared.vec bf16 v) : GTot (Shared.vec Shared.score v) =
  let values = widened_scores xs in
  S.init_ghost v (fun i -> score_key values (S.index values i))

inline_for_extraction noextract
let score_key_class
  (#v:Shared.size) (xs:Shared.vec f32 v) (x:f32)
  : Lemma (Shared.score_rank (score_key xs x) == score_class x) =
  ()

inline_for_extraction noextract
let score_key_zero
  (#v:Shared.size) (xs:Shared.vec f32 v) (x:f32)
  : Lemma
      (requires F.is_zero x)
      (ensures score_key xs x == Shared.FiniteScore 0.0R) =
  F.is_zero_spec x;
  F.neg_kind positive_infinity;
  if F.is_zero positive_infinity then F.is_zero_spec positive_infinity;
  if F.is_zero negative_infinity then F.is_zero_spec negative_infinity;
  F.eq_spec x positive_infinity;
  F.eq_spec x negative_infinity;
  finite_rank_zero xs x

// The canonical ordinal keys witness totality. Other real-valued keys are
// allowed only when they preserve the same categories, comparisons and ties.
inline_for_extraction noextract
let scores_represent
  (#v:Shared.size) (physical:Shared.vec bf16 v) (logical:Shared.vec Shared.score v)
  : prop =
  let values = widened_scores physical in
  (forall (i:nat{i < v}).
    Shared.score_rank (S.index logical i) == score_class (S.index values i) /\
    (F.is_zero (S.index values i) ==> S.index logical i == Shared.FiniteScore 0.0R)) /\
  (forall (i:nat{i < v}) (j:nat{j < v}).
    Shared.score_le (S.index logical i) (S.index logical j) <==>
    priority_le (S.index values i) (S.index values j))

inline_for_extraction noextract
let logical_scores_represent
  (#v:Shared.size) (physical:Shared.vec bf16 v)
  : Lemma (scores_represent physical (logical_scores physical)) =
  let values = widened_scores physical in
  let cells (i:nat{i < v})
    : Lemma (
        Shared.score_rank (S.index (logical_scores physical) i) ==
          score_class (S.index values i) /\
        (F.is_zero (S.index values i) ==>
          S.index (logical_scores physical) i == Shared.FiniteScore 0.0R)) =
    score_key_class values (S.index values i);
    if F.is_zero (S.index values i) then score_key_zero values (S.index values i)
  in
  FStar.Classical.forall_intro cells;
  let pairs (i:nat{i < v}) (j:nat{j < v})
    : Lemma (
        Shared.score_le
          (S.index (logical_scores physical) i)
          (S.index (logical_scores physical) j) <==>
        priority_le (S.index values i) (S.index values j)) =
    score_key_order values i j
  in
  FStar.Classical.forall_intro_2 pairs

inline_for_extraction noextract
let native_first_maximum
  (#v:Shared.size) (scores:Shared.vec bf16 v) (winner:Shared.token v) : prop =
  let values = widened_scores scores in
  forall (j:nat{j < v}).
    priority_le (S.index values j) (S.index values winner) /\
    (priority_equal (S.index values j) (S.index values winner) ==> winner <= j)

inline_for_extraction noextract
let native_prefix_correct
  (#v:Shared.size) (scores:Shared.vec bf16 v)
  (n:Shared.size{n <= v}) (winner:nat{winner < n})
  : Lemma
      (requires winner == Shared.argmax_prefix (logical_scores scores) n)
      (ensures (
        let values = widened_scores scores in
        forall (j:nat{j < n}).
          priority_le (S.index values j) (S.index values winner) /\
          (priority_equal (S.index values j) (S.index values winner) ==> winner <= j))) =
  Shared.argmax_prefix_correct (logical_scores scores) n;
  logical_scores_represent scores

inline_for_extraction noextract
let native_maximum
  (#v:Shared.size) (scores:Shared.vec bf16 v) (winner:Shared.token v) : prop =
  let values = widened_scores scores in
  forall (j:Shared.token v). priority_le (S.index values j) (S.index values winner)

inline_for_extraction noextract
let any_maximum_representation
  (#v:Shared.size) (physical:Shared.vec bf16 v)
  (logical:Shared.vec Shared.score v) (winner:Shared.token v)
  : Lemma
      (requires scores_represent physical logical)
      (ensures native_maximum physical winner <==> Shared.maximum logical winner) =
  ()

inline_for_extraction noextract
let maximum_representation
  (#v:Shared.size) (physical:Shared.vec bf16 v)
  (logical:Shared.vec Shared.score v) (winner:Shared.token v)
  : Lemma
      (requires scores_represent physical logical)
      (ensures
        native_first_maximum physical winner <==>
        Shared.first_maximum logical winner) =
  ()

inline_for_extraction noextract
let accumulated_tokens
  (#p:Shared.size) (completed:nat)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  : Shared.vec i64 (Shared.retained_count p (completed + 1)) =
  if completed < p - 1 then previous else S.append previous selected

inline_for_extraction noextract
let functional_post
  (#v:Shared.size) (p:Shared.size) (completed:nat)
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  : prop =
  (exists (winner:Shared.token v).
    I64.v (S.index selected 0) == winner /\
    Shared.first_maximum (logical_scores scores) winner) /\
  emitted == accumulated_tokens #p completed previous selected

inline_for_extraction noextract
let native_maximum_refines
  (#v:Shared.size) (p:Shared.size) (completed:nat)
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (winner:Shared.token v)
  : Lemma
      (requires
        I64.v (S.index selected 0) == winner /\
        native_first_maximum scores winner /\
        emitted == accumulated_tokens #p completed previous selected)
      (ensures functional_post p completed scores previous selected emitted) =
  logical_scores_represent scores;
  maximum_representation scores (logical_scores scores) winner

inline_for_extraction noextract
let selected_representation
  (#v:Shared.size) (p:Shared.size) (completed:nat)
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (logical:Shared.vec Shared.score v)
  : Lemma
      (requires
        functional_post p completed scores previous selected emitted /\
        scores_represent scores logical)
      (ensures I64.v (S.index selected 0) == Shared.argmax logical) =
  let winner : Shared.token v = I64.v (S.index selected 0) in
  logical_scores_represent scores;
  maximum_representation scores (logical_scores scores) winner;
  maximum_representation scores logical winner;
  Shared.argmax_correct logical;
  Shared.first_maximum_unique logical winner (Shared.argmax logical)

inline_for_extraction noextract
let selected_maximum
  (#v:Shared.size) (p:Shared.size) (completed:nat)
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (logical:Shared.vec Shared.score v) (choice:Shared.token v)
  : Lemma
      (requires
        functional_post p completed scores previous selected emitted /\
        scores_represent scores logical /\
        I64.v (S.index selected 0) == choice)
      (ensures Shared.maximum logical choice /\ native_maximum scores choice) =
  selected_representation p completed scores previous selected emitted logical;
  Shared.argmax_correct logical;
  Shared.first_maximum_refines logical choice;
  any_maximum_representation scores logical choice

inline_for_extraction noextract
let specification_total
  (p:Shared.size) (completed:nat)
  (scores:Shared.vec bf16 vocabulary)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  : Lemma (
      exists (selected:Shared.vec i64 1)
             (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1))).
        functional_post p completed scores previous selected emitted) =
  let keys = logical_scores scores in
  let winner = Shared.argmax keys in
  Shared.argmax_correct keys;
  let selected = S.create 1 (I64.int_to_t winner) in
  let emitted = accumulated_tokens #p completed previous selected in
  assert (functional_post p completed scores previous selected emitted)

inline_for_extraction noextract
let tokens_represent
  (#v:Shared.size) (#count:nat)
  (physical:Shared.vec i64 count) (logical:Shared.vec (Shared.token v) count)
  : prop =
  forall (j:nat{j < count}). I64.v (S.index physical j) == S.index logical j

inline_for_extraction noextract
let tokens_represent_append
  (#v:Shared.size) (#count:nat)
  (physical:Shared.vec i64 count) (logical:Shared.vec (Shared.token v) count)
  (selected:Shared.vec i64 1) (winner:Shared.token v)
  : Lemma
      (requires
        tokens_represent #v #count physical logical /\ I64.v (S.index selected 0) == winner)
      (ensures
        tokens_represent #v #(count + 1)
          (S.append physical selected) (S.append logical (S.create 1 winner))) =
  ()

inline_for_extraction noextract
let input_representation
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#completed:nat)
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (before:Shared.decode_state v h bf16 weights cache p completed)
  (observed:Shared.observation v h bf16 cache)
  : prop =
  scores_represent scores observed.logits /\
  tokens_represent #v #(Shared.retained_count p completed)
    previous before.returned.generated_tokens

inline_for_extraction noextract
let expected_hidden_frame
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#completed:nat)
  (before:Shared.decode_state v h bf16 weights cache p completed)
  (observed:Shared.observation v h bf16 cache)
  : Shared.vec (Shared.vec bf16 h) (Shared.retained_count p (completed + 1)) =
  if completed < p - 1 then before.returned.hidden_history
  else S.append before.returned.hidden_history (S.create 1 observed.residual)

inline_for_extraction noextract
let hidden_frame
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#completed:nat)
  (before:Shared.decode_state v h bf16 weights cache p completed)
  (observed:Shared.observation v h bf16 cache)
  (actual:Shared.vec (Shared.vec bf16 h) (Shared.retained_count p (completed + 1)))
  : prop =
  actual == expected_hidden_frame before observed

inline_for_extraction noextract
let output_representation
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#completed:nat)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h bf16 weights cache p completed)
  (observed:Shared.observation v h bf16 cache)
  (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec bf16 h) (Shared.retained_count p (completed + 1)))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (after:Shared.decode_state v h bf16 weights cache p (completed + 1))
  : prop =
  I64.v (S.index selected 0) == choice /\
  tokens_represent #v #(Shared.retained_count p (completed + 1))
    emitted after.returned.generated_tokens /\
  after.returned.hidden_history == hidden_after /\
  after.position == completed + 1 /\
  after.parameters == before.parameters /\
  after.model_cache == observed.cache_after /\
  after.captured == Some observed.residual /\
  (if completed + 1 < p then after.next_token == S.index prompt (completed + 1)
   else after.next_token == choice)

inline_for_extraction noextract
let post_refines_advance
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#completed:nat)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h bf16 weights cache p completed)
  (observed:Shared.observation v h bf16 cache)
  (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec bf16 h) (Shared.retained_count p (completed + 1)))
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  : Lemma
      (requires
        functional_post p completed scores previous selected emitted /\
        input_representation scores previous before observed /\
        hidden_frame before observed hidden_after /\
        I64.v (S.index selected 0) == choice)
      (ensures
        Shared.maximum observed.logits choice /\
        output_representation prompt before observed choice hidden_after selected emitted
          (Shared.advance prompt (Shared.prepare_call before) observed choice)) =
  selected_maximum p completed scores previous selected emitted observed.logits choice;
  if completed < p - 1 then ()
  else
    tokens_represent_append #v #(Shared.retained_count p completed)
      previous before.returned.generated_tokens
      selected choice

inline_for_extraction noextract
let output_representation_unique
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#completed:nat)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h bf16 weights cache p completed)
  (observed:Shared.observation v h bf16 cache) (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec bf16 h) (Shared.retained_count p (completed + 1)))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (left right:Shared.decode_state v h bf16 weights cache p (completed + 1))
  : Lemma
      (requires
        output_representation prompt before observed choice hidden_after selected emitted left /\
        output_representation prompt before observed choice hidden_after selected emitted right)
      (ensures left == right) =
  assert (S.equal left.returned.generated_tokens right.returned.generated_tokens);
  S.lemma_eq_elim left.returned.generated_tokens right.returned.generated_tokens

inline_for_extraction noextract
let post_refines_step
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#completed:nat)
  (model:Shared.model_operation v h bf16 weights cache)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h bf16 weights cache p completed)
  (observed:Shared.observation v h bf16 cache) (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec bf16 h) (Shared.retained_count p (completed + 1)))
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (after:Shared.decode_state v h bf16 weights cache p (completed + 1))
  : Lemma
      (requires
        functional_post p completed scores previous selected emitted /\
        input_representation scores previous before observed /\
        hidden_frame before observed hidden_after /\
        output_representation prompt before observed choice hidden_after selected emitted after /\
        model before.parameters before.model_cache before.position before.next_token ==
          Some observed)
      (ensures Shared.step_post model prompt before observed choice after) =
  post_refines_advance prompt before observed choice hidden_after scores previous selected emitted;
  output_representation_unique prompt before observed choice hidden_after selected emitted
    after (Shared.advance prompt (Shared.prepare_call before) observed choice)

inline_for_extraction noextract
let physical_trace_post
  (#v #h:Shared.size) (#cache:Type0) (#count:Shared.size)
  (p:Shared.size)
  (events:Shared.vec (Shared.observation v h bf16 cache) count)
  (scores:Shared.vec (Shared.vec bf16 v) count)
  (previous:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p t)))
  (selected:Shared.vec (Shared.vec i64 1) count)
  (emitted:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  (choices:Shared.vec (Shared.token v) count)
  : prop =
  forall (t:nat{t < count}).
    functional_post p t (S.index scores t) (previous t) (S.index selected t) (emitted t) /\
    scores_represent (S.index scores t) (S.index events t).logits /\
    I64.v (S.index (S.index selected t) 0) == S.index choices t

inline_for_extraction noextract
let prefixes_linked
  (p count:Shared.size)
  (previous:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p t)))
  (emitted:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  : prop =
  forall (t:nat{0 < t /\ t < count}). previous t == emitted (t - 1)

inline_for_extraction noextract
let posts_refine_selections
  (#v #h:Shared.size) (#cache:Type0) (#count:Shared.size)
  (p:Shared.size)
  (events:Shared.vec (Shared.observation v h bf16 cache) count)
  (scores:Shared.vec (Shared.vec bf16 v) count)
  (previous:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p t)))
  (selected:Shared.vec (Shared.vec i64 1) count)
  (emitted:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  (choices:Shared.vec (Shared.token v) count)
  : Lemma
      (requires physical_trace_post p events scores previous selected emitted choices)
      (ensures Shared.selections_admissible events choices) =
  let call_maximum (t:nat{t < count})
    : Lemma (Shared.maximum (S.index events t).logits (S.index choices t)) =
    selected_maximum p t (S.index scores t) (previous t) (S.index selected t) (emitted t)
      (S.index events t).logits (S.index choices t)
  in
  FStar.Classical.forall_intro call_maximum

inline_for_extraction noextract
let rec trace_prefix_correct
  (#v #h #p:Shared.size) (#weights #cache:Type0) (#count:Shared.size)
  (prompt:Shared.vec (Shared.token v) p) (parameters:weights) (initial_cache:cache)
  (events:Shared.vec (Shared.observation v h bf16 cache) count)
  (scores:Shared.vec (Shared.vec bf16 v) count)
  (previous:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p t)))
  (selected:Shared.vec (Shared.vec i64 1) count)
  (emitted:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  (choices:Shared.vec (Shared.token v) count)
  (t:nat{t < count})
  : Lemma
      (requires
        physical_trace_post p events scores previous selected emitted choices /\
        prefixes_linked p count previous emitted)
      (ensures
        tokens_represent #v (previous t)
          (Shared.state_at prompt parameters initial_cache events choices t)
            .returned.generated_tokens /\
        tokens_represent #v (emitted t)
          (Shared.state_at prompt parameters initial_cache events choices (t + 1))
            .returned.generated_tokens)
      (decreases t) =
  if t > 0 then
    trace_prefix_correct prompt parameters initial_cache events scores
      previous selected emitted choices (t - 1);
  let before = Shared.state_at prompt parameters initial_cache events choices t in
  let observed = S.index events t in
  post_refines_advance prompt before observed (S.index choices t)
    (expected_hidden_frame before observed)
    (S.index scores t) (previous t) (S.index selected t) (emitted t);
  Shared.state_at_successor prompt parameters initial_cache events choices t

inline_for_extraction noextract
let last_call (p g:Shared.size) : nat = Shared.call_count p g - 1

// All choices are bound to actual scalar outputs; prefix correctness follows
// from the physical calls and their links, not a presumed token history.
inline_for_extraction noextract
let post_refines_decode
  (#v #h #p #g:Shared.size) (#weights #cache:Type0)
  (model:Shared.model_operation v h bf16 weights cache)
  (prompt:Shared.vec (Shared.token v) p) (parameters:weights) (initial_cache:cache)
  (events:Shared.vec (Shared.observation v h bf16 cache) (Shared.call_count p g))
  (scores:Shared.vec (Shared.vec bf16 v) (Shared.call_count p g))
  (previous:(t:nat{t < Shared.call_count p g} -> Shared.vec i64 (Shared.retained_count p t)))
  (selected:Shared.vec (Shared.vec i64 1) (Shared.call_count p g))
  (emitted:(t:nat{t < Shared.call_count p g} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  (choices:Shared.vec (Shared.token v) (Shared.call_count p g))
  (hidden_after:Shared.vec (Shared.vec bf16 h) g)
  (output:Shared.histories v h bf16 g)
  (after:Shared.decode_state v h bf16 weights cache p (Shared.call_count p g))
  : Lemma
      (requires (
        let completed = last_call p g in
        let before = Shared.state_at prompt parameters initial_cache events choices completed in
        let observed = S.index events completed in
        physical_trace_post p events scores previous selected emitted choices /\
        prefixes_linked p (Shared.call_count p g) previous emitted /\
        Shared.observations_from_calls model prompt parameters initial_cache events choices /\
        hidden_frame before observed hidden_after /\
        output_representation prompt before observed (S.index choices completed) hidden_after
          (S.index selected completed) (emitted completed) after /\
        output == after.returned))
      (ensures
        Shared.selected_decode_post model prompt parameters initial_cache events choices output after /\
        Shared.decode_post model prompt parameters initial_cache events output after /\
        tokens_represent #v (emitted (last_call p g)) output.generated_tokens /\
        output.hidden_history == hidden_after) =
  let completed = last_call p g in
  let before = Shared.state_at prompt parameters initial_cache events choices completed in
  let observed = S.index events completed in
  posts_refine_selections p events scores previous selected emitted choices;
  trace_prefix_correct prompt parameters initial_cache events scores
    previous selected emitted choices completed;
  post_refines_step model prompt before observed (S.index choices completed) hidden_after
    (S.index scores completed) (previous completed) (S.index selected completed)
    (emitted completed) after;
  Shared.state_at_successor prompt parameters initial_cache events choices completed;
  Shared.choices_realize_contract #v #h #p #g #bf16 #weights #cache
    model prompt parameters initial_cache events choices

inline_for_extraction noextract
let reference_domain_covered
  (smoke:bool) (h p g:Shared.size) (completed:nat{completed < Shared.call_count p g})
  : Lemma
      (requires Shared.reference_domain smoke vocabulary h p g)
      (ensures
        completed < 2048 /\ 1 <= p /\ p <= 2048 /\
        Shared.retained_count p completed <= 2048 /\
        Shared.retained_count p (completed + 1) <= 2048) =
  ()

inline_for_extraction noextract
let active_global_array (#a:Type0) (n:nat) (arr:array a) : prop =
  n > 0 ==> (A.length arr == n /\ is_global_array arr)

inline_for_extraction noextract
let active_contents
  (#a:Type0) ([@@@mkey] arr:array a) (#f:perm) (values:S.seq a)
  : slprop =
  if S.length values = 0 then emp else A.pts_to arr #f values

inline_for_extraction noextract
let exclusive_input_permission = f:perm{f == 1.0R}

// External caller layout obligation; the unchanged wrapper does not check it.
// Let N = previous_count position prompt_length and E = output_count position prompt_length.
// The NONEMPTY half-open byte ranges must be pairwise disjoint:
// scores [s,s+2*vocabulary), previous [p,p+8*N), selected [o,o+8), emitted [e,e+8*E).
// BF16 cells occupy 2 bytes; int64 cells occupy 8. Disjoint subranges may share an allocation.
// Full separated typed ownership below represents this conditional physical boundary,
// not a heterogeneous byte-view conversion. Empty prefixes have emp ownership and
// no length, location or non-null requirement on their inactive pointers.
fn kplay_run
  (position:position_t)
  (prompt_length:prompt_length_t)
  (#score_fraction #previous_fraction:exclusive_input_permission)
  (#score_values:erased (Shared.vec bf16 vocabulary))
  (#previous_values:erased (Shared.vec i64 (previous_count position prompt_length)))
  (scores:larray bf16 vocabulary{is_global_array scores})
  (previous:array i64{active_global_array (previous_count position prompt_length) previous})
  (selected:larray i64 1{is_global_array selected})
  (emitted:array i64{active_global_array (output_count position prompt_length) emitted})
  preserves cpu **
    on gpu_loc (
      A.pts_to scores #score_fraction (FStar.Ghost.reveal score_values) **
      active_contents previous #previous_fraction (FStar.Ghost.reveal previous_values))
  requires
    exists* (old_selected:Shared.vec i64 1)
            (old_emitted:Shared.vec i64 (output_count position prompt_length)).
      on gpu_loc (A.pts_to selected old_selected ** active_contents emitted #1.0R old_emitted)
  ensures
    exists* (selected_values:Shared.vec i64 1)
            (emitted_values:Shared.vec i64 (output_count position prompt_length)).
      on gpu_loc (A.pts_to selected selected_values ** active_contents emitted #1.0R emitted_values) **
      pure (functional_post
        (prompt_size prompt_length) (position_index position)
        (FStar.Ghost.reveal score_values) (FStar.Ghost.reveal previous_values)
        selected_values emitted_values)
