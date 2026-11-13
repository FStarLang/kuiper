module KPlay.Functional

#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let vocabulary_size : Shared.size = 248320

inline_for_extraction noextract
type score_vector = lseq f32 vocabulary_size

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

// Native comparison/category helpers reuse the accepted BASE prior art.
// The supplied storage here remains FP32; no BF16 widening is performed.
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

inline_for_extraction noextract
let ieee_equal_comparisons (x y z:f32)
  : Lemma
      (requires F.ieee_eq x y)
      (ensures F.lt z x == F.lt z y /\ F.lt x z == F.lt y z) =
  native_order x y;
  native_order x z;
  native_order y z;
  if F.lt x z && F.lt z y then F32.lt_transitive x z y;
  if F.lt y z && F.lt z x then F32.lt_transitive y z x;
  if F.ieee_eq y z then ieee_equal_transitive x y z;
  if F.ieee_eq x z then ieee_equal_transitive y x z

inline_for_extraction noextract
let priority_reflexive (x:f32) : Lemma (priority_le x x) =
  native_irreflexive x

inline_for_extraction noextract
let priority_total (x y:f32) : Lemma (priority_le x y \/ priority_le y x) =
  native_order x y

inline_for_extraction noextract
let priority_transitive (x y z:f32)
  : Lemma
      (requires priority_le x y /\ priority_le y z)
      (ensures priority_le x z) =
  native_order x y;
  if ordered x && ordered y && ordered z then (
    if F.ieee_eq x y then ieee_equal_comparisons x y z
    else if F.lt z x then F32.lt_transitive z x y
  )

inline_for_extraction noextract
let better_candidate (incoming:f32) (incoming_index:int) (current:f32) (current_index:int)
  : GTot bool =
  incoming_index >= 0 && (current_index < 0 || not (priority_le incoming current))

inline_for_extraction noextract
let replace_best (incoming:f32) (incoming_index:i64) (current:f32) (current_index:i64)
  : result:bool{
      result == better_candidate incoming (I64.v incoming_index) current (I64.v current_index)} =
  native_order incoming current;
  F.eq_spec incoming incoming;
  F.eq_spec current current;
  I64.gte incoming_index 0L &&
    (I64.lt current_index 0L || F.lt current incoming ||
      (not (F.ieee_eq incoming incoming) && F.ieee_eq current current))

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
    (if F.lt (Seq.index xs (n - 1)) x then 1 else 0)

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
    let z = Seq.index xs (n - 1) in
    if F.lt x y && F.lt z x then F32.lt_transitive z x y;
    if F.ieee_eq x y then ieee_equal_comparisons x y z
  )

inline_for_extraction noextract
let rec count_less_strict
  (#v:Shared.size) (xs:Shared.vec f32 v) (n:Shared.size{n <= v})
  (i:nat{i < n}) (y:f32)
  : Lemma
      (requires F.lt (Seq.index xs i) y)
      (ensures count_less xs n (Seq.index xs i) < count_less xs n y)
      (decreases n) =
  let x = Seq.index xs i in
  if i = n - 1 then (
    count_less_monotone xs (n - 1) x y;
    native_irreflexive x
  )
  else (
    count_less_strict xs (n - 1) i y;
    let z = Seq.index xs (n - 1) in
    if F.lt z x then F32.lt_transitive z x y
  )

// These permutation-invariant ordinal keys preserve this row's order, not magnitudes.
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
      (requires F.lt (Seq.index xs i) y)
      (ensures finite_rank xs (Seq.index xs i) < finite_rank xs y) =
  let x = Seq.index xs i in
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
      Shared.score_le (score_key xs (Seq.index xs i)) (score_key xs (Seq.index xs j))
      <==> priority_le (Seq.index xs i) (Seq.index xs j)) =
  let x = Seq.index xs i in
  let y = Seq.index xs j in
  score_class_bounds x;
  score_class_bounds y;
  native_order x y;
  if F.ieee_eq x negative_infinity then ieee_equal_comparisons x negative_infinity y;
  if F.ieee_eq y negative_infinity then ieee_equal_comparisons y negative_infinity x;
  if F.ieee_eq x positive_infinity then ieee_equal_comparisons x positive_infinity y;
  if F.ieee_eq y positive_infinity then ieee_equal_comparisons y positive_infinity x;
  if F.ieee_eq x y then finite_rank_equal xs x y;
  if F.lt x y then finite_rank_strict xs i y;
  if F.lt y x then finite_rank_strict xs j x

inline_for_extraction noextract
let logical_scores
  (#v:Shared.size) (xs:Shared.vec f32 v) : GTot (Shared.vec Shared.score v) =
  Seq.init_ghost v (fun i -> score_key xs (Seq.index xs i))

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

inline_for_extraction noextract
let scores_represent
  (#v:Shared.size) (physical:Shared.vec f32 v) (logical:Shared.vec Shared.score v)
  : prop =
  (forall (i:nat{i < v}).
    Shared.score_rank (Seq.index logical i) == score_class (Seq.index physical i) /\
    (F.is_zero (Seq.index physical i) ==> Seq.index logical i == Shared.FiniteScore 0.0R)) /\
  (forall (i:nat{i < v}) (j:nat{j < v}).
    Shared.score_le (Seq.index logical i) (Seq.index logical j) <==>
    priority_le (Seq.index physical i) (Seq.index physical j))

inline_for_extraction noextract
let logical_scores_represent
  (#v:Shared.size) (physical:Shared.vec f32 v)
  : Lemma (scores_represent physical (logical_scores physical)) =
  let cells (i:nat{i < v})
    : Lemma (
        Shared.score_rank (Seq.index (logical_scores physical) i) ==
          score_class (Seq.index physical i) /\
        (F.is_zero (Seq.index physical i) ==>
          Seq.index (logical_scores physical) i == Shared.FiniteScore 0.0R)) =
    score_key_class physical (Seq.index physical i);
    if F.is_zero (Seq.index physical i) then score_key_zero physical (Seq.index physical i)
  in
  FStar.Classical.forall_intro cells;
  let pairs (i:nat{i < v}) (j:nat{j < v})
    : Lemma (
        Shared.score_le
          (Seq.index (logical_scores physical) i)
          (Seq.index (logical_scores physical) j) <==>
        priority_le (Seq.index physical i) (Seq.index physical j)) =
    score_key_order physical i j
  in
  FStar.Classical.forall_intro_2 pairs

inline_for_extraction noextract
let native_maximum
  (#v:Shared.size) (scores:Shared.vec f32 v) (winner:Shared.token v) : prop =
  forall (j:Shared.token v). priority_le (Seq.index scores j) (Seq.index scores winner)

inline_for_extraction noextract
let maximum_representation
  (#v:Shared.size) (physical:Shared.vec f32 v)
  (logical:Shared.vec Shared.score v) (winner:Shared.token v)
  : Lemma
      (requires scores_represent physical logical)
      (ensures native_maximum physical winner <==> Shared.maximum logical winner) =
  ()

inline_for_extraction noextract
let selected_token (scores : score_vector) (token : i64) : prop =
  exists (i : Shared.token vocabulary_size).
    I64.v token == i /\
    Shared.maximum (logical_scores scores) i

inline_for_extraction noextract
let valid_position (position prompt_length : i64) : prop =
  0 <= I64.v position /\ I64.v position < 2048 /\
  1 <= I64.v prompt_length /\ I64.v prompt_length <= 2048

inline_for_extraction noextract
let previous_count (position prompt_length : i64) : nat =
  if I64.v position < I64.v prompt_length
  then 0
  else I64.v position - I64.v prompt_length + 1

inline_for_extraction noextract
let output_count (position prompt_length : i64) : nat =
  if I64.v position < I64.v prompt_length - 1
  then 0
  else I64.v position - I64.v prompt_length + 2

inline_for_extraction noextract
let updated_history
  (previous : seq i64) (position prompt_length : i64) (token : i64)
  : seq i64 =
  if I64.v position < I64.v prompt_length - 1
  then previous
  else Seq.append previous (Seq.cons token Seq.empty)

inline_for_extraction noextract
let semantic_post
  (scores : score_vector)
  (previous : seq i64)
  (position prompt_length : i64)
  (token_values : lseq i64 1)
  (history_values : seq i64)
  : prop =
  selected_token scores (Seq.index token_values 0) /\
  Seq.length previous == previous_count position prompt_length /\
  Seq.length history_values == output_count position prompt_length /\
  history_values ==
    updated_history previous position prompt_length (Seq.index token_values 0)

inline_for_extraction noextract
let accumulated_tokens
  (#p:Shared.size) (completed:nat)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  : Shared.vec i64 (Shared.retained_count p (completed + 1)) =
  if completed < p - 1 then previous else Seq.append previous selected

inline_for_extraction noextract
let functional_post
  (#v:Shared.size) (p:Shared.size) (completed:nat)
  (scores:Shared.vec f32 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  : prop =
  (exists (winner:Shared.token v).
    I64.v (Seq.index selected 0) == winner /\
    Shared.maximum (logical_scores scores) winner) /\
  emitted == accumulated_tokens #p completed previous selected

inline_for_extraction noextract
let semantic_post_refines
  (position:i64{0 <= I64.v position /\ I64.v position < 2048})
  (prompt_length:i64{1 <= I64.v prompt_length /\ I64.v prompt_length <= 2048})
  (scores:score_vector)
  (previous:lseq i64 (previous_count position prompt_length))
  (selected:lseq i64 1)
  (emitted:lseq i64 (output_count position prompt_length))
  : Lemma
      (requires semantic_post scores previous position prompt_length selected emitted)
      (ensures
        functional_post #vocabulary_size (I64.v prompt_length) (I64.v position)
          scores previous selected emitted) =
  assert (Seq.equal selected (Seq.cons (Seq.index selected 0) Seq.empty));
  Seq.lemma_eq_elim selected (Seq.cons (Seq.index selected 0) Seq.empty)

inline_for_extraction noextract
let selected_maximum
  (#v:Shared.size) (p:Shared.size) (completed:nat)
  (scores:Shared.vec f32 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (logical:Shared.vec Shared.score v) (choice:Shared.token v)
  : Lemma
      (requires
        functional_post p completed scores previous selected emitted /\
        scores_represent scores logical /\
        I64.v (Seq.index selected 0) == choice)
      (ensures Shared.maximum logical choice /\ native_maximum scores choice) =
  logical_scores_represent scores;
  maximum_representation scores (logical_scores scores) choice;
  maximum_representation scores logical choice

inline_for_extraction noextract
let specification_total
  (p:Shared.size) (completed:nat) (scores:score_vector)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  : Lemma (
      exists (selected:Shared.vec i64 1)
             (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1))).
        functional_post #vocabulary_size p completed scores previous selected emitted) =
  let keys = logical_scores #vocabulary_size scores in
  let winner = Shared.argmax keys in
  Shared.argmax_correct keys;
  Shared.first_maximum_refines keys winner;
  let selected = Seq.create 1 (I64.int_to_t winner) in
  let emitted = accumulated_tokens #p completed previous selected in
  assert (functional_post #vocabulary_size p completed scores previous selected emitted)

inline_for_extraction noextract
let tokens_represent
  (#v:Shared.size) (#count:nat)
  (physical:Shared.vec i64 count) (logical:Shared.vec (Shared.token v) count)
  : prop =
  forall (j:nat{j < count}). I64.v (Seq.index physical j) == Seq.index logical j

inline_for_extraction noextract
let tokens_represent_append
  (#v:Shared.size) (#count:nat)
  (physical:Shared.vec i64 count) (logical:Shared.vec (Shared.token v) count)
  (selected:Shared.vec i64 1) (winner:Shared.token v)
  : Lemma
      (requires
        tokens_represent #v #count physical logical /\ I64.v (Seq.index selected 0) == winner)
      (ensures
        tokens_represent #v #(count + 1)
          (Seq.append physical selected) (Seq.append logical (Seq.create 1 winner))) =
  ()

inline_for_extraction noextract
let input_representation
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#completed:nat)
  (scores:Shared.vec f32 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (before:Shared.decode_state v h cell weights cache p completed)
  (observed:Shared.observation v h cell cache)
  : prop =
  scores_represent scores observed.logits /\
  tokens_represent #v #(Shared.retained_count p completed)
    previous before.returned.generated_tokens

inline_for_extraction noextract
let expected_hidden_frame
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#completed:nat)
  (before:Shared.decode_state v h cell weights cache p completed)
  (observed:Shared.observation v h cell cache)
  : Shared.vec (Shared.vec cell h) (Shared.retained_count p (completed + 1)) =
  if completed < p - 1 then before.returned.hidden_history
  else Seq.append before.returned.hidden_history (Seq.create 1 observed.residual)

inline_for_extraction noextract
let hidden_frame
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#completed:nat)
  (before:Shared.decode_state v h cell weights cache p completed)
  (observed:Shared.observation v h cell cache)
  (actual:Shared.vec (Shared.vec cell h) (Shared.retained_count p (completed + 1)))
  : prop =
  actual == expected_hidden_frame before observed

inline_for_extraction noextract
let output_representation
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#completed:nat)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h cell weights cache p completed)
  (observed:Shared.observation v h cell cache)
  (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec cell h) (Shared.retained_count p (completed + 1)))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (after:Shared.decode_state v h cell weights cache p (completed + 1))
  : prop =
  I64.v (Seq.index selected 0) == choice /\
  tokens_represent #v #(Shared.retained_count p (completed + 1))
    emitted after.returned.generated_tokens /\
  after.returned.hidden_history == hidden_after /\
  after.position == completed + 1 /\
  after.parameters == before.parameters /\
  after.model_cache == observed.cache_after /\
  after.captured == Some observed.residual /\
  (if completed + 1 < p then after.next_token == Seq.index prompt (completed + 1)
   else after.next_token == choice)

inline_for_extraction noextract
let post_refines_advance
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#completed:nat)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h cell weights cache p completed)
  (observed:Shared.observation v h cell cache)
  (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec cell h) (Shared.retained_count p (completed + 1)))
  (scores:Shared.vec f32 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  : Lemma
      (requires
        functional_post p completed scores previous selected emitted /\
        input_representation scores previous before observed /\
        hidden_frame before observed hidden_after /\
        I64.v (Seq.index selected 0) == choice)
      (ensures
        Shared.maximum observed.logits choice /\
        output_representation prompt before observed choice hidden_after selected emitted
          (Shared.advance prompt (Shared.prepare_call before) observed choice)) =
  selected_maximum p completed scores previous selected emitted observed.logits choice;
  if completed < p - 1 then ()
  else tokens_represent_append #v #(Shared.retained_count p completed)
    previous before.returned.generated_tokens selected choice

inline_for_extraction noextract
let output_representation_unique
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#completed:nat)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h cell weights cache p completed)
  (observed:Shared.observation v h cell cache) (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec cell h) (Shared.retained_count p (completed + 1)))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (left right:Shared.decode_state v h cell weights cache p (completed + 1))
  : Lemma
      (requires
        output_representation prompt before observed choice hidden_after selected emitted left /\
        output_representation prompt before observed choice hidden_after selected emitted right)
      (ensures left == right) =
  assert (Seq.equal left.returned.generated_tokens right.returned.generated_tokens);
  Seq.lemma_eq_elim left.returned.generated_tokens right.returned.generated_tokens

inline_for_extraction noextract
let post_refines_step
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#completed:nat)
  (model:Shared.model_operation v h cell weights cache)
  (prompt:Shared.vec (Shared.token v) p)
  (before:Shared.decode_state v h cell weights cache p completed)
  (observed:Shared.observation v h cell cache) (choice:Shared.token v)
  (hidden_after:Shared.vec (Shared.vec cell h) (Shared.retained_count p (completed + 1)))
  (scores:Shared.vec f32 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (Shared.retained_count p (completed + 1)))
  (after:Shared.decode_state v h cell weights cache p (completed + 1))
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
  (#v #h:Shared.size) (#cell #cache:Type0) (#count:Shared.size)
  (p:Shared.size)
  (events:Shared.vec (Shared.observation v h cell cache) count)
  (scores:Shared.vec (Shared.vec f32 v) count)
  (previous:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p t)))
  (selected:Shared.vec (Shared.vec i64 1) count)
  (emitted:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  (choices:Shared.vec (Shared.token v) count)
  : prop =
  forall (t:nat{t < count}).
    functional_post p t (Seq.index scores t) (previous t) (Seq.index selected t) (emitted t) /\
    scores_represent (Seq.index scores t) (Seq.index events t).logits /\
    I64.v (Seq.index (Seq.index selected t) 0) == Seq.index choices t

inline_for_extraction noextract
let prefixes_linked
  (p count:Shared.size)
  (previous:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p t)))
  (emitted:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  : prop =
  forall (t:nat{0 < t /\ t < count}). previous t == emitted (t - 1)

inline_for_extraction noextract
let posts_refine_selections
  (#v #h:Shared.size) (#cell #cache:Type0) (#count:Shared.size)
  (p:Shared.size)
  (events:Shared.vec (Shared.observation v h cell cache) count)
  (scores:Shared.vec (Shared.vec f32 v) count)
  (previous:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p t)))
  (selected:Shared.vec (Shared.vec i64 1) count)
  (emitted:(t:nat{t < count} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  (choices:Shared.vec (Shared.token v) count)
  : Lemma
      (requires physical_trace_post p events scores previous selected emitted choices)
      (ensures Shared.selections_admissible events choices) =
  let call_maximum (t:nat{t < count})
    : Lemma (Shared.maximum (Seq.index events t).logits (Seq.index choices t)) =
    selected_maximum p t (Seq.index scores t) (previous t) (Seq.index selected t) (emitted t)
      (Seq.index events t).logits (Seq.index choices t)
  in
  FStar.Classical.forall_intro call_maximum

inline_for_extraction noextract
let rec trace_prefix_correct
  (#v #h #p:Shared.size) (#cell #weights #cache:Type0) (#count:Shared.size)
  (prompt:Shared.vec (Shared.token v) p) (parameters:weights) (initial_cache:cache)
  (events:Shared.vec (Shared.observation v h cell cache) count)
  (scores:Shared.vec (Shared.vec f32 v) count)
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
  if t > 0 then trace_prefix_correct prompt parameters initial_cache events scores
    previous selected emitted choices (t - 1);
  let before = Shared.state_at prompt parameters initial_cache events choices t in
  let observed = Seq.index events t in
  post_refines_advance prompt before observed (Seq.index choices t)
    (expected_hidden_frame before observed)
    (Seq.index scores t) (previous t) (Seq.index selected t) (emitted t);
  Shared.state_at_successor prompt parameters initial_cache events choices t

inline_for_extraction noextract
let last_call (p g:Shared.size) : nat = Shared.call_count p g - 1

// Choices come from both actual outputs at every call, including prefill.
// Hidden payloads and the complete caller/producer state remain explicit frames.
inline_for_extraction noextract
let post_refines_decode
  (#v #h #p #g:Shared.size) (#cell #weights #cache:Type0)
  (model:Shared.model_operation v h cell weights cache)
  (prompt:Shared.vec (Shared.token v) p) (parameters:weights) (initial_cache:cache)
  (events:Shared.vec (Shared.observation v h cell cache) (Shared.call_count p g))
  (scores:Shared.vec (Shared.vec f32 v) (Shared.call_count p g))
  (previous:(t:nat{t < Shared.call_count p g} -> Shared.vec i64 (Shared.retained_count p t)))
  (selected:Shared.vec (Shared.vec i64 1) (Shared.call_count p g))
  (emitted:(t:nat{t < Shared.call_count p g} -> Shared.vec i64 (Shared.retained_count p (t + 1))))
  (choices:Shared.vec (Shared.token v) (Shared.call_count p g))
  (hidden_after:Shared.vec (Shared.vec cell h) g)
  (output:Shared.histories v h cell g)
  (after:Shared.decode_state v h cell weights cache p (Shared.call_count p g))
  : Lemma
      (requires (
        let completed = last_call p g in
        let before = Shared.state_at prompt parameters initial_cache events choices completed in
        let observed = Seq.index events completed in
        physical_trace_post p events scores previous selected emitted choices /\
        prefixes_linked p (Shared.call_count p g) previous emitted /\
        Shared.observations_from_calls model prompt parameters initial_cache events choices /\
        hidden_frame before observed hidden_after /\
        output_representation prompt before observed (Seq.index choices completed) hidden_after
          (Seq.index selected completed) (emitted completed) after /\
        output == after.returned))
      (ensures
        Shared.selected_decode_post model prompt parameters initial_cache events choices output after /\
        Shared.decode_post model prompt parameters initial_cache events output after /\
        tokens_represent #v (emitted (last_call p g)) output.generated_tokens /\
        output.hidden_history == hidden_after) =
  let completed = last_call p g in
  let before = Shared.state_at prompt parameters initial_cache events choices completed in
  let observed = Seq.index events completed in
  posts_refine_selections p events scores previous selected emitted choices;
  trace_prefix_correct prompt parameters initial_cache events scores
    previous selected emitted choices completed;
  post_refines_step model prompt before observed (Seq.index choices completed) hidden_after
    (Seq.index scores completed) (previous completed) (Seq.index selected completed)
    (emitted completed) after;
  Shared.state_at_successor prompt parameters initial_cache events choices completed;
  Shared.choices_realize_contract #v #h #p #g #cell #weights #cache
    model prompt parameters initial_cache events choices

inline_for_extraction noextract
let reference_domain_covered
  (smoke:bool) (h p g:Shared.size) (completed:nat{completed < Shared.call_count p g})
  : Lemma
      (requires Shared.reference_domain smoke vocabulary_size h p g)
      (ensures
        completed < 2048 /\ 1 <= p /\ p <= 2048 /\
        Shared.retained_count p completed <= 2048 /\
        Shared.retained_count p (completed + 1) <= 2048) =
  ()

(* Empty histories have no live allocation; their pointers are never read or written. *)
inline_for_extraction noextract
let history_ownership (pointer : array i64) (contents : seq i64) : slprop =
  if Seq.length contents = 0
  then emp
  else on gpu_loc (A.pts_to pointer #1.0R contents) ** pure (is_global_array pointer)

// External layout precondition, not a check performed by the native wrapper.
// For V=248320, N=previous_count input2 input3 and E=output_count input2 input3,
// the NONEMPTY accessed byte ranges are pairwise disjoint:
// scores [s,s+4*V), previous [p,p+8*N), selected [o,o+8), emitted [e,e+8*E).
// FP32 uses 4 bytes and int64 uses 8. Disjoint subranges may share an allocation.
// Full separated typed resources represent that assumed physical boundary;
// pointer inequality alone and overlapping heterogeneous views do not suffice.
// Empty prefixes own emp, with no inactive pointer length/location/non-null condition.
fn kplay_run
  (input0 : larray f32 vocabulary_size{is_global_array input0})
  (input1 : array i64)
  (input2 input3 : i64)
  (output0 : larray i64 1{is_global_array output0})
  (output1 : array i64)
  (#scores : erased score_vector)
  (#previous : erased (lseq i64 (previous_count input2 input3)))
  preserves
    cpu **
    on gpu_loc (A.pts_to input0 #1.0R scores) **
    history_ownership input1 previous
  requires
    pure (valid_position input2 input3) **
    (exists* (before_token : lseq i64 1).
      on gpu_loc (A.pts_to output0 #1.0R before_token)) **
    (exists* (before_history : lseq i64 (output_count input2 input3)).
      history_ownership output1 before_history)
  ensures
    exists* (token_values : lseq i64 1)
            (history_values : lseq i64 (output_count input2 input3)).
      on gpu_loc (output0 |-> token_values) **
      history_ownership output1 history_values **
      pure (semantic_post scores previous input2 input3 token_values history_values)
