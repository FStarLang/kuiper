module KPlay.Proof.Base.SelectionAlgebra

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open FStar.Real
module S = FStar.Seq
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module Shared = KPlay.Proof.Shared.Selection

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
