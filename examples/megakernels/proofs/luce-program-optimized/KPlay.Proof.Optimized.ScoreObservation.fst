module KPlay.Proof.Optimized.ScoreObservation

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module Seq = FStar.Seq
module FP = Kuiper.Float32
module F = Kuiper.Floating
module D = KPlay.Proof.Shared.Selection
module O = KPlay.Proof.Shared.Observations
module Order = KPlay.Proof.Optimized.FamilySelection

inline_for_extraction noextract
let keys (values:D.vec FP.t 248320) : GTot (D.vec D.score 248320) =
  Seq.init_ghost 248320 (fun i -> Order.score_key values (Seq.index values i))

inline_for_extraction noextract
let category (values:D.vec FP.t 248320) (value:FP.t)
  : Lemma (
      match F.kind value with
      | F.NaN -> Order.score_key values value == D.NotANumber
      | F.Infinite -> Order.score_key values value ==
          (if F.lt value (F.zero #FP.t) then D.NegativeInfinity else D.PositiveInfinity)
      | F.Finite ->
          D.FiniteScore? (Order.score_key values value) /\
          (F.is_zero value ==> Order.score_key values value == D.FiniteScore 0.0R)) =
  F.neg_kind (F.infinity #FP.t);
  if F.is_zero Order.positive_infinity then F.is_zero_spec Order.positive_infinity;
  if F.is_zero Order.negative_infinity then F.is_zero_spec Order.negative_infinity;
  F.eq_spec value Order.positive_infinity;
  F.eq_spec value Order.negative_infinity;
  Order.infinity_bounds ();
  Order.native_order value (F.zero #FP.t);
  if not (F.is_nan value) then FP.non_nan_classification value;
  if F.is_zero value then Order.score_key_zero values value

inline_for_extraction noextract
let represented (values:D.vec FP.t 248320)
  : Lemma (O.priority_keys (fun i -> Seq.index values i) (keys values)) =
  let at_index (i:D.token 248320)
    : Lemma (
        let value = Seq.index values i in
        let key = Seq.index (keys values) i in
        match F.kind value with
        | F.NaN -> key == D.NotANumber
        | F.Infinite -> key == (if F.lt value (F.zero #FP.t) then D.NegativeInfinity else D.PositiveInfinity)
        | F.Finite -> D.FiniteScore? key /\ (F.is_zero value ==> key == D.FiniteScore 0.0R)) =
    category values (Seq.index values i)
  in
  FStar.Classical.forall_intro at_index;
  let pair (i j:D.token 248320)
    : Lemma (
        ~(F.NaN? (F.kind (Seq.index values i))) /\ ~(F.NaN? (F.kind (Seq.index values j))) ==>
        (D.score_le (Seq.index (keys values) i) (Seq.index (keys values) j) <==>
          F.lte (Seq.index values i) (Seq.index values j))) =
    Order.score_key_order values i j;
    if not (F.is_nan (Seq.index values i)) && not (F.is_nan (Seq.index values j)) then
      F.negate_lt_is_lte (Seq.index values j) (Seq.index values i)
  in
  FStar.Classical.forall_intro_2 pair

