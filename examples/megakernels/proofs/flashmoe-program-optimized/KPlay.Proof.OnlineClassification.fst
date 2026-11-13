module KPlay.Proof.OnlineClassification

open Kuiper

module F = Kuiper.Floating
module B = Kuiper.Floating.Bounds
module Bounds = Kuiper.Float32.Bounds
module Order = KPlay.Proof.Shared.Capacity

inline_for_extraction noextract
let negative_infinity : f32 = F.neg F.infinity

inline_for_extraction noextract
let finite_above_seed (value:f32{F.is_finite value})
  : Lemma (F.lt negative_infinity value)
  =
  F.neg_kind #f32 F.infinity;
  F.neg_kind #f32 negative_infinity;
  F.neg_neg #f32 F.infinity;
  if F.is_zero #f32 F.infinity then F.is_zero_spec #f32 F.infinity;
  F.eq_spec #f32 (F.neg negative_infinity) F.infinity;
  assert (F.neg negative_infinity == F.infinity);
  F.neg_kind #f32 value;
  F.infinity_val_spec #f32 (F.neg value);
  F.lte_is_lt_or_eq #f32 (F.neg value) F.infinity;
  F.eq_spec #f32 (F.neg value) F.infinity;
  F.lt_neg_flip #f32 negative_infinity value

inline_for_extraction noextract
let equal_magnitude (n:B.bound) (x y:f32)
  : Lemma
      (requires B.magnitude n x /\ F.eq x y)
      (ensures B.magnitude n y)
  =
  F.eq_spec x y;
  F.bit_eq_spec x y;
  if not (F.bit_eq x y) then (
    F.is_zero_spec x;
    F.is_zero_spec y;
    Order.native_order2 F.zero x;
    Order.native_order2 F.zero y;
    Bounds.integer_bounds n;
    Order.native_order3 y x (B.integer #f32 n)
  )

inline_for_extraction noextract
let maximum_step (old x:f32)
  : Lemma
      (requires
        (old == negative_infinity \/ B.magnitude 65536 old) /\
        B.magnitude 65536 x)
      (ensures (
        let next = F.fmax old x in
        B.magnitude 65536 next /\
        F.lte old next /\ F.lte x next /\
        (F.eq next old \/ F.eq next x)))
  =
  F.neg_kind #f32 F.infinity;
  F.fmax_spec old x;
  if F.is_inf old then finite_above_seed x;
  if F.lt old x then (
    equal_magnitude 65536 x (F.fmax old x);
    Order.native_order3 old x (F.fmax old x);
    Order.native_order2 x (F.fmax old x)
  ) else (
    equal_magnitude 65536 old (F.fmax old x);
    Order.native_order3 x old (F.fmax old x);
    Order.native_order2 old (F.fmax old x)
  )

inline_for_extraction noextract
let exp_shift (x maximum delta term:f32)
  : Lemma
      (requires B.magnitude 65536 x /\ B.magnitude 65536 maximum /\
        F.lte x maximum /\ B.sub_post x maximum delta /\ B.exp_post delta term)
      (ensures B.unit term /\ (F.eq x maximum ==> term == F.one))
  =
  assert (B.magnitude 131072 delta);
  assert (F.lte delta F.zero)

inline_for_extraction noextract
let seed_shift (maximum delta term:f32)
  : Lemma
      (requires B.magnitude 65536 maximum /\
        delta == negative_infinity /\ B.exp_post delta term)
      (ensures term == F.zero /\ B.unit term)
  =
  Bounds.zero ();
  Bounds.one ();
  Order.native_order2 F.zero F.one

inline_for_extraction noextract
let denominator_step (n:nat{n < 8})
  (old maximum x sum adjustment numerator result:f32)
  : Lemma
      (requires
        (if n = 0 then old == negative_infinity /\ sum == F.zero
         else B.magnitude 65536 old /\ F.lte F.one sum) /\
        B.magnitude 65536 x /\ B.magnitude 65536 maximum /\
        (F.eq old maximum \/ F.eq x maximum) /\
        B.nonnegative n sum /\ B.unit adjustment /\ B.unit numerator /\
        (F.eq old maximum ==> adjustment == F.one) /\
        (F.eq x maximum ==> numerator == F.one) /\
        B.fma_post sum adjustment numerator result)
      (ensures B.nonnegative (n + 1) result /\ F.lte F.one result)
  =
  B.nonnegative_magnitude n sum;
  B.nonnegative_unit adjustment;
  B.nonnegative_magnitude 1 adjustment;
  B.nonnegative_unit numerator;
  B.nonnegative_magnitude 1 numerator;
  assert (B.magnitude (n + 1) result /\ F.lte F.zero result);
  assert (F.lte numerator result);
  if F.eq x maximum then ()
  else (
    F.eq_spec old maximum;
    F.neg_kind #f32 F.infinity;
    if n = 0 then (
      if F.is_zero old then F.is_zero_spec old;
      assert (False)
    );
    assert (adjustment == F.one);
    assert (F.lte sum result);
    Order.native_order3 F.one sum result
  )
