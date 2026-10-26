module Kuiper.Test.Floating

open Kuiper.Floating
open Kuiper.Approximates.Base

let zeros_compare_equal #t {| floating t |} (x y:t)
  : Lemma (requires is_zero x /\ is_zero y)
          (ensures ieee_eq x y)
  = is_zero_spec x; is_zero_spec y; eq_spec x y

let nan_equalities #t {| floating t |} (x:t{is_nan x})
  : Lemma (~(ieee_eq x x) /\ bit_eq x x)
  = ieee_eq_refl x; bit_eq_refl x

let exact_substitution #t {| floating t |} (f:t -> t) (x y:t)
  : Lemma (requires bit_eq x y) (ensures f x == f y)
  = bit_eq_spec x y

let nonzero_substitution #t {| floating t |} (f:t -> t) (x y:t)
  : Lemma (requires ieee_eq x y /\ ~(is_zero x)) (ensures f x == f y)
  = ieee_eq_nonzero_is_exact x y

let numerical_approximation #t {| floating t, real_like t, floating_real_like t |}
  (x y:t) (r:Kuiper.Real.real)
  : Lemma (requires ieee_eq x y /\ v_approximates y r)
          (ensures v_approximates x r)
  = eq_approx x y r

let zero_sub_classification #t {| floating t |} (x:t)
  : Lemma (kind (sub zero x) == kind x)
  = zero_sub_kind x

let zero_sub_twice #t {| floating t |} (x:t{not_nan x})
  : Lemma (ieee_eq (sub zero (sub zero x)) x)
  = zero_sub_involutive x

let zero_sub_order #t {| floating t |} (x y:t)
  : Lemma (requires not_nan x /\ not_nan y)
          (ensures lt x y <==> lt (sub zero y) (sub zero x))
  = lt_zero_sub_flip x y; ()

let subtraction_as_add_zero_sub #t {| floating t |} (x y:t)
  : Lemma (requires not_nan x /\ not_nan y /\ not_nan (sub x y))
          (ensures ieee_eq (sub x y) (add x (sub zero y)))
  = sub_is_add_zero_sub x y

let zero_sub_neg_numerical #t {| floating t |} (x:t{not_nan x})
  : Lemma (ieee_eq (sub zero x) (neg x))
  = zero_sub_is_neg x

let zero_sub_neg_nonzero #t {| floating t |} (x:t{not_nan x /\ ~(is_zero x)})
  : Lemma (bit_eq (sub zero x) (neg x))
  = zero_sub_is_neg x; bit_eq_spec (sub zero x) (neg x)

let zero_sub_neg_zeros #t {| floating t |} (x:t{is_zero x})
  : Lemma (ieee_eq (sub zero x) (neg x))
  = is_zero_spec x; zero_sub_is_neg x

[@@expect_failure [19]]
let zero_sub_neg_not_exact_at_zero #t {| floating t |} (x:t{is_zero x})
  : Lemma (sub zero x == neg x)
  = is_zero_spec x; zero_sub_is_neg x

[@@expect_failure [19]]
let zero_sub_neg_not_exact_for_nan #t {| floating t |} (x:t{is_nan x /\ ~(is_zero x)})
  : Lemma (sub zero x == neg x)
  = ()

[@@expect_failure [19]]
let zero_sub_twice_is_not_exact #t {| floating t |} (x:t{not_nan x})
  : Lemma (sub zero (sub zero x) == x)
  = zero_sub_involutive x

[@@expect_failure [19]]
let approximation_does_not_allow_substitution #t
  {| floating t, real_like t, floating_real_like t |} (f:t -> t) (x y:t)
  : Lemma (requires ieee_eq x y) (ensures f x == f y)
  = ()

[@@expect_failure [19]]
let numerical_equality_is_not_identity #t {| floating t |} (x y:t)
  : Lemma (requires ieee_eq x y) (ensures x == y)
  = eq_spec x y

[@@expect_failure [19]]
let numerical_equality_does_not_support_substitution #t {| floating t |}
  (f:t -> t) (x y:t)
  : Lemma (requires ieee_eq x y) (ensures f x == f y)
  = eq_spec x y

[@@expect_failure [19]]
let neg_neg_is_not_exact #t {| floating t |} (x:t{not_nan x})
  : Lemma (neg (neg x) == x)
  = neg_neg x

// These aliases exercise native F* interoperability at the implementation boundary.
let native_f32 (x:FStar.Float32.t) : Kuiper.Float32.Base.t = x
let native_f64 (x:FStar.Float64.t) : Kuiper.Float64.Base.t = x

// No generic executable equality: callers must choose ieee_eq or bit_eq.
[@@expect_failure [19]]
let no_generic_f32_equality (x y:Kuiper.Float32.Base.t) : bool = x = y

[@@expect_failure [19]]
let no_generic_f64_equality (x y:Kuiper.Float64.Base.t) : bool = x = y

// Import the actual axioms for each format and reject the original exploit.
module F16 = Kuiper.Float16.Base
module BF16 = Kuiper.BFloat16.Base
module F32 = Kuiper.Float32.Base
module F64 = Kuiper.Float64.Base

let zero_sub_twice_f16 (x:F16.t{~(NaN? (F16.kind x))})
  : Lemma (F16.eq (F16.sub F16.zero (F16.sub F16.zero x)) x)
  = F16.zero_sub_involutive x

let zero_sub_twice_bf16 (x:BF16.t{~(NaN? (BF16.kind x))})
  : Lemma (BF16.eq (BF16.sub BF16.zero (BF16.sub BF16.zero x)) x)
  = BF16.zero_sub_involutive x

let zero_sub_twice_f32 (x:F32.t{~(NaN? (F32.kind x))})
  : Lemma (F32.eq (F32.sub F32.zero (F32.sub F32.zero x)) x)
  = F32.zero_sub_involutive x

let zero_sub_twice_f64 (x:F64.t{~(NaN? (F64.kind x))})
  : Lemma (F64.eq (F64.sub F64.zero (F64.sub F64.zero x)) x)
  = F64.zero_sub_involutive x

[@@expect_failure [19]]
let reject_zero_sub_signed_zero_f16 () : Lemma
  (F16.div F16.one (F16.mul (F16.sub F16.zero F16.one) F16.zero)
    == F16.div F16.one F16.zero)
  = let _ = F16.kind_one in
    F16.zero_sub_kind F16.one;
    F16.mul_zero (F16.sub F16.zero F16.one)

[@@expect_failure [19]]
let reject_zero_sub_signed_zero_bf16 () : Lemma
  (BF16.div BF16.one (BF16.mul (BF16.sub BF16.zero BF16.one) BF16.zero)
    == BF16.div BF16.one BF16.zero)
  = let _ = BF16.kind_one in
    BF16.zero_sub_kind BF16.one;
    BF16.mul_zero (BF16.sub BF16.zero BF16.one)

[@@expect_failure [19]]
let reject_zero_sub_signed_zero_f32 () : Lemma
  (F32.div F32.one (F32.mul (F32.sub F32.zero F32.one) F32.zero)
    == F32.div F32.one F32.zero)
  = let _ = F32.kind_one in
    F32.zero_sub_kind F32.one;
    F32.mul_zero (F32.sub F32.zero F32.one)

[@@expect_failure [19]]
let reject_zero_sub_signed_zero_f64 () : Lemma
  (F64.div F64.one (F64.mul (F64.sub F64.zero F64.one) F64.zero)
    == F64.div F64.one F64.zero)
  = let _ = F64.kind_one in
    F64.zero_sub_kind F64.one;
    F64.mul_zero (F64.sub F64.zero F64.one)

[@@expect_failure [19]]
let reject_signed_zero_f16 () : Lemma
  (F16.div F16.one (F16.mul (F16.neg F16.one) F16.zero)
    == F16.div F16.one F16.zero)
  = let _ = F16.kind_one in
    F16.neg_kind F16.one;
    F16.mul_zero (F16.neg F16.one)

[@@expect_failure [19]]
let reject_signed_zero_bf16 () : Lemma
  (BF16.div BF16.one (BF16.mul (BF16.neg BF16.one) BF16.zero)
    == BF16.div BF16.one BF16.zero)
  = let _ = BF16.kind_one in
    BF16.neg_kind BF16.one;
    BF16.mul_zero (BF16.neg BF16.one)

[@@expect_failure [19]]
let reject_signed_zero_f32 () : Lemma
  (F32.div F32.one (F32.mul (F32.neg F32.one) F32.zero)
    == F32.div F32.one F32.zero)
  = let _ = F32.kind_one in
    F32.neg_kind F32.one;
    F32.mul_zero (F32.neg F32.one)

[@@expect_failure [19]]
let reject_signed_zero_f64 () : Lemma
  (F64.div F64.one (F64.mul (F64.neg F64.one) F64.zero)
    == F64.div F64.one F64.zero)
  = let _ = F64.kind_one in
    F64.neg_kind F64.one;
    F64.mul_zero (F64.neg F64.one)

[@@expect_failure [19]]
let axioms_do_not_prove_false () : Lemma False = ()
