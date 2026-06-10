module Kuiper.Floating.Base

include Kuiper.Scalars.Base
open FStar.Tactics.Easy
open FStar.Tactics.Typeclasses { solve, tcinstance }

(* Kinds of floating point numbers. For spec only. *)
[@@erasable]
noeq
type fkind =
  | Finite
  | Infinite
  | NaN

inline_for_extraction noextract
class floating (t : Type) = {
  [@@@tcinstance]
  is_scalar : scalar t;

  sub : t -> t -> t;
  div : t -> t -> t;

  of_int : Int64.t -> t;

  #[easy_fill ()] of_int_zero : squash (of_int 0L == zero);
  #[easy_fill ()] of_int_one  : squash (of_int 1L == one);

  kind : t -> fkind;

  (* NOTE: We do not model a "smallest positive value" here. Whether that
     means the smallest subnormal or the smallest normal depends on whether
     flush-to-zero (FTZ) mode is active, which varies by type (fp16/bf16
     typically use FTZ) and by compiler flags. *)
  largest  : t; (* largest (positive) representable value. *)
  infinity : t; (* positive infinity *)

  #[easy_fill()] kind_one      : squash (kind one == Finite);
  #[easy_fill()] kind_zero     : squash (kind zero == Finite);
  #[easy_fill()] kind_largest  : squash (kind largest  == Finite);
  #[easy_fill()] kind_infinity : squash (kind infinity == Infinite);

  (* Laws.

     NOTE: These axioms assume IEEE 754 default rounding mode
     (round-to-nearest-even). They may not hold under CUDA's --use_fast_math
     or explicit rounding-mode intrinsics (__fadd_rd, __fmul_ru, etc.).

     NOTE: We intentionally do not distinguish +0 and -0. The abstract type
     identifies them (i.e., propositional equality == conflates both zeros).
     This is sound for most GPU kernel verification but means copysign and
     signbit cannot be faithfully axiomatized without extending the model.
     See the note on copysign below.
  *)

  (* Equality is sound, at least for non-NaNs. *)
  #[easy_fill ()]
  eq_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq x y <==> x == y)
          [SMTPat (eq x y)];

  (* x <= y <==> x < y or x == y *)
  #[easy_fill ()]
  lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lte x y <==> lt x y \/ x == y)
          [SMTPat (lte x y)];

  #[easy_fill ()]
  neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x)
          [SMTPat (zero `sub` x)];

  (* -(-x) == x . *)
  #[easy_fill ()]
  neg_neg : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures zero `sub` (zero `sub` x) == x)
          [SMTPat (zero `sub` (zero `sub` x))];

  (* x < y <==> -y <= -x.  NOTE: This is sound because we identify +0 and -0.
     Under strict IEEE 754 with distinct signed zeros, this would fail:
     lt (-0) (+0) is false, but lte (0-(+0)) (0-(-0)) = lte 0 0 = true. *)
  #[easy_fill ()]
  lt_neg_flip : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> lte (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)];

  (* x < y <==> not (y <= x) *)
  #[easy_fill ()]
  negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)];

  (* Addition commutes. *)
  #[easy_fill ()]
  add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures add x y == add y x)
          [SMTPat (add x y)];

  #[easy_fill ()]
  mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures mul x y == mul y x)
          [SMTPat (mul x y)];

  #[easy_fill ()]
  add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures add x zero == x)
          [SMTPat (add x zero)];

  #[easy_fill ()]
  mul_zero : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures mul x zero == zero)
          [SMTPat (mul x zero)];

  #[easy_fill ()]
  mul_one : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures mul x one == x)
          [SMTPat (mul x one)];

  (* sub is add-of-negation. FIXME: adding the pattern breaks proofs. *)
  #[easy_fill ()]
  sub_is_add_neg : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures sub x y == add x (zero `sub` y));
          // [SMTPat (sub x y)]

  #[easy_fill ()]
  largest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures lte x largest)
          [SMTPat (lte x largest)];

  #[easy_fill ()]
  infinity_val_spec : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures lte x infinity)
          [SMTPat (lte x infinity)];

  fmax : t -> t -> t;

  // This spec could be strengthened: fmax returns the non-NaN if one the args is NaN
  #[easy_fill ()]
  fmax_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures fmax x y == (if lt x y then y else x))
          [SMTPat (fmax x y)];

  fexp : t -> t;
  flog : t -> t;
  sqrt : t -> t;
  rsqrt : t -> t;
  sin : t -> t;
  cos : t -> t;
  tan : t -> t;
  asin : t -> t;
  acos : t -> t;
  atan : t -> t;
  sinh : t -> t;
  cosh : t -> t;
  tanh : t -> t;
  ceil : t -> t;
  floor : t -> t;
  round : t -> t;
  fabs : t -> t;
  erf : t -> t;
  log2 : t -> t;
  log10 : t -> t;
  exp2 : t -> t;
  pow : t -> t -> t;
  atan2 : t -> t -> t;
  fmin : t -> t -> t;
  fmod : t -> t -> t;
  (* NOTE: copysign is inherently about the sign bit, which our model cannot
     faithfully express since we identify +0 and -0. Any axiomatization of
     copysign would require extending fkind or the abstract type to
     distinguish signs. For now, copysign is provided as an unaxiomatized
     primitive for extraction purposes only. *)
  copysign : t -> t -> t;
  fma : t -> t -> t -> t;
}
