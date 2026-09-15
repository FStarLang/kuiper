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

  (* Exact representation comparison, including NaN signs and payloads.
     The scalar superclass's eq remains IEEE numerical comparison. *)
  bit_eq : t -> t -> bool;

  of_int : Int64.t -> t;

  (* Build a value from a decimal literal string. Extracts to a C
     floating constant of the appropriate type. *)
  of_literal : string -> t;

  #[easy_fill ()] of_int_zero : squash (of_int 0L == zero);
  #[easy_fill ()] of_int_one  : squash (of_int 1L == one);

  kind : t -> fkind;
  is_zero : t -> GTot bool;

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
  #[easy_fill()] zero_is_zero : squash (is_zero zero);
  #[easy_fill()] one_is_nonzero : squash (~(is_zero one));

  (* Laws.

     NOTE: These axioms assume IEEE 754 default rounding mode
     (round-to-nearest-even). They may not hold under CUDA's --use_fast_math
     or explicit rounding-mode intrinsics (__fadd_rd, __fmul_ru, etc.).

     Propositional equality distinguishes every representation, including
     signed zeros and NaN payloads. IEEE comparison identifies the two zero
     signs and is false for NaNs, so it does not imply substitutable equality.
  *)

  #[easy_fill ()]
  bit_eq_spec : (x : t) -> (y : t) ->
    Lemma (bit_eq x y <==> x == y)
          [SMTPat (bit_eq x y)];

  #[easy_fill ()]
  is_zero_spec : (x : t) ->
    Lemma (requires is_zero x)
          (ensures kind x == Finite)
          [SMTPat (is_zero x)];

  #[easy_fill ()]
  eq_spec : (x : t) -> (y : t) ->
    Lemma (eq x y <==>
      (~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
       (x == y \/ (is_zero x /\ is_zero y))))
          [SMTPat (eq x y)];

  (* x <= y <==> x < y or IEEE equality *)
  #[easy_fill ()]
  lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lte x y <==> lt x y \/ eq x y)
          [SMTPat (lte x y)];

  #[easy_fill ()]
  neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x)
          [SMTPat (zero `sub` x)];

  (* Negation expressed as zero subtraction preserves numerical values. *)
  #[easy_fill ()]
  neg_neg : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (zero `sub` (zero `sub` x)) x)
          [SMTPat (zero `sub` (zero `sub` x))];

  (* x < y <==> -y < -x, using numerical ordering. *)
  #[easy_fill ()]
  lt_neg_flip : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> lt (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)];

  (* x < y <==> not (y <= x) *)
  #[easy_fill ()]
  negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)];

  (* Exact commutativity excludes NaN results, whose payloads are unspecified. *)
  #[easy_fill ()]
  add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(NaN? (kind (add x y))))
          (ensures add x y == add y x)
          [SMTPat (add x y)];

  #[easy_fill ()]
  mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(NaN? (kind (mul x y))))
          (ensures mul x y == mul y x)
          [SMTPat (mul x y)];

  #[easy_fill ()]
  add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)];

  #[easy_fill ()]
  mul_zero : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures eq (mul x zero) zero)
          [SMTPat (mul x zero)];

  #[easy_fill ()]
  mul_one : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures mul x one == x)
          [SMTPat (mul x one)];

  (* sub is add-of-negation. FIXME: adding the pattern breaks proofs. *)
  #[easy_fill ()]
  sub_is_add_neg : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(NaN? (kind (sub x y))))
          (ensures eq (sub x y) (add x (zero `sub` y)));
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
          (ensures eq (fmax x y) (if lt x y then y else x))
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
  (* Additional sign-bit laws can be specified without collapsing signed zeros. *)
  copysign : t -> t -> t;
  fma : t -> t -> t -> t;
}
