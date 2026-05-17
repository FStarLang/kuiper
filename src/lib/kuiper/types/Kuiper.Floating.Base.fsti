module Kuiper.Floating.Base

open Kuiper.Real
open Kuiper.Scalars.Base
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

  kind : t -> fkind;

  smallest : t; (* smallest (positive) epsilon *)
  largest  : t; (* largest (positive) representable value. *)
  infinity : t; (* positive infinity *)

  #[easy_fill()] kind_smallest : squash (kind smallest == Finite);
  #[easy_fill()] kind_largest  : squash (kind largest  == Finite);
  #[easy_fill()] kind_infinity : squash (kind infinity == Infinite);

  (* Laws. *)

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
          (ensures lte x y <==> lt x y \/ eq x y)
          [SMTPat (lte x y)];

  #[easy_fill ()]
  neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x)
          [SMTPat (zero `sub` x)];

  (* -(-x) == x . *)
  #[easy_fill ()]
  neg_neg : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (zero `sub` (zero `sub` x)) x)
          [SMTPat (zero `sub` (zero `sub` x))];

  (* x < y <==> -y <= -x.  FIXME: This is not exactly true due to signed zeros! *)
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
          (ensures eq (add x y) (add y x))
          [SMTPat (add x y)];

  #[easy_fill ()]
  mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq (mul x y) (mul y x))
          [SMTPat (mul x y)];

  #[easy_fill ()]
  add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)];

  #[easy_fill ()]
  smallest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x) /\ zero `lt` x)
          (ensures lte smallest x)
          [SMTPat (lte smallest x)];

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

  exp : t -> t;
  log : t -> t;
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
  fmax : t -> t -> t;
  fmod : t -> t -> t;
  copysign : t -> t -> t;
  fma : t -> t -> t -> t;
}

(* Derived methods *)

inline_for_extraction noextract
let neg (#t:Type) {| floating t |} (x : t) : t =
  zero `sub` x

inline_for_extraction noextract
let gt (#t:Type) {| floating t |} (x y : t) : bool =
  lt y x

inline_for_extraction noextract
let gte (#t:Type) {| floating t |} (x y : t) : bool =
  lte y x

inline_for_extraction noextract
let neq (#t:Type) {| floating t |} (x y : t) : bool =
  not (eq x y)

inline_for_extraction noextract
let abs (#t:Type) {| floating t |} (x : t) : t =
  if x `gte` zero then x else sub zero x

inline_for_extraction noextract
let max_float (#et : Type0) {| floating et |} (x y : et) : et =
  if x `gt` y then x else y

(* We could provide executable versions for these if needed. *)
let is_nan (#t:Type) {| floating t |} (x : t) : GTot bool =
  NaN? (kind x)

let not_nan (#t:Type) {| floating t |} (x : t) : GTot bool =
  ~(is_nan x)

let is_inf (#t:Type) {| floating t |} (x : t) : GTot bool =
  Infinite? (kind x)

let is_finite (#t:Type) {| floating t |} (x : t) : GTot bool =
  Finite? (kind x)

// val max_float_approximates_max_real (#et: Type0) {| floating et |}
//   (x: et) (y: et) (xr: real) (yr: real):
//     Lemma
//       (requires x %~ xr /\ y %~ yr)
//       (ensures max_float #et x y %~ max_real xr yr)
//       // [SMTPat (max_float x y); SMTPat (max_real xr yr);
//       //  SMTPat (x %~ xr); SMTPat (y %~ yr)]
