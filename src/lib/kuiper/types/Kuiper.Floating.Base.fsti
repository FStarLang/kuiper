module Kuiper.Floating.Base

open Kuiper.Scalars.Base
open FStar.Tactics.Easy
open FStar.Tactics.Typeclasses { solve, tcinstance }

inline_for_extraction noextract
class floating (t : Type) = {
  [@@@tcinstance]
  is_scalar : scalar t;

  (* Is this a mathematically valid element? I.e., not a NaN. *)
  valid : t -> bool;

  min_val : t;
  max_val : t;

  #[easy_fill()] min_val_is_valid : squash (valid min_val);
  #[easy_fill()] max_val_is_valid : squash (valid max_val);

  (* Laws. *)

  (* Equality is sound, at least for valid terms. *)
  #[easy_fill ()]
  eq_spec : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq x y <==> x == y)
          [SMTPat (eq x y)];

  (* x <= y <==> x < y or x == y *)
  #[easy_fill ()]
  lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures lte x y <==> lt x y \/ eq x y)
          [SMTPat (lte x y)];

  (* x < y <==> not (y <= x) *)
  #[easy_fill ()]
  negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)];

  (* Addition commutes. Note: this is true even for NaNs. *)
  #[easy_fill ()]
  add_comm : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq (add x y) (add y x))
          [SMTPat (add x y)];

  #[easy_fill ()]
  mul_comm : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq (mul x y) (mul y x))
          [SMTPat (mul x y)];

  #[easy_fill ()]
  add_zero : (x : t) ->
    Lemma (requires valid x)
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)];

  (* min and max are correct. *)
  #[easy_fill ()]
  min_max_val_spec : (x : t) ->
    Lemma (requires valid x)
          (ensures lte min_val x /\ lte x max_val)
          [SMTPat (lte min_val x)];


  sub : t -> t -> t;
  div : t -> t -> t;
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

inline_for_extraction noextract
let abs (#t:Type) {| floating t |} (x : t) : t =
  if x `gte` zero then x else sub zero x
