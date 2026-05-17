module Kuiper.Float64

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Approximates.Base

new
val t : Type0

val zero : t
val one : t

inline_for_extraction noextract
instance _ : sized t = { size = 8sz; default = zero }

val lt : t -> t -> bool
val lte : t -> t -> bool
val eq : t -> t -> bool

val add : t -> t -> t
val mul : t -> t -> t
val sub : t -> t -> t
val div : t -> t -> t

val valid : t -> bool

val min_val : (x:t{valid x})
val max_val : (x:t{valid x})

val eq_spec : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq x y <==> x == y)
          [SMTPat (eq x y)]

val lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures lte x y <==> lt x y \/ eq x y)
          [SMTPat (lte x y)]

val negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)]

val add_comm : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq (add x y) (add y x))
          [SMTPat (add x y)]

val mul_comm : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq (mul x y) (mul y x))
          [SMTPat (mul x y)]

val add_zero : (x : t) ->
    Lemma (requires valid x)
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)]

val min_max_val_spec : (x : t) ->
    Lemma (requires valid x)
          (ensures lte min_val x /\ lte x max_val)
          [SMTPat (lte min_val x)]

inline_for_extraction noextract
instance _ : scalar t = {
  is_sized = solve;
  add; mul; zero; one; lt; lte; eq;
}

val exp : t -> t
val log : t -> t

val sqrt : t -> t
val rsqrt : t -> t
val sin : t -> t
val cos : t -> t
val tan : t -> t
val asin : t -> t
val acos : t -> t
val atan : t -> t
val sinh : t -> t
val cosh : t -> t
val tanh : t -> t
val ceil : t -> t
val floor : t -> t
val round : t -> t
val fabs : t -> t
val erf : t -> t
val log2 : t -> t
val log10 : t -> t
val exp2 : t -> t

val pow : t -> t -> t
val atan2 : t -> t -> t
val fmin : t -> t -> t
val fmax : t -> t -> t
val fmod : t -> t -> t
val copysign : t -> t -> t

val fma : t -> t -> t -> t

inline_for_extraction noextract
instance _ : floating t = {
  is_scalar = solve;
  valid;
  min_val; max_val;
  eq_spec; lte_is_lt_or_eq; negate_lt_is_lte;
  add_comm; mul_comm; add_zero; min_max_val_spec;
  div; sub;
  exp; log; sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
  sinh; cosh; tanh; ceil; floor; round; fabs; erf; log2;
  log10; exp2; pow; atan2; fmin; fmax; fmod; copysign;
  fma;
}

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t
