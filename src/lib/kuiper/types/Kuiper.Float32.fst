module Kuiper.Float32

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Approximates.Base

assume val t0 : Type0
let t = t0

assume val zero : t
assume val one : t

inline_for_extraction noextract
instance _ : sized t = { size = 4sz; default = zero }

assume val add : t -> t -> t
assume val mul : t -> t -> t

assume val lt : t -> t -> bool
assume val lte : t -> t -> bool
assume val eq : t -> t -> bool

inline_for_extraction noextract
instance _ : scalar t = {
  is_sized = solve;
  add; mul; zero; one; lt; lte; eq;
}

assume val sub : t -> t -> t
assume val div : t -> t -> t

assume val kind : t -> fkind

assume val smallest : t
assume val largest : t
assume val infinity : t

assume val kind_smallest : squash (kind smallest == Finite)
assume val kind_largest  : squash (kind largest  == Finite)
assume val kind_infinity : squash (kind infinity == Infinite)

assume val eq_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq x y <==> x == y)
          [SMTPat (eq x y)]

assume val lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lte x y <==> lt x y \/ eq x y)
          [SMTPat (lte x y)]

assume val neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x)
          [SMTPat (zero `sub` x)]

assume val neg_neg : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (zero `sub` (zero `sub` x)) x)
          [SMTPat (zero `sub` (zero `sub` x))]

assume val lt_neg_flip : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> lte (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)]

assume val negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)]

assume val add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq (add x y) (add y x))
          [SMTPat (add x y)]

assume val mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq (mul x y) (mul y x))
          [SMTPat (mul x y)]

assume val add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)]

assume val smallest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x) /\ zero `lt` x)
          (ensures lte smallest x)
          [SMTPat (lte smallest x)]

assume val largest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures lte x largest)
          [SMTPat (lte x largest)]

assume val infinity_val_spec : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures lte x infinity)
          [SMTPat (lte x infinity)]

assume val exp : t -> t
assume val log : t -> t
assume val sqrt : t -> t
assume val rsqrt : t -> t
assume val sin : t -> t
assume val cos : t -> t
assume val tan : t -> t
assume val asin : t -> t
assume val acos : t -> t
assume val atan : t -> t
assume val sinh : t -> t
assume val cosh : t -> t
assume val tanh : t -> t
assume val ceil : t -> t
assume val floor : t -> t
assume val round : t -> t
assume val fabs : t -> t
assume val erf : t -> t
assume val log2 : t -> t
assume val log10 : t -> t
assume val exp2 : t -> t
assume val pow : t -> t -> t
assume val atan2 : t -> t -> t
assume val fmin : t -> t -> t
assume val fmax : t -> t -> t
assume val fmod : t -> t -> t
assume val copysign : t -> t -> t
assume val fma : t -> t -> t -> t

inline_for_extraction noextract
instance is_floating : floating t = {
  is_scalar = solve;
  sub; div;
  kind;
  smallest; largest; infinity;
  kind_smallest; kind_largest; kind_infinity;
  exp; log; sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
  sinh; cosh; tanh; ceil; floor; round; fabs; erf; log2;
  log10; exp2; pow; atan2; fmin; fmax; fmod; copysign;
  fma;
}

instance is_real_like : real_like t = magic()
instance is_floating_real_like : floating_real_like t = magic()

let lem_sizeof () = ()
