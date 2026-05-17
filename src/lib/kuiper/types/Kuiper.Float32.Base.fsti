module Kuiper.Float32.Base

(* All assumptions about Float32. *)

open Kuiper.Floating.Base

new
val t : Type0

val zero : t
val one : t

val add : t -> t -> t
val mul : t -> t -> t

val lt : t -> t -> bool
val lte : t -> t -> bool
val eq : t -> t -> bool

val sub : t -> t -> t
val div : t -> t -> t

val kind : t -> fkind

val smallest : t
val largest : t
val infinity : t

val kind_smallest : squash (kind smallest == Finite)
val kind_largest  : squash (kind largest  == Finite)
val kind_infinity : squash (kind infinity == Infinite)

val eq_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq x y <==> x == y)
          [SMTPat (eq x y)]

val lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lte x y <==> lt x y \/ eq x y)
          [SMTPat (lte x y)]

val neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x)
          [SMTPat (zero `sub` x)]

val neg_neg : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (zero `sub` (zero `sub` x)) x)
          [SMTPat (zero `sub` (zero `sub` x))]

val lt_neg_flip : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> lte (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)]

val negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)]

val add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq (add x y) (add y x))
          [SMTPat (add x y)]

val mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq (mul x y) (mul y x))
          [SMTPat (mul x y)]

val add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)]

val smallest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x) /\ zero `lt` x)
          (ensures lte smallest x)
          [SMTPat (lte smallest x)]

val largest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures lte x largest)
          [SMTPat (lte x largest)]

val infinity_val_spec : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures lte x infinity)
          [SMTPat (lte x infinity)]

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
