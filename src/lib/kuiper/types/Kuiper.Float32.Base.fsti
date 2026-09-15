module Kuiper.Float32.Base

(* All assumptions about Float32. *)

open Kuiper.Floating.Base

(* Reuse F* native types and primitives. The laws below specify the
   CUDA interpretation of these otherwise abstract operations. *)
module F = FStar.Float32

inline_for_extraction let t = F.t
inline_for_extraction let zero = F.zero
inline_for_extraction let one = F.one
inline_for_extraction let add = F.add
inline_for_extraction let mul = F.mul
inline_for_extraction let lt = F.lt
inline_for_extraction let lte = F.lte
inline_for_extraction let sub = F.sub
inline_for_extraction let div = F.div
inline_for_extraction let of_int = F.of_int
inline_for_extraction let of_literal = F.of_literal
inline_for_extraction let bit_eq = F.bit_eq
inline_for_extraction let ieee_eq = F.ieee_eq
inline_for_extraction let eq = ieee_eq

val of_int_zero : squash (of_int 0L == zero)
val of_int_one : squash (of_int 1L == one)

val kind : t -> fkind
val is_zero : t -> GTot bool

val largest : t
val infinity : t

val kind_one      : squash (kind one == Finite)
val kind_zero     : squash (kind zero == Finite)
val kind_largest  : squash (kind largest  == Finite)
val kind_infinity : squash (kind infinity == Infinite)
val zero_is_zero : squash (is_zero zero)
val one_is_nonzero : squash (~(is_zero one))

val bit_eq_spec : (x:t) -> (y:t) ->
    Lemma (bit_eq x y <==> x == y)
          [SMTPat (bit_eq x y)]

val is_zero_spec : (x:t) ->
    Lemma (requires is_zero x)
          (ensures kind x == Finite)
          [SMTPat (is_zero x)]

val eq_spec : (x : t) -> (y : t) ->
    Lemma (eq x y <==>
      (~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
       (x == y \/ (is_zero x /\ is_zero y))))
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
          (ensures lt x y <==> lt (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)]

val negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)]

val add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(NaN? (kind (add x y))))
          (ensures add x y == add y x)
          [SMTPat (add x y)]

val mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(NaN? (kind (mul x y))))
          (ensures mul x y == mul y x)
          [SMTPat (mul x y)]

val add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)]

val  mul_zero : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures eq (mul x zero) zero)
          [SMTPat (mul x zero)]

val  mul_one : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures mul x one == x)
          [SMTPat (mul x one)]

val sub_is_add_neg : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(NaN? (kind (sub x y))))
          (ensures eq (sub x y) (add x (zero `sub` y)))
          [SMTPat (sub x y)]

val largest_val_spec : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures lte x largest)
          [SMTPat (lte x largest)]

val infinity_val_spec : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures lte x infinity)
          [SMTPat (lte x infinity)]

val fmax : t -> t -> t

val fmax_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq (fmax x y) (if lt x y then y else x))
          [SMTPat (fmax x y)]

val fexp : t -> t
val flog : t -> t
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
val fmod : t -> t -> t
val copysign : t -> t -> t
val fma : t -> t -> t -> t
