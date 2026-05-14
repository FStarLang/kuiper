module Kuiper.Float64

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Scalars.Base

new
val t : Type0

val zero : t
val one : t

inline_for_extraction noextract
instance _ : sized t = { size = 8sz; default = zero }

val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val div : t -> t -> t

val eq : t -> t -> bool
val lt : t -> t -> bool
val lte : t -> t -> bool

val valid : t -> bool

val lte_is_lt_or_eq (x y : t) :
  Lemma (requires valid x /\ valid y) (ensures lte x y <==> lt x y \/ eq x y)

val negate_lt_is_lte (x y : t) :
  Lemma (requires valid x /\ valid y) (ensures lt x y <==> not (lte y x))

inline_for_extraction noextract
instance _ : scalar t = {
  is_sized = solve;
  add; mul; zero; one; lt; lte; eq; valid;
  lte_is_lt_or_eq; negate_lt_is_lte;
}

val exp : t -> t
val log : t -> t

(* Transcendental and math primitives *)
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

(* Binary *)
val pow : t -> t -> t
val atan2 : t -> t -> t
val fmin : t -> t -> t
val fmax : t -> t -> t
val fmod : t -> t -> t
val copysign : t -> t -> t

(* Ternary *)
val fma : t -> t -> t -> t

val add_comm (x y : t) : Lemma (add x y == add y x) [SMTPat (add x y)]
