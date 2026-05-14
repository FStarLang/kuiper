module Kuiper.Float64

new
val t : Type0

val zero : t
val one : t
val minus_inf : t

val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val div : t -> t -> t

val lt : t -> t -> bool
val lte : t -> t -> bool
val gt : t -> t -> bool
val gte : t -> t -> bool

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
