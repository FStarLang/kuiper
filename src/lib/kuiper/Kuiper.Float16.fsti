module Kuiper.Float16

new
val t : Type0

val zero : t
val one : t

val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val div : t -> t -> t

val gt : t -> t -> bool
val gte : t -> t -> bool

val exp : t -> t
val log : t -> t

val add_comm (x y : t) : Lemma (add x y == add y x) [SMTPat (add x y)]
