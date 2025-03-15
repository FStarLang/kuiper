module Kuiper.Float16

new
val t : Type0

[@@noextract_to "krml"] val zero : t
[@@noextract_to "krml"] val one : t

[@@noextract_to "krml"] val add : t -> t -> t
[@@noextract_to "krml"] val sub : t -> t -> t
[@@noextract_to "krml"] val mul : t -> t -> t
[@@noextract_to "krml"] val div : t -> t -> t

[@@noextract_to "krml"] val exp : t -> t

val add_comm (x y : t) : Lemma (add x y == add y x) [SMTPat (add x y)]
