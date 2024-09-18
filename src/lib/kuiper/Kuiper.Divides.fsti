module Kuiper.Divides

open FStar.Mul

let divides (x y : int) : prop =
  exists (z:int). x * z == y

unfold
let ( /? ) = divides

val lemma_divides_mod (x:pos) (y : int)
  : Lemma (x /? y <==> y % x == 0)
          [SMTPat (x /? y)]

val lemma_divides_exact (x:pos) (y:int)
  : Lemma (x /? y <==> x * (y/x) == y)
          [SMTPat (x /? y)]
