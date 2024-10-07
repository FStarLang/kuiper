module Kuiper.Divides

open FStar.Mul

let divides (x y : int) : prop =
  exists (z:int). x * z == y

unfold
let ( /? ) = divides

val get_factor (x y : int)
  : Ghost int (requires x /? y)
              (ensures fun z -> x * z == y)

val lemma_divides_mod (x:pos) (y : int)
  : Lemma (x /? y <==> y % x == 0)
          [SMTPat (x /? y)]

val lemma_divides_exact (x:pos) (y:int)
  : Lemma (x /? y <==> x * (y/x) == y)
          [SMTPat (x /? y)]

val lemma_divides_le (x : nat) (y : pos)
  : Lemma (requires x /? y)
          (ensures x <= y)
          [SMTPat (x /? y)]

val lemma_divides_trans (x y z : pos)
  : Lemma (requires x /? y /\ y /? z)
          (ensures x /? z)
          [SMTPat (x /? y); SMTPat (y /? z)]

val lemma_pow2_div (x y : nat)
  : Lemma (requires x <= y)
          (ensures pow2 x /? pow2 y)
          [SMTPat (pow2 x /? pow2 y)]
