module Kuiper.Divides

open FStar.Mul

let divides (x y : int) : prop =
  exists (z:int). x * z == y

unfold
let ( /? ) = divides

unfold
let ( /?+ ) (x : pos) (y : int) =
  y % x = 0

val get_factor (x y : int)
  : Ghost int (requires x /? y)
              (ensures fun z -> x * z == y)

val lemma_divides_mod (x:pos) (y : int)
  : Lemma (x /? y <==> y % x == 0)
          [SMTPat (x /? y)]

val lemma_divides_product (x y : int)
  : Lemma (x /? (x * y)  /\  x /? (y * x))
          [SMTPatOr [[SMTPat (x /? (x * y))];
                     [SMTPat (x /? (y * x))]]]

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

let divup (m:nat) (k:pos) : int =
  (m + (k-1)) / k

val lem_divup_back (m:nat) (k:pos)
  : Lemma (k * divup m k >= m)
          [SMTPat (divup m k)]

val lem_divup_divides (m:nat) (k:pos)
  : Lemma (k /? m ==> divup m k === m / k)
          [SMTPat (divup m k)]

val lemma_divides_sum (d : pos) (a b : int)
  : Lemma (requires d /? a /\ d /? b)
          (ensures d /? (a + b))

val lemma_divides_product_l (d : pos) (a c : int)
  : Lemma (requires d /? a)
          (ensures d /? (a * c))

val lemma_divides_product_r (d : pos) (a c : int)
  : Lemma (requires d /? a)
          (ensures d /? (c * a))

val lemma_divides_chain (a b c : pos)
  : Lemma (requires a /? b /\ b /? c)
          (ensures a /? c)

val lemma_divides_mod_op (d: pos) (a: int) (b: pos)
  : Lemma (requires d /? a /\
            d /? b)
          (ensures d /? (a % b))
