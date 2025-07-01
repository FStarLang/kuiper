module Kuiper.Math

open FStar.Mul

include Kuiper.Divides

let even (n : int) : GTot bool = n % 2 = 0
let odd (n : int)  : GTot bool = n % 2 = 1

val even_odd (n : int) :
  Lemma (even n <==> not (odd n))
        [SMTPat (even n); SMTPat (odd n)]

(* Cannot use patterns here, + and * are builtin
   so they would not be reliable. Just call them. *)
val even_2x (n : int) :
  Lemma (ensures even (2 * n))

val odd_2x1 (n : int) :
  Lemma (ensures odd (2 * n + 1))

let rec log2 (n:pos) : GTot (r:nat{r < n}) =
  if n = 1 then 0 else 1 + log2 (n / 2)

val lemma_log2_pow2 (n: nat)
: Lemma (log2 (pow2 n) == n)
        [SMTPat (log2 (pow2 n))]

val lemma_pow2_log2 (n: pos)
: Lemma (pow2 (log2 n) <= n)
        [SMTPat (pow2 (log2 n))]

val log2_mono (n m : pos)
: Lemma (requires n <= m)
        (ensures  log2 n <= log2 m)
        [SMTPat (log2 n); SMTPat (log2 m)]

val pow2_mono (n m : nat)
: Lemma (requires n <= m)
        (ensures  pow2 n <= pow2 m)
        [SMTPat (pow2 n); SMTPat (pow2 m)]

val mul_pow2 (n m : nat)
: Lemma (ensures pow2 n * pow2 m == pow2 (n + m))

(* Note the strict < on the left. It is not true that
log2 n == m ==> n <= pow2 m (e.g. n=3, m=1) *)
val lemma_log2_le1 (n:pos) (m:nat)
: Lemma (log2 n < m ==> n <= pow2 m)
  [SMTPat (log2 n); SMTPat (pow2 m)]

val lemma_log2_le2 (n:pos) (m:nat)
: Lemma (n <= pow2 m ==> log2 n <= m)
  [SMTPat (log2 n); SMTPat (pow2 m)]

let min (a b: int) : GTot int =
  if a < b then a else b

(* x is a multiple of 2^i. Or: x has i zeroes at the end of its
binary repr.

This is in GTot bool instead of prop to
use if_ and friends, but it coincides exactly with `pow2 i /? x`. We
state in the refinement to automatically use all SMTPats about /?.
*)
let div_pow2 (i:nat) (x:nat) : GTot (b:bool{b <==> pow2 i /? x}) =
  (x % pow2 i) = 0

val div_pow2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures  div_pow2 j tid ==> div_pow2 i tid)

(* Adding 2^n does not affect having n zeroes. *)
val div_pow2_lemma_3 (n tid: nat)
  : Lemma (div_pow2 n tid <==> div_pow2 n (tid + pow2 n))

(* If tid has n zeroes, then either tid or tid+2^n has n+1 zeroes, and not both. *)
val div_pow2_lemma_4 (n tid: nat)
  : Lemma (requires div_pow2 n tid)
          (ensures  div_pow2 (n + 1) tid <==> ~(div_pow2 (n + 1) (tid + pow2 n)))

val div_pow2_lemma_2 (it tid: nat)
  : Lemma (
      ~(div_pow2 (it + 1) (tid + pow2 it)) /\ div_pow2 it (tid + pow2 it) // TODO: remove + pow2 it? It's equivalent
      <==>
      div_pow2 (it + 1) tid
    )

(* This proves that 1<<n == pow2 n for every machine int *)
val shift_left_1_n (n:pos) (s:nat{s < n})
: Lemma (UInt.shift_left #n 1 s == pow2 s)
        [SMTPat (UInt.shift_left #n 1 s)]

val add_mod_assoc (#n:nat) (a b c : UInt.uint_t n)
: Lemma (UInt.add_mod a (UInt.add_mod b c) == UInt.add_mod (UInt.add_mod a b) c)
        [SMTPat (UInt.add_mod a (UInt.add_mod b c))]
