module GPU.Math

open FStar.Mul

include Kuiper.Divides

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

(* Note the strict < on the left. It is not true that
log2 n == m ==> n <= pow2 m *)
val lemma_log2_le1 (n:pos) (m:nat)
: Lemma (log2 n < m ==> n <= pow2 m)
  [SMTPat (log2 n); SMTPat (pow2 m)]

val lemma_log2_le2 (n:pos) (m:nat)
: Lemma (n <= pow2 m ==> log2 n <= m)
  [SMTPat (log2 n); SMTPat (pow2 m)]

let min (a b: int) : GTot int =
  if a < b then a else b

(* x is a multiple of 2^i *)
let div_pow2 (i x : nat) : GTot bool =
  (x % pow2 i) = 0

val div_pow2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures (div_pow2 j tid) ==> (div_pow2 i tid))

val div_pow2_lemma_2 (it tid: nat):
  Lemma (
    (not (div_pow2 (it + 1) (tid + pow2 it)) && div_pow2 it (tid + pow2 it))
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
