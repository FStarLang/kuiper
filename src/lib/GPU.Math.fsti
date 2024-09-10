module GPU.Math

let rec log2 (n:pos) : GTot (r:nat{r < n}) =
  if n = 1 then 0 else 1 + log2 (n / 2)

val pow_log_lemma (n: nat)
: Lemma (log2 (pow2 n) == n)
  [SMTPat (log2 (pow2 n))]

let min (a b: int) : GTot int =
  if a < b then a else b

(* x is a multiple of 2^i *)
let div_pow2 (i x : nat) : GTot bool =
  (x % pow2 i) = 0

val div_pow2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures (div_pow2 j tid) ==> (div_pow2 i tid))
