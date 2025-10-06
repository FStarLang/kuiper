module Kuiper.Math

module M = FStar.Math.Lemmas

let even_odd (n : int) :
  Lemma (even n <==> not (odd n))
        [SMTPat (even n); SMTPat (odd n)]
        = ()

let even_2x (n : int) :
  Lemma (ensures even (2 * n))
  = ()

let odd_2x1 (n : int) :
  Lemma (ensures odd (2 * n + 1))
  = ()

let rec lemma_log2_pow2 (n: nat) : Lemma (log2 (pow2 n) = n) =
  if n = 0 then () else lemma_log2_pow2 (n - 1)

let rec lemma_pow2_log2 (n: pos) : Lemma (pow2 (log2 n) <= n) =
  if n > 1 then
    lemma_pow2_log2 (n / 2)

let rec log2_mono n m =
  if n > 1 && m > 1 then
    log2_mono (n/2) (m/2)

let pow2_mono n m = M.pow2_le_compat m n

let mul_pow2 n m = M.pow2_plus n m

let lemma_log2_le1 (n:pos) (m : nat) : Lemma (log2 n < m ==> n <= pow2 m) =
  calc (==>) {
    log2 n < m <: prop;
    ==> { lemma_log2_pow2 m }
    log2 n < log2 (pow2 m) <: prop;
    ==> { pow2_mono (log2 n) (log2 (pow2 m)) }
    pow2 (log2 n) < pow2 (log2 (pow2 m)) <: prop;
    ==> { lemma_log2_pow2 m }
    pow2 (log2 n) < pow2 m <: prop;
    ==> {}
    n <= pow2 m <: prop;
  }

let lemma_log2_le2 (n:pos) (m:nat) : Lemma (n <= pow2 m ==> log2 n <= m) =
  calc (==>) {
    n <= pow2 m <: prop;
    ==> { lemma_pow2_log2 n }
    pow2 (log2 n) <= pow2 m <: prop;
    ==> { lemma_log2_pow2 m }
    pow2 (log2 n) <= pow2 (log2 (pow2 m)) <: prop;
    ==> { pow2_mono (log2 n) (log2 (pow2 m)) }
    log2 n <= log2 (pow2 m) <: prop;
    ==> { lemma_log2_pow2 m }
    log2 n <= m <: prop;
  }

let div_pow2_lemma (i j tid: nat) :
  Lemma
    (requires i < j)
    (ensures div_pow2 j tid ==> div_pow2 i tid)
  = assert (pow2 i /? pow2 j) // from lemma_pow2_div and lemma_divides_trans, nice

val lemma_div_exact: a:int -> p:pos -> Lemma
  (a % p = 0 <==> a = p * (a / p))
let lemma_div_exact a p = ()

let div_pow2_lemma_3 (n tid: nat)
: Lemma (div_pow2 n tid <==> div_pow2 n (tid + pow2 n))
= M.modulo_addition_lemma tid (pow2 n) 1;
  assert (tid % pow2 n == (tid + pow2 n) % pow2 n);
  ()

let div_pow2_lemma_4 (n tid: nat)
: Lemma (requires div_pow2 n tid)
        (ensures  div_pow2 (n + 1) tid <==> ~(div_pow2 (n + 1) (tid + pow2 n)))
=
  assert (tid % pow2 n == 0);
  let k = get_factor (pow2 n) tid in
  assert (tid == k * pow2 n);
  assert (tid % pow2 (n+1) == (k * pow2 n) % (pow2 n * 2));
  M.modulo_scale_lemma k (pow2 n) 2;
  M.modulo_scale_lemma (k+1) (pow2 n) 2;
  assert (tid % pow2 (n+1) == 0 \/ tid % pow2 (n+1) == pow2 n);
  assert (tid % pow2 (n+1) == 0 ==> (tid + pow2 n) % pow2 (n+1) == pow2 n);
  assert (tid % pow2 (n+1) == pow2 n ==> (tid + pow2 n) % pow2 (n+1) == 0);
  ()

let div_pow2_lemma_2 (it tid : nat)
  : Lemma (
      ~(div_pow2 (it + 1) (tid + pow2 it)) /\ div_pow2 it (tid + pow2 it)
      <==>
      div_pow2 (it + 1) tid
    )
  =
  div_pow2_lemma_3 it tid;
  FStar.Classical.move_requires (div_pow2_lemma_4 it) tid;
  div_pow2_lemma it (it + 1) tid;
  ()

let shift_left_1_n (n:pos) (s:nat{s < n}) :
  Lemma (UInt.shift_left #n 1 s == pow2 s) =
  calc (==) {
    UInt.shift_left #n 1 s;
    == { UInt.shift_left_value_lemma #n 1 s }
    (1 * pow2 s) % pow2 n;
    == {}
    pow2 s % pow2 n;
    == { M.small_mod (pow2 s) (pow2 n) }
    pow2 s;
  }

let add_mod_assoc (#n:nat) (a b c : UInt.uint_t n)
: Lemma (UInt.add_mod a (UInt.add_mod b c) == UInt.add_mod (UInt.add_mod a b) c)
        [SMTPat (UInt.add_mod a (UInt.add_mod b c))]
= calc (==) {
    UInt.add_mod a (UInt.add_mod b c);
    == {}
    UInt.add_mod a  ((b + c) % pow2 n);
    == {}
    (a + ((b + c) % pow2 n)) % pow2 n;
    == { M.lemma_mod_add_distr a (b + c) (pow2 n) }
    (a + (b + c)) % pow2 n;
    == {}
    (c + (a + b)) % pow2 n;
    == { M.lemma_mod_add_distr c (a + b) (pow2 n) }
    (c + ((a + b) % pow2 n)) % pow2 n;
    == {}
    UInt.add_mod (UInt.add_mod a b) c;
  }
