module GPU.Math

module M = FStar.Math.Lemmas

let rec lemma_log2_pow2 (n: nat) : Lemma (log2 (pow2 n) = n) =
  if n = 0 then () else lemma_log2_pow2 (n - 1)

let rec lemma_pow2_log2 (n: pos) : Lemma (pow2 (log2 n) <= n) =
  if n > 1 then
    lemma_pow2_log2 (n / 2)
  
let rec log2_mono n m =
  if n > 1 && m > 1 then
    log2_mono (n/2) (m/2)

let pow2_mono n m = M.pow2_le_compat m n

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

let rec div_pow2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures (div_pow2 j tid) ==> (div_pow2 i tid))
  = if not (div_pow2 j tid) then () else (
      if i = j - 1 then () else div_pow2_lemma i (j - 1) tid;
      M.mod_mult_exact tid (pow2 (j - 1)) 2
  )

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
