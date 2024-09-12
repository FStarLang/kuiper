module GPU.Math

let rec pow_log_lemma (n: nat) : Lemma (log2 (pow2 n) = n) =
  if n = 0 then () else pow_log_lemma (n - 1)

let rec div_pow2_lemma (i j tid: nat):
  Lemma
    (requires i < j)
    (ensures (div_pow2 j tid) ==> (div_pow2 i tid))
  = if not (div_pow2 j tid) then () else (
      if i = j - 1 then () else div_pow2_lemma i (j - 1) tid;
      FStar.Math.Lemmas.mod_mult_exact tid (pow2 (j - 1)) 2
  )
