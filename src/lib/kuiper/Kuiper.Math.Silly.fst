module Kuiper.Math.Silly

open FStar.Mul

let lemma_nonneg_mul (x y : int)
  : Lemma (requires x >= 0 /\ y >= 0)
          (ensures x * y >= 0)
= ()

let two_times_succ (x: int)
  : Lemma (requires x >= 0)
          (ensures 2 * (x + 1) = 2 * x + 2) = ()

let lemma_pos_times_pos (a b: pos)
  : Lemma (a * b > 0) = ()

let lemma_nat_times_nat (a b: nat)
  : Lemma (a * b >= 0) = ()

let lemma_sq_mono (a b : nat)
: Lemma (a <= b <==> a * a <= b * b)
= ()

let lemma_sq_mono' (a b : nat)
: Lemma (a < b <==> a * a < b * b)
= ()
