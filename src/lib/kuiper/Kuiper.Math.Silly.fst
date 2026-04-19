module Kuiper.Math.Silly

open Kuiper.Common
module M = FStar.Math.Lemmas

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

let stupid_mul_mono (x y z w : nat)
: Lemma (requires x <= z /\ y <= w)
        (ensures x * y <= z * w)
= ()

let stupid_divides (x:nat) (y:nonzero)
: Lemma (x/y <= x)
= ()

let p4_assoc = ez

let mod_prod (a b : int) (k : pos) :
  Lemma (ensures (a % k) * (b % k) % k == (a * b) % k)
  = M.lemma_mod_mul_distr_l a b k;
    M.lemma_mod_mul_distr_r (a % k) b k;
    ()

let lemma_mul_pos_recip (a b: nat) :
  Lemma (requires a * b > 0) (ensures a > 0 /\ b > 0)
= ()

let lemma_le_plus_lt (x y z w: int)
: Lemma (requires (x + z <= w /\
    y < z))
    (ensures (x + y < w))
= ()
