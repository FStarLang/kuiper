module Kuiper.Math.Silly

(* A collection of silly lemmas to deal with
bad non-linear arithmetic in Z3. *)

open FStar.Mul

val lemma_nonneg_mul (x y : int)
  : Lemma (requires x >= 0 /\ y >= 0)
          (ensures x * y >= 0)

val two_times_succ (x: int)
  : Lemma (requires x >= 0)
          (ensures 2 * (x + 1) = 2 * x + 2)

val lemma_pos_times_pos (a b: pos)
  : Lemma (a * b > 0)

val lemma_nat_times_nat (a b: nat)
  : Lemma (a * b >= 0)

val lemma_sq_mono (a b : nat)
  : Lemma (a <= b <==> a * a <= b * b)

val lemma_sq_mono' (a b : nat)
  : Lemma (a < b <==> a * a < b * b)

val stupid_mul_mono (x y z w : nat)
  : Lemma (requires x <= z /\ y <= w)
          (ensures x * y <= z * w)

val stupid_divides (x:nat) (y:nonzero)
  : Lemma (x/y <= x)

val p4_assoc
  (x y z w : nat)
  : Lemma ((x * y) * (z * w) == x * z * y * w)

