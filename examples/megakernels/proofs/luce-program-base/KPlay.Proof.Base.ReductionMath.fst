module KPlay.Proof.Base.ReductionMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module S = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Sum = Kuiper.Sum
module Algebra = KPlay.Proof.Base.Algebra

inline_for_extraction noextract
let sum_zero (f:S.vector 0) : Lemma (S.sum 0 f == 0.0R) =
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum"]] (S.sum 0 f)

inline_for_extraction noextract
let sum_step (n:pos) (f:S.vector n)
  : Lemma (S.sum n f == S.sum (n - 1) (fun i -> f i) +. f (n - 1)) =
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum"]] (S.sum n f)

inline_for_extraction noextract
let rec sum_ext (n:nat) (f g:S.vector n)
  : Lemma
      (requires forall (i:S.index n). f i == g i)
      (ensures S.sum n f == S.sum n g)
      (decreases n) =
  if n = 0 then (sum_zero f; sum_zero g)
  else (
    sum_ext (n - 1) (fun i -> f i) (fun i -> g i);
    sum_step n f;
    sum_step n g
  )

inline_for_extraction noextract
let rec sum_plus (n:nat) (f g:S.vector n)
  : Lemma (
      S.sum n (fun i -> f i +. g i) == S.sum n f +. S.sum n g)
      (decreases n) =
  if n = 0 then (
    sum_zero f;
    sum_zero g;
    sum_zero (fun i -> f i +. g i)
  ) else (
    sum_plus (n - 1) (fun i -> f i) (fun i -> g i);
    sum_step n f;
    sum_step n g;
    sum_step n (fun i -> f i +. g i)
  )

inline_for_extraction noextract
let rec sum_split (n m:nat) (f:S.vector (n + m))
  : Lemma (
      S.sum (n + m) f ==
      S.sum n (fun i -> f i) +. S.sum m (fun i -> f (n + i)))
      (decreases m) =
  if m = 0 then (
    sum_zero (fun i -> f (n + i));
    sum_ext n f (fun i -> f i)
  ) else (
    let lower : S.vector (n + m - 1) = fun i -> f i in
    sum_split n (m - 1) lower;
    sum_ext (n + m - 1) lower (fun i -> f i);
    sum_ext n (fun i -> lower i) (fun i -> f i);
    sum_ext (m - 1) (fun i -> lower (n + i)) (fun i -> f (n + i));
    assert (S.sum (n + m - 1) (fun i -> f i) ==
      S.sum n (fun i -> f i) +. S.sum (m - 1) (fun i -> f (n + i)));
    sum_step (n + m) f;
    sum_step m (fun i -> f (n + i));
    assert (f (n + m - 1) == f (n + (m - 1)));
    assert (S.sum m (fun i -> f (n + i)) ==
      S.sum (m - 1) (fun i -> f (n + i)) +. f (n + m - 1))
  )

inline_for_extraction noextract
let sum_halves (n:pos) (f:S.vector (2 * n))
  : Lemma (
      S.sum n (fun i -> f i +. f (i + n)) == S.sum (2 * n) f) =
  sum_split n n f;
  sum_plus n (fun i -> f i) (fun i -> f (i + n));
  sum_ext n (fun i -> f (i + n)) (fun i -> f (n + i))

inline_for_extraction noextract
let rec sum_groups4 (n:nat) (f:S.vector (4 * n))
  : Lemma (
      S.sum n (fun g -> S.sum 4 (fun j -> f (4 * g + j))) ==
      S.sum (4 * n) f)
      (decreases n) =
  if n = 0 then (
    sum_zero (fun g -> S.sum 4 (fun j -> f (4 * g + j)));
    sum_zero f
  ) else (
    sum_groups4 (n - 1) (fun i -> f i);
    sum_step n (fun g -> S.sum 4 (fun j -> f (4 * g + j)));
    sum_split (4 * (n - 1)) 4 f
  )

inline_for_extraction noextract
let rec sum_kuiper_prefix (#n:nat) (f:S.vector n) (count:nat{count <= n})
  : Lemma (ensures S.sum count (fun i -> f i) ==
      Sum.sum #real #Sum.real_add_semigroup 0 count f)
      (decreases count) =
  if count = 0 then (
    sum_zero (fun i -> f i);
    assert_norm (Sum.sum #real #Sum.real_add_semigroup 0 0 f == 0.0R)
  )
  else (
    sum_kuiper_prefix f (count - 1);
    sum_step count (fun i -> f i);
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 count f
  )

inline_for_extraction noextract
let sum_kuiper (n:nat) (f:S.vector n)
  : Lemma (S.sum n f == Sum.sum #real #Sum.real_add_semigroup 0 n f) =
  sum_kuiper_prefix f n;
  sum_ext n f (fun i -> f i)

inline_for_extraction noextract
let sum_window (#n:nat) (f:P.vector real n)
  (start:nat{start <= n}) (count:nat{start + count <= n})
  : Lemma (
      Sum.sum start (start + count) f == S.sum count (fun i -> f (start + i))) =
  Algebra.shift_sum f start count;
  sum_kuiper_prefix #(n - start) (Algebra.shifted #n f start) count;
  sum_ext count (fun i -> Algebra.shifted #n f start i) (fun i -> f (start + i))
