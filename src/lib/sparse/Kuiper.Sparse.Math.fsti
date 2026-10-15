module Kuiper.Sparse.Math

open Kuiper
open Kuiper.Array.Vectorized
module SZ = FStar.SizeT

let divup (n : int) (d : pos {n + d > 0}) : Tot nat = (n + d - 1) / d

(* sdivup is implemented as (n + (d-1))/d. Associating
that way usually performs more partial evaluation as d is usually
known. *)
[@@"opaque_to_smt"] // Important to prevent a trigger cascade apparently... investigate
inline_for_extraction noextract
let divup_ (n : sz) (d : szp)
: Pure sz (requires fits (n + d)) (ensures fun r -> SZ.v r == divup n d)
= sdivup n d

let round2 (k : pos) (n : nat) : natle n = (n / k) * k

(* round2 especializado para potencias de 2: n & (k - 1) *)
inline_for_extraction noextract
let round2_  (k : szp) (n : sz)
: Pure sz (requires true) (ensures fun r -> SZ.v r == round2 k n)
= (n /^ k) *^ k

val round2_lemma (a b n k : nat)
: Lemma
  (requires a /? pow2 k /\  b /? pow2 k /\ a <= b)
  (ensures a /? round2 b n)

val round2_chunk_lemma
  (a b : Type0) {| sized a, has_vec_cpy a, sized b, has_vec_cpy b |}
  (n : nat)
: Lemma
    (ensures
      chunk a /? round2 (max (chunk a) (chunk b)) n /\
      chunk b /? round2 (max (chunk a) (chunk b)) n
    )

val intro_divides (a b c : nat)
: Lemma (requires a * b = c) (ensures a /? c)

val prod_divides (a b c : pos)
: Lemma (requires (a * b) /? c) (ensures a /? c /\ b /? c) //[SMTPat ((a * b) /? c)]

val lineal_divides (d : pos) (a b k : nat)
: Lemma (requires d /? a /\ d /? b) (ensures d /? (a + k * b) /\ d /? (k * b + a))

val prod_preserves_divides (c d : pos) (a : nat)
: Lemma (requires c /? a) (ensures (c * d) /? (a * d))

val divides_leq (d : pos) (a b : nat)
: Lemma (requires d /? a /\ d /? b) (ensures b < a <==> b + d <= a)

val divides_chain (a b c : nat)
: Lemma (requires a /? b /\ b /? c) (ensures a /? c)

val prod_cancel_divides (a : nat) (c d : pos)
: Lemma (requires (c * d) /? a) (ensures c /? (a / d))

val block_lemma whole block k
: Lemma (requires block /? whole /\ k * block < whole) (ensures  k * block + block <= whole)

val rounded_offset_aligned
  (#t1 t2 : Type0) {| sized t1, has_vec_cpy t1, sized t2, has_vec_cpy t2 |}
  (#n : nat)
  (x : larray t1 n { aligned 16 x })
  (offset : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x (round2 (max (chunk t1) (chunk t2)) offset))

val step_aligned
  (#t1 : Type0) {| sized t1, has_vec_cpy t1 |}
  (#n : nat)
  (x : larray t1 n { aligned 16 x })
  (offset : nat)
  (step : nat { chunk t1 /? step })
  (k : nat)
: Lemma
  (requires aligned' 16 x offset)
  (ensures  aligned' 16 x (offset + k * step))