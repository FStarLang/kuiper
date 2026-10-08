module Kuiper.Sparse.Math

open Kuiper
open Kuiper.Array.Vectorized
module SZ = FStar.SizeT

let div2_lemma (a : nat)
: Lemma (requires a % 2 <> 0) (ensures a % 2 == 1)
= ()

let div2_prod_odd (a b : nat)
: Lemma
  (requires a % 2 == 1 /\ b % 2 == 1)
  (ensures (a * b) % 2 == 1)
=
  let p = a / 2 in
  let q = b / 2 in

  assert a * b = 2 * (p + q + 2*p*q) + 1;
  assert (a * b) % 2 == 1


let div2_lemma_prod (a b : nat)
: Lemma (requires 2 /? (a * b)) (ensures 2 /? a \/ 2 /? b)
=
  if 2 /? a then ()
  else if 2 /? b then ()
  else (
    div2_lemma a;
    div2_lemma b;
    div2_prod_odd a b
  )

let rec factor_pow2 (n : nat) (a b : nat)
: Ghost (nat & nat)
  (requires a * b == pow2 n)
  (ensures fun (p, q) -> pow2 p == a /\ pow2 q == b)
=
  if n = 0
    then (0, 0)
    else (
      assert a * b == 2 * pow2 (n - 1);
      div2_lemma_prod a b;
      if 2 /? a
        then let (p, q) = factor_pow2 (n - 1) (a / 2) b in (p + 1, q)
        else let (p, q) = factor_pow2 (n - 1) a (b / 2) in (p, q + 1)
    )


let pow2_div_log (n a : nat)
: Lemma (requires a /? pow2 n) (ensures exists r. pow2 r == a)
= let _ = (factor_pow2 n a (pow2 n / a)) in ()

let pow_div_lemma (n : nat) (a b : nat)
: Lemma
  (requires a /? (pow2 n) /\ b /? (pow2 n) /\ a <= b)
  (ensures a /? b)
= pow2_div_log n a; pow2_div_log n b

let round2_lemma (a b n k : nat)
: Lemma
  (requires a /? pow2 k /\  b /? pow2 k /\ a <= b)
  (ensures a /? round2 b n)
=
  pow_div_lemma k a b;
  assert a /? b;
  lemma_divides_product_r a (n / b) b;
  assert a /? ((n / b) * b);
  assert a /? round2 b n;
  ()

let round2_chunk_lemma
  (a b : Type0) {| sized a, has_vec_cpy a, sized b, has_vec_cpy b |}
  (n : nat)
: Lemma
    (ensures
      chunk a /? round2 (max (chunk a) (chunk b)) n /\
      chunk b /? round2 (max (chunk a) (chunk b)) n
    )
=
  let m = max (chunk a) (chunk b) in
  round2_lemma (chunk a) m n 4;
  round2_lemma (chunk b) m n 4;
  ()

let intro_divides (a b c : nat)
: Lemma (requires a * b = c) (ensures a /? c)
= ()

let prod_divides (a b c : pos)
: Lemma (requires (a * b) /? c) (ensures a /? c /\ b /? c)
=
  let k = c / (a * b) in
  intro_divides a (k * b) c;
  intro_divides b (k * a) c;
  ()

let lineal_divides (d : pos) (a b k : nat)
: Lemma (requires d /? a /\ d /? b) (ensures d /? (a + k * b) /\ d /? (k * b + a))
=
  lemma_divides_product_r d k b;
  lemma_divides_sum d a (k * b)

let prod_preserves_divides (c d : pos) (a : nat)
: Lemma (requires c /? a) (ensures (c * d) /? (a * d))
=
  lemma_divides_product_l c a d;
  lemma_divides_exact c (a * d);
  intro_divides (c * d) (a / c) (a * d);
  admit()

let divides_leq
  (d : pos)
  (a b : nat)
: Lemma
  (requires d /? a /\ d /? b)
  (ensures b < a <==> b + d <= a)
= lemma_divides_exact d a;
  lemma_divides_exact d b;
  admit()

let divides_chain (a b c : nat)
: Lemma (requires a /? b /\ b /? c) (ensures a /? c)
= ()

let prod_cancel_divides (a : nat) (c d : pos)
: Lemma (requires (c * d) /? a) (ensures c /? (a / d))
=
  let open FStar.Math.Lemmas in
  calc (==) {
    a;
    == { cancel_mul_div a (c * d) }
    a / (c * d) * (c * d);
    == { paren_mul_right (a / (c * d)) c d; paren_mul_right (a / (c * d)) c d }
    (a / (c * d) * c) * d;
  };
  calc (==) {
    a / d;
    == {}
    ((a / (c * d) * c) * d) / d;
    == { cancel_mul_div (a / (c * d) * c) d }
    a / (c * d) * c;
  };
  intro_divides c (a / (c * d)) (a / d)

let block_lemma whole block k
  : Lemma (requires block /? whole /\ k * block < whole)
          (ensures k * block + block <= whole)
  = if block > 0 then begin
      lemma_divides_exact block whole;
      FStar.Math.Lemmas.multiplication_order_lemma k (whole / block) block;
      FStar.Math.Lemmas.lemma_mult_le_right block (k + 1) (whole / block)
    end

#push-options "--z3rlimit 30"
let rounded_offset_aligned
  (#t1 t2 : Type0) {| sized t1, has_vec_cpy t1, sized t2, has_vec_cpy t2 |}
  (#n : nat)
  (x : larray t1 n { aligned 16 x })
  (offset : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x (round2 (max (chunk t1) (chunk t2)) offset))
=
  let offset' = round2 (max (chunk t1) (chunk t2)) offset in
  round2_chunk_lemma t1 t2 offset;
  assert chunk t1 /? offset'

let step_aligned
  (#t1 : Type0) {| sized t1, has_vec_cpy t1 |}
  (#n : nat)
  (x : larray t1 n { aligned 16 x })
  (offset : nat)
  (step : nat { chunk t1 /? step })
  (k : nat)
: Lemma
  (requires aligned' 16 x offset)
  (ensures  aligned' 16 x (offset + k * step))
=
  assert chunk t1 /? offset;
  lineal_divides (chunk t1) offset step k
#pop-options