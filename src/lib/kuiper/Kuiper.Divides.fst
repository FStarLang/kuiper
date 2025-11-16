module Kuiper.Divides

open FStar.Mul
module M = FStar.Math.Lemmas

let get_factor (x y : int)
  : Ghost int (requires x /? y)
              (ensures fun z -> x * z == y)
  = FStar.IndefiniteDescription.indefinite_description_ghost int (fun z -> x * z == y)

let lemma_divides_mod1 (x:pos) (y:int)
  : Lemma (x /? y ==> y % x == 0)
= let aux (z:int) : Lemma (requires z * x == y) (ensures y % x == 0) =
    calc (==) {
      y % x;
      == {}
      (z * x) % x;
      == { M.cancel_mul_mod z x }
      0;
    }
  in
  Classical.forall_intro (Classical.move_requires aux)

let lemma_divides_mod2 (x:pos) (y:int)
  : Lemma (requires y % x == 0) (ensures x /? y)
= M.lemma_div_exact y x;
  assert (x * (y/x) == y)

let lemma_divides_mod (x:pos) (y : int)
  : Lemma (x /? y <==> y % x == 0)
          [SMTPat (x /? y)]
= lemma_divides_mod1 x y;
  let aux (x:pos) (y:int) : Lemma (y%x == 0 ==> x /? y) =
    Classical.move_requires (lemma_divides_mod2 x) y
  in
  Classical.forall_intro_2 aux;
  ()

let lemma_divides_product (x y : int)
  : Lemma (x /? (x * y)  /\  x /? (y * x))
          [SMTPatOr [[SMTPat (x /? (x * y))];
                     [SMTPat (x /? (y * x))]]]
  = ()

let lemma_divides_exact (x:pos) (y:int)
  : Lemma (x /? y <==> x * (y/x) == y)
= lemma_divides_mod1 x y;
  Classical.move_requires (M.lemma_div_exact y) x

let lemma_nat_divides_pos_divides (x: pos) (y: int)
: Lemma (x /? y <==> x /?+ y)
= ()

let lemma_divides_le (x : nat) (y : pos)
  : Lemma (requires x /? y)
          (ensures x <= y)
          [SMTPat (x /? y)]
= ()

let lemma_divides_trans (x y z : pos)
  : Lemma (requires x /? y /\ y /? z)
          (ensures x /? z)
          [SMTPat (x /? y); SMTPat (y /? z)]
  = let f1 = get_factor x y in
    let f2 = get_factor y z in
    assert (x * (f1*f2) == z);
    ()

let rec lemma_pow2_div (x y : nat)
  : Lemma (requires x <= y)
          (ensures pow2 x /? pow2 y)
          (decreases (y-x))
          [SMTPat (pow2 x /? pow2 y)]
= if x = y then ()
  else (
    lemma_pow2_div (x+1) y;
    lemma_divides_trans (pow2 x) (pow2 (x+1)) (pow2 y)
  )

let lem_divup_back (m:nat) (k:pos)
  : Lemma (k * divup m k >= m)
          [SMTPat (divup m k)]
  = ()

let lem_divup_divides (m:nat) (k:pos)
  : Lemma (k /? m ==> divup m k === m / k)
          [SMTPat (divup m k)]
  = ()

let lemma_divides_sum (d : pos) (a b : int)
  : Lemma (requires d /? a /\ d /? b)
          (ensures d /? (a + b))
  = calc (==) {
      d * (a/d + b/d);
      == {}
      d * (a/d) + d * (b/d);
      == {}
      a + b;
  }

let lemma_divides_product_l (d : pos) (a c : int)
  : Lemma (requires d /? a)
          (ensures d /? (a * c))
  = calc (==) {
      d * ((a/d) * c);
      == {}
      (d * (a/d)) * c;
      == {}
      a * c;
  }

let lemma_divides_product_r (d : pos) (a b : int)
  : Lemma (requires d /? b)
          (ensures d /? (a * b))
  = lemma_divides_product_l d b a

let lemma_divides_chain (a b c : pos)
  : Lemma (requires a /? b /\ b /? c)
          (ensures a /? c)
  = ()

let lemma_divides_mod_op (d: pos) (a: int) (b: pos)
  : Lemma (requires d /? a /\
            d /? b)
          (ensures d /? (a % b))
= assert ((a % b) == a + ((-1) * (a / b)) * b);
  lemma_divides_product_r d ((-1) * (a / b)) b;
  lemma_divides_sum d a (((-1) * (a / b)) * b);
  ()

let lemma_eucl_unique
  (b: pos)
  (q r q' r': nat)
: Lemma (requires q * b + r == q' * b + r' /\
    r < b /\
    r' < b
  )
  (ensures q == q' /\ r == r')
= ()
