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
