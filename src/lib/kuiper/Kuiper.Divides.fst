module Kuiper.Divides

open FStar.Mul
module M = FStar.Math.Lemmas

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

(* add reciprocal too? *)
let lemma_divides_exact (x:pos) (y:int)
  : Lemma (x /? y <==> x * (y/x) == y)
= lemma_divides_mod1 x y;
  Classical.move_requires (M.lemma_div_exact y) x

let lemma_divides_le (x : nat) (y : pos)
  : Lemma (requires x /? y)
          (ensures x <= y)
          [SMTPat (x /? y)]
= ()
