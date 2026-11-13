module Luce.RoPE.Constants.Power

open FStar.Real
module Pow = FStar.Math.Pow

noextract
let rec ipow (x:pos) (n:nat) : Tot pos (decreases n) =
  if n = 0 then 1 else x * ipow x (n - 1)

let rec one_power (n:nat)
  : Lemma (ensures ipow 1 n == 1) (decreases n) =
  if n > 0 then one_power (n - 1)

let integer_product (x y:pos)
  : Lemma (of_int (x * y) == of_int x *. of_int y) = ()

let div_cancel (x:real) (y:Pow.rpos)
  : Lemma ((x /. y) *. y == x) = ()

let multiply_cancel (x y:real) (d:Pow.rpos)
  : Lemma (requires x *. d == y *. d) (ensures x == y) = ()

let fractions_product (a c:real) (b d:Pow.rpos)
  : Lemma ((a /. b) *. (c /. d) == (a *. c) /. (b *. d)) =
  div_cancel a b;
  div_cancel c d;
  div_cancel (a *. c) (b *. d);
  let lhs = (a /. b) *. (c /. d) in
  let rhs = (a *. c) /. (b *. d) in
  assert (lhs *. (b *. d) == a *. c);
  multiply_cancel lhs rhs (b *. d)

let rec rational_power (num den:pos) (n:nat)
  : Lemma
      (ensures Pow.pow (of_int num /. of_int den) (of_int n) ==
        of_int (ipow num n) /. of_int (ipow den n))
      (decreases n) =
  if n = 0 then Pow.pow_zero (of_int num /. of_int den)
  else (
    rational_power num den (n - 1);
    let pn = ipow num (n - 1) in
    let pd = ipow den (n - 1) in
    integer_product num pn;
    integer_product den pd;
    fractions_product (of_int num) (of_int pn) (of_int den) (of_int pd);
    assert (of_int (ipow num n) == of_int num *. of_int pn);
    assert (of_int (ipow den n) == of_int den *. of_int pd);
    Pow.pow_succ (of_int num /. of_int den) (n - 1)
  )

let rec power_strict_mono (x:Pow.rpos) (y:Pow.rpos{x <. y}) (n:pos)
  : Lemma
      (ensures Pow.pow x (of_int n) <. Pow.pow y (of_int n))
      (decreases n) =
  if n = 1 then (
    Pow.pow_one x; Pow.pow_one y
  ) else (
    power_strict_mono x y (n - 1);
    Pow.pow_succ x (n - 1);
    Pow.pow_succ y (n - 1)
  )

let enclosure (base:pos) (m:nat) (n:pos) (lo hi den:pos)
  : Lemma
      (requires
        ipow lo n < ipow base m * ipow den n /\
        ipow base m * ipow den n < ipow hi n)
      (ensures
        of_int lo /. of_int den <=. Pow.pow (of_int base) (of_int m /. of_int n) /\
        Pow.pow (of_int base) (of_int m /. of_int n) <=. of_int hi /. of_int den) =
  let lower = of_int lo /. of_int den in
  let upper = of_int hi /. of_int den in
  let value = Pow.pow (of_int base) (of_int m /. of_int n) in
  rational_power lo den n;
  rational_power hi den n;
  rational_power base 1 m;
  one_power m;
  integer_product (ipow base m) (ipow den n);
  Pow.pow_rational (of_int base) m n;
  assert (Pow.pow lower (of_int n) <. Pow.pow (of_int base) (of_int m));
  assert (Pow.pow (of_int base) (of_int m) <. Pow.pow upper (of_int n));
  if value <. lower then power_strict_mono value lower n;
  if upper <. value then power_strict_mono upper value n
