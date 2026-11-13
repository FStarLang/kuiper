module FStar.Math.Exp

open FStar.Real

(* Exponentiation *)
val exp (x:real) : real

(* Logarithm *)
val log (x:real{x >. 0.0R}) : real

(* Usual math laws about exponentiation and logarithms. *)

val exp_positive (x : real)
  : Lemma (ensures exp x >. 0.0R)
          [SMTPat (exp x)]

val exp_base ()
  : Lemma (exp 0.0R == 1.0R)

val exp_add (x y : real)
  : Lemma (ensures exp (x +. y) == exp x *. exp y)
          [SMTPat (exp (x +. y))]

val log_exp (x : real)
  : Lemma (ensures log (exp x) == x)
          [SMTPat (log (exp x))]

val exp_log (x : real{x >. 0.0R})
  : Lemma (ensures exp (log x) == x)
          [SMTPat (exp (log x))]

val exp_sub (x y : real)
  : Lemma (ensures exp (x -. y) == exp x /. exp y)
          [SMTPat (exp (x -. y))]

val log_mul (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures log (x *. y) == log x +. log y)
          [SMTPat (log (x *. y))]

val log_div (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures log (x /. y) == log x -. log y)
          [SMTPat (log (x /. y))]

(* Checked consequences of the constructed exponential normalization. *)
val exp2_as_exp (x:real)
  : Lemma (FStar.Math.Pow.exp2 x == exp (x *. log 2.0R))

val log_two_positive ()
  : Lemma (log 2.0R >. 0.0R)

val log_two_decimal_bounds ()
  : Lemma (
      69314718055994530.0R /. 100000000000000000.0R <. log 2.0R /\
      log 2.0R <. 69314718055994532.0R /. 100000000000000000.0R)
