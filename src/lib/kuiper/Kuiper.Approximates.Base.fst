module Kuiper.Approximates.Base

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base

(* This class is meant for scalar types that can "approximate" or
"model" real numbers. *)
[@@FStar.Tactics.Typeclasses.fundeps [1]]
// ^ This is odd, but needed. Otherwise we cannot use a hypothesis
// like `real_like a #d` to solve a goal `real_like a #?u`.
class real_like (a:Type) {| scalar a |} = {
  to_real : a -> real;

  v_approximates : a -> real -> prop;

  to_real_ok : x:a ->
    Lemma (ensures x `v_approximates` to_real x);

  a0 : squash (v_approximates zero 0.0R);
  a1 : squash (v_approximates one 1.0R);

  // It would be nice if we could directly write SMT patterns on the lemmas below.
  a_add : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires v_approximates x r /\ v_approximates y s)
                (ensures v_approximates (x `add` y) (r +. s));
                // [SMTPat (v_approximates x r); SMTPat (v_approximates y s)]

  a_mul : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires v_approximates x r /\ v_approximates y s)
                (ensures v_approximates (x `mul` y) (r *. s));
                //[SMTPat (v_approximates x r); SMTPat (v_approximates y s)];
}

let to_real_ok_pat 
  (a:Type) {| scalar a, real_like a |}
  (x : a) :
          Lemma (ensures x `v_approximates` to_real x)
                [SMTPat (v_approximates x (to_real x))]
  = to_real_ok x

let a_add_pat
  (a:Type) {| scalar a, real_like a, ar : real_like a |}
  (x y : a) (r s : real) :
          Lemma (requires v_approximates x r /\ v_approximates y s)
                (ensures v_approximates (x `add` y) (r +. s))
                [SMTPat (x `add` y);
                 SMTPat (v_approximates x r);
                 SMTPat (v_approximates y s);
                 SMTPat (has_type ar (real_like a))]
  = a_add x y r s

let a_mul_pat
  (a:Type) {| scalar a, real_like a, ar : real_like a |}
  (x y : a) (r s : real) :
          Lemma (requires v_approximates x r /\ v_approximates y s)
                (ensures v_approximates (x `mul` y) (r *. s))
                [SMTPat (x `mul` y);
                 SMTPat (v_approximates x r);
                 SMTPat (v_approximates y s);
                 SMTPat (has_type ar (real_like a))]
  = a_mul x y r s

[@@FStar.Tactics.Typeclasses.fundeps [1]]
class precise_real_like (a:Type) {| scalar a, real_like a |} = {
  v_approximates_inj : (x: a -> y: a -> r: real ->
    Lemma (requires v_approximates x r /\ v_approximates y r)
          (ensures x == y));
}

let approx_to_real #a {| scalar a, real_like a, precise_real_like a |} (x y: a) :
    Lemma (requires v_approximates x (to_real y))
          (ensures x == y) =
  to_real_ok y;
  v_approximates_inj x y (to_real y)

(* Extra rules for types supporting division and exponentiation. *)
class floating_real_like (a:Type) {| scalar a, floating a, real_like a |} = {
  sub_approx : x:a -> y:a -> r:real -> s:real ->
                Lemma (requires v_approximates x r /\ v_approximates y s)
                      (ensures v_approximates (sub x y) (r -. s));

  exp_approx : x:a -> r:real ->
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (exp x) (rexp r));

  div_approx : x:a -> y:a -> r:real -> s:real{s =!= 0.0R} ->
                Lemma (requires v_approximates x r /\ v_approximates y s)
                      (ensures v_approximates (div x y) (r /. s));

  log_approx : x:a -> r:real{r >. 0.0R} ->
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (log x) (rlog r));
}

let sub_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x y : a) (r s : real) :
                Lemma (requires v_approximates x r /\ v_approximates y s)
                      (ensures v_approximates (sub x y) (r -. s))
                      [SMTPat (sub x y);
                       SMTPat (v_approximates x r);
                       SMTPat (v_approximates y s);
                       SMTPat (has_type rr (floating_real_like a))]
  = sub_approx x y r s

let exp_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x : a) (r : real) :
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (exp x) (rexp r))
                      [SMTPat (exp x);
                       SMTPat (v_approximates x r);
                       SMTPat (has_type rr (floating_real_like a))]
  = exp_approx x r

let div_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x y : a) (r s : real{s =!= 0.0R}) :
                Lemma (requires v_approximates x r /\ v_approximates y s)
                      (ensures v_approximates (div x y) (r /. s))
                      [SMTPat (div x y);
                       SMTPat (v_approximates x r);
                       SMTPat (v_approximates y s);
                       SMTPat (has_type rr (floating_real_like a))]
  = div_approx x y r s

let log_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x : a) (r : real{r >. 0.0R}) :
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (log x) (rlog r))
                      [SMTPat (log x);
                       SMTPat (v_approximates x r);
                       SMTPat (has_type rr (floating_real_like a))]
  = log_approx x r
