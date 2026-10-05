module Kuiper.Approximates.Scalar

include Kuiper.Approximates.Core

open Kuiper.Real
open Kuiper.Float.Realops
open Kuiper.Scalars

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

  (* SMT triggers live on wrappers below, not these fields: putting them on
     the fields introduces matching loops (FStarLang/FStar#4264). *)

  a0 : squash (v_approximates zero r_zero);

  a1 : squash (v_approximates one r_one);

  a_add : (x:a) -> (y:a) -> (r:real) -> (s:real) -> Lemma (requires v_approximates x r /\ v_approximates y s)
        (ensures v_approximates (add x y) (r_add r s));

  a_mul : (x:a) -> (y:a) -> (r:real) -> (s:real) -> Lemma (requires v_approximates x r /\ v_approximates y s)
        (ensures v_approximates (mul x y) (r_mul r s));

}

unfold instance real_like_can_approximate (#a:Type) (_ : scalar a) (_ : real_like a)
  : can_approximate a real = {
  approximates = v_approximates;
}

(* Refined real domains (log, division, sqrt) use the same scalar relation.
   This instance has no recursive can_approximate premise. *)
unfold instance refined_real_can_approximate (#a:Type) (p:real -> prop)
  (_ : scalar a) (_ : real_like a)
  : can_approximate a (r:real{p r}) = {
  approximates = (fun (x:a) (r:real{p r}) -> v_approximates x r);
}

let to_real_ok_pat
  (a:Type) {| scalar a, real_like a |}
  (x : a) :
          Lemma (ensures x `v_approximates` to_real x)
                [SMTPat (v_approximates x (to_real x))]
  = to_real_ok x

let a_add_pat
  (a:Type) {| scalar a, real_like a, ar : real_like a |}
  (x:a) (y:a) (r:real) (s:real)
  : Lemma (requires v_approximates x r /\ v_approximates y s)
      (ensures v_approximates (add x y) (r_add r s))
          [SMTPat (add x y);
           SMTPat (v_approximates x r);
           SMTPat (v_approximates y s);
           SMTPat (has_type ar (real_like a))]
  = a_add x y r s

let a_mul_pat
  (a:Type) {| scalar a, real_like a, ar : real_like a |}
  (x:a) (y:a) (r:real) (s:real)
  : Lemma (requires v_approximates x r /\ v_approximates y s)
      (ensures v_approximates (mul x y) (r_mul r s))
          [SMTPat (mul x y);
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
