module Kuiper.Approximates.Base

open Kuiper
open Kuiper.Scalars

(* This class is meant for scalar types that can "approximate" or
"model" real numbers. *)
[@@FStar.Tactics.Typeclasses.fundeps [1]]
// ^ This is odd, but needed. Otherwise we cannot use a hypothesis
// like `real_like a #d` to solve a goal `real_like a #?u`.
class real_like (a:Type) {| scalar a |} = {
  to_real : a -> real;

  v_approximates : a -> real -> prop;

  to_real_ok :
    x:a ->
    Lemma (ensures x `v_approximates` to_real x);
      //     [SMTPat (x `v_approximates` to_real x)];

  (* We assume these two values exist and approximate correctly *)

  a0 : squash (v_approximates zero 0.0R);
  a1 : squash (v_approximates one 1.0R);

  // It would be nice if we could directly write SMT patterns on
  // the lemmas below, but they do not trigger.

  a_add : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires v_approximates x r /\ v_approximates y s)
                (ensures v_approximates (x `add` y) (r +. s));
                // [SMTPat (v_approximates x r); SMTPat (v_approximates y s)]

  a_mul : x:a -> y:a -> r:real -> s:real ->
          Lemma (requires v_approximates x r /\ v_approximates y s)
                (ensures v_approximates (x `mul` y) (r *. s));
                //[SMTPat (v_approximates x r); SMTPat (v_approximates y s)];
}
