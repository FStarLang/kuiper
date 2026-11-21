module Kuiper.Test.FloatingModel

friend Kuiper.Float32.Base
friend Kuiper.Float32

open Kuiper.Real
open Kuiper.Float.Model
open Kuiper.Float32
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
module Fast = Kuiper.Float32.FastMath

[@@expect_failure [19]]
let model_does_not_prove_false () : Lemma False = ()

(* These are proofs about the implementation, not additional interface axioms. *)
let approximation_is_functional (x:t) (r s:real)
  : Lemma (requires v_approximates x r /\ v_approximates x s)
          (ensures r == s)
  = ()

let infinity_subtraction_changes_only_the_ideal ()
  : Lemma (
      let x = infinity #t in
      let y = Fast.pure_sub_rn x one in
      bit_eq x y /\ x =!= y /\ v_approximates y (0.0R -. 1.0R))
  =
  let x = infinity #t in
  let y = Fast.pure_sub_rn x one in
  assert (v_approximates x 0.0R);
  assert (v_approximates y (0.0R -. 1.0R));
  if x == y then approximation_is_functional x 0.0R (0.0R -. 1.0R);
  bit_eq_spec y x;
  bit_eq_spec x y

let exponential_zero_is_not_canonical_zero ()
  : Lemma (
      let x = fexp (neg (infinity #t)) in
      bit_eq x (zero #t) /\ x =!= (zero #t) /\ v_approximates x 1.0R /\
      ~(v_approximates (zero #t) 1.0R))
  =
  exp_special_values (neg (infinity #t));
  FStar.Math.Exp.exp_base ();
  let x = fexp (neg (infinity #t)) in
  assert (v_approximates x 1.0R);
  if x == (zero #t) then approximation_is_functional x 1.0R 0.0R;
  if v_approximates (zero #t) 1.0R then
    approximation_is_functional (zero #t) 0.0R 1.0R
