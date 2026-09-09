module Kuiper.Spec.Sigmoid

open Kuiper.Real

(* Mathematical sigmoid, independent of any floating-point implementation. *)
let sigmoid_real (x : real) : Tot real =
  exp_positive (0.0R -. x);
  1.0R /. (1.0R +. exp (0.0R -. x))

let sigmoid_unit_interval (x : real) : Lemma
  (0.0R <. sigmoid_real x /\ sigmoid_real x <. 1.0R)
= exp_positive (0.0R -. x)

let sigmoid_at_zero () : Lemma (sigmoid_real 0.0R == 0.5R)
= exp_base ()

let sigmoid_symmetry (x : real) : Lemma
  (sigmoid_real (0.0R -. x) == 1.0R -. sigmoid_real x)
=
  exp_base ();
  exp_sub 0.0R x;
  exp_positive x;
  exp_positive (0.0R -. x)
