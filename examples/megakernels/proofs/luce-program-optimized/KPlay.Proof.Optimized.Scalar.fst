module KPlay.Proof.Optimized.Scalar

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open FStar.Real
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Neg = Kuiper.Float32.NativeNeg
module Approx = Kuiper.Approximates.Base
module Exp = FStar.Math.Exp
module Pow = FStar.Math.Pow
module Lit = Kuiper.Float32.Literals
module R = KPlay.Proof.Shared.Recurrent

inline_for_extraction noextract
let log2e () : real =
  Exp.log_two_positive ();
  1.0R /. Exp.log 2.0R

let log2e_bounds ()
  : Lemma (
      14426950408889633.0R /. 10000000000000000.0R <=. log2e () /\
      log2e () <=. 14426950408889635.0R /. 10000000000000000.0R) =
  Exp.log_two_decimal_bounds ();
  Exp.log_two_positive ()

let log2e_literal ()
  : Lemma (Approx.v_approximates
      (F.of_literal #FP.t "1.44269504088896340736") (log2e ())) =
  assert_norm (Lit.common_normal_cell "1.44269504088896340736"
    14426950408889633 10000000000000000
    14426950408889635 10000000000000000 12102203 (-23));
  log2e_bounds ();
  FP.literal_interval_approx "1.44269504088896340736"
    14426950408889633 10000000000000000
    14426950408889635 10000000000000000 12102203 (-23) (log2e ())

let exponential_identity (x:real)
  : Lemma (Pow.exp2 (x *. log2e ()) == Exp.exp x) =
  Exp.log_two_positive ();
  Exp.exp2_as_exp (x *. log2e ())

let exponential_result (x scaled result:FP.t)
  : Lemma
      (requires
        (forall (xr yr:real).
          Approx.v_approximates x xr /\
          Approx.v_approximates (F.of_literal #FP.t "1.44269504088896340736") yr ==>
          Approx.v_approximates scaled (xr *. yr)) /\
        (forall (r:real).
          Approx.v_approximates scaled r ==>
          Approx.v_approximates result (Pow.exp2 r)))
      (ensures forall (r:real).
        Approx.v_approximates x r ==>
        Approx.v_approximates result (Exp.exp r)) =
  log2e_literal ();
  let related (r:real{Approx.v_approximates x r})
    : Lemma (Approx.v_approximates result (Exp.exp r)) =
    exponential_identity r;
    assert (Approx.v_approximates scaled (r *. log2e ()))
  in FStar.Classical.forall_intro related

(* reduced.cu:44-49, including the FTZ multiplication before EX2. *)
noextract
fn exp32 (x:FP.t)
  preserves gpu
  returns result:FP.t
  ensures pure (forall (r:real).
    Approx.v_approximates x r ==> Approx.v_approximates result (Exp.exp r))
{
  let scaled = FP.mul_rn_ftz x (F.of_literal #FP.t "1.44269504088896340736");
  let result = FP.exp2_approx_ftz scaled;
  exponential_result x scaled result;
  result
}

let sigmoid_argument (x exponential argument:FP.t)
  : Lemma
      (requires
        (forall (r:real).
          Approx.v_approximates (Neg.neg x) r ==>
          Approx.v_approximates exponential (Exp.exp r)) /\
        (forall (a b:real).
          Approx.v_approximates (F.one #FP.t) a /\
          Approx.v_approximates exponential b ==>
          Approx.v_approximates argument (a +. b)))
      (ensures forall (r:real).
        Approx.v_approximates x r ==>
        Approx.v_approximates argument (1.0R +. Exp.exp (0.0R -. r)) /\
        1.0R +. Exp.exp (0.0R -. r) >. 0.0R) =
  let related (r:real{Approx.v_approximates x r})
    : Lemma (
        Approx.v_approximates argument (1.0R +. Exp.exp (0.0R -. r)) /\
        1.0R +. Exp.exp (0.0R -. r) >. 0.0R) =
    Neg.neg_approx x r;
    Exp.exp_positive (0.0R -. r)
  in FStar.Classical.forall_intro related

noextract
fn sigmoid32 (x:FP.t)
  preserves gpu
  returns result:FP.t
  ensures pure (forall (r:real).
    Approx.v_approximates x r ==> Approx.v_approximates result (R.sigmoid r))
{
  let exponential = exp32 (Neg.neg x);
  let argument = FP.add_rn_ftz (F.one #FP.t) exponential;
  sigmoid_argument x exponential argument;
  FP.rcp_approx_ftz argument
}

noextract
fn silu32 (x:FP.t)
  preserves gpu
  returns result:FP.t
  ensures pure (forall (r:real).
    Approx.v_approximates x r ==> Approx.v_approximates result (R.silu r))
{
  let sigmoid = sigmoid32 x;
  FP.mul_rn_ftz x sigmoid
}
