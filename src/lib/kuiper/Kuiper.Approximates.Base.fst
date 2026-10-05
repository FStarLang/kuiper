module Kuiper.Approximates.Base

include Kuiper.Approximates.Scalar

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base

(* Extra rules for types supporting division and exponentiation. *)
class floating_real_like (a:Type) {| scalar a, floating a, real_like a |} = {
  (* Real approximation ignores the sign of zero. This is a property of
     this relation, not permission to substitute IEEE-equal floats anywhere. *)
  eq_approx : (x:a) -> (y:a) -> (r:real) ->
    Lemma (requires eq x y /\ v_approximates y r)
          (ensures v_approximates x r);

  of_int_approx : (x : Int64.t) ->
    squash (v_approximates (of_int #a x) (Real.of_int (Int64.v x)));

  fmax_approx : (x: a) -> (y: a) -> (xr: real) -> (yr: real) ->
    Lemma (requires v_approximates x xr /\ v_approximates y yr)
          (ensures v_approximates (fmax x y) (rmax xr yr));

  sub_approx : x:a -> y:a -> r:real -> s:real ->
                Lemma (requires v_approximates x r /\ v_approximates y s)
                      (ensures v_approximates (sub x y) (r -. s));

  exp_approx : x:a -> r:real ->
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (fexp x) (exp r));

  div_approx : x:a -> y:a -> r:real -> s:real{s =!= 0.0R} ->
                Lemma (requires v_approximates x r /\ v_approximates y s)
                      (ensures v_approximates (div x y) (r /. s));

  log_approx : x:a -> r:real{r >. 0.0R} ->
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (flog x) (log r));

  sqrt_approx : x:a -> r:FStar.Math.Sqrt.rnonneg ->
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (sqrt x) (FStar.Math.Sqrt.sqrt r));

  rsqrt_approx : x:a -> r:FStar.Math.Sqrt.rpos ->
                Lemma (requires v_approximates x r)
                      (ensures v_approximates (rsqrt x) (1.0R /. FStar.Math.Sqrt.sqrt r));
}

let eq_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x y:a) (r:real)
  : Lemma (requires eq x y /\ v_approximates y r)
          (ensures v_approximates x r)
          [SMTPat (eq x y);
           SMTPat (v_approximates y r);
           SMTPat (v_approximates x r);
           SMTPat (has_type rr (floating_real_like a))]
  = eq_approx x y r

let fmax_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x y : a) (xr yr : real) :
    Lemma (requires v_approximates x xr /\ v_approximates y yr)
          (ensures v_approximates (fmax x y) (rmax xr yr))
          [SMTPat (fmax x y);
           SMTPat (v_approximates x xr);
           SMTPat (v_approximates y yr);
           SMTPat (has_type rr (floating_real_like a))]
  = fmax_approx x y xr yr

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
                      (ensures v_approximates (fexp x) (exp r))
                      [SMTPat (fexp x);
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
                      (ensures v_approximates (flog x) (log r))
                      [SMTPat (flog x);
                       SMTPat (v_approximates x r);
                       SMTPat (has_type rr (floating_real_like a))]
  = log_approx x r

let sqrt_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x : a) (r : FStar.Math.Sqrt.rnonneg) :
    Lemma (requires v_approximates x r)
          (ensures v_approximates (sqrt x) (FStar.Math.Sqrt.sqrt r))
          [SMTPat (sqrt x);
           SMTPat (v_approximates x r);
           SMTPat (has_type rr (floating_real_like a))]
  = sqrt_approx x r

let rsqrt_approx_pat
  (a:Type) {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  (x : a) (r : FStar.Math.Sqrt.rpos) :
    Lemma (requires v_approximates x r)
          (ensures v_approximates (rsqrt x) (1.0R /. FStar.Math.Sqrt.sqrt r))
          [SMTPat (rsqrt x);
           SMTPat (v_approximates x r);
           SMTPat (has_type rr (floating_real_like a))]
  = rsqrt_approx x r
