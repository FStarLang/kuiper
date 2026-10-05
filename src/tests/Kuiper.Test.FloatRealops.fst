module Kuiper.Test.FloatRealops

open Kuiper
open Kuiper.Float.Realops

(* The public operations share the same models across all four formats. *)
let expm1_models () : Lemma (
    Kuiper.Float16.fexpm1 %~ r_expm1 /\
    Kuiper.BFloat16.fexpm1 %~ r_expm1 /\
    Kuiper.Float32.fexpm1 %~ r_expm1 /\
    Kuiper.Float64.fexpm1 %~ r_expm1) = ()

let log1p_models () : Lemma (
    Kuiper.Float16.flog1p %~ r_log1p /\
    Kuiper.BFloat16.flog1p %~ r_log1p /\
    Kuiper.Float32.flog1p %~ r_log1p /\
    Kuiper.Float64.flog1p %~ r_log1p) = ()

(* Clients can still prove the original pointwise mathematical statements. *)
let log1p_application (x:f16) (r:real{r >. 0.0R -. 1.0R})
  : Lemma (requires x %~ r)
          (ensures Kuiper.Float16.flog1p x %~ log (1.0R +. r)) = ()

let generic_models #a {| scalar a, floating a, real_like a, floating_real_like a |} ()
  : Lemma ((fmax #a) %~ r_fmax /\ (sub #a) %~ r_sub /\
           (fexp #a) %~ r_exp /\ (div #a) %~ r_div /\
           (flog #a) %~ r_log /\ (sqrt #a) %~ r_sqrt /\
           (rsqrt #a) %~ r_rsqrt) = ()

let division_application #a
  {| scalar a, floating a, real_like a, floating_real_like a |}
  (x y:a) (r:real) (s:real{s =!= 0.0R})
  : Lemma (requires x %~ r /\ y %~ s)
          (ensures div x y %~ (r /. s)) = ()

let sqrt_application #a
  {| scalar a, floating a, real_like a, floating_real_like a |}
  (x:a) (r:FStar.Math.Sqrt.rnonneg)
  : Lemma (requires x %~ r)
          (ensures sqrt x %~ FStar.Math.Sqrt.sqrt r) = ()

let integer_application #a
  {| scalar a, floating a, real_like a, floating_real_like a |}
  (x:FStar.Int64.t)
  : Lemma (Kuiper.Floating.Base.of_int #a x %~ r_of_int x) =
  approx_apply (Kuiper.Floating.Base.of_int #a) r_of_int x x

[@@expect_failure [19]]
let log1p_requires_domain (x:real) : real = r_log1p x

[@@expect_failure [19]]
let division_requires_nonzero (x y:real) : real = r_div x y
