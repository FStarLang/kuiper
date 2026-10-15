module Kuiper.Approximates.Base

include Kuiper.Approximates.Scalar

open Kuiper.Real
open Kuiper.Float.Realops
open Kuiper.Scalars
open Kuiper.Floating.Base

(* Extra rules for types supporting division and exponentiation. *)
class floating_real_like (a:Type) {| scalar a, floating a, real_like a |} = {
  (* Real approximation ignores the sign of zero. This is a property of
     this relation, not permission to substitute IEEE-equal floats anywhere. *)
  eq_approx : (x:a) -> (y:a) -> (r:real) ->
    Lemma (requires eq x y /\ v_approximates y r)
          (ensures v_approximates x r);

  approx_of_int : squash ((of_int #a) %~ r_of_int);
  approx_fmax : squash ((fmax #a) %~ r_fmax);
  approx_sub : squash ((sub #a) %~ r_sub);
  approx_fexp : squash ((fexp #a) %~ r_exp);
  approx_div : squash ((div #a) %~ r_div);
  approx_flog : squash ((flog #a) %~ r_log);
  approx_sqrt : squash ((sqrt #a) %~ r_sqrt);
  approx_rsqrt : squash ((rsqrt #a) %~ r_rsqrt);
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

(* Expose each function contract when its operation occurs in a proof. *)

let of_int_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (of_int #a) %~ r_of_int)
      [SMTPat (of_int #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_of_int #a

let fmax_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (fmax #a) %~ r_fmax)
      [SMTPat (fmax #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_fmax #a

let sub_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (sub #a) %~ r_sub)
      [SMTPat (sub #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_sub #a

let fexp_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (fexp #a) %~ r_exp)
      [SMTPat (fexp #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_fexp #a

let div_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (div #a) %~ r_div)
      [SMTPat (div #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_div #a

let flog_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (flog #a) %~ r_log)
      [SMTPat (flog #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_flog #a

let sqrt_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (sqrt #a) %~ r_sqrt)
      [SMTPat (sqrt #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_sqrt #a

let rsqrt_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (rsqrt #a) %~ r_rsqrt)
      [SMTPat (rsqrt #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_rsqrt #a
