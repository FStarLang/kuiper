module Kuiper.Approximates.Base

include Kuiper.Approximates.Scalar

open Kuiper.Real
open Kuiper.Float.Realops
open Kuiper.Scalars
open Kuiper.Floating.Base

(* Floating approximation admits an interpretation with a single ideal real
   value, not the exact real interpretation of the rounded bits. In this model,
   a semantic float is (bits, ideal):
   primitives compute their usual floating result in the first component and
   their real specification in the second. The ideal component is ghost state
   and is erased by extraction; this relation alone gives no error bound.

   Canonical zero and one have ideals 0 and 1. Other values with the same bits
   can have different ideals, including values with zero, infinity or NaN bits.
   Infinity's ideal is an ordinary, unspecified real, not an extended real.
   For example, exp(-infinity) has zero bits but a strictly positive ideal.

   Leibniz equality compares both components. IEEE equality and bit_eq observe
   only the bits, so neither permits substitution or transfer of approximation.
   In particular, there is deliberately no eq_approx rule. Special-value laws
   must constrain bits unless both components agree; exact laws such as
   commutativity or multiplication by canonical one can still be sound.

   to_real_ok supplies a real approximation for every value. We deliberately
   do not assume uniqueness: the contracts are compatible with this single-value
   model without requiring it or excluding other approximation relations.

   The concrete floating instances implement this model using Float.Model and
   prove their approximation contracts. The Raw interfaces specify trusted IEEE
   representation behavior; CUDA extraction must still realize those operations.
   The semantic record is hidden behind the public interfaces. *)
class floating_real_like (a:Type) {| scalar a, floating a, real_like a |} = {
  approx_of_int : squash ((of_int #a) %~ r_of_int);
  approx_fmax : squash ((fmax #a) %~ r_fmax);
  approx_neg : squash ((neg #a) %~ r_neg);
  approx_sub : squash ((sub #a) %~ r_sub);
  approx_fexp : squash ((fexp #a) %~ r_exp);
  approx_div : squash ((div #a) %~ r_div);
  approx_flog : squash ((flog #a) %~ r_log);
  approx_sqrt : squash ((sqrt #a) %~ r_sqrt);
  approx_rsqrt : squash ((rsqrt #a) %~ r_rsqrt);
}

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

let neg_models_pat (a:Type)
  {| scalar a, floating a, real_like a, rr : floating_real_like a |}
  : Lemma (ensures (neg #a) %~ r_neg)
      [SMTPat (neg #a); SMTPat (has_type rr (floating_real_like a))]
  = approx_neg #a

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
