module Kuiper.Float32.ConstantFold

open FStar.Real
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module A = Kuiper.Approximates.Base
module Cert = Kuiper.Float32.ConstantFold.Certificates

inline_for_extraction let () = ()

(* Trusted literal interpretation in the existing approximation model.
   Unlike the general decimal rule, this requires an exact binary32 literal.
   The real enclosure is strictly inside its actual nearest-rounding cell. *)
val literal_approx
  (s:string) (lo_num:nat) (lo_den:pos) (hi_num:nat) (hi_den:pos)
  (mantissa:nat) (exponent:int) (r:real)
  : Lemma
      (requires Cert.exact_interval s lo_num lo_den hi_num hi_den mantissa exponent /\
        of_int lo_num /. of_int lo_den <=. r /\
        r <=. of_int hi_num /. of_int hi_den)
      (ensures A.v_approximates (F.of_literal #FP.t s) r)

(* Trusted native constant-folding correspondence for ordinary correctly
   rounded binary32 division, not fast reciprocal. Both literals are exact,
   positive and normal; the checked rational result is away from all ties.
   The certificate also covers a correctly rounded binary64 intermediate
   from unsuffixed extracted literals, with a strict 2^-30 ULP margin.
   No approximation relation or mathematical target appears in this rule. *)
val reciprocal_literal_eq
  (input output:string) (im:nat) (ie:int) (om:nat) (oe:int)
  : Lemma
      (requires Cert.reciprocal input output im ie om oe)
      (ensures F.of_literal #FP.t output ==
        F.div (F.one #FP.t) (F.of_literal #FP.t input))
