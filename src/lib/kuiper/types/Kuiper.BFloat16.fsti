module Kuiper.BFloat16

open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real

inline_for_extraction noextract
val t : Type0

inline_for_extraction noextract
instance val is_floating : floating t

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t

inline_for_extraction noextract
val fexpm1 : t -> t

inline_for_extraction noextract
val flog1p : t -> t

val expm1_approx
  (x : t)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (fexpm1 x) (exp r -. 1.0R))

val log1p_approx
  (x : t)
  (r : real { r >. 0.0R -. 1.0R })
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (flog1p x) (log (1.0R +. r)))

val lem_sizeof () : Lemma (Sized.size #t == 2sz)
