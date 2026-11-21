module Kuiper.Float16

open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
open Kuiper.Float.Realops

inline_for_extraction noextract
val t : Type0

inline_for_extraction noextract
instance val is_floating : floating t

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t

inline_for_extraction noextract
val fexpm1 : t -> t
val fexpm1_models (x:t) (r:real)
  : Lemma (requires v_approximates x r)
          (ensures v_approximates (fexpm1 x) (r_expm1 r))
          [SMTPat (fexpm1 x); SMTPat (v_approximates x r)]

inline_for_extraction noextract
val flog1p : t -> t
val flog1p_models (x:t) (r:real{r >. 0.0R -. 1.0R})
  : Lemma (requires v_approximates x r)
          (ensures v_approximates (flog1p x) (r_log1p r))
          [SMTPat (flog1p x); SMTPat (v_approximates x r)]

(* Trusted real approximation contracts; no rounding or error bounds are modeled. *)
val pow_approx
  (x y : t)
  (xr : Pow.rpos)
  (yr : real)
  : Lemma
      (requires v_approximates x xr /\ v_approximates y yr)
      (ensures v_approximates (pow x y) (Pow.pow xr yr))

val sin_approx
  (x : t)
  (xr : real)
  : Lemma
      (requires v_approximates x xr)
      (ensures v_approximates (sin x) (Trig.sin xr))

val cos_approx
  (x : t)
  (xr : real)
  : Lemma
      (requires v_approximates x xr)
      (ensures v_approximates (cos x) (Trig.cos xr))

val lem_sizeof () : Lemma (Sized.size #t == 2sz)
