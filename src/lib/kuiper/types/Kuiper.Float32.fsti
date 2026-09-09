module Kuiper.Float32

open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real

inline_for_extraction noextract
val t : Type0

inline_for_extraction noextract
instance val is_floating : floating t

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t

(* Explicit CUDA operations on the ordinary float type. Their approximation
   contracts use the same abstract relation as floating_real_like. *)
inline_for_extraction noextract
val neg : t -> Tot t

inline_for_extraction noextract
val add_rn : t -> t -> Tot t

inline_for_extraction noextract
val div_rn : t -> t -> Tot t

val neg_approx (x : t) (xr : real) : Lemma
  (requires v_approximates x xr)
  (ensures v_approximates (neg x) (0.0R -. xr))

val add_rn_approx (x y : t) (xr yr : real) : Lemma
  (requires v_approximates x xr /\ v_approximates y yr)
  (ensures v_approximates (add_rn x y) (xr +. yr))

val div_rn_approx (x y : t) (xr : real) (yr : real{yr =!= 0.0R}) : Lemma
  (requires v_approximates x xr /\ v_approximates y yr)
  (ensures v_approximates (div_rn x y) (xr /. yr))

val lem_sizeof () : Lemma (Sized.size #t == 4sz)
