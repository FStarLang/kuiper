module Kuiper.BFloat16

open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Float.Realops

inline_for_extraction noextract
val t : Type0

inline_for_extraction noextract
instance val is_floating : floating t

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t

inline_for_extraction noextract
val fexpm1 : t -> t
assume Fexpm1Models : fexpm1 %~ r_expm1

inline_for_extraction noextract
val flog1p : t -> t
assume Flog1pModels : flog1p %~ r_log1p

val lem_sizeof () : Lemma (Sized.size #t == 2sz)
