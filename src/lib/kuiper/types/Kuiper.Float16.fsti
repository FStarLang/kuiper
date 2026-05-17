module Kuiper.Float16

open Kuiper.Floating.Base
open Kuiper.Approximates.Base

inline_for_extraction noextract
val t : Type0

inline_for_extraction noextract
instance val is_floating : floating t

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t

val lem_sizeof () : Lemma (Sized.size #t == 2sz)
