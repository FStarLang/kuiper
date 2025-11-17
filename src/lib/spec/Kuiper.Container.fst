module Kuiper.Container

open FStar.Ghost
open Kuiper

module F = FStar.FunctionalExtensionality

let oplus_lemma
  (#ct #it #vt : Type)
  {| gm : container ct it vt |}
  (c : ct)
  (i : it)
  (v : vt)
  : Lemma (acc (upd c i v)
           `F.feq_g`
            oplus (acc c) i v)
  = let aux (i':it) :
      Lemma (acc (upd c i v) i' ==
             oplus (acc c) i v i') =
      if FStar.StrongExcludedMiddle.strong_excluded_middle (i == i')
      then
        gm.l1 c i v
      else
        gm.l2 c i' i v
    in
    Classical.forall_intro aux
