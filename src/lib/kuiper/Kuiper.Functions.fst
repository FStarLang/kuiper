module Kuiper.Functions

include FStar.Functions

val __pigeon (n1:nat) (n2:nat{n2 < n1})
  (f : natlt n1 -> natlt n2)
  : Lemma (requires is_inj f) (ensures False)
let __pigeon n1 n2 f = admit()

let pigeon (n1:nat) (n2:nat{n2 < n1})
  (f : natlt n1 -> natlt n2)
  : Lemma (~ (is_inj f))
  =
  Classical.forall_intro (Classical.move_requires (__pigeon n1 n2))
