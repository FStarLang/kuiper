module Kuiper.Functions

include FStar.Functions
module F = FStar.Fin
module S = FStar.Seq

module F = FStar.Fin

let pigeon (n1:nat) (n2:nat{n2 < n1}) (f : natlt n1 -> GTot (natlt n2))
: Lemma (~ (is_inj f))
= if n2 = 0 
  then let _ = f 0 in ()
  else (
    let holes  = S.init_ghost #(F.under n2) n1 f in
    let i, j = F.pigeonhole holes in
    ()
  )