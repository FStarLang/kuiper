module Kuiper.Functions

include FStar.Functions
module F = FStar.Fin
module S = FStar.Seq

module F = FStar.Fin
let pigeonhole_from_fstar_main (#n: pos) (s: Seq.seq (F.under n))
  : Pure (F.in_ s & F.in_ s)
      (requires Seq.length s > n)
      (ensures (fun (i1, i2) -> i1 < i2 /\ Seq.index s i1 = Seq.index s i2))
  = admit() //todo, take from F* main after updating ulib

let pigeon (n1:nat) (n2:nat{n2 < n1}) (f : natlt n1 -> GTot (natlt n2))
: Lemma (~ (is_inj f))
= if n2 = 0 
  then let _ = f 0 in ()
  else (
    let holes  = S.init_ghost #(F.under n2) n1 f in
    let i, j = pigeonhole_from_fstar_main holes in
    ()
  )