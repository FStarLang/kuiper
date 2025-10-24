module Kuiper.Functions

include FStar.Functions
module F = FStar.Fin
module S = FStar.Seq

(* To be taken from an update to FStar.Fin, coming in F* master *)
assume
val ipigeonhole (#n: pos) (s: S.seq (F.under n))
    : Pure (F.in_ s & F.in_ s)
      (requires S.length s > n)
      (ensures (fun (i1, i2) -> i1 < i2 /\ S.index s i1 = S.index s i2))

let pigeon (n1:nat) (n2:nat{n2 < n1}) (f : natlt n1 -> GTot (natlt n2))
: Lemma (~ (is_inj f))
= if n2 = 0 
  then let _ = f 0 in ()
  else (
    let holes  = S.init_ghost #(F.under n2) n1 f in
    let i, j = ipigeonhole holes in
    ()
  )