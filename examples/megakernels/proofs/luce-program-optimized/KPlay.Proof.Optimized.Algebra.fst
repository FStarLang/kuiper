module KPlay.Proof.Optimized.Algebra

#set-options "--fuel 1 --ifuel 1 --z3rlimit 20"

open FStar.Real

let multiply_commutes (a b:real) : Lemma (a *. b == b *. a) = ()

let multiply_associates (a b c:real)
  : Lemma ((a *. b) *. c == a *. (b *. c)) = ()

let multiply_distributes (a b c:real)
  : Lemma ((a +. b) *. c == a *. c +. b *. c) = ()

let delta_readout (decay old change key query:real)
  : Lemma (
      (old *. decay +. key *. change) *. query ==
      decay *. (old *. query) +. change *. (key *. query)) =
  multiply_distributes (old *. decay) (key *. change) query;
  multiply_commutes old decay;
  multiply_commutes key change;
  multiply_associates decay old query;
  multiply_associates change key query
