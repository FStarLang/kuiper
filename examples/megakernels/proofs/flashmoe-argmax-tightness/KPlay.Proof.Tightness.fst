module KPlay.Proof.Tightness

module P = KPlay.SharedSpec

let witness (#s:nat) (#e:pos) (before:P.probabilities s e)
  : result:(P.probabilities s e & P.outputs s e){
      let (after, out) = result in P.post before after out} =
  P.transition_satisfies_post before;
  P.transition before

let unique (#s:nat) (#e:pos) (before:P.probabilities s e)
  (left_after right_after:P.probabilities s e) (left right:P.outputs s e)
  : Lemma
      (requires P.post before left_after left /\ P.post before right_after right)
      (ensures
        (forall (t:P.index s) (j:P.index e).
          left_after t j == right_after t j) /\
        (forall (t:P.index s).
          left.selected_expert t == right.selected_expert t /\
          left.selected_probability t == right.selected_probability t)) =
  ()
