module KPlay.Proof.Tightness

module P = KPlay.SharedSpec

let witness (#state:Type0) (input:P.inputs) (before:state)
  : GTot (result:(P.outputs & state){
      P.postcondition input before result._1 result._2}) =
  P.transition_satisfies_postcondition input before;
  P.transition input before

let unique (#state:Type0) (input:P.inputs) (before:state)
  (left right:P.outputs) (left_state right_state:state)
  : Lemma
      (requires P.postcondition input before left left_state /\
                P.postcondition input before right right_state)
      (ensures
        (forall (p:P.prompt_index) (d:P.hidden_index).
          left.normalized_hidden p d == right.normalized_hidden p d) /\
        (forall (v:P.vocab_index). left.final_logits v == right.final_logits v) /\
        left_state == right_state) =
  ()
