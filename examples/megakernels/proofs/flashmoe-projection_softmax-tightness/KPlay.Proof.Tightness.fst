module KPlay.Proof.Tightness

module P = KPlay.SharedSpec

let witness (smoke:bool)
  (tokens:P.token_matrix smoke) (weights:P.weight_matrix smoke)
  (_:squash (P.input_domain smoke tokens weights))
  : GTot (result:(P.token_matrix smoke & P.weight_matrix smoke & P.probability_matrix smoke){
      let (tokens_after, weights_after, probabilities) = result in
      P.postcondition smoke tokens weights tokens_after weights_after probabilities}) =
  P.projection_softmax_satisfies_postcondition smoke tokens weights;
  (tokens, weights, P.projection_softmax smoke tokens weights)

let unique (smoke:bool)
  (tokens:P.token_matrix smoke) (weights:P.weight_matrix smoke)
  (left_tokens right_tokens:P.token_matrix smoke)
  (left_weights right_weights:P.weight_matrix smoke)
  (left right:P.probability_matrix smoke)
  : Lemma
      (requires P.postcondition smoke tokens weights left_tokens left_weights left /\
                P.postcondition smoke tokens weights right_tokens right_weights right)
      (ensures left_tokens == right_tokens /\ left_weights == right_weights /\ left == right) =
  ()
