module KPlay.Proof.Tightness

module P = KPlay.SharedSpec

let witness (#a:Type0) (weights:P.embedding_weights a) (token:P.token_id)
  : result:P.hidden_vector a{P.embedding_post weights token result} =
  P.embedding_correct weights token;
  P.embedding weights token

let unique (#a:Type0) (weights:P.embedding_weights a) (token:P.token_id)
  (left right:P.hidden_vector a)
  : Lemma
      (requires P.embedding_post weights token left /\ P.embedding_post weights token right)
      (ensures left == right) =
  P.embedding_post_unique weights token left;
  P.embedding_post_unique weights token right
