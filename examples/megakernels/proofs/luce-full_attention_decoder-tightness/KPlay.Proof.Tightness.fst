module KPlay.Proof.Tightness

module P = KPlay.SharedSpec

let witness (call:P.invocation) (weights:P.layer_weights) (input:P.vector 1024)
  (before:P.hybrid_cache{P.domain call before})
  : result:P.layer_result{P.postcondition call weights input before result} =
  P.full_attention_decoder call weights input before

let unique (call:P.invocation) (weights:P.layer_weights) (input:P.vector 1024)
  (before:P.hybrid_cache{P.domain call before}) (left right:P.layer_result)
  : Lemma
      (requires P.postcondition call weights input before left /\
                P.postcondition call weights input before right)
      (ensures
        (forall (i:P.index 1024). left.hidden i == right.hidden i) /\
        left.cache == right.cache) =
  ()
