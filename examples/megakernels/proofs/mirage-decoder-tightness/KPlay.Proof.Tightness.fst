module KPlay.Proof.Tightness

module P = KPlay.SharedSpec

let witness (input:P.prefill_input)
  : GTot (result:P.layer_result{P.postcondition input result}) =
  P.decoder_layer_satisfies_postcondition input;
  P.decoder_layer input

let unique (input:P.prefill_input) (left right:P.layer_result)
  : Lemma
      (requires P.postcondition input left /\ P.postcondition input right)
      (ensures
        (forall (t:P.index 7) (h:P.index 1024).
          left.hidden_after t h == right.hidden_after t h) /\
        (forall (l:P.index 28).
          P.same_cache_slot (left.cache_after l) (right.cache_after l))) =
  P.postcondition_determines_result input left right

let cache_contents (left right:P.cache_slot)
  : Lemma
      (requires P.same_cache_slot left right)
      (ensures
        P.cache_initialized left == P.cache_initialized right /\
        P.cache_length left == P.cache_length right /\
        (match left, right with
         | P.Prefilled l, P.Prefilled r ->
           forall (t:P.index 7) (h:P.index 8) (d:P.index 128).
             l.keys t h d == r.keys t h d /\ l.values t h d == r.values t h d
         | P.Empty, P.Empty -> True
         | _ -> False)) =
  ()
