module KPlay.Proof.Tightness

module P = KPlay.SharedSpec
module C = FStar.Classical

let same_convolution (left right:option P.convolution_state) : GTot prop =
  match left, right with
  | None, None -> True
  | Some l, Some r ->
    forall (channel:P.fin 6144) (tap:P.fin 4). l channel tap == r channel tap
  | _ -> False

let same_recurrent (left right:option P.recurrent_state) : GTot prop =
  match left, right with
  | None, None -> True
  | Some l, Some r ->
    forall (head:P.fin 16) (i j:P.fin 128). l head i j == r head i j
  | _ -> False

let witness (call:P.invocation) (weights:P.weights) (input:P.vector 1024)
  (before:P.model_cache{P.cache_domain call before})
  : GTot (result:P.result{P.postcondition call weights input before result}) =
  P.recurrent_decoder_satisfies_postcondition call weights input before;
  P.recurrent_decoder call weights input before

let unique (call:P.invocation) (weights:P.weights) (input:P.vector 1024)
  (before:P.model_cache{P.cache_domain call before}) (left right:P.result)
  : Lemma
      (requires P.postcondition call weights input before left /\
                P.postcondition call weights input before right)
      (ensures
        (forall (i:P.fin 1024). left.hidden i == right.hidden i) /\
        (forall (layer:P.fin 24).
          same_convolution (left.cache.conv_slots layer) (right.cache.conv_slots layer) /\
          same_recurrent (left.cache.recurrent_slots layer) (right.cache.recurrent_slots layer) /\
          left.cache.kv_slots layer == right.cache.kv_slots layer)) =
  let each (layer:P.fin 24)
    : Lemma (
        same_convolution (left.cache.conv_slots layer) (right.cache.conv_slots layer) /\
        same_recurrent (left.cache.recurrent_slots layer) (right.cache.recurrent_slots layer) /\
        left.cache.kv_slots layer == right.cache.kv_slots layer) =
    if layer = call.layer_index then () else ()
  in
  C.forall_intro each
