module KPlay.Proof.Tightness

module P = KPlay.SharedSpec

let witness (#a:Type0) (input:P.inputs a)
  : result:P.outputs a{P.postcondition input result} =
  P.prefill_satisfies_postcondition input;
  P.prefill input

let unique (#a:Type0) (input:P.inputs a) (left right:P.outputs a)
  : Lemma
      (requires P.postcondition input left /\ P.postcondition input right)
      (ensures
        (forall (p:P.prompt_index) (h:P.hidden_index).
          left.hidden_rows p h == right.hidden_rows p h) /\
        (forall (p:P.prompt_index).
          left.position_ids p == right.position_ids p /\
          left.cache_positions p == right.cache_positions p) /\
        (forall (p k:P.prompt_index).
          left.causal_bias p k == right.causal_bias p k) /\
        (forall (p:P.prompt_index) (d:P.rotary_index).
          left.rope_cos p d == right.rope_cos p d /\
          left.rope_sin p d == right.rope_sin p d) /\
        (forall (l:P.layer_index).
          left.decoder_state.layer_history l == right.decoder_state.layer_history l) /\
        (forall (r:P.frequency_index).
          left.decoder_state.inv_freq r == right.decoder_state.inv_freq r /\
          left.decoder_state.original_inv_freq r == right.decoder_state.original_inv_freq r) /\
        left.decoder_state.attention_scaling == right.decoder_state.attention_scaling /\
        left.decoder_state.max_seq_len_cached == right.decoder_state.max_seq_len_cached /\
        left.decoder_state.original_max_seq_len == right.decoder_state.original_max_seq_len) =
  ()
