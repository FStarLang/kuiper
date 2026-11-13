module KPlay.Proof.Tightness

module P = KPlay.SharedSpec
module C = FStar.Classical

let witness (prompt:P.prompt_tokens) (logits:P.prompt_logits) (steps:P.decode_steps)
  : GTot (result:P.result{P.postcondition prompt logits steps result}) =
  P.argmax_append_satisfies_postcondition prompt logits steps;
  P.argmax_append prompt logits steps

let choice_witness (prompt:P.prompt_tokens) (logits:P.prompt_logits) (steps:P.decode_steps)
  (chosen:P.token_id{P.maximal_index (P.final_logits logits) chosen})
  : GTot (result:P.result{
      P.postcondition prompt logits steps result /\ result._1 == chosen}) =
  P.maximal_index_append_allowed prompt logits steps chosen;
  (chosen, P.append_token prompt chosen)

let no_extra_results (prompt:P.prompt_tokens) (logits:P.prompt_logits)
  (steps:P.decode_steps) (result:P.result)
  : Lemma
      (requires P.postcondition prompt logits steps result)
      (ensures exists (chosen:P.token_id).
        P.maximal_index (P.final_logits logits) chosen /\
        result == (chosen, P.append_token prompt chosen)) =
  C.exists_intro
    (fun chosen ->
      P.maximal_index (P.final_logits logits) chosen /\
      result == (chosen, P.append_token prompt chosen))
    result._1

let fixed_choice_unique (prompt:P.prompt_tokens) (logits:P.prompt_logits)
  (steps:P.decode_steps) (left right:P.result)
  : Lemma
      (requires P.postcondition prompt logits steps left /\
                P.postcondition prompt logits steps right /\ left._1 == right._1)
      (ensures left == right) =
  ()
