module KPlay.Proof.Tightness

module P = KPlay.SharedSpec
module S = FStar.Seq

let witness (#v #h #p #g:P.size) (#cell #weights #cache:Type0)
  (model:P.model_operation v h cell weights cache)
  (prompt:P.vec (P.token v) p) (parameters:weights) (initial:cache)
  (events:P.vec (P.observation v h cell cache) (P.call_count p g))
  (selected:P.vec (P.token v) (P.call_count p g))
  (_:squash (P.selections_admissible events selected /\
             P.observations_from_calls model prompt parameters initial events selected))
  : GTot (result:(P.histories v h cell g &
      P.decode_state v h cell weights cache p (P.call_count p g)){
        P.decode_post model prompt parameters initial events result._1 result._2}) =
  P.choices_realize_contract model prompt parameters initial events selected;
  (P.decode prompt parameters initial events selected,
   P.state_at prompt parameters initial events selected (P.call_count p g))

let full_contract (#v #h #p #g:P.size) (#cell #weights #cache:Type0)
  (model:P.model_operation v h cell weights cache)
  (prompt:P.vec (P.token v) p) (parameters:weights) (initial:cache)
  (events:P.vec (P.observation v h cell cache) (P.call_count p g))
  (output:P.histories v h cell g)
  (after:P.decode_state v h cell weights cache p (P.call_count p g))
  : Lemma (
      P.decode_post model prompt parameters initial events output after <==>
      (exists (selected:P.vec (P.token v) (P.call_count p g)).
        P.selections_admissible events selected /\
        P.observations_from_calls model prompt parameters initial events selected /\
        output == P.decode prompt parameters initial events selected /\
        after == P.state_at prompt parameters initial events selected (P.call_count p g))) =
  ()

(* Compare independent model observations, not two copies sharing one event trace. *)
let rec states_unique (#v #h #p:P.size) (#cell #weights #cache:Type0) (#n:nat)
  (model:P.model_operation v h cell weights cache)
  (prompt:P.vec (P.token v) p) (parameters:weights) (initial:cache)
  (left right:P.vec (P.observation v h cell cache) n)
  (selected:P.vec (P.token v) n) (completed:nat{completed <= n})
  : Lemma
      (requires P.observations_from_calls model prompt parameters initial left selected /\
                P.observations_from_calls model prompt parameters initial right selected)
      (ensures
        P.state_at prompt parameters initial left selected completed ==
        P.state_at prompt parameters initial right selected completed)
      (decreases completed) =
  if completed = 0 then (
    let l = P.state_at prompt parameters initial left selected 0 in
    let r = P.state_at prompt parameters initial right selected 0 in
    assert (S.equal l.returned.generated_tokens r.returned.generated_tokens);
    S.lemma_eq_elim l.returned.generated_tokens r.returned.generated_tokens;
    assert (S.equal l.returned.hidden_history r.returned.hidden_history);
    S.lemma_eq_elim l.returned.hidden_history r.returned.hidden_history
  ) else (
    let t = completed - 1 in
    states_unique model prompt parameters initial left right selected t;
    assert (S.index left t == S.index right t);
    P.state_at_successor prompt parameters initial left selected t;
    P.state_at_successor prompt parameters initial right selected t
  )

let fixed_choices_unique (#v #h #p #g:P.size) (#cell #weights #cache:Type0)
  (model:P.model_operation v h cell weights cache)
  (prompt:P.vec (P.token v) p) (parameters:weights) (initial:cache)
  (left_events right_events:P.vec (P.observation v h cell cache) (P.call_count p g))
  (selected:P.vec (P.token v) (P.call_count p g))
  (left_output right_output:P.histories v h cell g)
  (left_state right_state:P.decode_state v h cell weights cache p (P.call_count p g))
  : Lemma
      (requires P.selected_decode_post model prompt parameters initial left_events selected
                  left_output left_state /\
                P.selected_decode_post model prompt parameters initial right_events selected
                  right_output right_state)
      (ensures left_output == right_output /\ left_state == right_state) =
  states_unique model prompt parameters initial left_events right_events selected (P.call_count p g)
