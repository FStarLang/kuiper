module KPlay.Proof.Optimized.Trace

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Shared = KPlay.SharedSpec
module Cache = KPlay.Proof.Optimized.Cache

inline_for_extraction noextract
let observation = O.model_observation O.cache_configuration BF.t FP.t FP.t BF.t FP.t FP.t

noeq type prefix (count:nat) = {
  observations:D.vec observation count;
  selected:D.vec (D.token 248320) count;
}

inline_for_extraction noextract
let empty : prefix 0 = {observations = Seq.empty; selected = Seq.empty}

inline_for_extraction noextract
let append (#count:nat) (before:prefix count) (observed:observation) (selected:D.token 248320) : prefix (count + 1) = {
  observations = Seq.append before.observations (Seq.create 1 observed);
  selected = Seq.append before.selected (Seq.create 1 selected);
}

inline_for_extraction noextract
let events (#count:nat) (p:prefix count) : GTot (D.vec (D.observation 248320 1024 BF.t (Cache.native_cache O.cache_configuration)) count) =
  Seq.init_ghost count (fun i -> Shared.event (Seq.index p.observations i))

inline_for_extraction noextract
let state (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix count) (completed:nat{completed <= count})
  : GTot (Shared.caller_state call O.cache_configuration BF.t FP.t FP.t BF.t completed) =
  D.state_at #248320 #1024 #(call.Model.prompt_length)
    prompt weights (O.empty_cache metadata) (events p) p.selected completed

inline_for_extraction noextract
let selected (#count:nat) (call:Model.invocation) (p:prefix count{count <= Model.calls call})
  : GTot (D.vec (D.token 248320) (Model.calls call)) =
  Seq.init_ghost #(D.token 248320) (Model.calls call) (fun i -> if i < count then Seq.index p.selected i else 0)

inline_for_extraction noextract
let consumed (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix count{count <= Model.calls call}) (completed:nat{completed <= count /\ completed < Model.calls call})
  : Lemma (
      (state call weights prompt metadata p completed).D.next_token ==
      Model.consumed call prompt (selected call p) completed) = ()

inline_for_extraction noextract
let same_prefix (#n #m:nat) (left:prefix n) (right:prefix m) (count:nat{count <= n /\ count <= m}) : prop =
  forall (i:nat{i < count}).
    Seq.index left.observations i == Seq.index right.observations i /\
    Seq.index left.selected i == Seq.index right.selected i

inline_for_extraction noextract
let state_prefix (#n #m:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (left:prefix n) (right:prefix m) (completed:nat{completed <= n /\ completed <= m})
  : Lemma
      (requires same_prefix left right completed)
      (ensures state call weights prompt metadata left completed == state call weights prompt metadata right completed) =
  let l = state call weights prompt metadata left completed in
  let r = state call weights prompt metadata right completed in
  assert (Seq.equal l.D.returned.D.generated_tokens r.D.returned.D.generated_tokens);
  Seq.lemma_eq_elim l.D.returned.D.generated_tokens r.D.returned.D.generated_tokens;
  assert (Seq.equal l.D.returned.D.hidden_history r.D.returned.D.hidden_history);
  Seq.lemma_eq_elim l.D.returned.D.hidden_history r.D.returned.D.hidden_history

inline_for_extraction noextract
let rec reference_prefix (call:Model.invocation) (weights:Model.parameters)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (left right:D.vec (D.token 248320) (Model.calls call)) (completed:nat{completed <= Model.calls call})
  : Lemma
      (requires forall (i:nat{i < completed}). Seq.index left i == Seq.index right i)
      (ensures Model.cache_at call weights prompt left completed == Model.cache_at call weights prompt right completed)
      (decreases completed) =
  if completed > 0 then (
    reference_prefix call weights prompt left right (completed - 1);
    assert (Model.consumed call prompt left (completed - 1) == Model.consumed call prompt right (completed - 1));
    Model.cache_at_successor call weights prompt left (completed - 1);
    Model.cache_at_successor call weights prompt right (completed - 1)
  )

inline_for_extraction noextract
let safety (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix count{count <= Model.calls call}) : prop =
  forall (t:nat{t < count}).
    let before = state call weights prompt metadata p t in
    let observed = Seq.index p.observations t in
    O.structure weights t before.D.next_token before.D.model_cache observed /\
    D.maximum observed.O.score_keys (Seq.index p.selected t)

inline_for_extraction noextract
let numerical (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (p:prefix count{count <= Model.calls call}) : prop =
  forall (t:nat{t < count}).
    O.numerical call (O.model_parameters weights) t
      (Model.consumed call prompt (selected call p) t)
      (Model.cache_at call (O.model_parameters weights) prompt (selected call p) t)
      (Seq.index p.observations t)

inline_for_extraction noextract
let append_safety (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix count{count < Model.calls call}) (observed:observation) (winner:D.token 248320)
  : Lemma
      (requires safety call weights prompt metadata p /\
        O.structure weights count (state call weights prompt metadata p count).D.next_token
          (state call weights prompt metadata p count).D.model_cache observed /\
        D.maximum observed.O.score_keys winner)
      (ensures safety call weights prompt metadata (append p observed winner)) =
  let at_index (t:nat{t < count + 1}) : Lemma (
      let before = state call weights prompt metadata (append p observed winner) t in
      let result = Seq.index (append p observed winner).observations t in
      O.structure weights t before.D.next_token before.D.model_cache result /\
      D.maximum result.O.score_keys (Seq.index (append p observed winner).selected t)) =
    state_prefix call weights prompt metadata p (append p observed winner) t;
    if t < count then () else assert (t == count)
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let append_numerical (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  (p:prefix count{count < Model.calls call}) (observed:observation) (winner:D.token 248320)
  : Lemma
      (requires numerical call weights prompt p /\
        O.numerical call (O.model_parameters weights) count
          (Model.consumed call prompt (selected call p) count)
          (Model.cache_at call (O.model_parameters weights) prompt (selected call p) count) observed)
      (ensures numerical call weights prompt (append p observed winner)) =
  let at_index (t:nat{t < count + 1}) : Lemma (
      O.numerical call (O.model_parameters weights) t
        (Model.consumed call prompt (selected call (append p observed winner)) t)
        (Model.cache_at call (O.model_parameters weights) prompt (selected call (append p observed winner)) t)
        (Seq.index (append p observed winner).observations t)) =
    reference_prefix call (O.model_parameters weights) prompt (selected call p) (selected call (append p observed winner)) t;
    assert (Model.consumed call prompt (selected call p) t == Model.consumed call prompt (selected call (append p observed winner)) t);
    if t < count then () else assert (t == count)
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let append_state (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix count) (observed:observation) (winner:D.token 248320)
  : Lemma (
      state call weights prompt metadata (append p observed winner) (count + 1) ==
      D.advance prompt (D.prepare_call (state call weights prompt metadata p count)) (Shared.event observed) winner) =
  state_prefix call weights prompt metadata p (append p observed winner) count;
  D.state_at_successor #248320 #1024 #(call.Model.prompt_length)
    prompt weights (O.empty_cache metadata) (events (append p observed winner)) (append p observed winner).selected count

inline_for_extraction noextract
let rec cache_shape (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix count{count <= Model.calls call}) (completed:nat{completed <= count})
  : Lemma
      (requires safety call weights prompt metadata p)
      (ensures O.cache_shape completed 0 (state call weights prompt metadata p completed).D.model_cache /\
        (state call weights prompt metadata p completed).D.model_cache.O.metadata == metadata)
      (decreases completed) =
  if completed > 0 then (
    cache_shape call weights prompt metadata p (completed - 1);
    let before = state call weights prompt metadata p (completed - 1) in
    O.structure_boundary weights (completed - 1) before.D.next_token before.D.model_cache
      (Seq.index p.observations (completed - 1))
  )

inline_for_extraction noextract
let cache_approx (#count:nat) (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix count{count <= Model.calls call})
  : Lemma
      (requires safety call weights prompt metadata p /\ numerical call weights prompt p)
      (ensures O.cache_approximates (state call weights prompt metadata p count).D.model_cache
        (Model.cache_at call (O.model_parameters weights) prompt (selected call p) count)) =
  if count > 0 then (
    let t = count - 1 in
    let before = state call weights prompt metadata p t in
    let observed = Seq.index p.observations t in
    consumed call weights prompt metadata p t;
    O.numerical_cache_boundary call (O.model_parameters weights) t before.D.next_token
      (Model.cache_at call (O.model_parameters weights) prompt (selected call p) t) observed;
    Model.cache_at_successor call (O.model_parameters weights) prompt (selected call p) t
  )

inline_for_extraction noextract
let full (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix (Model.calls call)) : Shared.execution call O.cache_configuration BF.t FP.t FP.t BF.t FP.t FP.t = {
  Shared.initial_metadata = metadata;
  Shared.observations = p.observations;
  Shared.selected = p.selected;
  Shared.caller = (fun completed -> state call weights prompt metadata p completed);
  Shared.prepared = (fun t -> D.prepare_call (state call weights prompt metadata p t));
}

inline_for_extraction noextract
let full_events (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix (Model.calls call))
  : Lemma (
      Shared.events call (full call weights prompt metadata p) == events p /\
      selected call p == p.selected) =
  assert (Seq.equal (Shared.events call (full call weights prompt metadata p)) (events p));
  Seq.lemma_eq_elim (Shared.events call (full call weights prompt metadata p)) (events p);
  assert (Seq.equal (selected call p) p.selected);
  Seq.lemma_eq_elim (selected call p) p.selected

inline_for_extraction noextract
let full_postcondition (call:Model.invocation) (weights:O.parameters BF.t)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (metadata:O.cache_configuration)
  (p:prefix (Model.calls call))
  : Lemma
      (requires O.configuration_ok metadata /\ safety call weights prompt metadata p /\ numerical call weights prompt p)
      (ensures (
        let run = full call weights prompt metadata p in
        Shared.postcondition call (fun x -> x) weights prompt run
          (state call weights prompt metadata p (Model.calls call)).D.returned
          (state call weights prompt metadata p (Model.calls call)))) =
  full_events call weights prompt metadata p;
  let run = full call weights prompt metadata p in
  let at_index (t:nat{t < Model.calls call}) : Lemma (Shared.structural_step call weights prompt run t) =
    D.state_at_successor #248320 #1024 #(call.Model.prompt_length)
      prompt weights (O.empty_cache metadata) (events p) p.selected t
  in
  FStar.Classical.forall_intro at_index;
  assert (Shared.safety_prefix call weights prompt run (Model.calls call));
  Shared.safety_invariant_from_steps call weights prompt run (Model.calls call)
