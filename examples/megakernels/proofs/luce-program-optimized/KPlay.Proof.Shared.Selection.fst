module KPlay.Proof.Shared.Selection

open FStar.Real
module S = FStar.Seq
module C = FStar.Classical

// decode_reference supplies one full observation per chronological model call.
// policy.txt relaxes only the preference between equal-priority maxima.
inline_for_extraction noextract
let size = n:nat{0 < n}

inline_for_extraction noextract
let vec (a:Type0) (n:nat) = s:S.seq a{S.length s == n}

inline_for_extraction noextract
let token (v:size) = i:nat{i < v}

// Algebraic finite keys do not assert IEEE arithmetic equivalence.
// Each role must preserve categories, ordering and ties of the SAME logits;
// real_like/v_approximates alone does not supply this discrete obligation.
// Signed zeros have key FiniteScore 0.0R; hidden payloads remain exact.
inline_for_extraction noextract
noeq type score =
  | FiniteScore : value:real -> score
  | NegativeInfinity : score
  | PositiveInfinity : score
  | NotANumber : score

inline_for_extraction noextract
let score_rank (x:score) : nat =
  match x with
  | NegativeInfinity -> 0
  | FiniteScore _ -> 1
  | PositiveInfinity -> 2
  | NotANumber -> 3

inline_for_extraction noextract
let score_value (x:score) : real =
  match x with
  | FiniteScore r -> r
  | _ -> 0.0R

// Argmax priority, not IEEE <= on NaNs.
inline_for_extraction noextract
let score_le (x y:score) : prop =
  score_rank x < score_rank y \/
  (score_rank x == score_rank y /\ score_value x <=. score_value y)

inline_for_extraction noextract
let same_score (x y:score) : prop =
  score_le x y /\ score_le y x

inline_for_extraction noextract
let better (candidate current:score) : GTot bool =
  t2b (~(score_le candidate current))

// Canonical first-maximum prior art, unchanged from the authorized contract.
inline_for_extraction noextract
let first_maximum_prefix
  (#v:size) (xs:vec score v) (n:size{n <= v}) (winner:nat{winner < n})
  : prop =
  forall (j:nat{j < n}).
    score_le (S.index xs j) (S.index xs winner) /\
    (same_score (S.index xs j) (S.index xs winner) ==> winner <= j)

inline_for_extraction noextract
let rec argmax_prefix
  (#v:size) (xs:vec score v) (n:size{n <= v})
  : GTot (i:nat{i < n}) (decreases n) =
  if n = 1 then 0
  else
    let previous = argmax_prefix xs (n - 1) in
    if better (S.index xs (n - 1)) (S.index xs previous)
    then n - 1
    else previous

inline_for_extraction noextract
let argmax (#v:size) (xs:vec score v) : GTot (token v) =
  argmax_prefix xs v

inline_for_extraction noextract
let first_maximum (#v:size) (xs:vec score v) (winner:token v) : prop =
  first_maximum_prefix xs v winner

inline_for_extraction noextract
let score_le_transitive (x y z:score)
  : Lemma
      (requires score_le x y /\ score_le y z)
      (ensures score_le x z) =
  ()

inline_for_extraction noextract
let rec argmax_prefix_correct
  (#v:size) (xs:vec score v) (n:size{n <= v})
  : Lemma
      (ensures first_maximum_prefix xs n (argmax_prefix xs n))
      (decreases n) =
  if n = 1 then ()
  else (
    argmax_prefix_correct xs (n - 1);
    let previous = argmax_prefix xs (n - 1) in
    if better (S.index xs (n - 1)) (S.index xs previous)
    then ()
    else ()
  )

inline_for_extraction noextract
let argmax_correct (#v:size) (xs:vec score v)
  : Lemma (first_maximum xs (argmax xs)) =
  argmax_prefix_correct xs v

inline_for_extraction noextract
let first_maximum_unique
  (#v:size) (xs:vec score v) (i j:token v)
  : Lemma
      (requires first_maximum xs i /\ first_maximum xs j)
      (ensures i == j) =
  ()

// PUBLIC selector: every entry participates; there is no index tie-break.
inline_for_extraction noextract
let maximum (#v:size) (xs:vec score v) (winner:token v) : prop =
  forall (j:token v). score_le (S.index xs j) (S.index xs winner)

inline_for_extraction noextract
let first_maximum_refines
  (#v:size) (xs:vec score v) (winner:token v)
  : Lemma
      (requires first_maximum xs winner)
      (ensures maximum xs winner) =
  ()

inline_for_extraction noextract
let maximum_iff_reference_tie (#v:size) (xs:vec score v) (winner:token v)
  : Lemma
      (maximum xs winner <==>
       same_score (S.index xs winner) (S.index xs (argmax xs))) =
  argmax_correct xs

inline_for_extraction noextract
let nan_is_maximum (#v:size) (xs:vec score v) (winner:token v)
  : Lemma
      (requires S.index xs winner == NotANumber)
      (ensures maximum xs winner) =
  ()

inline_for_extraction noextract
let nan_excludes_numerical_winners
  (#v:size) (xs:vec score v) (nan_index winner:token v)
  : Lemma
      (requires S.index xs nan_index == NotANumber /\ maximum xs winner)
      (ensures S.index xs winner == NotANumber) =
  ()

inline_for_extraction noextract
let uniform_scores (v:size) (key:score) (winner:token v)
  : Lemma
      (maximum #v (S.create v key) winner /\ argmax #v (S.create v key) == 0) =
  argmax_correct #v (S.create v key)

inline_for_extraction noextract
let call_count (p g:size) : size = p - 1 + g

inline_for_extraction noextract
let retained_count (p:size) (completed:nat) : nat =
  if completed < p then 0 else completed - p + 1

inline_for_extraction noextract
noeq type observation (v h:size) (cell cache:Type0) = {
  logits: vec score v;
  residual: vec cell h;
  cache_after: cache;
}

// No arithmetic, normalization, conversion or approximation of hidden cells.
inline_for_extraction noextract
noeq type histories (v h:size) (cell:Type0) (count:nat) = {
  generated_tokens: vec (token v) count;
  hidden_history: vec (vec cell h) count;
}

inline_for_extraction noextract
noeq type decode_state
  (v h:size) (cell weights cache:Type0) (p:size) (completed:nat) = {
  position: n:nat{n == completed};
  next_token: token v;
  parameters: weights;
  model_cache: cache;
  captured: option (vec cell h);
  returned: histories v h cell (retained_count p completed);
}

inline_for_extraction noextract
let selections_admissible
  (#v #h:size) (#cell #cache:Type0)
  (events:S.seq (observation v h cell cache))
  (selected:vec (token v) (S.length events))
  : prop =
  forall (t:nat{t < S.length events}).
    maximum (S.index events t).logits (S.index selected t)

inline_for_extraction noextract
let histories_at
  (#v #h #p:size) (#cell #cache:Type0)
  (events:S.seq (observation v h cell cache))
  (selected:vec (token v) (S.length events))
  (completed:nat{completed <= S.length events})
  : GTot (histories v h cell (retained_count p completed)) =
  {
    generated_tokens =
      S.init (retained_count p completed)
        (fun g -> S.index selected (p - 1 + g));
    hidden_history =
      S.init (retained_count p completed)
        (fun g -> (S.index events (p - 1 + g)).residual);
  }

// These are the ACTUAL selections, including when the next call consumes one.
inline_for_extraction noextract
let state_at
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  (selected:vec (token v) (S.length events))
  (completed:nat{completed <= S.length events})
  : GTot (decode_state v h cell weights cache p completed) =
  {
    position = completed;
    next_token =
      if completed < p then S.index prompt completed
      else S.index selected (completed - 1);
    parameters = parameters;
    model_cache =
      if completed = 0 then initial_cache
      else (S.index events (completed - 1)).cache_after;
    captured =
      if completed = 0 then None
      else Some (S.index events (completed - 1)).residual;
    returned = histories_at #v #h #p events selected completed;
  }

inline_for_extraction noextract
let prepare_call
  (#v #h #p:size) (#cell #weights #cache:Type0) (#completed:nat)
  (before:decode_state v h cell weights cache p completed)
  : decode_state v h cell weights cache p completed =
  { before with captured = None }

inline_for_extraction noextract
let advance
  (#v #h #p:size) (#cell #weights #cache:Type0) (#completed:nat)
  (prompt:vec (token v) p)
  (before:decode_state v h cell weights cache p completed)
  (observed:observation v h cell cache)
  (selected:token v)
  : GTot (decode_state v h cell weights cache p (completed + 1)) =
  let accumulated : histories v h cell (retained_count p (completed + 1)) =
    if completed < p - 1 then {
      generated_tokens = before.returned.generated_tokens;
      hidden_history = before.returned.hidden_history;
    }
    else {
      generated_tokens =
        S.append before.returned.generated_tokens (S.create 1 selected);
      hidden_history =
        S.append before.returned.hidden_history (S.create 1 observed.residual);
    }
  in
  {
    position = completed + 1;
    next_token =
      if completed + 1 < p then S.index prompt (completed + 1)
      else selected;
    parameters = before.parameters;
    model_cache = observed.cache_after;
    captured = Some observed.residual;
    returned = accumulated;
  }

inline_for_extraction noextract
let state_at_successor
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  (selected:vec (token v) (S.length events))
  (completed:nat{completed < S.length events})
  : Lemma
      (state_at prompt parameters initial_cache events selected (completed + 1) ==
       advance prompt
         (prepare_call
           (state_at prompt parameters initial_cache events selected completed))
         (S.index events completed) (S.index selected completed)) =
  let lhs = state_at prompt parameters initial_cache events selected (completed + 1) in
  let rhs =
    advance prompt
      (prepare_call
        (state_at prompt parameters initial_cache events selected completed))
      (S.index events completed) (S.index selected completed)
  in
  assert (S.equal lhs.returned.generated_tokens rhs.returned.generated_tokens);
  S.lemma_eq_elim lhs.returned.generated_tokens rhs.returned.generated_tokens;
  assert (S.equal lhs.returned.hidden_history rhs.returned.hidden_history);
  S.lemma_eq_elim lhs.returned.hidden_history rhs.returned.hidden_history

// Explicit compositional input, not an assumed model-correctness predicate.
// None denotes a failed call or absent norm pre-hook capture.
inline_for_extraction noextract
let model_operation (v h:size) (cell weights cache:Type0) =
  weights -> cache -> nat -> token v ->
  GTot (option (observation v h cell cache))

inline_for_extraction noextract
let observations_from_calls
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  (selected:vec (token v) (S.length events))
  : prop =
  forall (t:nat{t < S.length events}).
    let before =
      prepare_call (state_at prompt parameters initial_cache events selected t)
    in
    model before.parameters before.model_cache before.position before.next_token ==
      Some (S.index events t)

inline_for_extraction noextract
let step_post
  (#v #h #p:size) (#cell #weights #cache:Type0) (#completed:nat)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p)
  (before:decode_state v h cell weights cache p completed)
  (observed:observation v h cell cache) (selected:token v)
  (after:decode_state v h cell weights cache p (completed + 1))
  : prop =
  maximum observed.logits selected /\
  model before.parameters before.model_cache before.position before.next_token ==
    Some observed /\
  after == advance prompt (prepare_call before) observed selected

inline_for_extraction noextract
let selected_trace_step
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  (selected:vec (token v) (S.length events))
  (t:nat{t < S.length events})
  : Lemma
      (requires selections_admissible events selected /\
                observations_from_calls model prompt parameters initial_cache events selected)
      (ensures
        step_post model prompt
          (state_at prompt parameters initial_cache events selected t)
          (S.index events t) (S.index selected t)
          (state_at prompt parameters initial_cache events selected (t + 1))) =
  state_at_successor prompt parameters initial_cache events selected t

inline_for_extraction noextract
let decode
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  (selected:vec (token v) (call_count p g))
  : GTot (histories v h cell g) =
  (state_at prompt parameters initial_cache events selected (call_count p g)).returned

inline_for_extraction noextract
let selected_decode_post
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  (selected:vec (token v) (call_count p g))
  (output:histories v h cell g)
  (final_state:decode_state v h cell weights cache p (call_count p g))
  : prop =
  selections_admissible events selected /\
  observations_from_calls model prompt parameters initial_cache events selected /\
  output == decode prompt parameters initial_cache events selected /\
  final_state ==
    state_at prompt parameters initial_cache events selected (call_count p g)

// PUBLIC set-valued contract; only the actual valid maximum choices vary.
// Outputs and every state field are fixed once those choices are supplied.
inline_for_extraction noextract
let decode_post
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  (output:histories v h cell g)
  (final_state:decode_state v h cell weights cache p (call_count p g))
  : prop =
  exists (selected:vec (token v) (call_count p g)).
    selected_decode_post model prompt parameters initial_cache events selected output final_state

inline_for_extraction noextract
let choices_realize_contract
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  (selected:vec (token v) (call_count p g))
  : Lemma
      (requires selections_admissible events selected /\
                observations_from_calls model prompt parameters initial_cache events selected)
      (ensures
        decode_post model prompt parameters initial_cache events
          (decode prompt parameters initial_cache events selected)
          (state_at prompt parameters initial_cache events selected (call_count p g))) =
  C.exists_intro
    (fun actual ->
      selected_decode_post model prompt parameters initial_cache events actual
        (decode prompt parameters initial_cache events selected)
        (state_at prompt parameters initial_cache events selected (call_count p g)))
    selected

inline_for_extraction noextract
let decode_fields
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  (selected:vec (token v) (call_count p g))
  : Lemma
      (let final_state =
         state_at prompt parameters initial_cache events selected (call_count p g)
       in
       final_state.position == call_count p g /\
       final_state.parameters == parameters /\
       final_state.next_token == S.index selected (call_count p g - 1) /\
       final_state.model_cache == (S.index events (call_count p g - 1)).cache_after /\
       final_state.captured == Some (S.index events (call_count p g - 1)).residual /\
       (forall (k:nat{k < g}).
         S.index final_state.returned.generated_tokens k == S.index selected (p - 1 + k) /\
         S.index final_state.returned.hidden_history k == (S.index events (p - 1 + k)).residual /\
         (forall (j:nat{j < h}).
           S.index (S.index final_state.returned.hidden_history k) j ==
             S.index (S.index events (p - 1 + k)).residual j))) =
  ()

// Independent canonical definitions retain the original deterministic formulas.
// They are prior art, not requirements hidden in the public output/state map.
inline_for_extraction noextract
let reference_histories_at
  (#v #h #p:size) (#cell #cache:Type0)
  (events:S.seq (observation v h cell cache))
  (completed:nat{completed <= S.length events})
  : GTot (histories v h cell (retained_count p completed)) =
  {
    generated_tokens =
      S.init_ghost (retained_count p completed)
        (fun g -> argmax (S.index events (p - 1 + g)).logits);
    hidden_history =
      S.init (retained_count p completed)
        (fun g -> (S.index events (p - 1 + g)).residual);
  }

inline_for_extraction noextract
let reference_state_at
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  (completed:nat{completed <= S.length events})
  : GTot (decode_state v h cell weights cache p completed) =
  {
    position = completed;
    next_token =
      if completed < p then S.index prompt completed
      else argmax (S.index events (completed - 1)).logits;
    parameters = parameters;
    model_cache =
      if completed = 0 then initial_cache
      else (S.index events (completed - 1)).cache_after;
    captured =
      if completed = 0 then None
      else Some (S.index events (completed - 1)).residual;
    returned = reference_histories_at #v #h #p events completed;
  }

inline_for_extraction noextract
let reference_advance
  (#v #h #p:size) (#cell #weights #cache:Type0) (#completed:nat)
  (prompt:vec (token v) p)
  (before:decode_state v h cell weights cache p completed)
  (observed:observation v h cell cache)
  : GTot (decode_state v h cell weights cache p (completed + 1)) =
  let selected = argmax observed.logits in
  let accumulated : histories v h cell (retained_count p (completed + 1)) =
    if completed < p - 1 then {
      generated_tokens = before.returned.generated_tokens;
      hidden_history = before.returned.hidden_history;
    }
    else {
      generated_tokens =
        S.append before.returned.generated_tokens (S.create 1 selected);
      hidden_history =
        S.append before.returned.hidden_history (S.create 1 observed.residual);
    }
  in
  {
    position = completed + 1;
    next_token =
      if completed + 1 < p then S.index prompt (completed + 1)
      else selected;
    parameters = before.parameters;
    model_cache = observed.cache_after;
    captured = Some observed.residual;
    returned = accumulated;
  }

inline_for_extraction noextract
let reference_decode
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  : GTot (histories v h cell g) =
  (reference_state_at prompt parameters initial_cache events (call_count p g)).returned

inline_for_extraction noextract
let reference_decode_post
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  (output:histories v h cell g)
  (final_state:decode_state v h cell weights cache p (call_count p g))
  : prop =
  output == reference_decode prompt parameters initial_cache events /\
  final_state == reference_state_at prompt parameters initial_cache events (call_count p g)

inline_for_extraction noextract
let reference_calls
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  : prop =
  forall (t:nat{t < S.length events}).
    let before = prepare_call (reference_state_at prompt parameters initial_cache events t) in
    model before.parameters before.model_cache before.position before.next_token ==
      Some (S.index events t)

inline_for_extraction noextract
let reference_selections
  (#v #h:size) (#cell #cache:Type0)
  (events:S.seq (observation v h cell cache))
  : GTot (vec (token v) (S.length events)) =
  S.init_ghost (S.length events) (fun t -> argmax (S.index events t).logits)

inline_for_extraction noextract
let reference_selections_admissible
  (#v #h:size) (#cell #cache:Type0)
  (events:S.seq (observation v h cell cache))
  : Lemma (selections_admissible events (reference_selections events)) =
  let each (t:nat{t < S.length events})
    : Lemma (maximum (S.index events t).logits (S.index (reference_selections events) t)) =
    argmax_correct (S.index events t).logits;
    first_maximum_refines (S.index events t).logits (argmax (S.index events t).logits)
  in
  C.forall_intro each

inline_for_extraction noextract
let reference_state_agrees
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  (completed:nat{completed <= S.length events})
  : Lemma
      (reference_state_at prompt parameters initial_cache events completed ==
       state_at prompt parameters initial_cache events (reference_selections events) completed) =
  let lhs = reference_histories_at #v #h #p events completed in
  let rhs = histories_at #v #h #p events (reference_selections events) completed in
  assert (S.equal lhs.generated_tokens rhs.generated_tokens);
  S.lemma_eq_elim lhs.generated_tokens rhs.generated_tokens;
  assert (S.equal lhs.hidden_history rhs.hidden_history);
  S.lemma_eq_elim lhs.hidden_history rhs.hidden_history

inline_for_extraction noextract
let reference_state_at_successor
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  (completed:nat{completed < S.length events})
  : Lemma
      (reference_state_at prompt parameters initial_cache events (completed + 1) ==
       reference_advance prompt
         (prepare_call (reference_state_at prompt parameters initial_cache events completed))
         (S.index events completed)) =
  reference_state_agrees prompt parameters initial_cache events completed;
  reference_state_agrees prompt parameters initial_cache events (completed + 1);
  state_at_successor prompt parameters initial_cache events (reference_selections events) completed

inline_for_extraction noextract
let reference_calls_agree
  (#v #h #p:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:S.seq (observation v h cell cache))
  : Lemma
      (requires reference_calls model prompt parameters initial_cache events)
      (ensures observations_from_calls model prompt parameters initial_cache events
                 (reference_selections events)) =
  let each (t:nat{t < S.length events})
    : Lemma
        (let before =
           prepare_call
             (state_at prompt parameters initial_cache events (reference_selections events) t)
         in
         model before.parameters before.model_cache before.position before.next_token ==
           Some (S.index events t)) =
    reference_state_agrees prompt parameters initial_cache events t
  in
  C.forall_intro each

// Every result of the unchanged first-maximum reference is included, with its
// whole state and its own successful model observations, not merely its tokens.
inline_for_extraction noextract
let unchanged_reference_refines
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (model:model_operation v h cell weights cache)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (events:vec (observation v h cell cache) (call_count p g))
  (output:histories v h cell g)
  (final_state:decode_state v h cell weights cache p (call_count p g))
  : Lemma
      (requires reference_calls model prompt parameters initial_cache events /\
                reference_decode_post prompt parameters initial_cache events output final_state)
      (ensures decode_post model prompt parameters initial_cache events output final_state) =
  reference_selections_admissible events;
  reference_calls_agree model prompt parameters initial_cache events;
  reference_state_agrees prompt parameters initial_cache events (call_count p g);
  choices_realize_contract model prompt parameters initial_cache events (reference_selections events)

// Symbolic non-vacuity family, not a domain restriction or an HF model claim.
// Scores may be NaN, either infinity, zero, or any finite key at each position.
// Hidden cells and cache frames are arbitrary exact, token-sensitive payloads.
inline_for_extraction noextract
let tied_observation
  (#v #h:size) (#cell #cache:Type0)
  (keys:nat -> GTot score)
  (payloads:nat -> token v -> GTot (vec cell h))
  (frames:nat -> token v -> GTot cache)
  (position:nat) (consumed:token v)
  : GTot (observation v h cell cache) =
  {
    logits = S.create v (keys position);
    residual = payloads position consumed;
    cache_after = frames position consumed;
  }

inline_for_extraction noextract
let tied_model
  (#v #h:size) (#cell #weights #cache:Type0)
  (keys:nat -> GTot score)
  (payloads:nat -> token v -> GTot (vec cell h))
  (frames:nat -> token v -> GTot cache)
  : model_operation v h cell weights cache =
  fun _ _ position consumed -> Some (tied_observation keys payloads frames position consumed)

inline_for_extraction noextract
let tied_events
  (#v #h #p #g:size) (#cell #cache:Type0)
  (prompt:vec (token v) p)
  (keys:nat -> GTot score)
  (payloads:nat -> token v -> GTot (vec cell h))
  (frames:nat -> token v -> GTot cache)
  (winner:token v)
  : GTot (vec (observation v h cell cache) (call_count p g)) =
  S.init_ghost (call_count p g)
    (fun t ->
      tied_observation keys payloads frames t
        (if t < p then S.index prompt t else winner))

inline_for_extraction noextract
let nonfirst_tied_run_admissible
  (#v #h #p #g:size) (#cell #weights #cache:Type0)
  (prompt:vec (token v) p) (parameters:weights) (initial_cache:cache)
  (keys:nat -> GTot score)
  (payloads:nat -> token v -> GTot (vec cell h))
  (frames:nat -> token v -> GTot cache)
  (winner:token v{0 < winner})
  : Lemma
      (let events = tied_events #v #h #p #g prompt keys payloads frames winner in
       let selected = S.create (call_count p g) winner in
       let model = tied_model #v #h #cell #weights #cache keys payloads frames in
       let output = decode prompt parameters initial_cache events selected in
       let final_state =
         state_at prompt parameters initial_cache events selected (call_count p g)
       in
       selected_decode_post model prompt parameters initial_cache events selected output final_state /\
       decode_post model prompt parameters initial_cache events output final_state /\
       final_state.next_token == winner /\
       (forall (k:nat{k < g}).
         S.index output.generated_tokens k == winner /\
         S.index output.hidden_history k ==
           payloads (p - 1 + k) (if k = 0 then S.index prompt (p - 1) else winner)) /\
       final_state.model_cache ==
         frames (call_count p g - 1) (if g = 1 then S.index prompt (p - 1) else winner) /\
       final_state.captured ==
         Some (payloads (call_count p g - 1)
           (if g = 1 then S.index prompt (p - 1) else winner)) /\
       ~(output == reference_decode prompt parameters initial_cache events) /\
       ~(final_state ==
          reference_state_at prompt parameters initial_cache events (call_count p g))) =
  let events = tied_events #v #h #p #g prompt keys payloads frames winner in
  let selected = S.create (call_count p g) winner in
  let model = tied_model #v #h #cell #weights #cache keys payloads frames in
  let each (t:nat{t < call_count p g})
    : Lemma (maximum (S.index events t).logits (S.index selected t)) =
    uniform_scores v (keys t) winner
  in
  C.forall_intro each;
  assert (observations_from_calls model prompt parameters initial_cache events selected);
  choices_realize_contract model prompt parameters initial_cache events selected;
  decode_fields prompt parameters initial_cache events selected;
  uniform_scores v (keys (call_count p g - 1)) winner;
  let output = decode #v #h #p #g prompt parameters initial_cache events selected in
  let canonical = reference_decode #v #h #p #g prompt parameters initial_cache events in
  assert (argmax (S.index events (call_count p g - 1)).logits == 0);
  assert (S.index output.generated_tokens (g - 1) == winner);
  assert (S.index canonical.generated_tokens (g - 1) == 0)

// base.py overrides the helper defaults; frozen workloads are not a domain.
inline_for_extraction noextract
let reference_domain (smoke:bool) (v h p g:nat) : prop =
  v == 248320 /\ h == 1024 /\
  1 <= p /\ p <= 520 /\
  (if smoke then g == 1 else (g == 1 \/ g == 5 \/ g == 30 \/ g == 128)) /\
  p + g - 1 <= 2048
