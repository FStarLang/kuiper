module KPlay.Proof.Optimized.Execution

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module C = FStar.Classical
module I32 = FStar.Int32
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module D = KPlay.Proof.Shared.Selection
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Shared = KPlay.SharedSpec
module Layout = KPlay.Proof.Optimized.Layout
module Cache = KPlay.Proof.Optimized.Cache
module State = KPlay.Proof.Optimized.State
module Book = KPlay.Proof.Optimized.LayerBook
module Step = KPlay.Proof.Optimized.Step
module Trace = KPlay.Proof.Optimized.Trace
module History = KPlay.Proof.Optimized.History

noextract
noeq type recorded = {
  position: n:nat{n < 2048};
  value: Step.data position;
}

noextract
noeq type prefix (call:Model.invocation) = {
  count: n:nat{n <= Model.calls call /\ n <= 2048};
  entries: s:S.seq recorded{
    S.length s == count /\
    (forall (i:nat{i < count}). (S.index s i).position == i)
  };
}

inline_for_extraction noextract
let empty (call:Model.invocation) : prefix call = {count = 0; entries = S.empty}

inline_for_extraction noextract
let data_at (#call:Model.invocation) (p:prefix call) (t:nat{t < p.count}) : GTot (Step.data t) =
  (S.index p.entries t).value

inline_for_extraction noextract
let append (#call:Model.invocation) (p:prefix call{p.count < Model.calls call})
  (data:Step.data p.count) : GTot (prefix call) =
  Model.full_profile call;
  {count = p.count + 1; entries = S.append p.entries (S.create 1 {position = p.count; value = data})}

inline_for_extraction noextract
let common (#call:Model.invocation) (p:prefix call) : GTot (Trace.prefix p.count) = {
  Trace.observations = S.init_ghost #Trace.observation p.count (fun t -> Step.observed (data_at p t));
  Trace.selected = S.init_ghost #(D.token 248320) p.count (fun t -> I32.v (data_at p t).Step.chosen);
}

inline_for_extraction noextract
let caller (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call)
  (completed:nat{completed <= p.count})
  : GTot (Shared.caller_state call O.cache_configuration bf16 f32 f32 bf16 completed) =
  Trace.state call (Layout.parameter_values w) prompt Cache.configuration (common p) completed

inline_for_extraction noextract
let cache (#call:Model.invocation) (p:prefix call) : GTot State.native_cache =
  if p.count = 0 then O.empty_cache Cache.configuration
  else (Book.last (data_at p (p.count - 1)).Step.layers).State.cache

inline_for_extraction noextract
let physical (#call:Model.invocation) (p:prefix call) : GTot State.physical =
  if p.count = 0 then State.zeroed
  else (Book.last (data_at p (p.count - 1)).Step.layers).State.memory

inline_for_extraction noextract
let well_formed (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call) : prop =
  forall (t:nat{t < p.count}).
    Step.well_formed w (data_at p t) /\
    (data_at p t).Step.consumed == (caller w prompt p t).D.next_token /\
    (data_at p t).Step.layers.Book.initial.State.cache == (caller w prompt p t).D.model_cache

let common_append (#call:Model.invocation) (p:prefix call{p.count < Model.calls call})
  (data:Step.data p.count)
  : Lemma (
      common (append p data) == Trace.append (common p) (Step.observed data) (I32.v data.Step.chosen) /\
      data_at (append p data) p.count == data) =
  let left = common (append p data) in
  let right = Trace.append (common p) (Step.observed data) (I32.v data.Step.chosen) in
  assert (S.equal left.Trace.observations right.Trace.observations);
  S.lemma_eq_elim left.Trace.observations right.Trace.observations;
  assert (S.equal left.Trace.selected right.Trace.selected);
  S.lemma_eq_elim left.Trace.selected right.Trace.selected

let cache_is_caller (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call)
  : Lemma (cache p == (caller w prompt p p.count).D.model_cache) = ()

let input_token (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length)
  (physical:lseq I64.t call.prompt_length) (p:prefix call{p.count < Model.calls call})
  (previous:I32.t)
  : Lemma
      (requires
        (forall (i:nat{i < call.prompt_length}). I64.v (S.index physical i) == S.index prompt i) /\
        (p.count > 0 ==> I32.v previous == I32.v (data_at p (p.count - 1)).Step.chosen))
      (ensures (caller w prompt p p.count).D.next_token ==
        (if p.count < call.prompt_length then I64.v (S.index physical p.count) else I32.v previous)) =
  Trace.consumed call (Layout.parameter_values w) prompt Cache.configuration (common p) p.count

let safety (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call)
  : Lemma
      (requires well_formed w prompt p)
      (ensures Trace.safety call (Layout.parameter_values w) prompt Cache.configuration (common p)) =
  let at (t:nat{t < p.count}) : Lemma (
      let before = caller w prompt p t in
      O.structure (Layout.parameter_values w) t before.D.next_token before.D.model_cache
        (Step.observed (data_at p t)) /\
      D.maximum (Step.observed (data_at p t)).O.score_keys (I32.v (data_at p t).Step.chosen)) =
    Step.structure w (data_at p t)
  in C.forall_intro at

let append_well_formed (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call{p.count < Model.calls call})
  (data:Step.data p.count)
  : Lemma
      (requires well_formed w prompt p /\ Step.well_formed w data /\
        data.Step.consumed == (caller w prompt p p.count).D.next_token /\
        data.Step.layers.Book.initial.State.cache == cache p)
      (ensures well_formed w prompt (append p data)) =
  common_append p data;
  cache_is_caller w prompt p;
  let at (t:nat{t < p.count + 1}) : Lemma (
      Step.well_formed w (data_at (append p data) t) /\
      (data_at (append p data) t).Step.consumed == (caller w prompt (append p data) t).D.next_token /\
      (data_at (append p data) t).Step.layers.Book.initial.State.cache ==
        (caller w prompt (append p data) t).D.model_cache) =
    Trace.state_prefix call (Layout.parameter_values w) prompt Cache.configuration
      (common p) (common (append p data)) t;
    if t < p.count then (
      assert (data_at (append p data) t == data_at p t);
      assert (Step.well_formed w (data_at p t))
    ) else (
      assert (t == p.count);
      assert (data_at (append p data) t == data)
    )
  in C.forall_intro at

let state_represents (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call)
  : Lemma
      (requires well_formed w prompt p)
      (ensures State.represents p.count 0 (cache p) (physical p)) =
  Model.full_profile call;
  if p.count = 0 then State.initial ()
  else (
    let data = data_at p (p.count - 1) in
    Book.last_represents data.Step.layers;
    State.boundary (p.count - 1) (Book.last data.Step.layers).State.cache
      (Book.last data.Step.layers).State.memory
  )

inline_for_extraction noextract
let outputs_agree (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call)
  (tokens:lseq (option I64.t) call.decode_steps)
  (hidden:lseq (option bf16) (call.decode_steps * 1024)) : prop =
  let returned = (caller w prompt p p.count).D.returned in
  (forall (g:nat{g < D.retained_count call.prompt_length p.count}).
    (match S.index tokens g with
    | None -> False
    | Some value -> I64.v value == S.index returned.D.generated_tokens g) /\
    (forall (i:nat{i < 1024}).
      S.index hidden (g * 1024 + i) == Some (S.index (S.index returned.D.hidden_history g) i)))

let outputs_append (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call{p.count < Model.calls call})
  (data:Step.data p.count)
  (tokens:lseq (option I64.t) call.decode_steps)
  (hidden:lseq (option bf16) (call.decode_steps * 1024))
  : Lemma
      (requires outputs_agree w prompt p tokens hidden)
      (ensures outputs_agree w prompt (append p data)
        (History.tokens_after call p.count tokens (IC.int32_to_int64 data.Step.chosen))
        (History.hidden_after call p.count hidden (Book.last data.Step.layers).State.hidden)) =
  common_append p data;
  Trace.append_state call (Layout.parameter_values w) prompt Cache.configuration
    (common p) (Step.observed data) (I32.v data.Step.chosen)

inline_for_extraction noextract
let run (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call{p.count == Model.calls call})
  : GTot (Shared.execution call O.cache_configuration bf16 f32 f32 bf16 f32 f32) =
  Trace.full call (Layout.parameter_values w) prompt Cache.configuration (common p)

let full_safety (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call{p.count == Model.calls call})
  : Lemma
      (requires well_formed w prompt p)
      (ensures Shared.safety_postcondition call (fun x -> x) (Layout.parameter_values w) prompt (run w prompt p)
        (caller w prompt p p.count).D.returned (caller w prompt p p.count)) =
  Cache.configuration_checked ();
  safety w prompt p;
  Trace.full_events call (Layout.parameter_values w) prompt Cache.configuration (common p);
  let at (t:nat{t < Model.calls call})
    : Lemma (Shared.structural_step call (Layout.parameter_values w) prompt (run w prompt p) t) =
    D.state_at_successor #248320 #1024 #(call.prompt_length)
      prompt (Layout.parameter_values w) (O.empty_cache Cache.configuration)
      (Trace.events (common p)) (common p).Trace.selected t
  in C.forall_intro at
