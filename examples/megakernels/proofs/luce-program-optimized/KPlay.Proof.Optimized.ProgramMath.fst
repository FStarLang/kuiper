module KPlay.Proof.Optimized.ProgramMath

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
open KPlay.Proof.Optimized.Execution
module D = KPlay.Proof.Shared.Selection
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Shared = KPlay.SharedSpec
module Layout = KPlay.Proof.Optimized.Layout
module Cache = KPlay.Proof.Optimized.Cache
module Step = KPlay.Proof.Optimized.Step
module Trace = KPlay.Proof.Optimized.Trace
module Numerical = KPlay.Proof.Optimized.Numerical

inline_for_extraction noextract
let family_dependencies (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call{p.count == Model.calls call}) : prop =
  forall (t:nat{t < Model.calls call}).
    let data = data_at p t in
    Numerical.dependencies call w t data.Step.consumed
      (Model.cache_at call (O.model_parameters (Layout.parameter_values w)) prompt (common p).Trace.selected t)
      data.Step.layers

inline_for_extraction noextract
let numerical_scope (#call:Model.invocation) (p:prefix call) : prop =
  forall (t:nat{t < p.count}). Numerical.scope t (data_at p t).Step.layers

let rec numerical_prefix (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call{p.count == Model.calls call})
  (completed:nat{completed <= Model.calls call})
  : Lemma
      (requires well_formed w prompt p /\ family_dependencies w prompt p /\ numerical_scope p)
      (ensures Shared.numerical_prefix call (Layout.parameter_values w) prompt (run w prompt p) completed)
      (decreases completed) =
  full_safety w prompt p;
  Trace.full_events call (Layout.parameter_values w) prompt Cache.configuration (common p);
  if completed > 0 then (
    numerical_prefix w prompt p (completed - 1);
    let t = completed - 1 in
    let data = data_at p t in
    let execution = run w prompt p in
    let ideal = Model.cache_at call (O.model_parameters (Layout.parameter_values w)) prompt (common p).Trace.selected t in
    Shared.invariant_from_steps call (Layout.parameter_values w) prompt execution t;
    Shared.state_at_input call (Layout.parameter_values w) prompt execution t;
    Numerical.refines call w t data ideal
  )

let program_refines (#call:Model.invocation) (w:Layout.packed)
  (prompt:D.vec (D.token 248320) call.prompt_length) (p:prefix call{p.count == Model.calls call})
  : Lemma
      (requires well_formed w prompt p /\ family_dependencies w prompt p /\ numerical_scope p)
      (ensures Shared.postcondition call (fun x -> x) (Layout.parameter_values w) prompt (run w prompt p)
        (caller w prompt p p.count).D.returned (caller w prompt p p.count)) =
  Cache.configuration_checked ();
  safety w prompt p;
  numerical_prefix w prompt p (Model.calls call);
  Trace.full_events call (Layout.parameter_values w) prompt Cache.configuration (common p);
  Trace.full_postcondition call (Layout.parameter_values w) prompt Cache.configuration (common p)
