module KPlay.Proof.SourceRouting
#lang-pulse

open Kuiper
open Kuiper.Seq.Common

module S = FStar.Seq
module C = Kuiper.Chest
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base
module B = Kuiper.Floating.Bounds
module I32 = FStar.Int32
module PA = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module SM = Kuiper.Spec.Softmax
module Layout = KPlay.Proof.SourceLayouts
module Project = KPlay.Proof.SourceProjection
module Softmax = KPlay.Proof.SourceSoftmax
module Steps = KPlay.Proof.RouteSteps
module Shared = KPlay.SharedSpec
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module Capacity = KPlay.Proof.Shared.Capacity
module Family = KPlay.Proof.FamilySelection
module FamilyArgmax = KPlay.Proof.FamilyArgmax

inline_for_extraction noextract
let index = Softmax.index

noextract type prefix = n:nat{n <= 256}
noextract type history (n:prefix) = lseq Softmax.probability_row n
noextract type flags = lseq i32 2048
noextract type probabilities = lseq f32 2048

inline_for_extraction noextract
let row_certificate (x:Layout.token_values) (w:Layout.weight_values)
  (t:natlt 256) (row:Softmax.probability_row) : prop =
  forall (e:natlt 8).
    A.v_approximates (S.index row e)
      (C.acc2 (Projection.projection_softmax true (Layout.logical_tokens x)
        (Layout.logical_weights w)) t e)

inline_for_extraction noextract
let valid (x:Layout.token_values) (w:Layout.weight_values) (#n:prefix)
  (h:history n) : prop =
  forall (t:nat{t < n}). row_certificate x w t (S.index h t)

inline_for_extraction noextract
let routes_match (#n:prefix) (h:history n) (fs:flags) (ps:probabilities) : prop =
  forall (t:nat{t < n}) (e:natlt 8).
    S.index fs (t * 8 + e) == Steps.emitted_flag (S.index h t) e /\
    S.index ps (t * 8 + e) == Steps.emitted_probability (S.index h t) e

inline_for_extraction noextract
let score_row_certificate (x:Layout.token_values) (w:Layout.weight_values)
  (t:natlt 256)
  : Lemma
      (requires Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))
      (ensures
        (forall (e:natlt 8).
          B.magnitude 65536 (S.index (Layout.softmax_score_row (Project.scores x w) t) e)) /\
        seq_approximates (Layout.softmax_score_row (Project.scores x w) t)
          (C.chest1_to_seq (C.chest2_row
            (Projection.projection true (Layout.logical_tokens x) (Layout.logical_weights w)) t)))
  =
  let cell (e:natlt 8)
    : Lemma (
        B.magnitude 65536 (S.index (Layout.softmax_score_row (Project.scores x w) t) e) /\
        A.v_approximates (S.index (Layout.softmax_score_row (Project.scores x w) t) e)
          (C.acc2 (Projection.projection true (Layout.logical_tokens x) (Layout.logical_weights w)) t e))
    =
    Project.score_certificate x w t e;
    S.lemma_index_slice (Project.scores x w) (t * 8) (t * 8 + 8) e
  in FStar.Classical.forall_intro cell

inline_for_extraction noextract
let normalized_certificate (x:Layout.token_values) (w:Layout.weight_values)
  (t:natlt 256) (row:Softmax.probability_row)
  : Lemma
      (requires Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w) /\
        (forall (r:lseq real 8).
          seq_approximates (Layout.softmax_score_row (Project.scores x w) t) r ==>
          seq_approximates row (SM.softmax_real_seq r)))
      (ensures row_certificate x w t row)
  =
  score_row_certificate x w t;
  SM.lem_softmax_real_to_seq
    (C.chest2_row (Projection.projection true (Layout.logical_tokens x) (Layout.logical_weights w)) t)

inline_for_extraction noextract
let loaded_row_matches (values:Layout.score_values) (t:natlt 256) (loaded:lseq f32 8)
  : Lemma
      (requires forall (e:natlt 8). S.index loaded e == S.index values (t * 8 + e))
      (ensures loaded == Layout.softmax_score_row values t)
  =
  let cell (e:natlt 8)
    : Lemma (S.index loaded e == S.index (Layout.softmax_score_row values t) e)
    = S.lemma_index_slice values (t * 8) (t * 8 + 8) e
  in
  FStar.Classical.forall_intro cell;
  assert (S.equal loaded (Layout.softmax_score_row values t))

inline_for_extraction noextract
fn load_row (scores:larray f32 2048) (#values:erased Layout.score_values)
  (t:i32{0 <= I32.v t /\ I32.v t < 256}) (row:larray f32 8)
  preserves gpu ** PA.pts_to scores values
  requires PA.pts_to_uninit row 8
  ensures PA.pts_to row (Layout.softmax_score_row values (I32.v t))
{
  unfold PA.pts_to_uninit row 8;
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (e:i32{0 <= I32.v e /\ I32.v e <= 8})
      (raw:lseq (option f32) 8).
      expert |-> e ** PA.pts_to_mask row raw (fun _ -> True) **
      pure (forall (j:natlt 8). j < I32.v e ==>
        S.index raw j == Some (S.index values (I32.v t * 8 + j)))
    decreases (8 - I32.v !expert)
  {
    let e = !expert;
    let value = scores.(index t *^ 8sz +^ index e);
    PA.mask_write row (index e) value;
    expert := I32.add e 1l;
  };
  PA.from_mask row;
  with loaded. assert PA.pts_to row loaded;
  assert pure (S.length loaded == 8);
  assert pure (forall (e:natlt 8). S.index loaded e == S.index values (I32.v t * 8 + e));
  loaded_row_matches values (I32.v t) loaded;
  rewrite PA.pts_to row loaded as PA.pts_to row (Layout.softmax_score_row values (I32.v t));
}

inline_for_extraction noextract
fn source_select (row:larray f32 8) (#values:erased Softmax.probability_row)
  preserves gpu ** PA.pts_to row values
  returns result:(i32 & f32)
  ensures pure (
    fst result == (Steps.scan_prefix values 8).chosen /\
    snd result == (Steps.scan_prefix values 8).best /\
    FamilyArgmax.row_result values (fst result) (snd result))
{
  let mut expert = 0l;
  let mut selected = 0l;
  let mut best:f32 = Steps.negative_infinity;
  while (I32.lt !expert 8l)
    invariant exists* (e:i32{0 <= I32.v e /\ I32.v e <= 8}) (chosen:i32) (value:f32).
      expert |-> e ** selected |-> chosen ** best |-> value **
      pure (chosen == (Steps.scan_prefix values (I32.v e)).chosen /\
        value == (Steps.scan_prefix values (I32.v e)).best)
    decreases (8 - I32.v !expert)
  {
    let e = !expert;
    let value = row.(index e);
    let previous = !best;
    let old_choice = !selected;
    let improves = F.lt previous value;
    let next_choice:i32 = (if improves then (e <: i32) else (old_choice <: i32));
    let next_best:f32 = (if improves then value else previous);
    best := next_best;
    selected := next_choice;
    expert := I32.add e 1l;
  };
  let chosen = !selected;
  let value = !best;
  Family.source_selection values chosen value;
  (chosen, value)
}

inline_for_extraction noextract
fn emit_cell (t:i32{0 <= I32.v t /\ I32.v t < 256})
  (e:i32{0 <= I32.v e /\ I32.v e < 8})
  (row:larray f32 8) (#values:erased Softmax.probability_row)
  (selected:larray i32 2048) (routes:larray f32 2048)
  (chosen:i32{chosen == (Steps.scan_prefix values 8).chosen})
  (selected_sum:f32{selected_sum == F.add F.zero (Steps.scan_prefix values 8).best})
  (#old_flags:erased flags) (#old_probabilities:erased probabilities)
  preserves gpu ** PA.pts_to row values
  requires PA.pts_to selected old_flags ** PA.pts_to routes old_probabilities
  ensures
    PA.pts_to selected (S.upd old_flags (I32.v t * 8 + I32.v e)
      (Steps.emitted_flag values (I32.v e))) **
    PA.pts_to routes (S.upd old_probabilities (I32.v t * 8 + I32.v e)
      (Steps.emitted_probability values (I32.v e)))
{
  let offset = index t *^ 8sz +^ index e;
  let flag:i32 = (if I32.eq e chosen then 1l else 0l);
  selected.(offset) <- flag;
  if (I32.eq e chosen) {
    let probability = row.(index e);
    routes.(offset) <- F.div probability selected_sum;
  } else {
    routes.(offset) <- F.zero #f32;
  }
}

inline_for_extraction noextract
fn emit_row (t:i32{0 <= I32.v t /\ I32.v t < 256})
  (row:larray f32 8) (#values:erased Softmax.probability_row)
  (selected:larray i32 2048) (routes:larray f32 2048)
  (#old_flags:erased flags) (#old_probabilities:erased probabilities)
  preserves gpu ** PA.pts_to row values
  requires PA.pts_to selected old_flags ** PA.pts_to routes old_probabilities
  returns after:erased Steps.route_buffers
  ensures PA.pts_to selected after.Steps.selected ** PA.pts_to routes after.Steps.normalized **
    pure (forall (u:natlt 256) (e:natlt 8).
      S.index after.Steps.selected (u * 8 + e) ==
        (if u = I32.v t then Steps.emitted_flag values e else S.index old_flags (u * 8 + e)) /\
      S.index after.Steps.normalized (u * 8 + e) ==
        (if u = I32.v t then Steps.emitted_probability values e else S.index old_probabilities (u * 8 + e)))
{
  let (chosen, best) = source_select row #values;
  let selected_sum = F.add (F.zero #f32) best;
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (e:i32{0 <= I32.v e /\ I32.v e <= 8})
      (fs:flags) (ps:probabilities).
      expert |-> e ** PA.pts_to selected fs ** PA.pts_to routes ps **
      pure (forall (u:natlt 256) (j:natlt 8).
        S.index fs (u * 8 + j) ==
          (if u = I32.v t && j < I32.v e then Steps.emitted_flag values j
           else S.index old_flags (u * 8 + j)) /\
        S.index ps (u * 8 + j) ==
          (if u = I32.v t && j < I32.v e then Steps.emitted_probability values j
           else S.index old_probabilities (u * 8 + j)))
    decreases (8 - I32.v !expert)
  {
    let e = !expert;
    emit_cell t e row #values selected routes chosen selected_sum;
    expert := I32.add e 1l;
  };
  with fs ps. assert PA.pts_to selected fs ** PA.pts_to routes ps;
  FStar.Ghost.hide { Steps.selected = fs; Steps.normalized = ps }
}

inline_for_extraction noextract
fn source_token (t:i32{0 <= I32.v t /\ I32.v t < 256})
  (scores:larray f32 2048) (selected:larray i32 2048) (routes:larray f32 2048)
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values)
  (#old_flags:erased flags) (#old_probabilities:erased probabilities)
  preserves gpu ** PA.pts_to scores (Project.scores x w)
  requires PA.pts_to selected old_flags ** PA.pts_to routes old_probabilities **
    pure (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))
  returns observation:erased Softmax.probability_row
  ensures exists* (after:Steps.route_buffers).
    PA.pts_to selected after.selected ** PA.pts_to routes after.normalized **
    pure (row_certificate x w (I32.v t) observation /\
      (forall (u:natlt 256) (e:natlt 8).
        S.index after.selected (u * 8 + e) ==
          (if u = I32.v t then Steps.emitted_flag observation e else S.index old_flags (u * 8 + e)) /\
        S.index after.normalized (u * 8 + e) ==
          (if u = I32.v t then Steps.emitted_probability observation e else S.index old_probabilities (u * 8 + e))))
{
  let mut row:PA.array f32 = [| 8sz |];
  load_row scores #(FStar.Ghost.hide (Project.scores x w)) t row;
  score_row_certificate x w (I32.v t);
  let observation = Softmax.source_normalize row
    #(FStar.Ghost.hide (Layout.softmax_score_row (Project.scores x w) (I32.v t)));
  normalized_certificate x w (I32.v t) observation;
  let after = emit_row t row #observation selected routes #old_flags #old_probabilities;
  PA.to_mask row;
  observation
}

inline_for_extraction noextract
let push (#n:nat{n < 256}) (h:history n) (row:Softmax.probability_row) : history (n + 1) =
  S.append h (S.create 1 row)

inline_for_extraction noextract
let advance_history (x:Layout.token_values) (w:Layout.weight_values)
  (#n:nat{n < 256}) (h:history n) (row:Softmax.probability_row)
  (fs before_fs:flags) (ps before_ps:probabilities)
  : Lemma
      (requires valid x w h /\ row_certificate x w n row /\
        routes_match h before_fs before_ps /\
        (forall (u:natlt 256) (e:natlt 8).
          S.index fs (u * 8 + e) ==
            (if u = n then Steps.emitted_flag row e else S.index before_fs (u * 8 + e)) /\
          S.index ps (u * 8 + e) ==
            (if u = n then Steps.emitted_probability row e else S.index before_ps (u * 8 + e))))
      (ensures valid x w (push h row) /\ routes_match (push h row) fs ps)
  =
  let cell (t:nat{t < n + 1})
    : Lemma (row_certificate x w t (S.index (push h row) t) /\
      (forall (e:natlt 8).
        S.index fs (t * 8 + e) == Steps.emitted_flag (S.index (push h row) t) e /\
        S.index ps (t * 8 + e) == Steps.emitted_probability (S.index (push h row) t) e))
    = if t < n then () else assert (t == n)
  in FStar.Classical.forall_intro cell

inline_for_extraction noextract
let loop_payload (n:prefix) (x:Layout.token_values) (w:Layout.weight_values)
  (selected:larray i32 2048) (routes:larray f32 2048)
  (h:history n) (fs:flags) (ps:probabilities) : slprop =
  PA.pts_to selected fs ** PA.pts_to routes ps **
    pure (valid x w h /\ routes_match h fs ps)

inline_for_extraction noextract
let loop_state (n:prefix) (x:Layout.token_values) (w:Layout.weight_values)
  (selected:larray i32 2048) (routes:larray f32 2048) : slprop =
  exists* (h:history n) (fs:flags) (ps:probabilities).
    loop_payload n x w selected routes h fs ps

inline_for_extraction noextract
ghost
fn pack_loop (n:prefix) (x:Layout.token_values) (w:Layout.weight_values)
  (selected:larray i32 2048) (routes:larray f32 2048)
  (h:history n) (fs:flags) (ps:probabilities)
  requires PA.pts_to selected fs ** PA.pts_to routes ps **
    pure (valid x w h /\ routes_match h fs ps)
  ensures loop_state n x w selected routes
{
  fold loop_payload n x w selected routes h fs ps;
  Pulse.Lib.Core.intro_exists
    (fun (ps:probabilities) -> loop_payload n x w selected routes h fs ps) ps;
  Pulse.Lib.Core.intro_exists
    (fun (fs:flags) -> exists* (ps:probabilities).
      loop_payload n x w selected routes h fs ps) fs;
  Pulse.Lib.Core.intro_exists
    (fun (h:history n) -> exists* (fs:flags) (ps:probabilities).
      loop_payload n x w selected routes h fs ps) h;
  fold loop_state n x w selected routes;
}

inline_for_extraction noextract
fn source_router (scores:larray f32 2048)
  (selected:larray i32 2048) (routes:larray f32 2048)
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values)
  (#old_flags:erased flags) (#old_probabilities:erased probabilities) ()
  preserves gpu ** PA.pts_to scores (Project.scores x w)
  requires PA.pts_to selected old_flags ** PA.pts_to routes old_probabilities **
    pure (Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))
  returns observation:erased (history 256)
  ensures exists* (fs:flags) (ps:probabilities).
    PA.pts_to selected fs ** PA.pts_to routes ps **
    pure (valid x w observation /\ routes_match observation fs ps)
{
  pack_loop 0 x w selected routes (S.empty <: history 0) old_flags old_probabilities;
  let mut token = 0l;
  while (I32.lt !token 256l)
    invariant exists* (t:i32{0 <= I32.v t /\ I32.v t <= 256}).
      token |-> t ** loop_state (I32.v t) x w selected routes
    decreases (256 - I32.v !token)
  {
    let t = !token;
    unfold loop_state (I32.v t) x w selected routes;
    with h fs ps. assert loop_payload (I32.v t) x w selected routes h fs ps;
    unfold loop_payload (I32.v t) x w selected routes h fs ps;
    let row = source_token t scores selected routes #x #w #fs #ps;
    with after. assert PA.pts_to selected after.Steps.selected ** PA.pts_to routes after.Steps.normalized;
    advance_history x w h row after.selected fs after.normalized ps;
    pack_loop (I32.v (I32.add t 1l)) x w selected routes (push h row)
      after.selected after.normalized;
    token := I32.add t 1l;
  };
  let finished = !token;
  rewrite loop_state (I32.v finished) x w selected routes
    as loop_state 256 x w selected routes;
  unfold loop_state 256 x w selected routes;
  with h fs ps. assert loop_payload 256 x w selected routes h fs ps;
  unfold loop_payload 256 x w selected routes h fs ps;
  FStar.Ghost.hide (h <: history 256)
}
