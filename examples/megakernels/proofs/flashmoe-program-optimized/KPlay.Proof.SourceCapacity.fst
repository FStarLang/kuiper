module KPlay.Proof.SourceCapacity
#lang-pulse

open Kuiper

#set-options "--fuel 2 --ifuel 1 --z3rlimit 20"

module S = FStar.Seq
module F = Kuiper.Floating
module I32 = FStar.Int32
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module PA = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module A = Kuiper.Approximates.Base
module Steps = KPlay.Proof.RouteSteps
module Routing = KPlay.Proof.SourceRouting
module History = KPlay.Proof.SourceHistory
module Capacity = KPlay.Proof.Shared.Capacity
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let index = Routing.index

inline_for_extraction noextract
let slot_post (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (n:nat{n <= 256}) (e:natlt 8) (c:natlt 32) (state:Steps.capacity_buffers) : prop =
  let owner = Capacity.slot_owner b n e c in
  I64.v (S.index state.indices (e * 32 + c)) == owner /\
  S.index state.probabilities (e * 32 + c) ==
    (if owner < 0 then F.zero else History.combined h owner)

inline_for_extraction noextract
let slot_frame (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (n:nat{n <= 256}) (e:natlt 8) (c:natlt 32) (before after:Steps.capacity_buffers)
  : Lemma
      (requires slot_post h b n e c before /\
        S.index after.indices (e * 32 + c) == S.index before.indices (e * 32 + c) /\
        S.index after.probabilities (e * 32 + c) == S.index before.probabilities (e * 32 + c))
      (ensures slot_post h b n e c after)
  = ()

inline_for_extraction noextract
let completed (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (experts:nat{experts <= 8}) (state:Steps.capacity_buffers) : prop =
  (forall (e:natlt 8). e < experts ==>
    I32.v (S.index state.demand e) == Capacity.assigned_count b 256 e) /\
  (forall (e:natlt 8) (c:natlt 32). e < experts ==> slot_post h b 256 e c state) /\
  (forall (t:natlt 256) (e:natlt 8). e < experts ==>
    I32.v (S.index state.survival (t * 8 + e)) == Capacity.prefix_survival b 256 t e)

inline_for_extraction noextract
let scanning (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (e:natlt 8) (n:nat{n <= 256}) (count:i32) (state:Steps.capacity_buffers) : prop =
  completed h b e state /\
  I32.v count == Capacity.assigned_count b n e /\
  (forall (c:natlt 32). slot_post h b n e c state) /\
  (forall (t:natlt 256). t < n ==>
    I32.v (S.index state.survival (t * 8 + e)) == Capacity.prefix_survival b n t e)

inline_for_extraction noextract
let initial_scan (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (e:natlt 8) (state:Steps.capacity_buffers)
  : Lemma
      (requires completed h b e state /\
        (forall (c:natlt 32).
          S.index state.indices (e * 32 + c) == -1L /\
          S.index state.probabilities (e * 32 + c) == F.zero))
      (ensures scanning h b e 0 0l state)
  = ()

#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
inline_for_extraction noextract
let cell_advance (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (fs:Routing.flags) (qs:Routing.probabilities)
  (e:natlt 8) (t:natlt 256) (before:Steps.scan_state t)
  : Lemma
      (requires History.compatible h b /\ Routing.routes_match h fs qs /\
        scanning h b e t before.count before.buffers)
      (ensures (
        let after = Steps.capacity_cell_step t e (S.index fs (t * 8 + e))
          (S.index qs (t * 8 + e)) before in
        scanning h b e (t + 1) after.count after.buffers))
  =
  let routes:Steps.route_buffers = {selected = fs; normalized = qs} in
  let row = S.index h t in
  let after = Steps.consume_emitted_cell routes row t e before in
  History.selection h b t;
  Steps.source_bound_capacity_step routes row t e before;
  Steps.shared_capacity_step b routes row t e before;
  let old_expert (other:natlt 8)
    : Lemma (other < e ==>
        I32.v (S.index after.buffers.demand other) == Capacity.assigned_count b 256 other)
    = ()
  in FStar.Classical.forall_intro old_expert;
  let old_slot (other:natlt 8) (c:natlt 32)
    : Lemma (other < e ==> slot_post h b 256 other c after.buffers)
    =
    if other < e then (
      Steps.capacity_step_frame routes row t e before other c 0;
      assert (slot_post h b 256 other c before.buffers);
      slot_frame h b 256 other c before.buffers after.buffers
    )
  in FStar.Classical.forall_intro_2 old_slot;
  let old_flag (u:natlt 256) (other:natlt 8)
    : Lemma (other < e ==>
      I32.v (S.index after.buffers.survival (u * 8 + other)) ==
        Capacity.prefix_survival b 256 u other)
    =
    if other < e then Steps.capacity_step_frame routes row t e before other 0 u
  in FStar.Classical.forall_intro_2 old_flag;
  let slot (c:natlt 32)
    : Lemma (slot_post h b (t + 1) e c after.buffers)
    =
    Capacity.slot_step_law b t e c;
    if b.selected_expert t = e && I32.v before.count = c then (
      assert (Steps.emitted_probability row e == History.combined h t)
    ) else Steps.capacity_step_frame routes row t e before e c 0
  in FStar.Classical.forall_intro slot;
  let flag (u:natlt 256)
    : Lemma (u < t + 1 ==>
      I32.v (S.index after.buffers.survival (u * 8 + e)) ==
        Capacity.prefix_survival b (t + 1) u e)
    =
    if u < t then Steps.capacity_step_frame routes row t e before e 0 u
  in FStar.Classical.forall_intro flag
#pop-options

inline_for_extraction noextract
let finish_expert (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (e:natlt 8) (count:i32) (state:Steps.capacity_buffers)
  : Lemma
      (requires scanning h b e 256 count state)
      (ensures completed h b (e + 1) {
        state with demand = S.upd state.demand e count})
  =
  let after = { state with demand = S.upd state.demand e count } in
  let count_cell (other:natlt 8)
    : Lemma (other < e + 1 ==>
      I32.v (S.index after.demand other) == Capacity.assigned_count b 256 other)
    = if other = e then () else ()
  in FStar.Classical.forall_intro count_cell;
  let slot_cell (other:natlt 8) (c:natlt 32)
    : Lemma (other < e + 1 ==> slot_post h b 256 other c after)
    =
    if other < e + 1 then (
      assert (other < e \/ other == e);
      slot_frame h b 256 other c state after
    )
  in FStar.Classical.forall_intro_2 slot_cell;
  let flag_cell (t:natlt 256) (other:natlt 8)
    : Lemma (other < e + 1 ==>
      I32.v (S.index after.survival (t * 8 + other)) == Capacity.prefix_survival b 256 t other)
    =
    if other < e then (
      assert (completed h b e state);
      assert (I32.v (S.index state.survival (t * 8 + other)) ==
        Capacity.prefix_survival b 256 t other)
    ) else if other = e then (
      assert (I32.v (S.index state.survival (t * 8 + e)) ==
        Capacity.prefix_survival b 256 t e)
    )
  in FStar.Classical.forall_intro_2 flag_cell

inline_for_extraction noextract
let public_output (state:Steps.capacity_buffers) : Shared.public_outputs = {
  expert_demand = (fun e -> S.index state.demand e);
  token_indices = (fun e c -> S.index state.indices (e * 32 + c));
  probabilities = (fun e c -> S.index state.probabilities (e * 32 + c));
  survival = (fun t e -> S.index state.survival (t * 8 + e))
}

inline_for_extraction noextract
let complete_routing (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (state:Steps.capacity_buffers)
  : Lemma
      (requires History.compatible h b /\ completed h b 8 state)
      (ensures Capacity.route_post b (public_output state) /\
        Shared.packing_post b (History.normalization h) (public_output state))
  =
  let slot (e:natlt 8) (c:natlt 32)
    : Lemma (A.v_approximates (S.index state.probabilities (e * 32 + c))
        (Capacity.slot_probability b 256 e c))
    =
    let owner = Capacity.slot_owner b 256 e c in
    assert (slot_post h b 256 e c state);
    if owner >= 0 then (
      assert (S.index state.probabilities (e * 32 + c) == History.combined h owner);
      History.selection h b owner
    ) else assert (S.index state.probabilities (e * 32 + c) == F.zero)
  in
  FStar.Classical.forall_intro_2 slot;
  let owner (e:natlt 8) (c:natlt 32)
    : Lemma (I64.v ((public_output state).token_indices e c) == Capacity.slot_owner b 256 e c)
    = assert (slot_post h b 256 e c state)
  in FStar.Classical.forall_intro_2 owner;
  assert (forall (e:natlt 8).
    I32.v ((public_output state).expert_demand e) == Capacity.assigned_count b 256 e);
  assert (forall (t:natlt 256) (e:natlt 8).
    I32.v ((public_output state).survival t e) == Capacity.prefix_survival b 256 t e);
  assert (forall (e:natlt 8) (c:natlt 32).
    A.v_approximates ((public_output state).probabilities e c) (Capacity.slot_probability b 256 e c));
  assert (Capacity.state_representation (Capacity.prefix_state b 256) (public_output state));
  Capacity.final_prefix_law b (Capacity.prefix_state b 256) (public_output state);
  let packed (e:natlt 8) (c:natlt 32)
    : Lemma (
      let owner = Capacity.slot_owner b 256 e c in
      if owner < 0 then A.v_approximates ((public_output state).probabilities e c) 0.0R
      else (public_output state).probabilities e c ==
        (History.normalization h).combine_probabilities owner)
    = assert (slot_post h b 256 e c state)
  in FStar.Classical.forall_intro_2 packed

inline_for_extraction noextract
let points (demand:larray i32 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (state:Steps.capacity_buffers) : slprop =
  PA.pts_to demand state.demand ** PA.pts_to indices state.indices **
  PA.pts_to probabilities state.probabilities ** PA.pts_to survival state.survival

inline_for_extraction noextract
let initialize_slot_frame (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (e:natlt 8) (c:natlt 32) (before:Steps.capacity_buffers)
  : Lemma
      (requires completed h b e before)
      (ensures completed h b e {
        before with
          indices = S.upd before.indices (e * 32 + c) (-1L);
          probabilities = S.upd before.probabilities (e * 32 + c) F.zero})
  =
  let after = { before with
    indices = S.upd before.indices (e * 32 + c) (-1L);
    probabilities = S.upd before.probabilities (e * 32 + c) F.zero } in
  let slot (other:natlt 8) (j:natlt 32)
    : Lemma (other < e ==> slot_post h b 256 other j after)
    =
    if other < e then (
      assert (other * 32 + j <> e * 32 + c);
      slot_frame h b 256 other j before after
    )
  in FStar.Classical.forall_intro_2 slot

inline_for_extraction noextract
fn initialize_expert
  (e:i32{0 <= I32.v e /\ I32.v e < 8})
  (demand:larray i32 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#h:erased (Routing.history 256)) (#b:erased (Capacity.selected_boundary Capacity.Smoke))
  (#before:erased Steps.capacity_buffers)
  preserves gpu
  requires points demand indices probabilities survival before **
    pure (completed h b (I32.v e) before)
  returns state:erased Steps.capacity_buffers
  ensures points demand indices probabilities survival state **
    pure (scanning h b (I32.v e) 0 0l state)
{
  unfold points demand indices probabilities survival before;
  let mut slot = 0l;
  while (I32.lt !slot 32l)
    invariant exists* (c:i32{0 <= I32.v c /\ I32.v c <= 32})
      (ids:lseq i64 256) (ps:lseq f32 256).
      slot |-> c ** PA.pts_to indices ids ** PA.pts_to probabilities ps **
      pure (completed h b (I32.v e) {
          before with indices = ids; probabilities = ps} /\
        (forall (j:natlt 32). j < I32.v c ==>
          S.index ids (I32.v e * 32 + j) == -1L /\
          S.index ps (I32.v e * 32 + j) == F.zero))
    decreases (32 - I32.v !slot)
  {
    let c = !slot;
    with ids ps. assert PA.pts_to indices ids ** PA.pts_to probabilities ps;
    initialize_slot_frame h b (I32.v e) (I32.v c)
      { before with indices = ids; probabilities = ps };
    let offset = index e *^ 32sz +^ index c;
    indices.(offset) <- -1L;
    probabilities.(offset) <- F.zero #f32;
    slot := I32.add c 1l;
  };
  with ids ps. assert PA.pts_to indices ids ** PA.pts_to probabilities ps;
  let state:erased Steps.capacity_buffers = FStar.Ghost.hide {
    Steps.demand = (FStar.Ghost.reveal before).Steps.demand;
    Steps.indices = ids;
    Steps.probabilities = ps;
    Steps.survival = (FStar.Ghost.reveal before).Steps.survival
  };
  initial_scan h b (I32.v e) state;
  fold points demand indices probabilities survival state;
  state
}

inline_for_extraction noextract
let binary_at (t:natlt 256) (e:natlt 8) =
  values:Routing.flags{
    I32.v (S.index values (t * 8 + e)) == 0 \/
    I32.v (S.index values (t * 8 + e)) == 1}

inline_for_extraction noextract
fn source_cell
  (e:i32{0 <= I32.v e /\ I32.v e < 8})
  (t:i32{0 <= I32.v t /\ I32.v t < 256})
  (selected:larray i32 2048) (routes:larray f32 2048)
  (#fs:erased (binary_at (I32.v t) (I32.v e)))
  (#qs:erased Routing.probabilities)
  (count:i32{0 <= I32.v count /\ I32.v count <= I32.v t})
  (demand:larray i32 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#before:erased Steps.capacity_buffers)
  preserves gpu ** PA.pts_to selected fs ** PA.pts_to routes qs
  requires points demand indices probabilities survival before
  returns next:i32
  ensures (
    let after = Steps.capacity_cell_step (I32.v t) (I32.v e)
      (S.index fs (I32.v t * 8 + I32.v e)) (S.index qs (I32.v t * 8 + I32.v e))
      {count = count; buffers = before} in
    points demand indices probabilities survival after.buffers ** pure (next == after.count))
{
  unfold points demand indices probabilities survival before;
  let offset = index t *^ 8sz +^ index e;
  let flag = selected.(offset);
  let next = I32.add count flag;
  let slot = I32.sub next 1l;
  let keep = I32.ne flag 0l && I32.lt slot 32l;
  let survives:i32 = (if keep then 1l else 0l);
  survival.(offset) <- survives;
  if (keep) {
    let destination = index e *^ 32sz +^ index slot;
    indices.(destination) <- IC.int32_to_int64 t;
    let probability = routes.(offset);
    probabilities.(destination) <- probability;
    fold points demand indices probabilities survival
      (Steps.capacity_cell_step (I32.v t) (I32.v e)
        (S.index fs (I32.v t * 8 + I32.v e)) (S.index qs (I32.v t * 8 + I32.v e))
        {Steps.count = count; Steps.buffers = before}).buffers;
    next
  } else {
    fold points demand indices probabilities survival
      (Steps.capacity_cell_step (I32.v t) (I32.v e)
        (S.index fs (I32.v t * 8 + I32.v e)) (S.index qs (I32.v t * 8 + I32.v e))
        {Steps.count = count; Steps.buffers = before}).buffers;
    next
  }
}

inline_for_extraction noextract
fn source_expert
  (e:i32{0 <= I32.v e /\ I32.v e < 8})
  (selected:larray i32 2048) (routes:larray f32 2048)
  (#fs:erased Routing.flags) (#qs:erased Routing.probabilities)
  (demand:larray i32 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#h:erased (Routing.history 256)) (#b:erased (Capacity.selected_boundary Capacity.Smoke))
  (#before:erased Steps.capacity_buffers)
  preserves gpu ** PA.pts_to selected fs ** PA.pts_to routes qs
  requires points demand indices probabilities survival before **
    pure (History.compatible h b /\ Routing.routes_match h fs qs /\
      completed h b (I32.v e) before)
  returns after:erased Steps.capacity_buffers
  ensures points demand indices probabilities survival after **
    pure (completed h b (I32.v e + 1) after)
{
  let initial = initialize_expert e demand indices probabilities survival #h #b #before;
  let mut token = 0l;
  let mut count = 0l;
  while (I32.lt !token 256l)
    invariant exists* (t:i32{0 <= I32.v t /\ I32.v t <= 256})
      (n:i32{0 <= I32.v n /\ I32.v n <= I32.v t}) (state:Steps.capacity_buffers).
      token |-> t ** count |-> n ** points demand indices probabilities survival state **
      pure (scanning h b (I32.v e) (I32.v t) n state)
    decreases (256 - I32.v !token)
  {
    let t = !token;
    let next_token = I32.add t 1l;
    assert pure (I32.v next_token == I32.v t + 1);
    assert pure (256 - I32.v next_token << 256 - I32.v t);
    let n = !count;
    with state. assert points demand indices probabilities survival state;
    History.flag_and_weight h b fs qs (I32.v t) (I32.v e);
    let next = source_cell e t selected routes
      #(FStar.Ghost.hide ((FStar.Ghost.reveal fs) <: binary_at (I32.v t) (I32.v e))) #qs n
      demand indices probabilities survival #state;
    cell_advance h b fs qs (I32.v e) (I32.v t)
      {Steps.count = n; Steps.buffers = state};
    count := (next <: i32);
    token := next_token;
  };
  let count = !count;
  with state. assert points demand indices probabilities survival state;
  finish_expert h b (I32.v e) count state;
  unfold points demand indices probabilities survival state;
  demand.(index e) <- count;
  let after:erased Steps.capacity_buffers = FStar.Ghost.hide {
    Steps.demand = S.upd (FStar.Ghost.reveal state).Steps.demand (I32.v e) count;
    Steps.indices = (FStar.Ghost.reveal state).Steps.indices;
    Steps.probabilities = (FStar.Ghost.reveal state).Steps.probabilities;
    Steps.survival = (FStar.Ghost.reveal state).Steps.survival
  };
  fold points demand indices probabilities survival after;
  after
}

inline_for_extraction noextract
fn source_router
  (selected:larray i32 2048) (routes:larray f32 2048)
  (#fs:erased Routing.flags) (#qs:erased Routing.probabilities)
  (demand:larray i32 8) (indices:larray i64 256)
  (probabilities:larray f32 256) (survival:larray i32 2048)
  (#h:erased (Routing.history 256)) (#b:erased (Capacity.selected_boundary Capacity.Smoke))
  (#before:erased Steps.capacity_buffers) ()
  preserves gpu ** PA.pts_to selected fs ** PA.pts_to routes qs
  requires points demand indices probabilities survival before **
    pure (History.compatible h b /\ Routing.routes_match h fs qs)
  returns after:erased Steps.capacity_buffers
  ensures points demand indices probabilities survival after **
    pure (completed h b 8 after /\ Capacity.route_post b (public_output after) /\
      Shared.packing_post b (History.normalization h) (public_output after))
{
  let mut expert = 0l;
  while (I32.lt !expert 8l)
    invariant exists* (e:i32{0 <= I32.v e /\ I32.v e <= 8}) (state:Steps.capacity_buffers).
      expert |-> e ** points demand indices probabilities survival state **
      pure (completed h b (I32.v e) state)
    decreases (8 - I32.v !expert)
  {
    let e = !expert;
    with state. assert points demand indices probabilities survival state;
    let after = source_expert e selected routes #fs #qs
      demand indices probabilities survival #h #b #state;
    expert := I32.add e 1l;
  };
  with after. assert points demand indices probabilities survival after;
  complete_routing h b after;
  FStar.Ghost.hide (after <: Steps.capacity_buffers)
}
