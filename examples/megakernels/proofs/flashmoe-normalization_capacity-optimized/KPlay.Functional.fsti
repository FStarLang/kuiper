module KPlay.Functional

#lang-pulse

open Kuiper
module Shared = KPlay.SharedSpec
module S = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module C = FStar.Classical

#set-options "--fuel 2 --ifuel 1 --z3rlimit 80"

inline_for_extraction noextract
let row_length (p:Shared.model_profile) : nat =
  Shared.token_count p * Shared.expert_count p

inline_for_extraction noextract
let slot_length (p:Shared.model_profile) : nat =
  Shared.expert_count p * Shared.capacity p

inline_for_extraction noextract
let profile_bounds (p:Shared.model_profile)
  : Lemma (
      0 < Shared.token_count p /\ Shared.token_count p <= 1024 /\
      0 < Shared.expert_count p /\ Shared.expert_count p <= 16 /\
      0 < Shared.capacity p /\ Shared.capacity p <= 64 /\
      row_length p <= 2147483647 /\ slot_length p <= 2147483647 /\
      SZ.fits (row_length p) /\ SZ.fits (slot_length p))
  = ()

inline_for_extraction noextract
let selected_representation
  (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (selected:lseq I32.t (Shared.token_count p)) : prop =
  forall (t:Shared.token_index p).
    I32.v (S.index selected t) == b.selected_expert t

inline_for_extraction noextract
let input_representation
  (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p)) : prop =
  (forall (t:Shared.token_index p) (e:Shared.expert_index p).
    S.index rows (t * Shared.expert_count p + e) == b.probability_rows t e) /\
  selected_representation p b selected /\
  (forall (t:Shared.token_index p).
    S.index best t == b.selected_probability t)

inline_for_extraction noextract
let packed_coordinates (width:pos) (row:nat) (column:nat{column < width})
  : Lemma (
      (row * width + column) / width == row /\
      (row * width + column) % width == column)
  = ()

inline_for_extraction noextract
let boundary_rows (p:Shared.model_profile) (b:Shared.selected_boundary p)
  : lseq f32 (row_length p) =
  S.init (row_length p) (fun i ->
    b.probability_rows (i / Shared.expert_count p) (i % Shared.expert_count p))

inline_for_extraction noextract
let boundary_selected (p:Shared.model_profile) (b:Shared.selected_boundary p)
  : lseq I32.t (Shared.token_count p) =
  S.init (Shared.token_count p) (fun t -> I32.int_to_t (b.selected_expert t))

inline_for_extraction noextract
let boundary_best (p:Shared.model_profile) (b:Shared.selected_boundary p)
  : lseq f32 (Shared.token_count p) =
  S.init (Shared.token_count p) b.selected_probability

inline_for_extraction noextract
let boundary_representation_covers
  (p:Shared.model_profile) (b:Shared.selected_boundary p)
  : Lemma (
      input_representation p b (boundary_rows p b)
        (boundary_selected p b) (boundary_best p b))
  =
  let cell (t:Shared.token_index p) (e:Shared.expert_index p)
    : Lemma (
        S.index (boundary_rows p b) (t * Shared.expert_count p + e) ==
          b.probability_rows t e)
    = packed_coordinates (Shared.expert_count p) t e
  in
  C.forall_intro_2 cell

(* Retained candidate counting and arithmetic specifications. *)
inline_for_extraction noextract
let rec prefix_count
  (#tokens:nat) (selected:lseq I32.t tokens) (expert:nat)
  (limit:nat{limit <= tokens})
  : Tot (count:nat{count <= limit}) (decreases limit) =
  if limit = 0 then 0
  else prefix_count selected expert (limit - 1) +
    (if I32.v (S.index selected (limit - 1)) = expert then 1 else 0)

inline_for_extraction noextract
let retained
  (#tokens:nat) (selected:lseq I32.t tokens) (capacity:nat)
  (token:nat{token < tokens}) (expert:nat) : bool =
  I32.v (S.index selected token) = expert &&
    prefix_count selected expert token < capacity

inline_for_extraction noextract
let selected_weight_real
  (numerator:real) (denominator:real{denominator =!= 0.0R}) : real =
  numerator /. denominator

inline_for_extraction noextract
let selected_weight_f32 (numerator best:f32) : f32 =
  div numerator (add (zero #f32) best)

inline_for_extraction noextract
let selected_weight_real_model
  (numerator best:f32) (nr:real) (br:real{br =!= 0.0R})
  : Lemma
      (requires v_approximates numerator nr /\ v_approximates best br)
      (ensures v_approximates (selected_weight_f32 numerator best)
        (selected_weight_real nr br))
  =
  a_add (zero #f32) best 0.0R br;
  div_approx numerator (add (zero #f32) best) nr br

inline_for_extraction noextract
let normalized_weight_post (numerator best result:f32) : prop =
  result == selected_weight_f32 numerator best /\
  (forall (nr:real) (br:real{br =!= 0.0R}).
    v_approximates numerator nr /\ v_approximates best br ==>
    v_approximates result (selected_weight_real nr br))

inline_for_extraction noextract
let native_slot_post
  (p:Shared.model_profile)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  (limit:Shared.prefix_length p) (expert:Shared.expert_index p)
  (slot:nat) (token_id:I64.t) (probability:f32) : prop =
  if slot < prefix_count selected expert limit then
    let token = I64.v token_id in
    if 0 <= token && token < limit then
      I32.v (S.index selected token) == expert /\
      prefix_count selected expert token == slot /\
      normalized_weight_post
        (S.index rows (token * Shared.expert_count p + expert))
        (S.index best token) probability
    else False
  else I64.v token_id == -1 /\ probability == zero #f32

inline_for_extraction noextract
let native_prefix
  (p:Shared.model_profile)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  (limit:Shared.prefix_length p)
  (demand:lseq I32.t (Shared.expert_count p))
  (indices:lseq I64.t (slot_length p))
  (weights:lseq f32 (slot_length p))
  (flags:lseq I32.t (row_length p)) : prop =
  (forall (e:Shared.expert_index p).
    I32.v (S.index demand e) == prefix_count selected e limit) /\
  (forall (e:Shared.expert_index p) (c:Shared.slot_index p).
    native_slot_post p rows selected best limit e c
      (S.index indices (e * Shared.capacity p + c))
      (S.index weights (e * Shared.capacity p + c))) /\
  (forall (t:Shared.token_index p) (e:Shared.expert_index p).
    I32.v (S.index flags (t * Shared.expert_count p + e)) ==
      (if t < limit && retained selected (Shared.capacity p) t e then 1 else 0))

(* Every logical output is a view of an actual returned physical cell. *)
inline_for_extraction noextract
let logical_output
  (p:Shared.model_profile)
  (demand:lseq I32.t (Shared.expert_count p))
  (indices:lseq I64.t (slot_length p))
  (weights:lseq f32 (slot_length p))
  (flags:lseq I32.t (row_length p)) : Shared.public_outputs p = {
  expert_demand = (fun e -> S.index demand e);
  token_indices = (fun e c -> S.index indices (e * Shared.capacity p + c));
  probabilities = (fun e c -> S.index weights (e * Shared.capacity p + c));
  survival = (fun t e -> S.index flags (t * Shared.expert_count p + e))
}

inline_for_extraction noextract
let state_representation
  (p:Shared.model_profile)
  (demand:lseq I32.t (Shared.expert_count p))
  (indices:lseq I64.t (slot_length p))
  (weights:lseq f32 (slot_length p))
  (flags:lseq I32.t (row_length p))
  (state:Shared.routing_state p) : prop =
  Shared.state_representation state (logical_output p demand indices weights flags)

inline_for_extraction noextract
let rec prefix_count_represents
  (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (selected:lseq I32.t (Shared.token_count p))
  (limit:Shared.prefix_length p) (expert:Shared.expert_index p)
  : Lemma
      (requires selected_representation p b selected)
      (ensures prefix_count selected expert limit == Shared.assigned_count b limit expert)
      (decreases limit)
  =
  if limit = 0 then ()
  else prefix_count_represents p b selected (limit - 1) expert

inline_for_extraction noextract
let prefix_refines_shared
  (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  (limit:Shared.prefix_length p)
  (demand:lseq I32.t (Shared.expert_count p))
  (indices:lseq I64.t (slot_length p))
  (weights:lseq f32 (slot_length p))
  (flags:lseq I32.t (row_length p))
  : Lemma
      (requires
        input_representation p b rows selected best /\
        native_prefix p rows selected best limit demand indices weights flags)
      (ensures state_representation p demand indices weights flags
        (Shared.prefix_state b limit))
  =
  let demand_ok (e:Shared.expert_index p)
    : Lemma (I32.v (S.index demand e) == Shared.assigned_count b limit e)
    = prefix_count_represents p b selected limit e
  in
  C.forall_intro demand_ok;
  let slot_ok (e:Shared.expert_index p) (c:Shared.slot_index p)
    : Lemma (
        I64.v (S.index indices (e * Shared.capacity p + c)) ==
          Shared.slot_owner b limit e c /\
        v_approximates (S.index weights (e * Shared.capacity p + c))
          (Shared.slot_probability b limit e c))
    =
    prefix_count_represents p b selected limit e;
    Shared.slot_characterization b limit e c;
    if c < prefix_count selected e limit then (
      let t = I64.v (S.index indices (e * Shared.capacity p + c)) in
      assert (0 <= t /\ t < limit);
      prefix_count_represents p b selected t e;
      assert (b.selected_expert t == e /\ Shared.assigned_count b t e == c);
      Shared.retained_slot_law b limit t e;
      assert (S.index rows (t * Shared.expert_count p + e) ==
        b.selected_probability t);
      a_add (zero #f32) (S.index best t) 0.0R (b.selected_model t);
      Shared.normalization_approx b t (add (zero #f32) (S.index best t))
    )
  in
  C.forall_intro_2 slot_ok;
  let survival_ok (t:Shared.token_index p) (e:Shared.expert_index p)
    : Lemma (
        I32.v (S.index flags (t * Shared.expert_count p + e)) ==
          Shared.prefix_survival b limit t e)
    = prefix_count_represents p b selected t e
  in
  C.forall_intro_2 survival_ok

inline_for_extraction noextract
let semantic_post
  (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  (demand:lseq I32.t (Shared.expert_count p))
  (indices:lseq I64.t (slot_length p))
  (weights:lseq f32 (slot_length p))
  (flags:lseq I32.t (row_length p)) : prop =
  native_prefix p rows selected best (Shared.token_count p) demand indices weights flags /\
  Shared.route_post b (logical_output p demand indices weights flags)

fn kplay_run
  (#p:erased Shared.model_profile)
  (#b:erased (Shared.selected_boundary p))
  (token_count:sz{SZ.v token_count == Shared.token_count p})
  (expert_count:sz{SZ.v expert_count == Shared.expert_count p})
  (capacity:I64.t{I64.v capacity == Shared.capacity p})
  (normalized_rows:larray f32 (row_length p){is_global_array normalized_rows})
  (selected_experts:larray I32.t (Shared.token_count p){is_global_array selected_experts})
  (best_values:larray f32 (Shared.token_count p){is_global_array best_values})
  (expert_demand:larray I32.t (Shared.expert_count p){is_global_array expert_demand})
  (token_indices:larray I64.t (slot_length p){is_global_array token_indices})
  (probabilities:larray f32 (slot_length p){is_global_array probabilities})
  (survival:larray I32.t (row_length p){is_global_array survival})
  (#rows_perm #selected_perm #best_perm:perm)
  (#rows:erased (lseq f32 (row_length p)))
  (#selected:erased (lseq I32.t (Shared.token_count p)))
  (#best:erased (lseq f32 (Shared.token_count p)))
  preserves cpu **
    on gpu_loc (
      normalized_rows |-> Frac rows_perm (FStar.Ghost.reveal rows) **
      selected_experts |-> Frac selected_perm (FStar.Ghost.reveal selected) **
      best_values |-> Frac best_perm (FStar.Ghost.reveal best))
  requires pure (input_representation p b rows selected best) **
    (exists*
      (demand_before:lseq I32.t (Shared.expert_count p))
      (indices_before:lseq I64.t (slot_length p))
      (weights_before:lseq f32 (slot_length p))
      (flags_before:lseq I32.t (row_length p)).
      on gpu_loc (
        expert_demand |-> demand_before **
        token_indices |-> indices_before **
        probabilities |-> weights_before **
        survival |-> flags_before))
  ensures exists*
    (demand:lseq I32.t (Shared.expert_count p))
    (indices:lseq I64.t (slot_length p))
    (weights:lseq f32 (slot_length p))
    (flags:lseq I32.t (row_length p)).
    on gpu_loc (
      expert_demand |-> demand **
      token_indices |-> indices **
      probabilities |-> weights **
      survival |-> flags) **
    pure (semantic_post p b rows selected best demand indices weights flags)
