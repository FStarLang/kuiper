module KPlay.Proof.BaseCapacity

#lang-pulse

open Kuiper
module Shared = KPlay.Proof.Shared.Capacity
module I32 = FStar.Int32
module I64 = FStar.Int64
module U64 = FStar.UInt64
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module FP = Kuiper.Floating
module Classical = FStar.Classical

inline_for_extraction noextract
let reference_input (p:Shared.model_profile) =
  Shared.selected_boundary p

inline_for_extraction noextract
let slot_length (p:Shared.model_profile) : nat =
  Shared.expert_count p * Shared.capacity p

inline_for_extraction noextract
let survival_length (p:Shared.model_profile) (rows:Shared.prefix_length p) : nat =
  rows * Shared.expert_count p

inline_for_extraction noextract
let demand_length (token:nat) : nat =
  if token = 255 then 8 else 0

inline_for_extraction noextract
let slot_offset
  (p:Shared.model_profile) (e:Shared.expert_index p) (j:Shared.slot_index p)
  : n:nat{n < slot_length p}
  = e * Shared.capacity p + j

inline_for_extraction noextract
let survival_offset
  (p:Shared.model_profile) (rows:Shared.prefix_length p)
  (t:nat{t < rows}) (e:Shared.expert_index p)
  : n:nat{n < survival_length p rows}
  = t * Shared.expert_count p + e

inline_for_extraction noextract
let approved_profile_fits_native
  (p:Shared.model_profile) (t:Shared.token_index p)
  : Lemma (
      8 <= Shared.expert_count p /\ Shared.expert_count p <= 16 /\
      Shared.expert_count p <= 2147483648 /\
      Shared.token_count p <= 1024 /\ t < 9223372036854775807 /\
      slot_length p < 4294967296 /\
      survival_length p (t + 1) < 4294967296 /\
      SZ.fits (Shared.expert_count p) /\ SZ.fits (slot_length p) /\
      SZ.fits (survival_length p (t + 1)))
  = SZ.fits_ok (Shared.expert_count p);
    SZ.fits_ok (slot_length p);
    SZ.fits_ok (survival_length p (t + 1))

inline_for_extraction noextract
let both_profile_layouts ()
  : Lemma (
      slot_length Shared.Smoke == 256 /\
      survival_length Shared.Smoke 256 == 2048 /\
      slot_length Shared.Full == 1024 /\
      survival_length Shared.Full 1024 == 16384 /\
      demand_length 255 == 8 /\ demand_length 1023 == 0)
  = ()

inline_for_extraction noextract
noeq type physical_state
  (p:Shared.model_profile) (rows:Shared.prefix_length p) = {
  counts: lseq i64 (Shared.expert_count p);
  indices: lseq i64 (slot_length p);
  weights: lseq f32 (slot_length p);
  flags: lseq i32 (survival_length p rows)
}

inline_for_extraction noextract
let state_representation
  (p:Shared.model_profile) (rows:Shared.prefix_length p)
  (v:physical_state p rows) (s:Shared.routing_state p) : prop =
  (forall (e:Shared.expert_index p).
    I64.v (Seq.index v.counts e) == s.counts e) /\
  (forall (e:Shared.expert_index p) (j:Shared.slot_index p).
    let off = slot_offset p e j in
    I64.v (Seq.index v.indices off) == s.owners e j /\
    v_approximates (Seq.index v.weights off) (s.weights e j) /\
    (s.owners e j == -1 ==> Seq.index v.weights off == zero #f32)) /\
  (forall (t:Shared.token_index p) (e:Shared.expert_index p).
    if t < rows then
      I32.v (Seq.index v.flags (survival_offset p rows t e)) ==
        s.survival_cells t e
    else s.survival_cells t e == 0)

inline_for_extraction noextract
let selected_weight (q:f32) : f32 =
  div q (add (zero #f32) q)

inline_for_extraction noextract
let selected_weight_represents (q:f32) (r:Shared.positive_real)
  : Lemma
      (requires v_approximates q r)
      (ensures
        v_approximates (selected_weight q) (r /. Shared.selected_set_sum r))
  = a_add (zero #f32) q 0.0R r;
    div_approx q (add (zero #f32) q) r (Shared.selected_set_sum r)

(* Selection and gather refer to the producer's same stored native row. *)
inline_for_extraction noextract
let selected_boundary
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (selected:lseq i32 1) : prop =
  I32.v (Seq.index selected 0) == x.selected_expert t

inline_for_extraction noextract
let row_representation
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (row:lseq f32 (Shared.expert_count p)) : prop =
  forall (e:Shared.expert_index p).
    Seq.index row e == x.probability_rows t e

inline_for_extraction noextract
let input_representation
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (selected:lseq i32 1) (row:lseq f32 (Shared.expert_count p))
  (before:physical_state p (t + 1)) : prop =
  selected_boundary p x t selected /\
  row_representation p x t row /\
  state_representation p (t + 1) before (Shared.prefix_state x t)

inline_for_extraction noextract
let native_maximum_represents_selection
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (selected:lseq i32 1) (row:lseq f32 (Shared.expert_count p))
  (chosen:Shared.expert_index p)
  : Lemma
      (requires
        row_representation p x t row /\
        I32.v (Seq.index selected 0) == chosen /\
        (forall (e:Shared.expert_index p).
          FP.lte (Seq.index row e) (Seq.index row chosen) /\
          (e < chosen ==> FP.lt (Seq.index row e) (Seq.index row chosen))))
      (ensures
        selected_boundary p x t selected /\
        Seq.index row chosen == x.selected_probability t /\
        v_approximates (Seq.index row chosen) (x.selected_model t))
  = Shared.native_selection_law (x.probability_rows t) chosen;
    Shared.native_order2
      (x.probability_rows t chosen)
      (x.probability_rows t (x.selected_expert t))

inline_for_extraction noextract
let boundary_representation_covers
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (before:physical_state p (t + 1))
  : Lemma
      (requires
        state_representation p (t + 1) before (Shared.prefix_state x t))
      (ensures
        input_representation p x t
          (Seq.create 1 (I32.int_to_t (x.selected_expert t)))
          (Seq.init (Shared.expert_count p) (x.probability_rows t)) before)
  = approved_profile_fits_native p t

inline_for_extraction noextract
let native_step
  (p:Shared.model_profile) (t:Shared.token_index p) (chosen:Shared.expert_index p)
  (row:lseq f32 (Shared.expert_count p))
  (before after:physical_state p (t + 1)) : prop =
  (forall (e:Shared.expert_index p).
    I64.v (Seq.index after.counts e) ==
      I64.v (Seq.index before.counts e) + (if e = chosen then 1 else 0)) /\
  (forall (e:Shared.expert_index p) (j:Shared.slot_index p).
    let off = slot_offset p e j in
    if e = chosen && I64.v (Seq.index before.counts e) = j then
      I64.v (Seq.index after.indices off) == t /\
      Seq.index after.weights off == selected_weight (Seq.index row chosen)
    else
      Seq.index after.indices off == Seq.index before.indices off /\
      Seq.index after.weights off == Seq.index before.weights off) /\
  (forall (u:nat{u < t + 1}) (e:Shared.expert_index p).
    let off = survival_offset p (t + 1) u e in
    if u = t && e = chosen &&
       I64.v (Seq.index before.counts e) < Shared.capacity p then
      I32.v (Seq.index after.flags off) == 1
    else Seq.index after.flags off == Seq.index before.flags off)

inline_for_extraction noextract
let demand_representation
  (p:Shared.model_profile) (t:Shared.token_index p)
  (demand:lseq i32 (demand_length t))
  (after:physical_state p (t + 1)) : prop =
  if t = 255 then
    (forall (e:nat{e < 8}).
      let n = I64.v (Seq.index after.counts e) in
      0 <= n /\ n < 2147483648 /\ I32.v (Seq.index demand e) == n)
  else True

inline_for_extraction noextract
let native_step_refines_prefix
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (selected:lseq i32 1) (row:lseq f32 (Shared.expert_count p))
  (before after:physical_state p (t + 1))
  : Lemma
      (requires
        input_representation p x t selected row before /\
        native_step p t (x.selected_expert t) row before after)
      (ensures
        state_representation p (t + 1) after (Shared.prefix_state x (t + 1)))
  = let chosen = x.selected_expert t in
    selected_weight_represents (Seq.index row chosen) (x.selected_model t);
    let count_cell (e:Shared.expert_index p)
      : Lemma (
          I64.v (Seq.index after.counts e) == Shared.assigned_count x (t + 1) e)
      = Shared.count_step_law x t e
    in
    Classical.forall_intro count_cell;
    let slot_cell (e:Shared.expert_index p) (j:Shared.slot_index p)
      : Lemma (
          let off = slot_offset p e j in
          I64.v (Seq.index after.indices off) == Shared.slot_owner x (t + 1) e j /\
          v_approximates (Seq.index after.weights off)
            (Shared.slot_probability x (t + 1) e j) /\
          (Shared.slot_owner x (t + 1) e j == -1
            ==> Seq.index after.weights off == zero #f32))
      = Shared.slot_step_law x t e j
    in
    Classical.forall_intro_2 slot_cell;
    let flag_cell (u:Shared.token_index p) (e:Shared.expert_index p)
      : Lemma (
          if u < t + 1 then
            I32.v (Seq.index after.flags (survival_offset p (t + 1) u e)) ==
              Shared.prefix_survival x (t + 1) u e
          else Shared.prefix_survival x (t + 1) u e == 0)
      = Shared.survival_step_law x t u e
    in
    Classical.forall_intro_2 flag_cell

inline_for_extraction noextract
let incoming_count_bounds
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (before:physical_state p (t + 1))
  : Lemma
      (requires state_representation p (t + 1) before (Shared.prefix_state x t))
      (ensures forall (e:Shared.expert_index p).
        0 <= I64.v (Seq.index before.counts e) /\
        I64.v (Seq.index before.counts e) <= t)
  = ()

inline_for_extraction noextract
let packed_offset_injective
  (width:pos) (a b:nat) (i j:nat{i < width /\ j < width})
  : Lemma (a * width + i == b * width + j <==> a == b /\ i == j)
  = ()

inline_for_extraction noextract
let updated_state
  (p:Shared.model_profile)
  (token:i64{0 <= I64.v token /\ I64.v token < Shared.token_count p})
  (chosen:Shared.expert_index p) (weight:f32)
  (before:physical_state p (I64.v token + 1))
  (rank:i64{0 < I64.v rank})
  : physical_state p (I64.v token + 1) =
  let counts = Seq.upd before.counts chosen rank in
  if I64.v rank <= Shared.capacity p then
    let off = slot_offset p chosen (I64.v rank - 1) in
    {
      counts = counts;
      indices = Seq.upd before.indices off token;
      weights = Seq.upd before.weights off weight;
      flags = Seq.upd before.flags
        (survival_offset p (I64.v token + 1) (I64.v token) chosen) 1l
    }
  else {
    counts = counts;
    indices = before.indices;
    weights = before.weights;
    flags = before.flags
  }

inline_for_extraction noextract
let updated_state_is_native
  (p:Shared.model_profile)
  (token:i64{0 <= I64.v token /\ I64.v token < Shared.token_count p})
  (chosen:Shared.expert_index p)
  (row:lseq f32 (Shared.expert_count p)) (weight:f32)
  (before:physical_state p (I64.v token + 1))
  (rank:i64{0 < I64.v rank})
  : Lemma
      (requires
        I64.v rank == I64.v (Seq.index before.counts chosen) + 1 /\
        weight == selected_weight (Seq.index row chosen))
      (ensures
        native_step p (I64.v token) chosen row before
          (updated_state p token chosen weight before rank))
  =
  let after = updated_state p token chosen weight before rank in
  let count_cell (e:Shared.expert_index p)
    : Lemma (
        I64.v (Seq.index after.counts e) ==
          I64.v (Seq.index before.counts e) + (if e = chosen then 1 else 0))
    = ()
  in
  Classical.forall_intro count_cell;
  let slot_cell (e:Shared.expert_index p) (j:Shared.slot_index p)
    : Lemma (
        let off = slot_offset p e j in
        if e = chosen && I64.v (Seq.index before.counts e) = j then
          I64.v (Seq.index after.indices off) == I64.v token /\
          Seq.index after.weights off == selected_weight (Seq.index row chosen)
        else
          Seq.index after.indices off == Seq.index before.indices off /\
          Seq.index after.weights off == Seq.index before.weights off)
    =
    if I64.v rank <= Shared.capacity p then
      packed_offset_injective (Shared.capacity p) e chosen j (I64.v rank - 1)
    else ()
  in
  Classical.forall_intro_2 slot_cell;
  let flag_cell (u:nat{u < I64.v token + 1}) (e:Shared.expert_index p)
    : Lemma (
        let off = survival_offset p (I64.v token + 1) u e in
        if u = I64.v token && e = chosen &&
           I64.v (Seq.index before.counts e) < Shared.capacity p then
          I32.v (Seq.index after.flags off) == 1
        else Seq.index after.flags off == Seq.index before.flags off)
    =
    packed_offset_injective (Shared.expert_count p) u (I64.v token) e chosen
  in
  Classical.forall_intro_2 flag_cell

