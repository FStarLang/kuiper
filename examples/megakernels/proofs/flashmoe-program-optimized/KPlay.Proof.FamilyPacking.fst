module KPlay.Proof.FamilyPacking

open Kuiper

module S = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module F = Kuiper.Floating
module Family = KPlay.Proof.FamilyCapacity
module Capacity = KPlay.Proof.Shared.Capacity
module Routing = KPlay.Proof.SourceRouting
module History = KPlay.Proof.SourceHistory
module Observe = KPlay.Proof.SourceCapacity
module Steps = KPlay.Proof.RouteSteps

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

(* Erased views of the actual local rows and winning registers, not extra
   dense tensors or calls to the family's synchronous host wrapper. *)
inline_for_extraction noextract
let rows (h:Routing.history 256) : lseq f32 2048 =
  S.init #f32 2048 (fun n -> S.index (S.index h (n / 8)) (n % 8))

inline_for_extraction noextract
let selected (h:Routing.history 256) : lseq i32 256 =
  S.init #i32 256 (fun t -> (Steps.scan_prefix (S.index h t) 8).chosen)

inline_for_extraction noextract
let gathered (h:Routing.history 256) : lseq f32 256 =
  S.init #f32 256 (fun t -> (Steps.scan_prefix (S.index h t) 8).best)

inline_for_extraction noextract
let input_binding (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  : Lemma
      (requires History.compatible h b)
      (ensures Family.input_representation Capacity.Smoke b (rows h) (selected h) (gathered h))
  =
  let token (t:natlt 256)
    : Lemma (
      I32.v (S.index (selected h) t) == b.selected_expert t /\
      S.index (gathered h) t == b.selected_probability t)
    = History.selection h b t
  in FStar.Classical.forall_intro token;
  let cell (t:natlt 256) (e:natlt 8)
    : Lemma (S.index (rows h) (t * 8 + e) == b.probability_rows t e)
    = Family.packed_coordinates 8 t e
  in FStar.Classical.forall_intro_2 cell

inline_for_extraction noextract
let count_binding (h:Routing.history 256) (b:Capacity.selected_boundary Capacity.Smoke)
  (n:nat{n <= 256}) (e:natlt 8)
  : Lemma
      (requires History.compatible h b)
      (ensures Family.prefix_count (selected h) e n == Capacity.assigned_count b n e)
  =
  input_binding h b;
  Family.prefix_count_represents Capacity.Smoke b (selected h) n e

inline_for_extraction noextract
let native_prefix_from_source (h:Routing.history 256)
  (b:Capacity.selected_boundary Capacity.Smoke) (state:Steps.capacity_buffers)
  : Lemma
      (requires History.compatible h b /\ Observe.completed h b 8 state)
      (ensures Family.native_prefix Capacity.Smoke (rows h) (selected h) (gathered h) 256
        state.demand state.indices state.probabilities state.survival)
  =
  input_binding h b;
  let demand (e:natlt 8)
    : Lemma (I32.v (S.index state.demand e) == Family.prefix_count (selected h) e 256)
    = count_binding h b 256 e
  in FStar.Classical.forall_intro demand;
  let slot (e:natlt 8) (c:natlt 32)
    : Lemma (Family.native_slot_post Capacity.Smoke (rows h) (selected h) (gathered h)
      256 e c (S.index state.indices (e * 32 + c)) (S.index state.probabilities (e * 32 + c)))
    =
    count_binding h b 256 e;
    Capacity.slot_characterization b 256 e c;
    assert (Observe.slot_post h b 256 e c state);
    let owner = Capacity.slot_owner b 256 e c in
    if c < Family.prefix_count (selected h) e 256 then (
      assert (0 <= owner /\ owner < 256);
      count_binding h b owner e;
      History.selection h b owner;
      assert (History.choice h owner == e);
      assert (S.index (rows h) (owner * 8 + e) == b.selected_probability owner);
      assert (S.index (gathered h) owner == b.selected_probability owner);
      Family.weight_correct (S.index (rows h) (owner * 8 + e)) (S.index (gathered h) owner);
      assert (Family.selected_weight_f32 (S.index (rows h) (owner * 8 + e))
        (S.index (gathered h) owner) == History.combined h owner)
    )
  in FStar.Classical.forall_intro_2 slot;
  let survival (t:natlt 256) (e:natlt 8)
    : Lemma (I32.v (S.index state.survival (t * 8 + e)) ==
      (if t < 256 && Family.retained (selected h) 32 t e then 1 else 0))
    =
    count_binding h b t e;
    History.selection h b t
  in FStar.Classical.forall_intro_2 survival

inline_for_extraction noextract
let source_family_post (h:Routing.history 256)
  (b:Capacity.selected_boundary Capacity.Smoke) (state:Steps.capacity_buffers)
  : Lemma
      (requires History.compatible h b /\ Observe.completed h b 8 state)
      (ensures Family.semantic_post Capacity.Smoke b (rows h) (selected h) (gathered h)
        state.demand state.indices state.probabilities state.survival)
  =
  input_binding h b;
  native_prefix_from_source h b state;
  Family.prefix_finished Capacity.Smoke b (rows h) (selected h) (gathered h)
    state.demand state.indices state.probabilities state.survival
