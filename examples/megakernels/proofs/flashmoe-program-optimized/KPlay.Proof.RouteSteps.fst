module KPlay.Proof.RouteSteps

open Kuiper

module S = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base
module Argmax = KPlay.Proof.Shared.Argmax
module Capacity = KPlay.Proof.Shared.Capacity
module Shared = KPlay.SharedSpec
module Layout = KPlay.Proof.SourceLayouts
module Results = KPlay.Proof.SourceResults
module Input = Kuiper.Tensor.Input

inline_for_extraction noextract
let route_cell_readback before row t e =
  S.lemma_index_upd1 before.selected (t * 8 + e) (emitted_flag row e);
  S.lemma_index_upd1 before.normalized (t * 8 + e) (emitted_probability row e)

inline_for_extraction noextract
let route_cell_frame before row t e other_t other_e =
  assert (other_t * 8 + other_e <> t * 8 + e);
  S.lemma_index_upd2 before.selected (t * 8 + e) (emitted_flag row e)
    (other_t * 8 + other_e);
  S.lemma_index_upd2 before.normalized (t * 8 + e) (emitted_probability row e)
    (other_t * 8 + other_e)

inline_for_extraction noextract
let rec route_prefix_readback before rows cells t e =
  let n = cells - 1 in
  let previous = emit_route_prefix before rows n in
  if t * 8 + e = n then (
    assert (n / 8 == t /\ n % 8 == e);
    route_cell_readback previous (rows t) t e
  ) else (
    route_prefix_readback before rows n t e;
    route_cell_frame previous (rows (n / 8)) (n / 8) (n % 8) t e
  )

inline_for_extraction noextract
let consume_emitted_cell routes row t e before =
  let produced = emit_route_cell routes row t e in
  route_cell_readback routes row t e;
  capacity_cell_step t e
    (S.index produced.selected (t * 8 + e))
    (S.index produced.normalized (t * 8 + e)) before

inline_for_extraction noextract
let consume_completed_route_cell routes rows t e before =
  let produced = emit_route_prefix routes rows (256 * 8) in
  route_prefix_readback routes rows (256 * 8) t e;
  capacity_cell_step t e
    (S.index produced.selected (t * 8 + e))
    (S.index produced.normalized (t * 8 + e)) before

inline_for_extraction noextract
let finite_above_seed value =
  F.neg_kind #f32 F.infinity;
  F.neg_kind #f32 negative_infinity;
  F.neg_neg #f32 F.infinity;
  if F.is_zero #f32 F.infinity then F.is_zero_spec #f32 F.infinity;
  F.eq_spec #f32 (F.neg negative_infinity) F.infinity;
  assert (F.neg negative_infinity == F.infinity);
  F.neg_kind #f32 value;
  F.infinity_val_spec #f32 (F.neg value);
  F.lte_is_lt_or_eq #f32 (F.neg value) F.infinity;
  F.eq_spec #f32 (F.neg value) F.infinity;
  F.lt_neg_flip #f32 negative_infinity value

inline_for_extraction noextract
let rec scan_matches_native_prefix row n =
  if n = 1 then finite_above_seed (S.index row 0)
  else scan_matches_native_prefix row (n - 1)

inline_for_extraction noextract
let same_row_selection row =
  scan_matches_native_prefix row 8;
  Shared.selector_compatibility #8 (native_row row) 8;
  Argmax.best_prefix_correct (native_row row) 8

inline_for_extraction noextract
let source_bound_capacity_step routes row t e before =
  if e = I32.v (scan_prefix row 8).chosen then ()
  else ()

inline_for_extraction noextract
let capacity_step_frame routes row t e before other c u =
  source_bound_capacity_step routes row t e before;
  if other <> e || c <> I32.v before.count then (
    assert (other * 32 + c <> e * 32 + I32.v before.count \/
      I32.v before.count >= 32)
  );
  if u <> t || other <> e then
    assert (u * 8 + other <> t * 8 + e)

inline_for_extraction noextract
let source_scheduled_capacity_step routes rows t e before =
  source_bound_capacity_step routes (rows t) t e before

inline_for_extraction noextract
let shared_capacity_step b routes row t e before =
  same_row_selection row;
  Capacity.selection_prefix_extensional #8
    (fun j -> S.index row j) (b.probability_rows t) 8;
  source_bound_capacity_step routes row t e before;
  Capacity.count_step_law b t e;
  if b.selected_expert t = e && I32.v before.count < 32 then (
    assert ((scan_prefix row 8).best == b.selected_probability t);
    assert (S.index row e == b.selected_probability t);
    A.a_add #f32 F.zero (scan_prefix row 8).best 0.0R (b.selected_model t);
    Capacity.normalization_approx b t (F.add F.zero (scan_prefix row 8).best)
  )

inline_for_extraction noextract
let ordinary_input_views tokens weights =
  Input.view_metadata tokens;
  Input.view_metadata weights

inline_for_extraction noextract
let source_projection_to_softmax x w before tile r e =
  Layout.mma_prefix_bounds x w tile 32;
  Layout.mma_to_softmax_same_cell x w before tile 31
    (Layout.mma_prefix x w tile 31) r e

inline_for_extraction noextract
let source_projection_frame x w before tile t e =
  Layout.score_store_frame before tile (Layout.mma_prefix x w tile 32) t e

inline_for_extraction noextract
let source_scheduled_shared_capacity_step b routes rows t e before =
  source_scheduled_capacity_step routes rows t e before;
  shared_capacity_step b routes (rows t) t e before;
  source_bound_capacity_step routes (rows t) t e before

inline_for_extraction noextract
let source_return_preserves_output_binding handles values out value =
  Results.output_binding_preserved handles values out value
