module KPlay.Proof.RouteSteps

open Kuiper

module S = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module A = Kuiper.Approximates.Base
module Argmax = KPlay.Proof.Shared.Argmax
module Capacity = KPlay.Proof.Shared.Capacity
module Shared = KPlay.SharedSpec
module Layout = KPlay.Proof.SourceLayouts
module Results = KPlay.Proof.SourceResults
module Native = KPlay.Proof.Shared.Native
module RT = Kuiper.Tensor.Runtime
module Input = Kuiper.Tensor.Input
module PA = Pulse.Lib.Array
module C = Kuiper.Chest
module B = Kuiper.Floating.Bounds

noextract type token = Shared.token_index
noextract type expert = Shared.expert_index
noextract type slot = Shared.slot_index

inline_for_extraction noextract
let negative_infinity : f32 = F.neg F.infinity

noextract noeq
type scan_result = {
  chosen: value:I32.t{0 <= I32.v value /\ I32.v value < 8};
  best: f32
}

(* Exact strict-update order at optimized_s.cu:85-93, including the seed. *)
inline_for_extraction noextract
let rec scan_prefix (row:lseq f32 8) (n:nat{n <= 8})
  : Tot scan_result (decreases n) =
  if n = 0 then { chosen = 0l; best = negative_infinity }
  else
    let before = scan_prefix row (n - 1) in
    let value = S.index row (n - 1) in
    if F.lt before.best value
    then { chosen = I32.int_to_t (n - 1); best = value }
    else before

noextract type probability_row =
  row:lseq f32 8{forall (e:expert).
    F.kind (S.index row e) == F.Finite /\
    F.lte F.zero (S.index row e) /\ F.lte (S.index row e) F.one}

inline_for_extraction noextract
let native_row (row:probability_row) : Argmax.row 8 =
  fun e -> S.index row e

noextract noeq
type route_buffers = {
  selected: lseq I32.t (256 * 8);
  normalized: lseq f32 (256 * 8)
}

inline_for_extraction noextract
let emitted_flag (row:lseq f32 8) (e:expert)
  : flag:I32.t{I32.v flag == 0 \/ I32.v flag == 1} =
  if I32.eq (I32.int_to_t e) (scan_prefix row 8).chosen then 1l else 0l

inline_for_extraction noextract
let emitted_probability (row:lseq f32 8) (e:expert) : f32 =
  let result = scan_prefix row 8 in
  let selected_sum = F.add F.zero result.best in
  if I32.eq (I32.int_to_t e) result.chosen
  then F.div (S.index row e) selected_sum
  else F.zero

inline_for_extraction noextract
let emit_route_cell (before:route_buffers) (row:lseq f32 8) (t:token) (e:expert)
  : route_buffers = {
  selected = S.upd before.selected (t * 8 + e) (emitted_flag row e);
  normalized = S.upd before.normalized (t * 8 + e) (emitted_probability row e)
}

noextract type row_observations = token -> lseq f32 8

(* Observations of the fused token/expert loop; no dense row buffer is allocated. *)
inline_for_extraction noextract
let rec emit_route_prefix (before:route_buffers) (rows:row_observations)
  (cells:nat{cells <= 256 * 8}) : Tot route_buffers (decreases cells) =
  if cells = 0 then before
  else
    let n = cells - 1 in
    emit_route_cell (emit_route_prefix before rows n) (rows (n / 8))
      (n / 8) (n % 8)

noextract noeq
type capacity_buffers = {
  demand: lseq I32.t 8;
  indices: lseq I64.t (8 * 32);
  probabilities: lseq f32 (8 * 32);
  survival: lseq I32.t (256 * 8)
}

noextract noeq
type scan_state (prefix:nat{prefix <= 256}) = {
  count: value:I32.t{0 <= I32.v value /\ I32.v value <= prefix};
  buffers: capacity_buffers
}

(* One inner iteration at optimized_s.cu:120-129, not a token-major wrapper. *)
inline_for_extraction noextract
let capacity_cell_step (t:token) (e:expert)
  (selected:I32.t{I32.v selected == 0 \/ I32.v selected == 1})
  (probability:f32) (before:scan_state t) : scan_state (t + 1) =
  let count = I32.add before.count selected in
  let rank = I32.sub count 1l in
  let keep = I32.ne selected 0l && I32.lt rank 32l in
  {
    count = count;
    buffers = {
      demand = before.buffers.demand;
      indices = (if keep then
        S.upd before.buffers.indices (e * 32 + I32.v rank) (I64.int_to_t t)
        else before.buffers.indices);
      probabilities = (if keep then
        S.upd before.buffers.probabilities (e * 32 + I32.v rank) probability
        else before.buffers.probabilities);
      survival = S.upd before.buffers.survival (t * 8 + e)
        (if keep then 1l else 0l)
    }
  }

inline_for_extraction noextract
val route_cell_readback
  (before:route_buffers) (row:lseq f32 8) (t:token) (e:expert)
  : Lemma (
      let after = emit_route_cell before row t e in
      S.index after.selected (t * 8 + e) == emitted_flag row e /\
      S.index after.normalized (t * 8 + e) == emitted_probability row e)

inline_for_extraction noextract
val route_cell_frame (before:route_buffers) (row:lseq f32 8)
  (t:token) (e:expert) (other_t:token) (other_e:expert)
  : Lemma
      (requires other_t <> t \/ other_e <> e)
      (ensures (
        let after = emit_route_cell before row t e in
        S.index after.selected (other_t * 8 + other_e) ==
          S.index before.selected (other_t * 8 + other_e) /\
        S.index after.normalized (other_t * 8 + other_e) ==
          S.index before.normalized (other_t * 8 + other_e)))

inline_for_extraction noextract
val route_prefix_readback (before:route_buffers) (rows:row_observations)
  (cells:nat{cells <= 256 * 8}) (t:token) (e:expert)
  : Lemma
      (requires t * 8 + e < cells)
      (ensures (
        let after = emit_route_prefix before rows cells in
        S.index after.selected (t * 8 + e) == emitted_flag (rows t) e /\
        S.index after.normalized (t * 8 + e) == emitted_probability (rows t) e))

inline_for_extraction noextract
val consume_emitted_cell (routes:route_buffers) (row:lseq f32 8)
  (t:token) (e:expert) (before:scan_state t)
  : after:scan_state (t + 1){
      after == capacity_cell_step t e (emitted_flag row e)
        (emitted_probability row e) before}

inline_for_extraction noextract
val consume_completed_route_cell (routes:route_buffers) (rows:row_observations)
  (t:token) (e:expert) (before:scan_state t)
  : after:scan_state (t + 1){
      after == capacity_cell_step t e (emitted_flag (rows t) e)
        (emitted_probability (rows t) e) before}

inline_for_extraction noextract
val finite_above_seed (value:Argmax.finite32)
  : Lemma (F.lt negative_infinity value)

inline_for_extraction noextract
val scan_matches_native_prefix (row:probability_row) (n:pos{n <= 8})
  : Lemma (
      I32.v (scan_prefix row n).chosen == Argmax.best_prefix (native_row row) n /\
      (scan_prefix row n).best ==
        S.index row (Argmax.best_prefix (native_row row) n))

inline_for_extraction noextract
val same_row_selection (row:probability_row)
  : Lemma (
      let result = scan_prefix row 8 in
      I32.v result.chosen ==
        Capacity.reference_selection #8 (fun e -> S.index row e) /\
      result.best == S.index row (I32.v result.chosen) /\
      Argmax.stable_maximum (native_row row) 8 (I32.v result.chosen))

inline_for_extraction noextract
val source_bound_capacity_step (routes:route_buffers) (row:lseq f32 8)
  (t:token) (e:expert) (before:scan_state t)
  : Lemma (
      let after = consume_emitted_cell routes row t e before in
      let selected = e = I32.v (scan_prefix row 8).chosen in
      let keep = selected && I32.v before.count < 32 in
      I32.v after.count == I32.v before.count + (if selected then 1 else 0) /\
      after.buffers.demand == before.buffers.demand /\
      after.buffers.indices ==
        (if keep then S.upd before.buffers.indices
           (e * 32 + I32.v before.count) (I64.int_to_t t)
         else before.buffers.indices) /\
      after.buffers.probabilities ==
        (if keep then S.upd before.buffers.probabilities
           (e * 32 + I32.v before.count) (emitted_probability row e)
         else before.buffers.probabilities) /\
      after.buffers.survival ==
        S.upd before.buffers.survival (t * 8 + e) (if keep then 1l else 0l))

inline_for_extraction noextract
val capacity_step_frame (routes:route_buffers) (row:lseq f32 8)
  (t:token) (e:expert) (before:scan_state t)
  (other:expert) (c:slot) (u:token)
  : Lemma (
      let after = consume_emitted_cell routes row t e before in
      (other <> e \/ c <> I32.v before.count ==>
        S.index after.buffers.indices (other * 32 + c) ==
          S.index before.buffers.indices (other * 32 + c) /\
        S.index after.buffers.probabilities (other * 32 + c) ==
          S.index before.buffers.probabilities (other * 32 + c)) /\
      (u <> t \/ other <> e ==>
        S.index after.buffers.survival (u * 8 + other) ==
          S.index before.buffers.survival (u * 8 + other)))

inline_for_extraction noextract
val source_scheduled_capacity_step (routes:route_buffers) (rows:row_observations)
  (t:token) (e:expert) (before:scan_state t)
  : Lemma (
      let after = consume_completed_route_cell routes rows t e before in
      after == consume_emitted_cell routes (rows t) t e before /\
      I32.v after.count == I32.v before.count +
        (if e = I32.v (scan_prefix (rows t) 8).chosen then 1 else 0))

(* An internal frozen-boundary compatibility lemma, not producer correctness. *)
inline_for_extraction noextract
val shared_capacity_step
  (b:Capacity.selected_boundary Capacity.Smoke)
  (routes:route_buffers) (row:probability_row)
  (t:token) (e:expert) (before:scan_state t)
  : Lemma
      (requires
        (forall (j:expert). S.index row j == b.probability_rows t j) /\
        I32.v before.count == Capacity.assigned_count b t e)
      (ensures (
        let after = consume_emitted_cell routes row t e before in
        I32.v after.count == Capacity.assigned_count b (t + 1) e /\
        I32.v (S.index after.buffers.survival (t * 8 + e)) ==
          Capacity.prefix_survival b (t + 1) t e /\
        (b.selected_expert t == e /\ I32.v before.count < 32 ==>
          I64.v (S.index after.buffers.indices
            (e * 32 + I32.v before.count)) == t /\
          S.index after.buffers.probabilities (e * 32 + I32.v before.count) ==
            emitted_probability row e /\
          A.v_approximates
            (S.index after.buffers.probabilities (e * 32 + I32.v before.count))
            (Capacity.normalized_probability b t))))

inline_for_extraction noextract
val ordinary_input_views (tokens weights:Input.argument bf16)
  : Lemma (
      PA.offset_of (Input.data tokens) ==
        PA.offset_of (Input.backing tokens) + Input.offset tokens /\
      PA.offset_of (Input.data weights) ==
        PA.offset_of (Input.backing weights) + Input.offset weights)

inline_for_extraction noextract
val source_projection_to_softmax
  (x:Layout.token_values) (w:Layout.weight_values)
  (before:Layout.score_values) (tile:natlt 16) (r:natlt 16) (e:expert)
  : Lemma
      (requires Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))
      (ensures (
        let produced = Layout.mma_prefix x w tile 32 in
        let after = Layout.project_tile x w before tile in
        let consumed = S.index (Layout.softmax_score_row after (16 * tile + r)) e in
        consumed == C.acc2 produced r e /\
        B.magnitude 65536 consumed))

inline_for_extraction noextract
val source_projection_frame
  (x:Layout.token_values) (w:Layout.weight_values)
  (before:Layout.score_values) (tile:natlt 16) (t:token) (e:expert)
  : Lemma
      (requires t < 16 * tile \/ 16 * tile + 16 <= t)
      (ensures
        S.index (Layout.project_tile x w before tile) (t * 8 + e) ==
          S.index before (t * 8 + e))

(* The boundary premise is an internal producer obligation, not a caller filter. *)
inline_for_extraction noextract
val source_scheduled_shared_capacity_step
  (b:Capacity.selected_boundary Capacity.Smoke)
  (routes:route_buffers) (rows:token -> probability_row)
  (t:token) (e:expert) (before:scan_state t)
  : Lemma
      (requires
        (forall (j:expert). S.index (rows t) j == b.probability_rows t j) /\
        I32.v before.count == Capacity.assigned_count b t e)
      (ensures (
        let after = consume_completed_route_cell routes rows t e before in
        I32.v after.count == Capacity.assigned_count b (t + 1) e /\
        I32.v (S.index after.buffers.survival (t * 8 + e)) ==
          Capacity.prefix_survival b (t + 1) t e /\
        after.buffers.demand == before.buffers.demand /\
        (b.selected_expert t == e /\ I32.v before.count < 32 ==>
          I64.v (S.index after.buffers.indices
            (e * 32 + I32.v before.count)) == t /\
          S.index after.buffers.probabilities (e * 32 + I32.v before.count) ==
            emitted_probability (rows t) e /\
          A.v_approximates
            (S.index after.buffers.probabilities (e * 32 + I32.v before.count))
            (Capacity.normalized_probability b t))))

inline_for_extraction noextract
val source_return_preserves_output_binding
  (handles:Native.output_handles) (values:Native.output_values)
  (out:Shared.public_outputs) (value:Results.result_vector)
  : Lemma
      (requires Results.matches handles value /\ Native.output_binding handles values out)
      (ensures Native.output_binding (Results.vector_handles value) values out)
