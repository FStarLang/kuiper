module KPlay.Proof.SourceLayouts

open Kuiper

module S = FStar.Seq
module C = Kuiper.Chest
module MMA = Kuiper.TensorCore.MMA
module Input = Kuiper.Tensor.Input
module Native = KPlay.Proof.Shared.Native
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module F = Kuiper.Floating
module B = Kuiper.Floating.Bounds
module Bounds = Kuiper.Float32.Bounds
module MB = Kuiper.TensorCore.MMA.Bounds
module Shape = Kuiper.Shape

noextract type token_values = lseq bf16 (256 * 512)
noextract type weight_values = lseq bf16 (8 * 512)
noextract type score_values = lseq f32 (256 * 8)

inline_for_extraction noextract
let logical_tokens (values:token_values) =
  C.mk2 #bf16 #256 #512 (fun t h -> S.index values (t * 512 + h))

inline_for_extraction noextract
let logical_weights (values:weight_values) =
  C.mk2 #bf16 #512 #8 (fun h e -> S.index values (e * 512 + h))

inline_for_extraction noextract
let input_binding (tokens weights:Input.argument bf16)
  (x:token_values) (w:weight_values)
  : Lemma
      (requires Input.dimensions tokens == [256; 512] /\
                Input.dimensions weights == [8; 512])
      (ensures Native.input_binding tokens weights x w
        (logical_tokens x) (logical_weights w))
  = ()

inline_for_extraction noextract
let operand_a (x:token_values) (tile:natlt 16) (step:natlt 32)
  : GTot (C.chest2 bf16 16 16) =
  C.mk2 (fun r k -> S.index x ((16 * tile + r) * 512 + 16 * step + k))

inline_for_extraction noextract
let operand_b (w:weight_values) (step:natlt 32)
  : GTot (C.chest2 bf16 16 8) =
  C.mk2 (fun k e -> S.index w (e * 512 + 16 * step + k))

(* All four A packs and both B packs in optimized_s.cu:27-38. *)
inline_for_extraction noextract
let packed_operand_cells (x:token_values) (w:weight_values)
  (tile:natlt 16) (hidden_tile:natlt 8) (hidden_step:natlt 4)
  (lane:natlt 32) (row_half column_half packed_half:natlt 2)
  : Lemma (
      let step = 4 * hidden_tile + hidden_step in
      let group = lane / 4 in
      let pair = (lane % 4) * 2 in
      C.acc2 (operand_a x tile step) (group + 8 * row_half)
        (pair + 8 * column_half + packed_half) ==
        S.index x
          ((16 * tile + group) * 512 +
           64 * hidden_tile + 16 * hidden_step + pair +
           8 * 512 * row_half + 8 * column_half + packed_half) /\
      C.acc2 (operand_b w step)
        (pair + 8 * column_half + packed_half) group ==
        S.index w
          (group * 512 + 64 * hidden_tile + 16 * hidden_step +
           pair + 8 * column_half + packed_half))
  = ()

inline_for_extraction noextract
let mma_step (x:token_values) (w:weight_values)
  (tile:natlt 16) (step:natlt 32) (acc:C.chest2 f32 16 8)
  : GTot (C.chest2 f32 16 8) =
  MMA.emma acc (operand_a x tile step) (operand_b w step)

inline_for_extraction noextract
let register_row (lane:natlt 32) (reg:natlt 4) : natlt 16 =
  lane / 4 + 8 * (reg / 2)

inline_for_extraction noextract
let register_column (lane:natlt 32) (reg:natlt 4) : natlt 8 =
  (lane % 4) * 2 + reg % 2

inline_for_extraction noextract
let register_store_offset (tile:natlt 16) (lane:natlt 32) (reg:natlt 4)
  : Lemma (
      (16 * tile + register_row lane reg) * 8 + register_column lane reg ==
      (16 * tile + lane / 4) * 8 + (lane % 4) * 2 +
        (reg / 2) * 64 + reg % 2)
  = ()

inline_for_extraction noextract
let register_cover (r:natlt 16) (c:natlt 8)
  : Lemma (
      register_row ((r % 8) * 4 + c / 2) ((r / 8) * 2 + c % 2) == r /\
      register_column ((r % 8) * 4 + c / 2) ((r / 8) * 2 + c % 2) == c)
  = ()

inline_for_extraction noextract
let register_unique (lane other_lane:natlt 32) (reg other_reg:natlt 4)
  : Lemma
      (requires
        register_row lane reg == register_row other_lane other_reg /\
        register_column lane reg == register_column other_lane other_reg)
      (ensures lane == other_lane /\ reg == other_reg)
  = ()

(* Observation of the disjoint four-register stores, not an extra kernel. *)
inline_for_extraction noextract
let store_score_tile (before:score_values) (tile:natlt 16)
  (produced:C.chest2 f32 16 8) : GTot score_values =
  S.init_ghost (256 * 8) (fun n ->
    if 16 * tile <= n / 8 && n / 8 < 16 * tile + 16
    then C.acc2 produced (n / 8 - 16 * tile) (n % 8)
    else S.index before n)

inline_for_extraction noextract
let softmax_score_row (scores:score_values) (token:natlt 256) : lseq f32 8 =
  S.slice scores (token * 8) (token * 8 + 8)

inline_for_extraction noextract
let mma_to_softmax_same_cell
  (x:token_values) (w:weight_values) (before:score_values)
  (tile:natlt 16) (step:natlt 32) (acc:C.chest2 f32 16 8)
  (r:natlt 16) (c:natlt 8)
  : Lemma (
      let produced = mma_step x w tile step acc in
      let stored = store_score_tile before tile produced in
      S.index (softmax_score_row stored (16 * tile + r)) c ==
        C.acc2 produced r c)
  =
  let produced = mma_step x w tile step acc in
  let stored = store_score_tile before tile produced in
  let token = 16 * tile + r in
  S.lemma_index_slice stored (token * 8) (token * 8 + 8) c

inline_for_extraction noextract
let score_store_frame (before:score_values) (tile:natlt 16)
  (produced:C.chest2 f32 16 8) (token:natlt 256) (expert:natlt 8)
  : Lemma
      (requires token < 16 * tile \/ 16 * tile + 16 <= token)
      (ensures
        S.index (store_score_tile before tile produced) (token * 8 + expert) ==
        S.index before (token * 8 + expert))
  = ()

(* The source stores only after all 8 * 4 MMA steps, starting from +0. *)
inline_for_extraction noextract
let rec mma_prefix (x:token_values) (w:weight_values) (tile:natlt 16)
  (steps:nat{steps <= 32})
  : GTot (C.chest2 f32 16 8) (decreases steps) =
  if steps = 0 then C.mk2 (fun _ _ -> F.zero)
  else mma_step x w tile (steps - 1) (mma_prefix x w tile (steps - 1))

inline_for_extraction noextract
let source_operand_bounds (x:token_values) (w:weight_values)
  (tile:natlt 16) (step:natlt 32)
  : Lemma
      (requires Projection.input_domain true (logical_tokens x) (logical_weights w))
      (ensures
        C.chest_forall (B.magnitude 32) (operand_a x tile step) /\
        C.chest_forall (B.magnitude 4) (operand_b w step))
  =
  FStar.Int64.v_inj (FStar.Int64.int_to_t 32) 32L;
  FStar.Int64.v_inj (FStar.Int64.int_to_t 4) 4L;
  let token_cell (index:Shape.abs (Shape.ICons 16 (Shape.ICons 16 Shape.INil)))
    : Lemma (B.magnitude 32 (C.acc (operand_a x tile step) index))
    =
    let (r, (k, ())) = index in
    assert (Projection.finite_bounded_bf16 32L
      (C.acc2 (logical_tokens x) (16 * tile + r) (16 * step + k)))
  in
  FStar.Classical.forall_intro token_cell;
  let weight_cell (index:Shape.abs (Shape.ICons 16 (Shape.ICons 8 Shape.INil)))
    : Lemma (B.magnitude 4 (C.acc (operand_b w step) index))
    =
    let (k, (e, ())) = index in
    assert (Projection.finite_bounded_bf16 4L
      (C.acc2 (logical_weights w) (16 * step + k) e))
  in
  FStar.Classical.forall_intro weight_cell

inline_for_extraction noextract
let rec mma_prefix_bounds (x:token_values) (w:weight_values) (tile:natlt 16)
  (steps:nat{steps <= 32})
  : Lemma
      (requires Projection.input_domain true (logical_tokens x) (logical_weights w))
      (ensures C.chest_forall (B.magnitude (2048 * steps))
        (mma_prefix x w tile steps))
      (decreases steps)
  =
  if steps = 0 then Bounds.zero ()
  else (
    mma_prefix_bounds x w tile (steps - 1);
    source_operand_bounds x w tile (steps - 1);
    MB.bounded_emma (2048 * (steps - 1)) (mma_prefix x w tile (steps - 1))
      (operand_a x tile (steps - 1)) (operand_b w (steps - 1))
  )

inline_for_extraction noextract
let project_tile (x:token_values) (w:weight_values)
  (before:score_values) (tile:natlt 16) : GTot score_values =
  store_score_tile before tile (mma_prefix x w tile 32)
