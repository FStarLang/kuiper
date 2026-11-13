module KPlay.Proof.Base.Vocabulary

module S = FStar.Seq
module C = Kuiper.Chest
module F = Kuiper.Floating
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Transfer = KPlay.Proof.Base.Transfer
module Native = KPlay.Proof.Base.NativeProjection
module Scratch = KPlay.Proof.Base.Scratch
module Order = KPlay.Proof.Base.Order
module Approx = Kuiper.Approximates.Base

inline_for_extraction noextract
let negative_infinity : FP.t = F.neg (F.infinity #FP.t)

inline_for_extraction noextract
let seed : Order.selection = { Order.best = negative_infinity; Order.token = 0 }

inline_for_extraction noextract
let score_values (weights:Transfer.packed) (hidden:Transfer.scratch) : GTot (D.vec FP.t 248320) =
  S.init_ghost 248320 (fun token ->
    S.index (Transfer.consumed_scores
      (Transfer.block_accumulator weights hidden ((token / 16) * 16) 64)) (token % 16))

inline_for_extraction noextract
let rec scan_blocks (weights:Transfer.packed) (hidden:Transfer.scratch)
  (count:nat{count <= 15520}) : GTot Order.selection (decreases count) =
  if count = 0 then seed
  else
    let row = 16 * (count - 1) in
    Order.scan (Transfer.consumed_scores (Transfer.block_accumulator weights hidden row 64))
      row (scan_blocks weights hidden (count - 1)) 16

inline_for_extraction noextract
let score_cell (weights:Transfer.packed) (hidden:Transfer.scratch)
  (row:Transfer.row_block) (i:P.index 16)
  : Lemma (
    S.index (score_values weights hidden) (row + i) ==
    S.index (Transfer.consumed_scores (Transfer.block_accumulator weights hidden row 64)) i) =
  assert ((row + i) / 16 == row / 16);
  assert ((row + i) % 16 == i)

inline_for_extraction noextract
let rec scan_block_prefix (weights:Transfer.packed) (hidden:Transfer.scratch)
  (row:Transfer.row_block) (count:nat{count <= 16})
  : Lemma
      (ensures
        Order.scan (Transfer.consumed_scores (Transfer.block_accumulator weights hidden row 64))
          row (Order.scan (score_values weights hidden) 0 seed row) count ==
        Order.scan (score_values weights hidden) 0 seed (row + count))
      (decreases count) =
  if count > 0 then (
    scan_block_prefix weights hidden row (count - 1);
    score_cell weights hidden row (count - 1)
  )

inline_for_extraction noextract
let rec scan_blocks_agree (weights:Transfer.packed) (hidden:Transfer.scratch)
  (count:nat{count <= 15520})
  : Lemma
      (ensures scan_blocks weights hidden count ==
        Order.scan (score_values weights hidden) 0 seed (16 * count))
      (decreases count) =
  if count > 0 then (
    scan_blocks_agree weights hidden (count - 1);
    scan_block_prefix weights hidden (16 * (count - 1)) 16
  )

let negative_infinity_not_zero ()
  : Lemma (~(F.is_zero negative_infinity) /\ ~(F.is_nan negative_infinity)) =
  F.neg_kind (F.infinity #FP.t);
  if F.is_zero negative_infinity then F.is_zero_spec negative_infinity

let negative_infinity_lower (x:FP.t)
  : Lemma (Order.priority_le negative_infinity x) =
  if not (F.is_nan x) then (
    F.neg_kind (F.infinity #FP.t);
    F.neg_kind x;
    F.neg_kind (F.neg x);
    F.infinity_val_spec (F.neg x);
    F.negate_lt_is_lte (F.infinity #FP.t) (F.neg x);
    F.lt_neg_flip (F.infinity #FP.t) (F.neg x);
    F.neg_neg x;
    Order.ieee_equal_comparisons (F.neg (F.neg x)) x negative_infinity
  )

let scan_bounds_and_provenance (values:D.vec FP.t 248320)
  : Lemma (
    let result = Order.scan values 0 seed 248320 in
    result.token < 248320 /\
    result.best == S.index values result.token /\
    (forall (i:D.token 248320). Order.priority_le (S.index values i) result.best)) =
  Order.scan_dominates values 0 seed 248320;
  if Order.scan values 0 seed 248320 == seed then (
    let first = S.index values 0 in
    negative_infinity_not_zero ();
    negative_infinity_lower first;
    assert (Order.priority_le first negative_infinity);
    assert (~(F.is_nan first));
    Order.native_order first negative_infinity;
    F.eq_spec first negative_infinity;
    assert (first == negative_infinity)
  )

let complete_selection (weights:Transfer.packed) (hidden:Transfer.scratch)
  : Lemma (
    let values = score_values weights hidden in
    let result = scan_blocks weights hidden 15520 in
    result.token < 248320 /\
    result.best == S.index values result.token /\
    (forall (i:D.token 248320).
      (F.is_nan (S.index values i) ==> F.is_nan result.best) /\
      (~(F.is_nan result.best) ==>
        ~(F.is_nan (S.index values i)) /\ F.lte (S.index values i) result.best))) =
  scan_blocks_agree weights hidden 15520;
  scan_bounds_and_provenance (score_values weights hidden);
  Order.scan_special_values (score_values weights hidden) 0 seed

let complete_numerical_scores (weights:Transfer.packed) (hidden:Transfer.scratch)
  (normalized:P.vector FStar.Real.real 1024)
  : Lemma
      (requires Transfer.normalized_input hidden normalized)
      (ensures forall (token:D.token 248320).
        Approx.v_approximates (S.index (score_values weights hidden) token)
          (P.project normalized (Transfer.embedding_model weights) token)) =
  let at_token (token:D.token 248320)
    : Lemma (
      Approx.v_approximates (S.index (score_values weights hidden) token)
        (P.project normalized (Transfer.embedding_model weights) token)) =
    let row = (token / 16) * 16 in
    Transfer.block_handoff weights hidden normalized row;
    score_cell weights hidden row (token % 16)
  in
  FStar.Classical.forall_intro at_token

let same_normalized (left right:Transfer.scratch) : prop =
  forall (i:P.index 1024). S.index left (1024 + i) == S.index right (1024 + i)

let input_fragment_agrees (left right:Transfer.scratch) (k:Transfer.depth_block)
  : Lemma
      (requires same_normalized left right)
      (ensures Transfer.input_fragment left k == Transfer.input_fragment right k) =
  let at_cell (i:P.index 16) (j:P.index 8)
    : Lemma (
      C.acc2 (Transfer.input_fragment left k) i j ==
      C.acc2 (Transfer.input_fragment right k) i j) =
    assert (S.index left (1024 + (k + i)) == S.index right (1024 + (k + i)))
  in
  FStar.Classical.forall_intro_2 at_cell;
  C.ext (Transfer.input_fragment left k) (Transfer.input_fragment right k)

let rec accumulator_agrees (weights:Transfer.packed) (left right:Transfer.scratch)
  (row:Transfer.row_block) (count:nat{count <= 64})
  : Lemma
      (requires same_normalized left right)
      (ensures Transfer.block_accumulator weights left row count ==
        Transfer.block_accumulator weights right row count)
      (decreases count) =
  if count > 0 then (
    input_fragment_agrees left right (16 * (count - 1));
    accumulator_agrees weights left right row (count - 1)
  )

inline_for_extraction noextract
let scratch_after (weights:Transfer.packed) (hidden:C.chest2 BF.t 16000 1)
  (count:nat{count <= 15520}) : GTot (C.chest2 BF.t 16000 1) =
  if count = 0 then hidden
  else Scratch.replace_window hidden 12500 16 (Native.stored_column
    (Transfer.block_accumulator weights (Native.hidden_values hidden) (16 * (count - 1)) 64))

let scratch_normalized (weights:Transfer.packed) (hidden:C.chest2 BF.t 16000 1)
  (count:nat{count <= 15520})
  : Lemma (
    same_normalized (Native.hidden_values (scratch_after weights hidden count))
      (Native.hidden_values hidden)) = ()

let replace_score_twice (hidden:C.chest2 BF.t 16000 1) (left right:C.chest2 BF.t 16 1)
  : Lemma (
    Scratch.replace_window (Scratch.replace_window hidden 12500 16 left) 12500 16 right ==
    Scratch.replace_window hidden 12500 16 right) =
  C.ext (Scratch.replace_window (Scratch.replace_window hidden 12500 16 left) 12500 16 right)
    (Scratch.replace_window hidden 12500 16 right)

let scratch_step (weights:Transfer.packed) (hidden:C.chest2 BF.t 16000 1)
  (count:nat{count < 15520})
  : Lemma (
    let before = scratch_after weights hidden count in
    let produced = Transfer.block_accumulator weights (Native.hidden_values before) (16 * count) 64 in
    produced == Transfer.block_accumulator weights (Native.hidden_values hidden) (16 * count) 64 /\
    Scratch.replace_window before 12500 16 (Native.stored_column produced) ==
      scratch_after weights hidden (count + 1)) =
  scratch_normalized weights hidden count;
  accumulator_agrees weights (Native.hidden_values (scratch_after weights hidden count))
    (Native.hidden_values hidden) (16 * count) 64;
  if count > 0 then
    replace_score_twice hidden
      (Native.stored_column (Transfer.block_accumulator weights (Native.hidden_values hidden)
        (16 * (count - 1)) 64))
      (Native.stored_column (Transfer.block_accumulator weights (Native.hidden_values hidden)
        (16 * count) 64))
