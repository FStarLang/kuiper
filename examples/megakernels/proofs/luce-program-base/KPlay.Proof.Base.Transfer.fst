module KPlay.Proof.Base.Transfer

open FStar.Real
module S = FStar.Seq
module C = Kuiper.Chest
module EM = Kuiper.EMatrix
module F = Kuiper.Floating.Base
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module MMA = Kuiper.TensorCore.MMA
module Registers = Kuiper.TensorCore.MMA.Registers
module GEMM = Kuiper.Spec.GEMM
module Sum = Kuiper.Sum
module P = KPlay.Proof.Shared.Projection
module D = KPlay.Proof.Shared.Selection
module Algebra = KPlay.Proof.Base.Algebra

inline_for_extraction noextract
let packed = D.vec BF.t 752393024

inline_for_extraction noextract
let scratch = D.vec BF.t 16000

inline_for_extraction noextract
let row_block = row:nat{row % 16 == 0 /\ row + 16 <= 248320}

inline_for_extraction noextract
let depth_block = k:nat{k % 16 == 0 /\ k + 16 <= 1024}

(* kernel.cu: final=true, matrix=0, columns=1024, input_offset=1024.
   B's eight columns denote the SAME source cells, not a staged allocation. *)
inline_for_extraction noextract
let weight_fragment (weights:packed) (row:row_block) (k:depth_block)
  : C.chest2 BF.t 16 16 =
  C.mk2 (fun i j -> S.index weights ((row + i) * 1024 + k + j))

inline_for_extraction noextract
let input_fragment (hidden:scratch) (k:depth_block)
  : C.chest2 BF.t 16 8 =
  C.mk2 (fun i _ -> S.index hidden (1024 + k + i))

inline_for_extraction noextract
let ideal_input_fragment (normalized:P.vector real 1024) (k:depth_block)
  : C.chest2 real 16 8 =
  C.mk2 (fun i _ -> normalized (k + i))

inline_for_extraction noextract
let instruction (weights:packed) (hidden:scratch)
  (row:row_block) (k:depth_block) (accumulator:C.chest2 FP.t 16 8)
  : C.chest2 FP.t 16 8 =
  MMA.emma accumulator (weight_fragment weights row k) (input_fragment hidden k)

inline_for_extraction noextract
let ideal_instruction (weights:packed) (normalized:P.vector real 1024)
  (row:row_block) (k:depth_block) (accumulator:C.chest2 real 16 8)
  : C.chest2 real 16 8 =
  GEMM.matplus accumulator
    (GEMM.matmul (C.to_real_chest (weight_fragment weights row k))
      (ideal_input_fragment normalized k))

(* Keep total0=__fadd_rn(0,d0), including its actual result.
   Registers.add_values is the installed __fadd_rn register adapter. *)
inline_for_extraction noextract
let totals (produced:C.chest2 FP.t 16 8) : C.chest2 FP.t 16 8 =
  Registers.add_values (C.mk2 (fun _ _ -> F.zero #FP.t)) produced

inline_for_extraction noextract
let stored_scores (produced:C.chest2 FP.t 16 8) : GTot (D.vec BF.t 16) =
  S.init_ghost 16 (fun i -> Cast.cast_f32_to_bf16 (C.acc2 (totals produced) i 0))

inline_for_extraction noextract
let consumed_scores (produced:C.chest2 FP.t 16 8) : GTot (D.vec FP.t 16) =
  S.init_ghost 16 (fun i -> Cast.cast_bf16_to_f32 (S.index (stored_scores produced) i))

inline_for_extraction noextract
let normalized_input (hidden:scratch) (normalized:P.vector real 1024) : prop =
  forall (i:P.index 1024).
    Approx.v_approximates (S.index hidden (1024 + i)) (normalized i)

inline_for_extraction noextract
let handoff (produced:C.chest2 FP.t 16 8) (expected:C.chest2 real 16 8) : prop =
  forall (i:P.index 16).
    Approx.v_approximates (S.index (stored_scores produced) i) (C.acc2 expected i 0) /\
    Approx.v_approximates (S.index (consumed_scores produced) i) (C.acc2 expected i 0) /\
    S.index (consumed_scores produced) i ==
      Cast.cast_bf16_to_f32 (S.index (stored_scores produced) i)

inline_for_extraction noextract
let source_register_loads
  (weights:packed) (hidden:scratch) (row:row_block) (k:depth_block)
  (lane:P.index 32) (areg:P.index 4) (breg:P.index 2) (half:P.index 2)
  : Lemma (
      C.acc2 (weight_fragment weights row k)
        (lane / 4 + (areg % 2) * 8)
        ((lane % 4) * 2 + (areg / 2) * 8 + half) ==
      S.index weights
        ((row + lane / 4 + (areg % 2) * 8) * 1024 +
          k + (lane % 4) * 2 + (areg / 2) * 8 + half) /\
      C.acc2 (input_fragment hidden k)
        ((lane % 4) * 2 + breg * 8 + half) (lane / 4) ==
      S.index hidden (1024 + k + (lane % 4) * 2 + breg * 8 + half)) = ()

inline_for_extraction noextract
let source_register_stores (produced:C.chest2 FP.t 16 8) (group:P.index 8)
  : Lemma (
      S.index (stored_scores produced) group ==
        Cast.cast_f32_to_bf16 (F.add (F.zero #FP.t) (C.acc2 produced group 0)) /\
      S.index (stored_scores produced) (group + 8) ==
        Cast.cast_f32_to_bf16
          (F.add (F.zero #FP.t) (C.acc2 produced (group + 8) 0))) = ()

inline_for_extraction noextract
let forward_handoff (produced:C.chest2 FP.t 16 8) (expected:C.chest2 real 16 8)
  : Lemma
      (requires C.chest_approximates produced expected)
      (ensures handoff produced expected) =
  let each (i:P.index 16)
    : Lemma (
        Approx.v_approximates (S.index (stored_scores produced) i) (C.acc2 expected i 0) /\
        Approx.v_approximates (S.index (consumed_scores produced) i) (C.acc2 expected i 0) /\
        S.index (consumed_scores produced) i ==
          Cast.cast_bf16_to_f32 (S.index (stored_scores produced) i)) =
    let x = C.acc2 produced i 0 in
    let r = C.acc2 expected i 0 in
    assert (Approx.v_approximates x r);
    Approx.a_add (F.zero #FP.t) x 0.0R r;
    Cast.cast_f32_to_bf16_ok (F.add (F.zero #FP.t) x) r;
    Cast.cast_bf16_to_f32_ok (S.index (stored_scores produced) i) r
  in
  FStar.Classical.forall_intro each

inline_for_extraction noextract
let instruction_handoff
  (weights:packed) (hidden:scratch) (normalized:P.vector real 1024)
  (row:row_block) (k:depth_block)
  (accumulator:C.chest2 FP.t 16 8) (ideal_accumulator:C.chest2 real 16 8)
  : Lemma
      (requires normalized_input hidden normalized /\
                C.chest_approximates accumulator ideal_accumulator)
      (ensures
        handoff (instruction weights hidden row k accumulator)
          (ideal_instruction weights normalized row k ideal_accumulator)) =
  let a = weight_fragment weights row k in
  let b = input_fragment hidden k in
  let ra = C.to_real_chest a in
  let rb = ideal_input_fragment normalized k in
  C.lemma_to_real_chest_approximates a;
  assert (C.chest_approximates b rb);
  MMA.emma_approx_lemma accumulator a b ideal_accumulator ra rb;
  forward_handoff (instruction weights hidden row k accumulator)
    (ideal_instruction weights normalized row k ideal_accumulator)

inline_for_extraction noextract
let rec block_accumulator (weights:packed) (hidden:scratch)
  (row:row_block) (count:nat{count <= 64})
  : GTot (C.chest2 FP.t 16 8) (decreases count) =
  if count = 0 then C.mk2 (fun _ _ -> F.zero #FP.t)
  else instruction weights hidden row (16 * (count - 1))
    (block_accumulator weights hidden row (count - 1))

inline_for_extraction noextract
let rec block_model (weights:packed) (normalized:P.vector real 1024)
  (row:row_block) (count:nat{count <= 64})
  : GTot (C.chest2 real 16 8) (decreases count) =
  if count = 0 then C.mk2 (fun _ _ -> 0.0R)
  else ideal_instruction weights normalized row (16 * (count - 1))
    (block_model weights normalized row (count - 1))

inline_for_extraction noextract
let rec block_accumulator_approximates
  (weights:packed) (hidden:scratch) (normalized:P.vector real 1024)
  (row:row_block) (count:nat{count <= 64})
  : Lemma
      (requires normalized_input hidden normalized)
      (ensures C.chest_approximates (block_accumulator weights hidden row count)
        (block_model weights normalized row count))
      (decreases count) =
  if count > 0 then (
    block_accumulator_approximates weights hidden normalized row (count - 1);
    let k = 16 * (count - 1) in
    let a = weight_fragment weights row k in
    let b = input_fragment hidden k in
    let ra = C.to_real_chest a in
    let rb = ideal_input_fragment normalized k in
    C.lemma_to_real_chest_approximates a;
    assert (C.chest_approximates b rb);
    MMA.emma_approx_lemma
      (block_accumulator weights hidden row (count - 1)) a b
      (block_model weights normalized row (count - 1)) ra rb
  )

inline_for_extraction noextract
let embedding_model (weights:packed) : P.matrix real 248320 1024 =
  fun token i -> Approx.to_real (S.index weights (token * 1024 + i))

inline_for_extraction noextract
let to_real_cell (a:C.chest2 BF.t 16 16) (i j:P.index 16)
  : Lemma (
      C.acc2 (C.to_real_chest a) i j == Approx.to_real (C.acc2 a i j)) =
  C.acc_pat #2
    #(Kuiper.Shape.ICons 16 (Kuiper.Shape.ICons 16 Kuiper.Shape.INil))
    #real (fun index -> Approx.to_real (C.acc a index)) (i, (j, ()));
  norm_spec [delta_only ["Kuiper.Chest.to_real_chest"; "Kuiper.Chest.chest_map";
    "Kuiper.Chest.acc2"]] (C.acc2 (C.to_real_chest a) i j)

inline_for_extraction noextract
let immutable_weight_cell
  (weights:packed) (row:row_block) (k:depth_block) (i j:P.index 16)
  : Lemma (
      C.acc2 (C.to_real_chest (weight_fragment weights row k)) i j ==
        embedding_model weights (row + i) (k + j)) =
  to_real_cell (weight_fragment weights row k) i j;
  EM.macc_mkM #BF.t #16 #16
    (fun a b -> S.index weights ((row + a) * 1024 + k + b)) i j

#push-options "--fuel 1 --ifuel 1"

inline_for_extraction noextract
let terms (weights:packed) (normalized:P.vector real 1024) (token:D.token 248320)
  : P.vector real 1024 =
  Algebra.products #1024 (embedding_model weights token) normalized

inline_for_extraction noextract
let tile_model_sum
  (weights:packed) (normalized:P.vector real 1024)
  (row:row_block) (k:depth_block) (i:P.index 16) (count:nat{count <= 16})
  : Lemma
      (ensures
        GEMM.__matmul_single (C.to_real_chest (weight_fragment weights row k))
          (ideal_input_fragment normalized k) i 0 count ==
        Sum.sum k (k + count) (terms weights normalized (row + i))) =
  let a = C.to_real_chest (weight_fragment weights row k) in
  let b = ideal_input_fragment normalized k in
  let f : P.vector real 16 = Algebra.dot_terms a b i 0 in
  let g : P.vector real 16 = Algebra.shifted #1024 (terms weights normalized (row + i)) k in
  Algebra.matmul_sum a b i 0 count;
  let each (j:nat{j < count}) : Lemma (f j == g j) =
    immutable_weight_cell weights row k i j;
    EM.macc_mkM #real #16 #8 (fun l _ -> normalized (k + l)) j 0;
    assert (C.acc2 a i j == embedding_model weights (row + i) (k + j));
    norm_spec [delta_only ["KPlay.Proof.Base.Transfer.ideal_input_fragment"]]
      (ideal_input_fragment normalized k);
    assert (C.acc2 b j 0 == normalized (k + j));
    norm_spec [delta_only ["KPlay.Proof.Base.Algebra.dot_terms"]]
      (Algebra.dot_terms a b i 0 j);
    norm_spec [delta_only ["KPlay.Proof.Base.Algebra.shifted"]]
      (Algebra.shifted #1024 (terms weights normalized (row + i)) k j);
    norm_spec [delta_only ["KPlay.Proof.Base.Transfer.terms";
      "KPlay.Proof.Base.Algebra.products"]]
      (terms weights normalized (row + i) (k + j))
  in
  FStar.Classical.forall_intro each;
  Algebra.sum_ext #16 f g count;
  Algebra.shift_sum #1024 (terms weights normalized (row + i)) k count

inline_for_extraction noextract
let rec block_model_sum
  (weights:packed) (normalized:P.vector real 1024)
  (row:row_block) (i:P.index 16) (count:nat{count <= 64})
  : Lemma
      (ensures
        C.acc2 (block_model weights normalized row count) i 0 ==
        Sum.sum 0 (16 * count) (terms weights normalized (row + i)))
      (decreases count) =
  if count > 0 then (
    block_model_sum weights normalized row i (count - 1);
    let k = 16 * (count - 1) in
    tile_model_sum weights normalized row k i 16;
    GEMM.lemma_matmul_index
      (C.to_real_chest (weight_fragment weights row k))
      (ideal_input_fragment normalized k) i 0;
    Sum.sum_split 0 (16 * count) (terms weights normalized (row + i)) k
  )

inline_for_extraction noextract
let block_handoff
  (weights:packed) (hidden:scratch) (normalized:P.vector real 1024)
  (row:row_block)
  : Lemma
      (requires normalized_input hidden normalized)
      (ensures (
        let produced = block_accumulator weights hidden row 64 in
        handoff produced (block_model weights normalized row 64) /\
        (forall (i:P.index 16).
          Approx.v_approximates (S.index (stored_scores produced) i)
            (P.project normalized (embedding_model weights) (row + i)) /\
          Approx.v_approximates (S.index (consumed_scores produced) i)
            (P.project normalized (embedding_model weights) (row + i))))) =
  block_accumulator_approximates weights hidden normalized row 64;
  forward_handoff (block_accumulator weights hidden row 64)
    (block_model weights normalized row 64);
  let each (i:P.index 16)
    : Lemma (
        C.acc2 (block_model weights normalized row 64) i 0 ==
        P.project normalized (embedding_model weights) (row + i)) =
    block_model_sum weights normalized row i 64;
    Algebra.project_products #1024 #248320 normalized (embedding_model weights) (row + i);
    norm_spec [delta_once ["KPlay.Proof.Base.Transfer.terms"]]
      (terms weights normalized (row + i))
  in
  FStar.Classical.forall_intro each

#pop-options
