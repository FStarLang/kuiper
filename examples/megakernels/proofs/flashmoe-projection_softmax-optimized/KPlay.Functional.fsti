module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Shape

module Shared = KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module U64 = FStar.UInt64
module GEMM = Kuiper.Spec.GEMM
module Softmax = Kuiper.Spec.Softmax

inline_for_extraction noextract
let admissible_shape (tokens hidden experts:nat) : prop =
  0 < tokens /\ tokens % 16 == 0 /\
  0 < hidden /\ hidden % 64 == 0 /\
  0 < experts /\ experts % 8 == 0 /\
  tokens * hidden <= 2147483647 /\
  experts * hidden <= 2147483647 /\
  tokens * experts <= 2147483647

inline_for_extraction noextract
let row_major_view (#et:Type0) (rows columns:nat)
  (values:lseq et (rows * columns)) : GTot (chest2 et rows columns) =
  mk2 (fun row column -> S.index values (row * columns + column))

inline_for_extraction noextract
let weight_view (hidden experts:nat)
  (values:lseq bf16 (experts * hidden)) : GTot (chest2 bf16 hidden experts) =
  mk2 (fun k j -> S.index values (j * hidden + k))

inline_for_extraction noextract
let input_domain (tokens hidden experts:nat)
  (x:lseq bf16 (tokens * hidden)) (w:lseq bf16 (experts * hidden)) : prop =
  chest_forall (Shared.finite_bounded_bf16 32L) (row_major_view tokens hidden x) /\
  chest_forall (Shared.finite_bounded_bf16 4L) (weight_view hidden experts w)

inline_for_extraction noextract
let projection_real (#tokens #hidden #experts:nat)
  (x:chest2 real tokens hidden) (w:chest2 real experts hidden)
  : chest2 real tokens experts =
  GEMM.matmul x (mk2 (fun k j -> acc2 w j k))

inline_for_extraction noextract
let projection_softmax_real (#tokens #hidden:nat) (#experts:pos)
  (x:chest2 real tokens hidden) (w:chest2 real experts hidden)
  : chest2 real tokens experts =
  let scores = projection_real x w in
  mk2 (fun t j -> acc1 (Softmax.softmax_real (chest2_row scores t)) j)

inline_for_extraction noextract
let input_models_exist (tokens hidden experts:nat)
  (x:lseq bf16 (tokens * hidden)) (w:lseq bf16 (experts * hidden))
  : Lemma (
      row_major_view tokens hidden x %~ to_real_chest (row_major_view tokens hidden x) /\
      row_major_view experts hidden w %~ to_real_chest (row_major_view experts hidden w))
  = lemma_to_real_chest_approximates (row_major_view tokens hidden x);
    lemma_to_real_chest_approximates (row_major_view experts hidden w)

inline_for_extraction noextract
let projection_softmax_post (tokens hidden:nat) (experts:pos)
  (x:lseq bf16 (tokens * hidden)) (w:lseq bf16 (experts * hidden))
  (normalized:lseq f32 (tokens * experts)) : prop =
  forall (rx:chest2 real tokens hidden) (rw:chest2 real experts hidden).
    row_major_view tokens hidden x %~ rx /\
    row_major_view experts hidden w %~ rw ==>
    row_major_view tokens experts normalized %~ projection_softmax_real rx rw

(* The modeled output is related to every returned cell, not to_real of it. *)
inline_for_extraction noextract
let semantic_post (tokens hidden:nat) (experts:pos)
  (x x_after:lseq bf16 (tokens * hidden))
  (w w_after:lseq bf16 (experts * hidden))
  (normalized:lseq f32 (tokens * experts))
  (modeled:chest2 real tokens experts) : prop =
  input_domain tokens hidden experts x w /\
  x_after == x /\ w_after == w /\
  projection_softmax_post tokens hidden experts x w normalized /\
  row_major_view tokens experts normalized %~ modeled /\
  modeled == projection_softmax_real
    (to_real_chest (row_major_view tokens hidden x))
    (to_real_chest (row_major_view experts hidden w))

inline_for_extraction noextract
let completed_post (tokens hidden:nat) (experts:pos)
  (x:lseq bf16 (tokens * hidden)) (w:lseq bf16 (experts * hidden))
  (normalized:lseq f32 (tokens * experts))
  : Lemma
      (requires input_domain tokens hidden experts x w /\
        projection_softmax_post tokens hidden experts x w normalized)
      (ensures semantic_post tokens hidden experts x x w w normalized
        (projection_softmax_real
          (to_real_chest (row_major_view tokens hidden x))
          (to_real_chest (row_major_view experts hidden w))))
  = input_models_exist tokens hidden experts x w

inline_for_extraction noextract
let chest2_extensional (#a:Type0) (#m #n:nat) (left right:chest2 a m n)
  : Lemma
      (requires forall (i:natlt m) (j:natlt n). acc2 left i j == acc2 right i j)
      (ensures left == right)
  = let cell (index:Kuiper.Shape.abs (m @| n @| INil))
      : Lemma (acc left index == acc right index)
      = let (i, (j, ())) = index in
        assert (acc2 left i j == acc2 right i j);
        assert (acc2 left i j == acc left index);
        assert (acc2 right i j == acc right index)
    in
    FStar.Classical.forall_intro cell;
    lemma_equal_intro left right;
    ext left right

inline_for_extraction noextract
let weight_model_transpose (hidden experts:nat)
  (w:lseq bf16 (experts * hidden))
  : Lemma (
      to_real_chest (weight_view hidden experts w) ==
      mk2 (fun k j -> acc2 (to_real_chest (row_major_view experts hidden w)) j k))
  = chest2_extensional (to_real_chest (weight_view hidden experts w))
      (mk2 (fun k j -> acc2 (to_real_chest (row_major_view experts hidden w)) j k))

#push-options "--fuel 4 --z3rlimit 20"
inline_for_extraction noextract
let refines_shared (smoke:bool)
  (x x_after:lseq bf16 (Shared.token_count smoke * Shared.hidden_count smoke))
  (w w_after:lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (normalized:lseq f32 (Shared.token_count smoke * Shared.expert_count smoke))
  (modeled:Shared.probability_matrix smoke)
  : Lemma
      (requires semantic_post
        (Shared.token_count smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke)
        x x_after w w_after normalized modeled)
      (ensures
        Shared.postcondition smoke
          (row_major_view (Shared.token_count smoke) (Shared.hidden_count smoke) x)
          (weight_view (Shared.hidden_count smoke) (Shared.expert_count smoke) w)
          (row_major_view (Shared.token_count smoke) (Shared.hidden_count smoke) x_after)
          (weight_view (Shared.hidden_count smoke) (Shared.expert_count smoke) w_after)
          modeled /\
        row_major_view (Shared.token_count smoke) (Shared.expert_count smoke) normalized
          %~ modeled)
  = let tokens = row_major_view (Shared.token_count smoke) (Shared.hidden_count smoke) x in
    let weights = weight_view (Shared.hidden_count smoke) (Shared.expert_count smoke) w in
    let rx = to_real_chest tokens in
    let rw = to_real_chest (row_major_view (Shared.expert_count smoke)
      (Shared.hidden_count smoke) w) in
    weight_model_transpose (Shared.hidden_count smoke) (Shared.expert_count smoke) w;
    let transposed = mk2 (fun k j -> acc2 rw j k) in
    chest2_extensional transposed (to_real_chest weights);
    assert_norm (projection_real rx rw == GEMM.matmul rx (mk2 (fun k j -> acc2 rw j k)));
    assert_norm (Shared.projection smoke tokens weights ==
      GEMM.matmul (to_real_chest tokens) (to_real_chest weights));
    assert (projection_real rx rw == Shared.projection smoke tokens weights);
    chest2_extensional (projection_softmax_real rx rw)
      (Shared.projection_softmax smoke tokens weights);
    assert (modeled == Shared.projection_softmax smoke tokens weights);
    assert (Shared.input_domain smoke tokens weights)
#pop-options

inline_for_extraction noextract
let profile_dimensions (smoke:bool)
  : Lemma (admissible_shape
      (Shared.token_count smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke))
  = if smoke then () else ()

inline_for_extraction noextract
let token_flat_index (smoke:bool)
  (n:natlt (Shared.token_count smoke * Shared.hidden_count smoke))
  : Lemma (
      n / Shared.hidden_count smoke < Shared.token_count smoke /\
      n % Shared.hidden_count smoke < Shared.hidden_count smoke)
  = if smoke then () else ()

inline_for_extraction noextract
let weight_flat_index (smoke:bool)
  (n:natlt (Shared.expert_count smoke * Shared.hidden_count smoke))
  : Lemma (
      n / Shared.hidden_count smoke < Shared.expert_count smoke /\
      n % Shared.hidden_count smoke < Shared.hidden_count smoke)
  = if smoke then () else ()

inline_for_extraction noextract
let pack_tokens (smoke:bool) (tokens:Shared.token_matrix smoke)
  : GTot (lseq bf16 (Shared.token_count smoke * Shared.hidden_count smoke)) =
  S.init_ghost (Shared.token_count smoke * Shared.hidden_count smoke) (fun n ->
    token_flat_index smoke n;
    acc2 tokens (n / Shared.hidden_count smoke) (n % Shared.hidden_count smoke))

inline_for_extraction noextract
let pack_weights (smoke:bool) (weights:Shared.weight_matrix smoke)
  : GTot (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)) =
  S.init_ghost (Shared.expert_count smoke * Shared.hidden_count smoke) (fun n ->
    weight_flat_index smoke n;
    acc2 weights (n % Shared.hidden_count smoke) (n / Shared.hidden_count smoke))

inline_for_extraction noextract
let packed_token_entry (smoke:bool) (tokens:Shared.token_matrix smoke)
  (t:natlt (Shared.token_count smoke)) (k:natlt (Shared.hidden_count smoke))
  : Lemma (
      S.index (pack_tokens smoke tokens) (t * Shared.hidden_count smoke + k) ==
      acc2 tokens t k)
  = if smoke then () else ()

inline_for_extraction noextract
let packed_weight_entry (smoke:bool) (weights:Shared.weight_matrix smoke)
  (k:natlt (Shared.hidden_count smoke)) (j:natlt (Shared.expert_count smoke))
  : Lemma (
      S.index (pack_weights smoke weights) (j * Shared.hidden_count smoke + k) ==
      acc2 weights k j)
  = if smoke then () else ()

inline_for_extraction noextract
let packed_inputs_roundtrip (smoke:bool)
  (tokens:Shared.token_matrix smoke) (weights:Shared.weight_matrix smoke)
  : Lemma (
      row_major_view (Shared.token_count smoke) (Shared.hidden_count smoke)
        (pack_tokens smoke tokens) == tokens /\
      weight_view (Shared.hidden_count smoke) (Shared.expert_count smoke)
        (pack_weights smoke weights) == weights)
  = FStar.Classical.forall_intro_2 (packed_token_entry smoke tokens);
    FStar.Classical.forall_intro_2 (packed_weight_entry smoke weights);
    chest2_extensional
      (row_major_view (Shared.token_count smoke) (Shared.hidden_count smoke)
        (pack_tokens smoke tokens)) tokens;
    chest2_extensional
      (weight_view (Shared.hidden_count smoke) (Shared.expert_count smoke)
        (pack_weights smoke weights)) weights

inline_for_extraction noextract
let all_shared_inputs_admissible (smoke:bool)
  (tokens:Shared.token_matrix smoke) (weights:Shared.weight_matrix smoke)
  : Lemma
      (requires Shared.input_domain smoke tokens weights)
      (ensures
        admissible_shape
          (Shared.token_count smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) /\
        input_domain
          (Shared.token_count smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke)
          (pack_tokens smoke tokens) (pack_weights smoke weights))
  = profile_dimensions smoke;
    packed_inputs_roundtrip smoke tokens weights

fn kplay_run
  (token_count:u64{0 < U64.v token_count /\ U64.v token_count <= 2147483647})
  (hidden_count:u64{0 < U64.v hidden_count /\ U64.v hidden_count <= 2147483647})
  (expert_count:u64{admissible_shape
    (U64.v token_count) (U64.v hidden_count) (U64.v expert_count)})
  (tokens:larray bf16 (U64.v token_count * U64.v hidden_count){is_global_array tokens})
  (weights:larray bf16 (U64.v expert_count * U64.v hidden_count){is_global_array weights})
  (normalized_rows:larray f32 (U64.v token_count * U64.v expert_count){
    is_global_array normalized_rows})
  (#token_perm #weight_perm:perm)
  (#x:erased (lseq bf16 (U64.v token_count * U64.v hidden_count)))
  (#w:erased (lseq bf16 (U64.v expert_count * U64.v hidden_count)))
  preserves cpu ** on gpu_loc (
    tokens |-> Frac token_perm (FStar.Ghost.reveal x) **
    weights |-> Frac weight_perm (FStar.Ghost.reveal w))
  requires
    pure (input_domain (U64.v token_count) (U64.v hidden_count) (U64.v expert_count) x w) **
    (exists* (before:lseq f32 (U64.v token_count * U64.v expert_count)).
      on gpu_loc (normalized_rows |-> before))
  returns modeled:erased (chest2 real (U64.v token_count) (U64.v expert_count))
  ensures exists* (normalized:lseq f32 (U64.v token_count * U64.v expert_count)).
    on gpu_loc (normalized_rows |-> normalized) **
    pure (semantic_post (U64.v token_count) (U64.v hidden_count) (U64.v expert_count)
      x x w w normalized (FStar.Ghost.reveal modeled))

(* One invocation produces the entire matrix for either source-declared profile. *)
inline_for_extraction noextract
fn run_profile
  (smoke:bool)
  (#logical_tokens:erased (Shared.token_matrix smoke))
  (#logical_weights:erased (Shared.weight_matrix smoke))
  (tokens:larray bf16 (Shared.token_count smoke * Shared.hidden_count smoke){
    is_global_array tokens})
  (weights:larray bf16 (Shared.expert_count smoke * Shared.hidden_count smoke){
    is_global_array weights})
  (normalized_rows:larray f32 (Shared.token_count smoke * Shared.expert_count smoke){
    is_global_array normalized_rows})
  (#token_perm #weight_perm:perm)
  preserves cpu ** on gpu_loc (
    tokens |-> Frac token_perm (pack_tokens smoke logical_tokens) **
    weights |-> Frac weight_perm (pack_weights smoke logical_weights))
  requires
    pure (Shared.input_domain smoke logical_tokens logical_weights) **
    (exists* (before:lseq f32 (Shared.token_count smoke * Shared.expert_count smoke)).
      on gpu_loc (normalized_rows |-> before))
  returns modeled:erased (Shared.probability_matrix smoke)
  ensures exists* (normalized:lseq f32 (Shared.token_count smoke * Shared.expert_count smoke)).
    on gpu_loc (normalized_rows |-> normalized) **
    pure (Shared.postcondition smoke logical_tokens logical_weights
      logical_tokens logical_weights (FStar.Ghost.reveal modeled) /\
      row_major_view (Shared.token_count smoke) (Shared.expert_count smoke) normalized
        %~ FStar.Ghost.reveal modeled)
{
  all_shared_inputs_admissible smoke logical_tokens logical_weights;
  packed_inputs_roundtrip smoke logical_tokens logical_weights;
  let m = (if smoke then 256UL else 1024UL);
  let h = (if smoke then 512UL else 5120UL);
  let e = (if smoke then 8UL else 16UL);
  let modeled = kplay_run m h e tokens weights normalized_rows #token_perm #weight_perm
    #(FStar.Ghost.hide (pack_tokens smoke logical_tokens))
    #(FStar.Ghost.hide (pack_weights smoke logical_weights));
  with normalized. assert on gpu_loc (normalized_rows |-> normalized);
  refines_shared smoke
    (pack_tokens smoke logical_tokens) (pack_tokens smoke logical_tokens)
    (pack_weights smoke logical_weights) (pack_weights smoke logical_weights)
    normalized (FStar.Ghost.reveal modeled);
  modeled
}
