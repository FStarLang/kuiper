module KPlay.Functional

#lang-pulse

open Kuiper

module Shared = KPlay.SharedSpec
module C = Kuiper.Chest
module S = FStar.Seq
module A = Kuiper.Approximates.Base
module FP = Kuiper.Floating
module G = Kuiper.Spec.GEMM
module SM = Kuiper.Spec.Softmax
module R = Kuiper.Real
module I64 = FStar.Int64
module U64 = FStar.UInt64
module KA = Kuiper.Array
module PA = Pulse.Lib.Array

inline_for_extraction noextract
let chest1_extensional (#a:Type0) (#n:nat) (left right:C.chest1 a n)
  : Lemma
      (requires forall (j:natlt n). C.acc1 left j == C.acc1 right j)
      (ensures left == right)
  = let cell (index:Kuiper.Shape.abs (Kuiper.Shape.ICons n Kuiper.Shape.INil))
      : Lemma (C.acc left index == C.acc right index)
      = let (j, ()) = index in ()
    in
    FStar.Classical.forall_intro cell;
    C.lemma_equal_intro left right;
    C.ext left right

inline_for_extraction noextract
let chest2_extensional (#a:Type0) (#m #n:nat) (left right:C.chest2 a m n)
  : Lemma
      (requires forall (i:natlt m) (j:natlt n). C.acc2 left i j == C.acc2 right i j)
      (ensures left == right)
  = let cell (index:Kuiper.Shape.abs
        (Kuiper.Shape.ICons m (Kuiper.Shape.ICons n Kuiper.Shape.INil)))
      : Lemma (C.acc left index == C.acc right index)
      = let (i, (j, ())) = index in
        assert (C.acc2 left i j == C.acc2 right i j);
        assert (C.acc2 left i j == C.acc left index);
        assert (C.acc2 right i j == C.acc right index)
    in
    FStar.Classical.forall_intro cell;
    C.lemma_equal_intro left right;
    C.ext left right

inline_for_extraction noextract
let packed_size (e:pos) : pos = e * 32

inline_for_extraction noextract
let active_size (t n:nat) : nat = if t = 0 then 0 else n

inline_for_extraction noextract
let row_domain (#h #e:pos)
  (x:lseq bf16 h) (w:lseq bf16 (e * h)) : prop =
  (forall (k:natlt h). Shared.finite_bounded_bf16 32L (S.index x k)) /\
  (forall (n:natlt (e * h)). Shared.finite_bounded_bf16 4L (S.index w n))

inline_for_extraction noextract
let token_view (#h:pos) (x:lseq bf16 h) : C.chest2 bf16 1 h =
  C.mk2 (fun _ k -> S.index x k)

inline_for_extraction noextract
let expert_major_view (#h #e:pos) (w:lseq bf16 (e * h))
  : C.chest2 bf16 e h =
  C.mk2 (fun j k -> S.index w (j * h + k))

inline_for_extraction noextract
let weight_view (#h #e:pos) (w:lseq bf16 (e * h))
  : C.chest2 bf16 h e =
  C.mk2 (fun k j -> S.index w (j * h + k))

inline_for_extraction noextract
let weight_transpose (#h #e:pos) (w:lseq bf16 (e * h))
  (k:natlt h) (j:natlt e)
  : Lemma (
      C.acc2 (weight_view w) k j ==
      C.acc2 (expert_major_view w) j k)
  = ()

inline_for_extraction noextract
let projection_prefix (#h #e:pos)
  (x:lseq bf16 h) (w:lseq bf16 (e * h))
  (j:natlt e) (n:nat{n <= h}) : GTot real =
  G.__gmatmul_single 0.0R ( *. ) ( +. )
    (C.to_real_chest (token_view x))
    (C.to_real_chest (weight_view w)) 0 j n

inline_for_extraction noextract
let projection_prefix_zero (#h #e:pos)
  (x:lseq bf16 h) (w:lseq bf16 (e * h)) (j:natlt e)
  : Lemma (projection_prefix x w j 0 == 0.0R)
  = G.__gmatmul_single_zero_lemma 0.0R ( *. ) ( +. )
      (C.to_real_chest (token_view x))
      (C.to_real_chest (weight_view w)) 0 j

inline_for_extraction noextract
let projection_prefix_step (#h #e:pos)
  (x:lseq bf16 h) (w:lseq bf16 (e * h))
  (j:natlt e) (n:natlt h)
  : Lemma (
      projection_prefix x w j (n + 1) ==
      A.to_real (S.index x n) *.
        A.to_real (S.index w (j * h + n)) +.
        projection_prefix x w j n)
  = G.__gmatmul_single_lemma 0.0R ( *. ) ( +. )
      (C.to_real_chest (token_view x))
      (C.to_real_chest (weight_view w)) 0 j (n + 1)

inline_for_extraction noextract
let projection_row (#h #e:pos)
  (x:lseq bf16 h) (w:lseq bf16 (e * h)) : C.chest1 real e =
  C.mk1 (fun j -> projection_prefix x w j h)

inline_for_extraction noextract
let rec maximum_index (#e:pos) (row:lseq f32 e) (n:pos{n <= e})
  : Tot (natlt n) (decreases n) =
  if n = 1 then 0
  else
    let previous = maximum_index row (n - 1) in
    if FP.lt #f32 (S.index row previous) (S.index row (n - 1))
    then n - 1 else previous

inline_for_extraction noextract
let rec sum_prefix (#e:pos) (row:lseq f32 e) (n:nat{n <= e})
  : Tot f32 (decreases n) =
  if n = 0 then FP.zero #f32
  else FP.add #f32 (sum_prefix row (n - 1)) (S.index row (n - 1))

inline_for_extraction noextract
let exponentials (#e:pos) (scores:lseq f32 e) : lseq f32 e =
  let maximum = S.index scores (maximum_index scores e) in
  S.init e (fun j -> FP.fexp #f32 (FP.sub #f32 (S.index scores j) maximum))

inline_for_extraction noextract
let normalized_values (#e:pos) (scores:lseq f32 e) : lseq f32 e =
  let terms = exponentials scores in
  let inverse = FP.div #f32 (FP.one #f32) (sum_prefix terms e) in
  S.init e (fun j -> FP.mul #f32 (S.index terms j) inverse)

inline_for_extraction noextract
let real_prefix (#e:pos) (row:lseq real e) (n:nat{n <= e}) : real =
  R.rsum (S.slice row 0 n)

inline_for_extraction noextract
let real_prefix_step (#e:pos) (row:lseq real e) (n:natlt e)
  : Lemma (
      real_prefix row (n + 1) ==
      real_prefix row n +. S.index row n)
  = assert (S.equal (S.slice row 0 (n + 1))
      (S.append (S.slice row 0 n) (S.create 1 (S.index row n))));
    R.rsum_append (S.slice row 0 n) (S.create 1 (S.index row n))

inline_for_extraction noextract
let rec sum_prefix_models (#e:pos)
  (stored:lseq f32 e) (modeled:lseq real e) (n:nat{n <= e})
  : Lemma
      (requires forall (j:natlt e). S.index stored j %~ S.index modeled j)
      (ensures sum_prefix stored n %~ real_prefix modeled n)
      (decreases n)
  = if n > 0 then (
      sum_prefix_models stored modeled (n - 1);
      real_prefix_step modeled (n - 1);
      A.a_add #f32 (sum_prefix stored (n - 1)) (S.index stored (n - 1))
        (real_prefix modeled (n - 1)) (S.index modeled (n - 1)))

(* The shift is a model of the actual stored maximum, not an assumed real argmax. *)
inline_for_extraction noextract
let normalization_models (#e:pos)
  (scores:lseq f32 e) (modeled:C.chest1 real e)
  : Lemma
      (requires C.seq_to_chest1 scores %~ modeled)
      (ensures C.seq_to_chest1 (normalized_values scores) %~ SM.softmax_real modeled)
  = let maximum = S.index scores (maximum_index scores e) in
    let shift = A.to_real maximum in
    let shifted = C.chest_map (fun z -> z -. shift) modeled in
    let real_terms = C.chest1_to_seq (C.chest_map exp shifted) in
    let terms = exponentials scores in
    A.to_real_ok maximum;
    let term_model (j:natlt e)
      : Lemma (S.index terms j %~ S.index real_terms j)
      = A.sub_approx #f32 (S.index scores j) maximum (C.acc1 modeled j) shift;
        A.exp_approx #f32 (FP.sub #f32 (S.index scores j) maximum)
          (C.acc1 modeled j -. shift)
    in
    FStar.Classical.forall_intro term_model;
    sum_prefix_models terms real_terms e;
    assert (S.equal (S.slice real_terms 0 e) real_terms);
    R.sum_non_zero real_terms 0.0R;
    let denominator = R.rsum real_terms in
    let inverse = FP.div #f32 (FP.one #f32) (sum_prefix terms e) in
    A.div_approx #f32 (FP.one #f32) (sum_prefix terms e) 1.0R denominator;
    SM.softmax_shift modeled shift;
    let output_model (j:natlt e)
      : Lemma (
          S.index (normalized_values scores) j %~
            C.acc1 (SM.softmax_real modeled) j)
      = A.a_mul #f32 (S.index terms j) inverse
          (S.index real_terms j) (1.0R /. denominator);
        assert (S.index (normalized_values scores) j ==
          FP.mul #f32 (S.index terms j) inverse);
        assert (C.acc1 (SM.softmax_real shifted) j ==
          S.index real_terms j /. denominator);
        assert (S.index real_terms j *. (1.0R /. denominator) ==
          S.index real_terms j /. denominator);
        assert (SM.softmax_real shifted == SM.softmax_real modeled)
    in
    FStar.Classical.forall_intro output_model

inline_for_extraction noextract
noeq type input_frame (e:pos) (t:nat) = {
  previous_counts: lseq i64 (active_size t e);
  previous_indices: lseq i64 (active_size t (packed_size e));
  previous_probabilities: lseq f32 (active_size t (packed_size e));
  previous_survival: lseq i32 (t * e)
}

inline_for_extraction noextract
noeq type output_frame (e:pos) (t:nat) = {
  counts: lseq i64 e;
  indices: lseq i64 (packed_size e);
  probabilities: lseq f32 (packed_size e);
  survival: lseq i32 ((t + 1) * e)
}

inline_for_extraction noextract
let next_frame (#e:pos) (#t:nat) (before:input_frame e t)
  : output_frame e t =
  if t = 0 then {
    counts = S.create e 0L;
    indices = S.create (packed_size e) (-1L);
    probabilities = S.create (packed_size e) (FP.zero #f32);
    survival = S.append before.previous_survival (S.create e 0l)
  } else {
    counts = (before.previous_counts <: lseq i64 e);
    indices = (before.previous_indices <: lseq i64 (packed_size e));
    probabilities = (before.previous_probabilities <: lseq f32 (packed_size e));
    survival = S.append before.previous_survival (S.create e 0l)
  }

inline_for_extraction noextract
let frame_transition (#e:pos) (#t:nat)
  (before:input_frame e t) (after:output_frame e t) : prop =
  after.counts == (next_frame before).counts /\
  after.indices == (next_frame before).indices /\
  after.probabilities == (next_frame before).probabilities /\
  after.survival == (next_frame before).survival

(* Each returned real row is an erased relational model of output0. It is not
   to_real(output0), and never supplies rows absent from the observations. *)
inline_for_extraction noextract
let semantic_post (#h #e:pos) (#t:nat)
  (x:lseq bf16 h) (w:lseq bf16 (e * h)) (before:input_frame e t)
  (x_after:lseq bf16 h) (w_after:lseq bf16 (e * h))
  (row:lseq f32 e) (after:output_frame e t) (modeled:C.chest1 real e)
  : prop =
  row_domain x w /\
  x_after == x /\ w_after == w /\
  frame_transition before after /\
  (exists (scores:lseq f32 e).
    C.seq_to_chest1 scores %~ projection_row x w /\
    row == normalized_values scores) /\
  C.seq_to_chest1 row %~ modeled /\
  modeled == SM.softmax_real (projection_row x w)

inline_for_extraction noextract
let completed_row_models (#h #e:pos) (#t:nat)
  (x:lseq bf16 h) (w:lseq bf16 (e * h)) (before:input_frame e t)
  (scores:lseq f32 e)
  : Lemma
      (requires row_domain x w /\
        C.seq_to_chest1 scores %~ projection_row x w)
      (ensures semantic_post x w before x w
        (normalized_values scores) (next_frame before)
        (SM.softmax_real (projection_row x w)))
  = normalization_models scores (projection_row x w);
    assert (exists (s:lseq f32 e).
      C.seq_to_chest1 s %~ projection_row x w /\
      normalized_values scores == normalized_values s)

inline_for_extraction noextract
let selected_boundary (#e:pos) (row:lseq f32 e) (modeled:C.chest1 real e)
  : GTot (natlt e & f32 & real) =
  let j = maximum_index row e in
  (j, S.index row j, C.acc1 modeled j)

inline_for_extraction noextract
let selected_boundary_models (#e:pos)
  (row:lseq f32 e) (modeled:C.chest1 real e)
  : Lemma
      (requires C.seq_to_chest1 row %~ modeled)
      (ensures
        S.index row (maximum_index row e) %~
          C.acc1 modeled (maximum_index row e))
  = ()

inline_for_extraction noextract
noeq type row_observation (h e:pos) (t:nat) = {
  token_before: lseq bf16 h;
  weights_before: lseq bf16 (e * h);
  frame_before: input_frame e t;
  token_after: lseq bf16 h;
  weights_after: lseq bf16 (e * h);
  stored_row: lseq f32 e;
  frame_after: output_frame e t;
  modeled_row: C.chest1 real e
}

inline_for_extraction noextract
let observes_call (#h #e:pos) (#t:nat) (c:row_observation h e t) : prop =
  semantic_post c.token_before c.weights_before c.frame_before
    c.token_after c.weights_after c.stored_row c.frame_after c.modeled_row

inline_for_extraction noextract
let observe_result (#h #e:pos) (#t:nat)
  (x:lseq bf16 h) (w:lseq bf16 (e * h)) (before:input_frame e t)
  (row:lseq f32 e)
  (modeled:C.chest1 real e {
    semantic_post x w before x w row (next_frame before) modeled})
  : c:row_observation h e t {
      observes_call c /\
      c.token_before == x /\ c.weights_before == w /\
      c.frame_before == before /\ c.stored_row == row /\
      c.modeled_row == modeled
    } = {
  token_before = x;
  weights_before = w;
  frame_before = before;
  token_after = x;
  weights_after = w;
  stored_row = row;
  frame_after = next_frame before;
  modeled_row = modeled
}

inline_for_extraction noextract
unfold let row_collection (smoke:bool) : Type0 =
  t:natlt (Shared.token_count smoke) ->
    row_observation (Shared.hidden_count smoke) (Shared.expert_count smoke) t

inline_for_extraction noextract
let collected_tokens (smoke:bool) (calls:row_collection smoke)
  : Shared.token_matrix smoke =
  C.mk2 (fun t k -> S.index (calls t).token_before k)

inline_for_extraction noextract
let collected_tokens_after (smoke:bool) (calls:row_collection smoke)
  : Shared.token_matrix smoke =
  C.mk2 (fun t k -> S.index (calls t).token_after k)

inline_for_extraction noextract
let collected_weights (smoke:bool) (calls:row_collection smoke)
  : Shared.weight_matrix smoke =
  weight_view (calls 0).weights_before

inline_for_extraction noextract
let collected_weights_after (smoke:bool) (calls:row_collection smoke)
  : Shared.weight_matrix smoke =
  weight_view (calls 0).weights_after

inline_for_extraction noextract
let collected_output (smoke:bool) (calls:row_collection smoke)
  : Shared.probability_matrix smoke =
  C.mk2 (fun t j -> C.acc1 (calls t).modeled_row j)

inline_for_extraction noextract
let collected_storage (smoke:bool) (calls:row_collection smoke)
  : C.chest2 f32 (Shared.token_count smoke) (Shared.expert_count smoke) =
  C.mk2 (fun t j -> S.index (calls t).stored_row j)

inline_for_extraction noextract
let valid_collection (smoke:bool) (calls:row_collection smoke) : prop =
  forall (t:natlt (Shared.token_count smoke)).
    observes_call (calls t) /\
    (calls t).weights_before == (calls 0).weights_before

inline_for_extraction noextract
let collected_projection_row (smoke:bool) (calls:row_collection smoke)
  (t:natlt (Shared.token_count smoke))
  : Lemma
      (requires valid_collection smoke calls)
      (ensures
        projection_row (calls t).token_before (calls t).weights_before ==
        C.chest2_row
          (Shared.projection smoke
            (collected_tokens smoke calls) (collected_weights smoke calls)) t)
  = let x = collected_tokens smoke calls in
    let w = collected_weights smoke calls in
    let left = projection_row (calls t).token_before (calls t).weights_before in
    let right = C.chest2_row (Shared.projection smoke x w) t in
    let cell (j:natlt (Shared.expert_count smoke))
      : Lemma (C.acc1 left j == C.acc1 right j)
      = Shared.projection_entry smoke x w t j;
        G.__gmatmul_single_congr 0.0R ( *. ) ( +. )
          (C.to_real_chest (token_view (calls t).token_before))
          (C.to_real_chest (weight_view (calls t).weights_before))
          (C.to_real_chest x) (C.to_real_chest w)
          0 j t j (Shared.hidden_count smoke)
    in
    FStar.Classical.forall_intro cell;
    chest1_extensional left right

inline_for_extraction noextract
let collection_refines_shared (smoke:bool) (calls:row_collection smoke)
  : Lemma
      (requires valid_collection smoke calls)
      (ensures
        Shared.postcondition smoke
          (collected_tokens smoke calls) (collected_weights smoke calls)
          (collected_tokens_after smoke calls) (collected_weights_after smoke calls)
          (collected_output smoke calls) /\
        collected_storage smoke calls %~ collected_output smoke calls)
  = let x = collected_tokens smoke calls in
    let w = collected_weights smoke calls in
    let xa = collected_tokens_after smoke calls in
    let wa = collected_weights_after smoke calls in
    let p = collected_output smoke calls in
    let target = Shared.projection_softmax smoke x w in
    let row (t:natlt (Shared.token_count smoke))
      : Lemma (
          (calls t).token_after == (calls t).token_before /\
          row_domain (calls t).token_before (calls t).weights_before /\
          (forall (j:natlt (Shared.expert_count smoke)).
            C.acc2 p t j == C.acc2 target t j /\
            C.acc2 (collected_storage smoke calls) t j %~ C.acc2 p t j))
      = assert (observes_call (calls t));
        collected_projection_row smoke calls t
    in
    FStar.Classical.forall_intro row;
    chest2_extensional xa x;
    assert ((calls 0).weights_after == (calls 0).weights_before);
    chest2_extensional p target

inline_for_extraction noextract
let pack_token (smoke:bool) (tokens:Shared.token_matrix smoke)
  (t:natlt (Shared.token_count smoke))
  : GTot (lseq bf16 (Shared.hidden_count smoke)) =
  S.init_ghost (Shared.hidden_count smoke) (fun k -> C.acc2 tokens t k)

inline_for_extraction noextract
let profile_flat_index (smoke:bool)
  (n:natlt (Shared.expert_count smoke * Shared.hidden_count smoke))
  : Lemma (
      n % Shared.hidden_count smoke < Shared.hidden_count smoke /\
      n / Shared.hidden_count smoke < Shared.expert_count smoke)
  = if smoke then () else ()

inline_for_extraction noextract
let pack_weights (smoke:bool) (weights:Shared.weight_matrix smoke)
  : GTot (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)) =
  S.init_ghost (Shared.expert_count smoke * Shared.hidden_count smoke)
    (fun n ->
      profile_flat_index smoke n;
      C.acc2 weights (n % Shared.hidden_count smoke) (n / Shared.hidden_count smoke))

inline_for_extraction noextract
let packed_weight_entry (smoke:bool) (weights:Shared.weight_matrix smoke)
  (k:natlt (Shared.hidden_count smoke)) (j:natlt (Shared.expert_count smoke))
  : Lemma (
      S.index (pack_weights smoke weights) (j * Shared.hidden_count smoke + k) ==
      C.acc2 weights k j)
  = if smoke then () else ()

inline_for_extraction noextract
let pack_weights_transposes (smoke:bool) (weights:Shared.weight_matrix smoke)
  : Lemma (weight_view (pack_weights smoke weights) == weights)
  = let entry (k:natlt (Shared.hidden_count smoke))
      (j:natlt (Shared.expert_count smoke))
      : Lemma (C.acc2 (weight_view (pack_weights smoke weights)) k j == C.acc2 weights k j)
      = packed_weight_entry smoke weights k j
    in
    FStar.Classical.forall_intro_2 entry;
    chest2_extensional (weight_view (pack_weights smoke weights)) weights

inline_for_extraction noextract
let every_shared_row_is_admissible (smoke:bool)
  (tokens:Shared.token_matrix smoke) (weights:Shared.weight_matrix smoke)
  (t:natlt (Shared.token_count smoke))
  : Lemma
      (requires Shared.input_domain smoke tokens weights)
      (ensures
        row_domain (pack_token smoke tokens t) (pack_weights smoke weights) /\
        weight_view (pack_weights smoke weights) == weights /\
        (forall (k:natlt (Shared.hidden_count smoke)).
          S.index (pack_token smoke tokens t) k == C.acc2 tokens t k))
  = pack_weights_transposes smoke weights;
    let flat (n:natlt (Shared.expert_count smoke * Shared.hidden_count smoke))
      : Lemma (Shared.finite_bounded_bf16 4L (S.index (pack_weights smoke weights) n))
      = profile_flat_index smoke n
    in
    FStar.Classical.forall_intro flat

inline_for_extraction noextract
let packed_collection_refines_shared (smoke:bool)
  (tokens:Shared.token_matrix smoke) (weights:Shared.weight_matrix smoke)
  (calls:row_collection smoke)
  : Lemma
      (requires
        Shared.input_domain smoke tokens weights /\
        (forall (t:natlt (Shared.token_count smoke)).
          observes_call (calls t) /\
          (calls t).token_before == pack_token smoke tokens t /\
          (calls t).weights_before == pack_weights smoke weights))
      (ensures
        Shared.postcondition smoke tokens weights
          (collected_tokens_after smoke calls) (collected_weights_after smoke calls)
          (collected_output smoke calls) /\
        collected_storage smoke calls %~ collected_output smoke calls)
  = pack_weights_transposes smoke weights;
    chest2_extensional (collected_tokens smoke calls) tokens;
    collection_refines_shared smoke calls

inline_for_extraction noextract
unfold let device_array (a:Type0) (n:nat) : Type0 =
  ar:larray a n { KA.is_global_array ar }

inline_for_extraction noextract
unfold let inactive_or_device_array (a:Type0) (n:nat) : Type0 =
  ar:array a { n > 0 ==> PA.length ar == n /\ KA.is_global_array ar }

(* Null placeholders own no live cells. In particular, their dtype is never read. *)
[@@no_mkeys]
inline_for_extraction noextract
let input_frame_ownership (#e:pos) (#t:nat)
  (counts:inactive_or_device_array i64 (active_size t e))
  (indices:inactive_or_device_array i64 (active_size t (packed_size e)))
  (probabilities:inactive_or_device_array f32 (active_size t (packed_size e)))
  (survival:inactive_or_device_array i32 (t * e))
  (contents:input_frame e t) : slprop =
  if t = 0 then emp
  else
    PA.pts_to counts contents.previous_counts **
    PA.pts_to indices contents.previous_indices **
    PA.pts_to probabilities contents.previous_probabilities **
    PA.pts_to survival contents.previous_survival

fn kplay_run
  (hidden:u64 { 0 < U64.v hidden /\ U64.v hidden < 4294967296 })
  (experts:u64 {
    0 < U64.v experts /\
    U64.v experts * U64.v hidden < 4294967296 /\
    packed_size (U64.v experts) < 4294967296
  })
  (token:i64 {
    0 <= I64.v token /\ (I64.v token + 1) * U64.v experts < 4294967296
  })
  (#x:erased (lseq bf16 (U64.v hidden)))
  (#w:erased (lseq bf16 (U64.v experts * U64.v hidden)))
  (#before:erased (input_frame (U64.v experts) (I64.v token)))
  (input_token:device_array bf16 (U64.v hidden))
  (input_weights:device_array bf16 (U64.v experts * U64.v hidden))
  (input_counts:inactive_or_device_array i64 (active_size (I64.v token) (U64.v experts)))
  (input_indices:inactive_or_device_array i64
    (active_size (I64.v token) (packed_size (U64.v experts))))
  (input_probabilities:inactive_or_device_array f32
    (active_size (I64.v token) (packed_size (U64.v experts))))
  (input_survival:inactive_or_device_array i32 (I64.v token * U64.v experts))
  (output_row:device_array f32 (U64.v experts))
  (output_counts:device_array i64 (U64.v experts))
  (output_indices:device_array i64 (packed_size (U64.v experts)))
  (output_probabilities:device_array f32 (packed_size (U64.v experts)))
  (output_survival:device_array i32 ((I64.v token + 1) * U64.v experts))
  requires
    cpu ** pure (row_domain x w) ** on gpu_loc (
      PA.pts_to input_token x **
      PA.pts_to input_weights w **
      input_frame_ownership input_counts input_indices input_probabilities input_survival before **
      PA.pts_to output_row 'old_row **
      PA.pts_to output_counts 'old_counts **
      PA.pts_to output_indices 'old_indices **
      PA.pts_to output_probabilities 'old_probabilities **
      PA.pts_to output_survival 'old_survival)
  ensures exists* (row:lseq f32 (U64.v experts)).
    cpu ** on gpu_loc (
      PA.pts_to input_token x **
      PA.pts_to input_weights w **
      input_frame_ownership input_counts input_indices input_probabilities input_survival before **
      PA.pts_to output_row row **
      PA.pts_to output_counts (next_frame before).counts **
      PA.pts_to output_indices (next_frame before).indices **
      PA.pts_to output_probabilities (next_frame before).probabilities **
      PA.pts_to output_survival (next_frame before).survival) **
    pure (exists (modeled:C.chest1 real (U64.v experts)).
      semantic_post x w before x w row (next_frame before) modeled)

inline_for_extraction noextract
let profile_dimensions (smoke:bool) (t:natlt (Shared.token_count smoke))
  : Lemma (
      0 < Shared.hidden_count smoke /\ Shared.hidden_count smoke < 4294967296 /\
      0 < Shared.expert_count smoke /\
      Shared.expert_count smoke * Shared.hidden_count smoke < 4294967296 /\
      packed_size (Shared.expert_count smoke) < 4294967296 /\
      (t + 1) * Shared.expert_count smoke < 4294967296)
  = if smoke then () else ()

inline_for_extraction noextract
fn run_profile_row
  (smoke:bool)
  (token:i64 { 0 <= I64.v token /\ I64.v token < Shared.token_count smoke })
  (#x:erased (lseq bf16 (Shared.hidden_count smoke)))
  (#w:erased (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)))
  (#before:erased (input_frame (Shared.expert_count smoke) (I64.v token)))
  (input_token:device_array bf16 (Shared.hidden_count smoke))
  (input_weights:device_array bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (input_counts:inactive_or_device_array i64
    (active_size (I64.v token) (Shared.expert_count smoke)))
  (input_indices:inactive_or_device_array i64
    (active_size (I64.v token) (packed_size (Shared.expert_count smoke))))
  (input_probabilities:inactive_or_device_array f32
    (active_size (I64.v token) (packed_size (Shared.expert_count smoke))))
  (input_survival:inactive_or_device_array i32 (I64.v token * Shared.expert_count smoke))
  (output_row:device_array f32 (Shared.expert_count smoke))
  (output_counts:device_array i64 (Shared.expert_count smoke))
  (output_indices:device_array i64 (packed_size (Shared.expert_count smoke)))
  (output_probabilities:device_array f32 (packed_size (Shared.expert_count smoke)))
  (output_survival:device_array i32 ((I64.v token + 1) * Shared.expert_count smoke))
  requires
    cpu ** pure (row_domain x w) ** on gpu_loc (
      PA.pts_to input_token x **
      PA.pts_to input_weights w **
      input_frame_ownership input_counts input_indices input_probabilities input_survival before **
      PA.pts_to output_row 'old_row **
      PA.pts_to output_counts 'old_counts **
      PA.pts_to output_indices 'old_indices **
      PA.pts_to output_probabilities 'old_probabilities **
      PA.pts_to output_survival 'old_survival)
  ensures exists* (row:lseq f32 (Shared.expert_count smoke)).
    cpu ** on gpu_loc (
      PA.pts_to input_token x **
      PA.pts_to input_weights w **
      input_frame_ownership input_counts input_indices input_probabilities input_survival before **
      PA.pts_to output_row row **
      PA.pts_to output_counts (next_frame before).counts **
      PA.pts_to output_indices (next_frame before).indices **
      PA.pts_to output_probabilities (next_frame before).probabilities **
      PA.pts_to output_survival (next_frame before).survival) **
    pure (exists (modeled:C.chest1 real (Shared.expert_count smoke)).
      semantic_post x w before x w row (next_frame before) modeled)
{
  profile_dimensions smoke (I64.v token);
  let hidden = (if smoke then 512UL else 5120UL);
  let experts = (if smoke then 8UL else 16UL);
  kplay_run hidden experts token #x #w #before
    input_token input_weights input_counts input_indices input_probabilities input_survival
    output_row output_counts output_indices output_probabilities output_survival;
}
