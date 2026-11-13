module KPlay.Proof.BaseProjection

#lang-pulse

open Kuiper

module Shared = KPlay.Proof.Shared.ProjectionSoftmax
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

