module KPlay.Proof.Optimized.SelectionMath

#lang-pulse

open Kuiper
open KPlay.Proof.Optimized.FamilySelection
module Seq = FStar.Seq
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module Cast = FStar.Int.Cast
module Math = FStar.Math.Lemmas
module Shared = KPlay.Proof.Shared.Selection

#set-options "--fuel 1 --ifuel 1 --z3rlimit 100"

inline_for_extraction noextract
let minus_infinity : f32 = negative_infinity

inline_for_extraction noextract
type index = i:int{-1 <= i /\ i < vocabulary_size}

inline_for_extraction noextract
type region = natlt vocabulary_size -> prop

inline_for_extraction noextract
let index_nat (i : index) : natlt vocabulary_size =
  if i < 0 then 0 else i

inline_for_extraction noextract
let candidate (scores : score_vector) (value : f32) (i : index) : prop =
  if i < 0 then value == minus_infinity
  else value == Seq.index scores (index_nat i)

inline_for_extraction noextract
let best (scores : score_vector) (r : region) (value : f32) (i : index) : prop =
  candidate scores value i /\
  (i >= 0 ==> r (index_nat i)) /\
  (forall (j : natlt vocabulary_size).
    r j ==> i >= 0 /\ priority_le (Seq.index scores j) value)

inline_for_extraction noextract
let union (a b : region) : region = fun j -> a j \/ b j

inline_for_extraction noextract
let empty_best (scores : score_vector) (r : region)
  : Lemma
      (requires forall (j : natlt vocabulary_size). ~(r j))
      (ensures best scores r minus_infinity (-1))
  = ()

inline_for_extraction noextract
let singleton_best (scores : score_vector) (j : natlt vocabulary_size)
  : Lemma
      (ensures
        best scores (fun k -> k == j) (Seq.index scores j) j)
  = priority_reflexive (Seq.index scores j)

inline_for_extraction noextract
let merge_point
  (scores : score_vector) (a b : region)
  (av bv : f32) (ai bi : index)
  (j : natlt vocabulary_size)
  : Lemma
      (requires best scores a av ai /\ best scores b bv bi)
      (ensures
        union a b j ==>
          (if better_candidate bv bi av ai then bi else ai) >= 0 /\
          priority_le (Seq.index scores j)
            (if better_candidate bv bi av ai then bv else av))
  = priority_total av bv;
    if union a b j then (
      if better_candidate bv bi av ai then (
        if a j then priority_transitive (Seq.index scores j) av bv
      ) else (
        if b j then priority_transitive (Seq.index scores j) bv av
      )
    )

inline_for_extraction noextract
let merge_best
  (scores : score_vector) (a b : region)
  (av bv : f32) (ai bi : index)
  : Lemma
      (requires best scores a av ai /\ best scores b bv bi)
      (ensures
        best scores (union a b)
          (if better_candidate bv bi av ai then bv else av)
          (if better_candidate bv bi av ai then bi else ai))
  = FStar.Classical.forall_intro (merge_point scores a b av bv ai bi)

inline_for_extraction noextract
let add_best
  (scores : score_vector) (r : region)
  (value : f32) (i : index) (j : natlt vocabulary_size)
  : Lemma
      (requires best scores r value i)
      (ensures
        best scores (union r (fun k -> k == j))
          (if better_candidate (Seq.index scores j) j value i then Seq.index scores j else value)
          (if better_candidate (Seq.index scores j) j value i then j else i))
  = singleton_best scores j;
    merge_best scores r (fun k -> k == j) value (Seq.index scores j) i j

inline_for_extraction noextract
let best_ext
  (scores : score_vector) (a b : region) (value : f32) (i : index)
  : Lemma
      (requires best scores a value i /\
                (forall (j : natlt vocabulary_size). a j <==> b j))
      (ensures best scores b value i)
  = ()

inline_for_extraction noextract
let best_member
  (scores:score_vector) (r:region) (value:f32) (i:index) (j:natlt vocabulary_size)
  : Lemma
      (requires best scores r value i /\ r j)
      (ensures i >= 0 /\ priority_le (Seq.index scores j) value) =
  ()

inline_for_extraction noextract
let best_selected (scores : score_vector) (value : f32) (token : i64)
  : Lemma
      (requires -1 <= I64.v token /\ I64.v token < vocabulary_size /\
                best scores (fun _ -> True) value (I64.v token))
      (ensures selected_token scores token)
  = best_member scores (fun _ -> True) value (I64.v token) 0;
    let i = index_nat (I64.v token) in
    assert (native_maximum #vocabulary_size scores i);
    logical_scores_represent #vocabulary_size scores;
    maximum_representation #vocabulary_size scores (logical_scores #vocabulary_size scores) i

inline_for_extraction noextract
let half_stride (d : nat) : prop =
  d == 1 \/ d == 2 \/ d == 4 \/ d == 8 \/ d == 16 \/
  d == 32 \/ d == 64 \/ d == 128 \/ d == 256

inline_for_extraction noextract
let lane_region (block : nat) (stride : pos) (lane : nat) : region =
  fun j -> j / 485 == block /\ ((j % 485) % 8) % stride == lane

inline_for_extraction noextract
let scan_region (block warp limit : nat) : region =
  fun j -> j / 485 == block /\ (j % 485) % 8 == warp /\ j < limit

inline_for_extraction noextract
let block_region (stride : pos) (lane : nat) : region =
  fun j -> (j / 485) % stride == lane

inline_for_extraction noextract
let remainder_split (x : nat) (d : pos{half_stride d}) (lane : natlt d)
  : Lemma
      ((x % (2 * d) == lane \/ x % (2 * d) == lane + d) <==>
        x % d == lane)
  = Math.modulo_modulo_lemma x d 2

inline_for_extraction noextract
let lane_merge
  (scores : score_vector) (block : natlt 512)
  (d : pos{d == 1 \/ d == 2 \/ d == 4 \/ d == 8 \/ d == 16})
  (lane : natlt d) (av bv : f32) (ai bi : index)
  : Lemma
      (requires
        best scores (lane_region block (2 * d) lane) av ai /\
        best scores (lane_region block (2 * d) (lane + d)) bv bi)
      (ensures
        best scores (lane_region block d lane)
          (if better_candidate bv bi av ai then bv else av)
          (if better_candidate bv bi av ai then bi else ai))
  = merge_best scores
      (lane_region block (2 * d) lane) (lane_region block (2 * d) (lane + d))
      av bv ai bi;
    let point (j : natlt vocabulary_size)
      : Lemma
          (union (lane_region block (2 * d) lane)
                 (lane_region block (2 * d) (lane + d)) j <==>
           lane_region block d lane j)
      = remainder_split ((j % 485) % 8) d lane
    in
    FStar.Classical.forall_intro point;
    best_ext scores
      (union (lane_region block (2 * d) lane) (lane_region block (2 * d) (lane + d)))
      (lane_region block d lane)
      (if better_candidate bv bi av ai then bv else av)
      (if better_candidate bv bi av ai then bi else ai)

inline_for_extraction noextract
let block_merge
  (scores : score_vector) (d : pos{half_stride d}) (lane : natlt d)
  (av bv : f32) (ai bi : index)
  : Lemma
      (requires
        best scores (block_region (2 * d) lane) av ai /\
        best scores (block_region (2 * d) (lane + d)) bv bi)
      (ensures
        best scores (block_region d lane)
          (if better_candidate bv bi av ai then bv else av)
          (if better_candidate bv bi av ai then bi else ai))
  = merge_best scores (block_region (2 * d) lane) (block_region (2 * d) (lane + d))
      av bv ai bi;
    let point (j : natlt vocabulary_size)
      : Lemma
          (union (block_region (2 * d) lane) (block_region (2 * d) (lane + d)) j <==>
           block_region d lane j)
      = remainder_split (j / 485) d lane
    in
    FStar.Classical.forall_intro point;
    best_ext scores
      (union (block_region (2 * d) lane) (block_region (2 * d) (lane + d)))
      (block_region d lane)
      (if better_candidate bv bi av ai then bv else av)
      (if better_candidate bv bi av ai then bi else ai)

inline_for_extraction noextract
let scan_step
  (scores : score_vector) (block : natlt 512) (warp : natlt 8)
  (row : natlt vocabulary_size{
    block * 485 + warp <= row /\ row < (block + 1) * 485 /\
    (row - block * 485 - warp) % 8 == 0})
  (value : f32) (i : index)
  : Lemma
      (requires best scores (scan_region block warp row) value i)
      (ensures
        best scores (scan_region block warp (row + 8))
          (if better_candidate (Seq.index scores row) row value i then Seq.index scores row else value)
          (if better_candidate (Seq.index scores row) row value i then row else i))
  = assert (row / 485 == block /\ (row % 485) % 8 == warp);
    add_best scores (scan_region block warp row) value i row;
    assert (forall (j : natlt vocabulary_size).
      union (scan_region block warp row) (fun k -> k == row) j <==>
      scan_region block warp (row + 8) j);
    best_ext scores
      (union (scan_region block warp row) (fun k -> k == row))
      (scan_region block warp (row + 8))
      (if better_candidate (Seq.index scores row) row value i then Seq.index scores row else value)
      (if better_candidate (Seq.index scores row) row value i then row else i)
