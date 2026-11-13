module KPlay.Proof.Optimized.ScoreMath

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module C = FStar.Classical
module I32 = FStar.Int32
module SM = KPlay.Proof.Optimized.SelectionMath
module Sel = KPlay.Proof.Optimized.Selector
module FS = KPlay.Proof.Optimized.FamilySelection
module Dot = KPlay.Proof.Optimized.DotMath
module Memory = KPlay.Proof.Optimized.Memory

inline_for_extraction noextract
let certified (scores:FS.score_vector) (region:SM.region) (c:Sel.candidate) : prop =
  if -1 <= I32.v c.index && I32.v c.index < 248320
  then SM.best scores region c.value (I32.v c.index)
  else False

let certified_frame (before after:FS.score_vector) (region:SM.region) (c:Sel.candidate)
  : Lemma
      (requires certified before region c /\
        (forall (i:nat{i < 248320}). region i ==> S.index before i == S.index after i))
      (ensures certified after region c) = ()

let certified_ext (scores:FS.score_vector) (left right:SM.region) (c:Sel.candidate)
  : Lemma
      (requires certified scores left c /\ (forall (i:nat{i < 248320}). left i <==> right i))
      (ensures certified scores right c) = ()

let tree_certified (#n:Sel.size) (scores:FS.score_vector) (original:lseq Sel.candidate n)
  (regions:nat -> SM.region)
  : Lemma
      (requires forall (i:nat{i < n}). certified scores (regions i) (S.index original i))
      (ensures certified scores
        (fun j -> exists (i:nat{i < n}). regions i j) (Sel.tree original 1 0)) =
  Sel.tree_summary original;
  let root = Sel.tree original 1 0 in
  assert (-1 <= I32.v root.index /\ I32.v root.index < 248320);
  assert (SM.candidate scores root.value (I32.v root.index));
  assert (I32.v root.index >= 0 ==>
    (exists (i:nat{i < n}). regions i (SM.index_nat (I32.v root.index))));
  let at (j:nat{j < 248320})
    : Lemma (
        (exists (i:nat{i < n}). regions i j) ==>
          I32.v root.index >= 0 /\ FS.priority_le (S.index scores j) root.value) =
    let from (i:nat{i < n})
      : Lemma (regions i j ==>
          I32.v root.index >= 0 /\ FS.priority_le (S.index scores j) root.value) =
      let candidate = S.index original i in
      if regions i j then (
        assert (Sel.below candidate root);
        FS.priority_transitive (S.index scores j) candidate.value root.value
      )
    in C.forall_intro from
  in C.forall_intro at

let scan_record (before:FS.score_vector) (block:nat{block < 512}) (warp:nat{warp < 8})
  (row:nat{block * 485 + warp <= row /\ row < (block + 1) * 485 /\
    (row - block * 485 - warp) % 8 == 0})
  (current:Sel.candidate) (value:f32) (index:I32.t{I32.v index == row})
  : Lemma
      (requires certified before (SM.scan_region block warp row) current)
      (ensures certified (S.upd before row value) (SM.scan_region block warp (row + 8))
        (Sel.choose current {Sel.value = value; Sel.index = index})) =
  let after = S.upd before row value in
  certified_frame before after (SM.scan_region block warp row) current;
  SM.scan_step after block warp row current.value (I32.v current.index)

inline_for_extraction noextract
let produced (block warp limit:nat) : SM.region =
  fun row ->
    row / 485 < block \/
    (row / 485 == block /\
      ((row % 485) % 8 < warp \/ ((row % 485) % 8 == warp /\ row < limit)))

inline_for_extraction noextract
let before_warp (block warp:nat) : SM.region =
  produced block warp (block * 485 + warp)

inline_for_extraction noextract
let score_model (m:Dot.interpretation 752393024 1024) (row:nat{row < 248320})
  : GTot real =
  rsum (S.init_ghost 1024 (Dot.term m (row * 1024)))

inline_for_extraction noextract
let modeled (weights:lseq bf16 752393024) (normalized:lseq f32 1024)
  (scores:FS.score_vector) (region:SM.region) : prop =
  forall (m:Dot.interpretation 752393024 1024).
    Dot.float_inputs weights normalized m ==>
      (forall (row:nat{row < 248320}). region row ==>
        S.index scores row %~ score_model m row)

let modeled_ext (weights:lseq bf16 752393024) (normalized:lseq f32 1024)
  (scores:FS.score_vector) (left right:SM.region)
  : Lemma
      (requires modeled weights normalized scores left /\
        (forall (row:nat{row < 248320}). right row ==> left row))
      (ensures modeled weights normalized scores right) = ()

let model_record (weights:lseq bf16 752393024) (normalized:lseq f32 1024)
  (scores:FS.score_vector) (block:nat{block < 512}) (warp:nat{warp < 8})
  (row:nat{block * 485 + warp <= row /\ row < (block + 1) * 485 /\
    (row - block * 485 - warp) % 8 == 0}) (value:f32)
  : Lemma
      (requires modeled weights normalized scores (produced block warp row) /\
        (forall (m:Dot.interpretation 752393024 1024).
          Dot.float_inputs weights normalized m ==> value %~ score_model m row))
      (ensures modeled weights normalized (S.upd scores row value) (produced block warp (row + 8))) =
  assert (forall (i:nat{i < 248320}).
    produced block warp (row + 8) i ==> produced block warp row i \/ i == row)

let warp_finished (weights:lseq bf16 752393024) (normalized:lseq f32 1024)
  (scores:FS.score_vector) (block:nat{block < 512}) (warp:nat{warp < 8})
  (limit:nat{(block + 1) * 485 <= limit})
  (candidate:Sel.candidate)
  : Lemma
      (requires certified scores (SM.scan_region block warp limit) candidate /\
        modeled weights normalized scores (produced block warp limit))
      (ensures certified scores (SM.lane_region block 32 warp) candidate /\
        modeled weights normalized scores (before_warp block (warp + 1))) = ()

let block_tree (scores:FS.score_vector) (block:nat{block < 512}) (original:lseq Sel.candidate 32)
  : Lemma
      (requires forall (i:nat{i < 32}). certified scores (SM.lane_region block 32 i) (S.index original i))
      (ensures certified scores (fun row -> row / 485 == block) (Sel.tree original 1 0)) =
  tree_certified scores original (SM.lane_region block 32);
  let member (row:nat{row < 248320})
    : Lemma (
        (exists (i:nat{i < 32}). SM.lane_region block 32 i row) <==> row / 485 == block) =
    if row / 485 = block then
      C.exists_intro #(i:nat{i < 32}) (fun i -> SM.lane_region block 32 i row) ((row % 485) % 8)
  in
  C.forall_intro member;
  certified_ext scores (fun row -> exists (i:nat{i < 32}). SM.lane_region block 32 i row)
    (fun row -> row / 485 == block) (Sel.tree original 1 0)

let vocabulary_tree (scores:FS.score_vector) (original:lseq Sel.candidate 512)
  : Lemma
      (requires forall (i:nat{i < 512}). certified scores (fun row -> row / 485 == i) (S.index original i))
      (ensures certified scores (fun _ -> True) (Sel.tree original 1 0)) =
  tree_certified scores original (fun i row -> row / 485 == i);
  let member (row:nat{row < 248320})
    : Lemma (exists (i:nat{i < 512}). row / 485 == i) =
    C.exists_intro #(i:nat{i < 512}) (fun i -> row / 485 == i) (row / 485)
  in
  C.forall_intro member;
  certified_ext scores (fun row -> exists (i:nat{i < 512}). row / 485 == i)
    (fun _ -> True) (Sel.tree original 1 0)

inline_for_extraction noextract
let masked_certificates (#n:nat) (scores:FS.score_vector) (regions:nat -> SM.region)
  (values:lseq (option f32) n) (indices:lseq (option I32.t) n)
  (count:nat{count <= n}) : prop =
  Memory.ready values 0 count /\ Memory.ready indices 0 count /\
  (forall (i:nat{i < count}).
    match S.index values i, S.index indices i with
    | Some v, Some index -> certified scores (regions i) {Sel.value = v; Sel.index = index}
    | _ -> False)

let frame_certificates (#n:nat) (before after:FS.score_vector) (regions:nat -> SM.region)
  (values:lseq (option f32) n) (indices:lseq (option I32.t) n)
  (count:nat{count <= n})
  : Lemma
      (requires masked_certificates before regions values indices count /\
        (forall (i:nat{i < count}) (r:nat{r < 248320}).
          regions i r ==> S.index before r == S.index after r))
      (ensures masked_certificates after regions values indices count) =
  let at (i:nat{i < count})
    : Lemma (
        match S.index values i, S.index indices i with
        | Some v, Some index -> certified after (regions i) {Sel.value = v; Sel.index = index}
        | _ -> False) =
    match S.index values i, S.index indices i with
    | Some v, Some index ->
        certified_frame before after (regions i) {Sel.value = v; Sel.index = index}
    | _ -> ()
  in C.forall_intro at

let push_certificate (#n:nat) (scores:FS.score_vector) (regions:nat -> SM.region)
  (values:lseq (option f32) n) (indices:lseq (option I32.t) n)
  (count:nat{count < n}) (candidate:Sel.candidate)
  : Lemma
      (requires masked_certificates scores regions values indices count /\
        certified scores (regions count) candidate)
      (ensures masked_certificates #n scores regions
        (S.upd values count (Some candidate.value))
        (S.upd indices count (Some candidate.index)) (count + 1)) = ()

let plain_certificates (#n:nat) (scores:FS.score_vector) (regions:nat -> SM.region)
  (masked_values:lseq (option f32) n) (masked_indices:lseq (option I32.t) n)
  (values:lseq f32 n) (indices:lseq I32.t n)
  : Lemma
      (requires masked_certificates scores regions masked_values masked_indices n /\
        (forall (i:nat{i < n}).
          S.index masked_values i == Some (S.index values i) /\
          S.index masked_indices i == Some (S.index indices i)))
      (ensures forall (i:nat{i < n}).
        certified scores (regions i) (S.index (Sel.candidates values indices) i)) = ()
