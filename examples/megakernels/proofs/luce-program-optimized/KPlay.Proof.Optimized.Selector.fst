module KPlay.Proof.Optimized.Selector

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module I32 = FStar.Int32
module F = Kuiper.Floating
module FS = KPlay.Proof.Optimized.FamilySelection
module C = FStar.Classical
module Ref = Pulse.Lib.Reference

noextract
noeq type candidate = {
  value: f32;
  index: I32.t;
}

inline_for_extraction noextract
let valid (c:candidate) : prop = 0 <= I32.v c.index

inline_for_extraction noextract
let better (incoming:f32) (index:I32.t) (current:f32) (current_index:I32.t)
  : result:bool{result == FS.better_candidate incoming (I32.v index) current (I32.v current_index)} =
  FS.native_order incoming current;
  F.eq_spec incoming incoming;
  F.eq_spec current current;
  I32.gte index 0l &&
    (I32.lt current_index 0l || F.lt current incoming ||
      (not (F.ieee_eq incoming incoming) && F.ieee_eq current current))

inline_for_extraction noextract
let choose (left right:candidate) : GTot candidate =
  if better right.value right.index left.value left.index then right else left

inline_for_extraction noextract
let below (left right:candidate) : prop =
  valid left ==> valid right /\ FS.priority_le left.value right.value

let below_reflexive (x:candidate) : Lemma (below x x) =
  FS.priority_reflexive x.value

let below_transitive (x y z:candidate)
  : Lemma (requires below x y /\ below y z) (ensures below x z) =
  if valid x then FS.priority_transitive x.value y.value z.value

let choose_sound (left right:candidate)
  : Lemma (
      below left (choose left right) /\ below right (choose left right) /\
      (choose left right == left \/ choose left right == right) /\
      (valid (choose left right) <==> (valid left \/ valid right))) =
  FS.priority_reflexive left.value;
  FS.priority_reflexive right.value;
  FS.priority_total left.value right.value

inline_for_extraction noextract
let power (n:nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32 \/
  n = 64 \/ n = 128 \/ n = 256 \/ n = 512

inline_for_extraction noextract
let size = n:pos{n = 32 \/ n = 512}

inline_for_extraction noextract
let rec tree (#n:size) (original:lseq candidate n)
  (width:nat{power width /\ width <= n}) (lane:nat{lane < width})
  : GTot candidate (decreases n - width) =
  if width = n then S.index original lane
  else choose (tree original (2 * width) lane) (tree original (2 * width) (lane + width))

let rec tree_covers (#n:size) (original:lseq candidate n)
  (width:nat{power width /\ width <= n}) (lane:nat{lane < width})
  (j:nat{j < n /\ j % width == lane})
  : Lemma
      (ensures below (S.index original j) (tree original width lane))
      (decreases n - width) =
  if width = n then below_reflexive (S.index original j)
  else (
    let left = tree original (2 * width) lane in
    let right = tree original (2 * width) (lane + width) in
    choose_sound left right;
    if j % (2 * width) = lane then (
      tree_covers original (2 * width) lane j;
      below_transitive (S.index original j) left (choose left right)
    ) else (
      assert (j % (2 * width) == lane + width);
      tree_covers original (2 * width) (lane + width) j;
      below_transitive (S.index original j) right (choose left right)
    )
  )

let rec tree_origin (#n:size) (original:lseq candidate n)
  (width:nat{power width /\ width <= n}) (lane:nat{lane < width})
  : Lemma
      (ensures exists (j:nat{j < n}). tree original width lane == S.index original j)
      (decreases n - width) =
  if width = n then
    C.exists_intro (fun (j:nat{j < n}) -> tree original width lane == S.index original j) lane
  else (
    tree_origin original (2 * width) lane;
    tree_origin original (2 * width) (lane + width);
    choose_sound (tree original (2 * width) lane) (tree original (2 * width) (lane + width))
  )

let tree_summary (#n:size) (original:lseq candidate n)
  : Lemma (
      (forall (i:nat{i < n}). below (S.index original i) (tree original 1 0)) /\
      (exists (i:nat{i < n}). tree original 1 0 == S.index original i)) =
  let at (i:nat{i < n}) : Lemma (below (S.index original i) (tree original 1 0)) =
    tree_covers original 1 0 i
  in
  C.forall_intro at;
  tree_origin original 1 0

inline_for_extraction noextract
let cell (#n:nat) (values:lseq f32 n) (indices:lseq I32.t n) (i:nat{i < n}) : candidate =
  { value = S.index values i; index = S.index indices i }

inline_for_extraction noextract
let candidates (#n:nat) (values:lseq f32 n) (indices:lseq I32.t n)
  : GTot (lseq candidate n) =
  S.init_ghost n (cell values indices)

inline_for_extraction noextract
let level (#n:size) (original:lseq candidate n)
  (values:lseq f32 n) (indices:lseq I32.t n) (width:nat{power width /\ width <= n}) : prop =
  forall (i:nat{i < width}). cell values indices i == tree original width i

inline_for_extraction noextract
let advancing (#n:size) (original:lseq candidate n)
  (values:lseq f32 n) (indices:lseq I32.t n)
  (width:nat{power width /\ width < n}) (done:nat{done <= width}) : prop =
  forall (i:nat{i < 2 * width}).
    cell values indices i ==
      (if i < done then tree original width i else tree original (2 * width) i)

let advance_step (#n:size) (original:lseq candidate n)
  (values:lseq f32 n) (indices:lseq I32.t n)
  (width:nat{power width /\ width < n}) (lane:nat{lane < width})
  : Lemma
      (requires advancing original values indices width lane)
      (ensures (
        let take = better (S.index values (lane + width)) (S.index indices (lane + width))
          (S.index values lane) (S.index indices lane) in
        advancing original
          (if take then S.upd values lane (S.index values (lane + width)) else values)
          (if take then S.upd indices lane (S.index indices (lane + width)) else indices)
          width (lane + 1))) =
  let take = better (S.index values (lane + width)) (S.index indices (lane + width))
    (S.index values lane) (S.index indices lane) in
  let next_values = if take then S.upd values lane (S.index values (lane + width)) else values in
  let next_indices = if take then S.upd indices lane (S.index indices (lane + width)) else indices in
  assert (cell values indices lane == tree original (2 * width) lane);
  assert (cell values indices (lane + width) == tree original (2 * width) (lane + width));
  FStar.Pervasives.norm_spec [delta_once ["KPlay.Proof.Optimized.Selector.tree"]]
    (tree original width lane);
  let at (i:nat{i < 2 * width}) : Lemma (
      cell next_values next_indices i ==
        (if i < lane + 1 then tree original width i else tree original (2 * width) i)) =
    if i = lane then (
      assert (cell next_values next_indices lane ==
        choose (cell values indices lane) (cell values indices (lane + width)))
    ) else (
      assert (cell next_values next_indices i == cell values indices i)
    )
  in C.forall_intro at

noextract
fn write_if_better (#n:erased pos) (values:larray f32 n) (indices:larray I32.t n)
  (lane:sz{SZ.v lane < n}) (incoming current:f32) (incoming_index current_index:I32.t)
  (#vv:erased (lseq f32 n)) (#iv:erased (lseq I32.t n))
  preserves gpu
  requires values |-> vv ** indices |-> iv
  ensures values |-> (if better incoming incoming_index current current_index
    then S.upd (FStar.Ghost.reveal vv) (SZ.v lane) incoming else FStar.Ghost.reveal vv)
  ensures indices |-> (if better incoming incoming_index current current_index
    then S.upd (FStar.Ghost.reveal iv) (SZ.v lane) incoming_index else FStar.Ghost.reveal iv)
{
  if (better incoming incoming_index current current_index) {
    values.(lane) <- incoming;
    indices.(lane) <- incoming_index;
  } else {
    ();
  }
}

noextract
fn accumulate (maximum:Ref.ref f32) (index:Ref.ref I32.t)
  (incoming:f32) (incoming_index:I32.t) (#current:erased f32) (#current_index:erased I32.t)
  preserves gpu
  requires maximum |-> current ** index |-> current_index
  ensures maximum |-> (choose {value = current; index = current_index}
    {value = incoming; index = incoming_index}).value
  ensures index |-> (choose {value = current; index = current_index}
    {value = incoming; index = incoming_index}).index
{
  if (better incoming incoming_index (!maximum) (!index)) {
    maximum := incoming;
    index := incoming_index;
  } else {
    ();
  }
}

noextract
fn reduce_level (#n:erased size) (values:larray f32 n) (indices:larray I32.t n)
  (distance:sz{power (SZ.v distance) /\ SZ.v distance < n})
  (#original:erased (lseq candidate n))
  (#vv:erased (lseq f32 n)) (#iv:erased (lseq I32.t n))
  preserves gpu
  requires values |-> vv ** indices |-> iv **
    pure (level original vv iv (2 * SZ.v distance))
  ensures exists* (after_values:lseq f32 n) (after_indices:lseq I32.t n).
    values |-> after_values ** indices |-> after_indices **
    pure (level original after_values after_indices (SZ.v distance))
{
  let mut lane = 0sz;
  while (!lane <^ distance)
    invariant exists* (i:sz{i <= distance}) (cv:lseq f32 n) (ci:lseq I32.t n).
      lane |-> i ** values |-> cv ** indices |-> ci **
      pure (advancing original cv ci (SZ.v distance) (SZ.v i))
    decreases (SZ.v distance - SZ.v !lane)
  {
    let i = !lane;
    with current_values current_indices. assert values |-> current_values ** indices |-> current_indices;
    let incoming = values.(i +^ distance);
    let incoming_index = indices.(i +^ distance);
    let current = values.(i);
    let current_index = indices.(i);
    advance_step original current_values current_indices (SZ.v distance) (SZ.v i);
    write_if_better values indices i incoming current incoming_index current_index;
    lane := i +^ 1sz;
  }
}

noextract
fn reduce (#n:erased size) (values:larray f32 n) (indices:larray I32.t n)
  (count:sz{SZ.v count == n})
  (#vv:erased (lseq f32 n)) (#iv:erased (lseq I32.t n))
  preserves gpu
  requires values |-> vv ** indices |-> iv
  returns result:candidate
  ensures exists* (after_values:lseq f32 n) (after_indices:lseq I32.t n).
    values |-> after_values ** indices |-> after_indices
  ensures pure (result == tree (candidates vv iv) 1 0)
{
  let mut distance = count /^ 2sz;
  while (0sz <^ !distance)
    invariant exists* (d:sz{SZ.v d < n /\ (SZ.v d = 0 \/ power (SZ.v d))})
                     (cv:lseq f32 n) (ci:lseq I32.t n).
      distance |-> d ** values |-> cv ** indices |-> ci **
      pure (if SZ.v d = 0 then level (candidates vv iv) cv ci 1
        else level (candidates vv iv) cv ci (2 * SZ.v d))
    decreases (SZ.v !distance)
  {
    let d = !distance;
    reduce_level values indices d #(candidates vv iv);
    distance := d /^ 2sz;
  };
  let value = values.(0sz);
  let index = indices.(0sz);
  { value; index }
}
