module KPlay.Proof.Base.NormNative

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module S = FStar.Seq
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module BF = Kuiper.BFloat16
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module P = KPlay.Proof.Shared.Projection
module Norm = KPlay.Proof.Base.Norm
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let partials = lseq FP.t 256

inline_for_extraction noextract
let partial_values (values:partials) : P.vector FP.t 256 =
  fun i -> S.index values i

inline_for_extraction noextract
let grouped (before:partials) (hidden:Transfer.scratch) (count:nat{count <= 256})
  : GTot partials =
  S.init_ghost 256 (fun i ->
    if i < count then Norm.group 1024 (Norm.residual_values hidden) i 4 else S.index before i)

inline_for_extraction noextract
let grouped_zero (before:partials) (hidden:Transfer.scratch)
  : Lemma (grouped before hidden 0 == before) =
  assert (S.equal (grouped before hidden 0) before);
  S.lemma_eq_elim (grouped before hidden 0) before

inline_for_extraction noextract
let grouped_step (before:partials) (hidden:Transfer.scratch) (i:P.index 256)
  : Lemma (
      S.upd (grouped before hidden i) i (Norm.group 1024 (Norm.residual_values hidden) i 4) ==
      grouped before hidden (i + 1)) =
  assert (S.equal
    (S.upd (grouped before hidden i) i (Norm.group 1024 (Norm.residual_values hidden) i 4))
    (grouped before hidden (i + 1)));
  S.lemma_eq_elim
    (S.upd (grouped before hidden i) i (Norm.group 1024 (Norm.residual_values hidden) i 4))
    (grouped before hidden (i + 1))

inline_for_extraction noextract
fn square_group
  (hidden:larray BF.t 16000) (group:sz{SZ.v group < 256})
  (#values:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction values
  returns result:FP.t
  ensures pure (result == Norm.group 1024 (Norm.residual_values values) (SZ.v group) 4)
{
  let mut j = 0sz;
  let mut accum:FP.t = F.zero #FP.t;
  while (!j <^ 4sz)
    invariant exists* (count:sz{SZ.v count <= 4}) (current:FP.t).
      j |-> count ** accum |-> current **
      pure (current == Norm.group 1024 (Norm.residual_values values) (SZ.v group) (SZ.v count))
    invariant gpu ** A.pts_to hidden #fraction values
    decreases (4 - SZ.v !j)
  {
    let count = !j;
    let value = hidden.(SZ.add (SZ.mul group 4sz) count);
    let wide = Cast.cast_bf16_to_f32 value;
    let current = !accum;
    let next = F.add current (F.mul wide wide);
    norm_spec [delta_once ["KPlay.Proof.Base.Norm.group"]]
      (Norm.group 1024 (Norm.residual_values values) (SZ.v group) (SZ.v count + 1));
    accum := next;
    j := SZ.add count 1sz;
  };
  !accum
}

inline_for_extraction noextract
fn initialize_partials
  (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (#values:erased Transfer.scratch) (#before:erased partials) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction values
  requires partial |-> before
  ensures partial |-> grouped before values 256
{
  grouped_zero before values;
  let mut i = 0sz;
  while (!i <^ 256sz)
    invariant exists* (count:sz{SZ.v count <= 256}).
      i |-> count ** partial |-> grouped before values (SZ.v count)
    invariant gpu ** A.pts_to hidden #fraction values
    decreases (256 - SZ.v !i)
  {
    let count = !i;
    let value = square_group hidden count;
    grouped_step before values (SZ.v count);
    partial.(count) <- value;
    i := SZ.add count 1sz;
  }
}

inline_for_extraction noextract
let halved (before:partials) (half:pos{half <= 128}) (count:nat{count <= half})
  : GTot partials =
  S.init_ghost 256 (fun i ->
    if i < count then F.add (S.index before i) (S.index before (i + half))
    else S.index before i)

inline_for_extraction noextract
let halved_zero (before:partials) (half:pos{half <= 128})
  : Lemma (halved before half 0 == before) =
  assert (S.equal (halved before half 0) before);
  S.lemma_eq_elim (halved before half 0) before

inline_for_extraction noextract
let halved_step (before:partials) (half:pos{half <= 128}) (i:nat{i < half})
  : Lemma (
      S.index (halved before half i) i == S.index before i /\
      S.index (halved before half i) (i + half) == S.index before (i + half) /\
      S.upd (halved before half i) i (F.add (S.index before i) (S.index before (i + half))) ==
      halved before half (i + 1)) =
  assert (S.equal
    (S.upd (halved before half i) i (F.add (S.index before i) (S.index before (i + half))))
    (halved before half (i + 1)));
  S.lemma_eq_elim
    (S.upd (halved before half i) i (F.add (S.index before i) (S.index before (i + half))))
    (halved before half (i + 1))

inline_for_extraction noextract
fn halve_partial
  (partial:larray FP.t 256) (half:sz{0 < SZ.v half /\ SZ.v half <= 128})
  (#before:erased partials)
  preserves gpu
  requires partial |-> before
  ensures partial |-> halved before (SZ.v half) (SZ.v half)
{
  halved_zero before (SZ.v half);
  let mut i = 0sz;
  while (!i <^ half)
    invariant exists* (count:sz{SZ.v count <= SZ.v half}).
      i |-> count ** partial |-> halved before (SZ.v half) (SZ.v count)
    invariant gpu
    decreases (SZ.v half - SZ.v !i)
  {
    let count = !i;
    halved_step before (SZ.v half) (SZ.v count);
    let left = partial.(count);
    let right = partial.(SZ.add count half);
    partial.(count) <- F.add left right;
    i := SZ.add count 1sz;
  }
}

inline_for_extraction noextract
let rec tree_ext (n:pos{Norm.tree_size n}) (left right:P.vector FP.t 256)
  : Lemma
      (requires forall (i:nat{i < n}). left i == right i)
      (ensures Norm.tree n left == Norm.tree n right)
      (decreases n) =
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.tree"]] (Norm.tree n left);
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.tree"]] (Norm.tree n right);
  if n > 1 then (
    Norm.half_size n;
    let at_index (i:nat{i < n / 2})
      : Lemma (Norm.paired_float (n / 2) left i == Norm.paired_float (n / 2) right i) = ()
    in
    FStar.Classical.forall_intro at_index;
    tree_ext (n / 2) (Norm.paired_float (n / 2) left) (Norm.paired_float (n / 2) right)
  )

inline_for_extraction noextract
let halved_tree (before:partials) (n:pos{Norm.tree_size n /\ n > 1})
  : Lemma (
      Norm.tree (n / 2) (partial_values (halved before (n / 2) (n / 2))) ==
      Norm.tree n (partial_values before)) =
  Norm.half_size n;
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.tree"]] (Norm.tree n (partial_values before));
  tree_ext (n / 2) (partial_values (halved before (n / 2) (n / 2)))
    (Norm.paired_float (n / 2) (partial_values before))

inline_for_extraction noextract
fn reduce_partials
  (partial:larray FP.t 256) (#before:erased partials)
  preserves gpu
  requires partial |-> before
  returns result:FP.t
  ensures exists* (after:partials).
    partial |-> after ** pure (result == Norm.tree 256 (partial_values before))
{
  let mut width = 256sz;
  while (1sz <^ !width)
    invariant exists* (n:sz{0 < SZ.v n /\ Norm.tree_size (SZ.v n)}) (current:partials).
      width |-> n ** partial |-> current **
      pure (Norm.tree (SZ.v n) (partial_values current) == Norm.tree 256 (partial_values before))
    invariant gpu
    decreases (SZ.v !width)
  {
    let n = !width;
    Norm.half_size (SZ.v n);
    with current. assert partial |-> current;
    let half = n /^ 2sz;
    halve_partial partial half;
    halved_tree current (SZ.v n);
    width := half;
  };
  with current. assert partial |-> current;
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.tree"]] (Norm.tree 1 (partial_values current));
  partial.(0sz)
}

inline_for_extraction noextract
fn source_scale
  (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (#values:erased Transfer.scratch) (#before:erased partials) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction values
  requires partial |-> before
  returns result:FP.t
  ensures exists* (after:partials).
    partial |-> after **
    pure (result == Norm.source_scale 1024 (Norm.residual_values values))
{
  initialize_partials hidden partial;
  let sum = reduce_partials partial;
  tree_ext 256 (partial_values (grouped before values 256))
    (Norm.grouped 1024 (Norm.residual_values values));
  let mean = F.mul sum (Norm.reciprocal 1024);
  F.rsqrt (F.add mean Norm.epsilon)
}

inline_for_extraction noextract
let writes (before:Transfer.scratch) (weights:Transfer.packed) (count:nat{count <= 1024})
  : GTot Transfer.scratch =
  S.init_ghost 16000 (fun i ->
    if 1024 <= i && i < 1024 + count then Norm.final_value before weights i
    else S.index before i)

inline_for_extraction noextract
let writes_zero (before:Transfer.scratch) (weights:Transfer.packed)
  : Lemma (writes before weights 0 == before) =
  assert (S.equal (writes before weights 0) before);
  S.lemma_eq_elim (writes before weights 0) before

inline_for_extraction noextract
let writes_step (before:Transfer.scratch) (weights:Transfer.packed) (i:P.index 1024)
  : Lemma (
      S.index (writes before weights i) i == S.index before i /\
      S.upd (writes before weights i) (1024 + i)
        (Norm.source_cell (S.index before i) (S.index weights (254279680 + i))
          (Norm.source_scale 1024 (Norm.residual_values before))) ==
      writes before weights (i + 1)) =
  assert (S.equal
    (S.upd (writes before weights i) (1024 + i)
      (Norm.source_cell (S.index before i) (S.index weights (254279680 + i))
        (Norm.source_scale 1024 (Norm.residual_values before))))
    (writes before weights (i + 1)));
  S.lemma_eq_elim
    (S.upd (writes before weights i) (1024 + i)
      (Norm.source_cell (S.index before i) (S.index weights (254279680 + i))
        (Norm.source_scale 1024 (Norm.residual_values before))))
    (writes before weights (i + 1))

inline_for_extraction noextract
let writes_complete (before:Transfer.scratch) (weights:Transfer.packed)
  : Lemma (writes before weights 1024 == Norm.final_writes before weights) =
  let at_index (i:P.index 16000)
    : Lemma (S.index (writes before weights 1024) i ==
      S.index (Norm.final_writes before weights) i) =
    Norm.final_writes_cell before weights i
  in
  FStar.Classical.forall_intro at_index;
  assert (S.equal (writes before weights 1024) (Norm.final_writes before weights));
  S.lemma_eq_elim (writes before weights 1024) (Norm.final_writes before weights)

inline_for_extraction noextract
fn source_final_norm
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000) (partial:larray FP.t 256)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#before:erased partials) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden (reveal vh) ** A.pts_to partial (reveal before)
  ensures exists* (after:partials).
    A.pts_to hidden (Norm.final_writes vh vw) ** partial |-> after
{
  let scale = source_scale hidden partial;
  writes_zero vh vw;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (count:sz{SZ.v count <= 1024}).
      i |-> count ** A.pts_to hidden (writes vh vw (SZ.v count))
    invariant gpu ** A.pts_to weights #fraction vw
    decreases (1024 - SZ.v !i)
  {
    let count = !i;
    writes_step vh vw (SZ.v count);
    let x = hidden.(count);
    let weight = weights.(SZ.add (SZ.uint_to_t 254279680) count);
    let value = Norm.source_cell x weight scale;
    hidden.(SZ.add 1024sz count) <- value;
    i := SZ.add count 1sz;
  };
  let count = !i;
  assert pure (SZ.v count == 1024);
  writes_complete (reveal vh) (reveal vw);
  rewrite A.pts_to hidden (writes vh vw 1024) as A.pts_to hidden (Norm.final_writes vh vw);
}
