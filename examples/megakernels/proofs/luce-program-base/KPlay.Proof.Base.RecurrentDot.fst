module KPlay.Proof.Base.RecurrentDot

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module P = KPlay.Proof.Shared.Projection
module A = Pulse.Lib.Array
module Local = KPlay.Proof.Base.LocalArray
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let rec component (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128)
  (group:P.index 16) (count:nat{count <= 8}) : GTot FP.t (decreases count) =
  if count = 0 then F.zero #FP.t
  else
    let i = group + 16 * (count - 1) in
    F.add (component state vector offset column group (count - 1))
      (F.mul (Seq.index state (offset + i * 128 + column)) (Seq.index vector i))

inline_for_extraction noextract
let partial_value (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (y:P.index 4) : GTot FP.t =
  F.add (F.add (F.add
    (component state vector offset column y 8)
    (component state vector offset column (y + 4) 8))
    (component state vector offset column (y + 8) 8))
    (component state vector offset column (y + 12) 8)

inline_for_extraction noextract
let value (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) : GTot FP.t =
  F.add
    (F.add (partial_value state vector offset column 0) (partial_value state vector offset column 2))
    (F.add (partial_value state vector offset column 1) (partial_value state vector offset column 3))

inline_for_extraction noextract
let component_initial (before:lseq FP.t 4) (r:P.index 4)
  : Lemma
      (requires Seq.index before r == F.zero #FP.t)
      (ensures Seq.upd before r (F.zero #FP.t) == before) =
  assert (Seq.equal (Seq.upd before r (F.zero #FP.t)) before);
  Seq.lemma_eq_elim (Seq.upd before r (F.zero #FP.t)) before

inline_for_extraction noextract
let overwrite (before:lseq FP.t 4) (r:P.index 4) (old next:FP.t)
  : Lemma (Seq.upd (Seq.upd before r old) r next == Seq.upd before r next) =
  assert (Seq.equal (Seq.upd (Seq.upd before r old) r next) (Seq.upd before r next));
  Seq.lemma_eq_elim (Seq.upd (Seq.upd before r old) r next) (Seq.upd before r next)

inline_for_extraction noextract
fn accumulate
  (state:larray FP.t 4718592) (vector:larray FP.t 128) (sums:larray FP.t 4)
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (column:sz{SZ.v column < 128})
  (y r:sz{SZ.v y < 4 /\ SZ.v r < 4})
  (#state_fraction #vector_fraction:perm)
  (#ss:erased (lseq FP.t 4718592)) (#sv:erased (lseq FP.t 128)) (#before:erased (lseq FP.t 4))
  preserves gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
  requires sums |-> before
  requires pure (Seq.index before (SZ.v r) == F.zero #FP.t)
  ensures sums |-> Seq.upd before (SZ.v r)
    (component ss sv (SZ.v offset) (SZ.v column) (SZ.v y + 4 * SZ.v r) 8)
{
  component_initial before (SZ.v r);
  let first = SZ.add y (SZ.mul 4sz r);
  let mut i = first;
  while (!i <^ 128sz)
    invariant exists* (current:sz{SZ.v first <= SZ.v current /\
      SZ.v current <= SZ.v first + 128 /\ (SZ.v current - SZ.v first) % 16 == 0}).
      i |-> current **
      sums |-> Seq.upd before (SZ.v r)
        (component ss sv (SZ.v offset) (SZ.v column) (SZ.v first)
          ((SZ.v current - SZ.v first) / 16))
    invariant gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
    decreases (SZ.v first + 128 - SZ.v !i)
  {
    let current = !i;
    let left = state.(SZ.add (SZ.add offset (SZ.mul current 128sz)) column);
    let right = vector.(current);
    let old = sums.(r);
    let result = F.add old (F.mul left right);
    let next = SZ.add current 16sz;
    assert pure (SZ.v current == SZ.v first + 16 * ((SZ.v current - SZ.v first) / 16));
    assert pure ((SZ.v next - SZ.v first) / 16 == (SZ.v current - SZ.v first) / 16 + 1);
    norm_spec [delta_once ["KPlay.Proof.Base.RecurrentDot.component"]]
      (component ss sv (SZ.v offset) (SZ.v column) (SZ.v first) ((SZ.v next - SZ.v first) / 16));
    overwrite before (SZ.v r) old result;
    sums.(r) <- result;
    i := next;
  };
  let current = !i;
  assert pure ((SZ.v current - SZ.v first) / 16 == 8);
}

inline_for_extraction noextract
let sums_after (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (y:P.index 4)
  (count:nat{count <= 4}) : GTot (lseq FP.t 4) =
  Seq.init_ghost 4 (fun r -> if r < count then component state vector offset column (y + 4 * r) 8
    else F.zero #FP.t)

inline_for_extraction noextract
let sums_zero (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (y:P.index 4)
  : Lemma (sums_after state vector offset column y 0 == Seq.create 4 (F.zero #FP.t)) =
  assert (Seq.equal (sums_after state vector offset column y 0) (Seq.create 4 (F.zero #FP.t)));
  Seq.lemma_eq_elim (sums_after state vector offset column y 0) (Seq.create 4 (F.zero #FP.t))

inline_for_extraction noextract
let sums_step (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (y r:P.index 4)
  : Lemma (
      Seq.index (sums_after state vector offset column y r) r == F.zero #FP.t /\
      Seq.upd (sums_after state vector offset column y r) r
        (component state vector offset column (y + 4 * r) 8) ==
      sums_after state vector offset column y (r + 1)) =
  assert (Seq.equal
    (Seq.upd (sums_after state vector offset column y r) r
      (component state vector offset column (y + 4 * r) 8))
    (sums_after state vector offset column y (r + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (sums_after state vector offset column y r) r
      (component state vector offset column (y + 4 * r) 8))
    (sums_after state vector offset column y (r + 1))

inline_for_extraction noextract
fn sum_groups
  (state:larray FP.t 4718592) (vector:larray FP.t 128)
  (sums:larray FP.t 4)
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (column:sz{SZ.v column < 128}) (y:sz{SZ.v y < 4})
  (#state_fraction #vector_fraction:perm)
  (#ss:erased (lseq FP.t 4718592)) (#sv:erased (lseq FP.t 128))
  preserves gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
  requires sums |-> Seq.create 4 (F.zero #FP.t)
  returns result:FP.t
  ensures sums |-> sums_after ss sv (SZ.v offset) (SZ.v column) (SZ.v y) 4
  ensures pure (result == partial_value ss sv (SZ.v offset) (SZ.v column) (SZ.v y))
{
  sums_zero ss sv (SZ.v offset) (SZ.v column) (SZ.v y);
  let mut r = 0sz;
  while (!r <^ 4sz)
    invariant exists* (current:sz{SZ.v current <= 4}).
      r |-> current ** sums |-> sums_after ss sv (SZ.v offset) (SZ.v column) (SZ.v y) (SZ.v current)
    invariant gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
    decreases (4 - SZ.v !r)
  {
    let current = !r;
    sums_step ss sv (SZ.v offset) (SZ.v column) (SZ.v y) (SZ.v current);
    accumulate state vector sums offset column y current;
    r := SZ.add current 1sz;
  };
  let a = sums.(0sz);
  let b = sums.(1sz);
  let c = sums.(2sz);
  let d = sums.(3sz);
  F.add (F.add (F.add a b) c) d
}

inline_for_extraction noextract
fn partial
  (state:larray FP.t 4718592) (vector:larray FP.t 128)
  (sums partial:larray FP.t 4)
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (column:sz{SZ.v column < 128}) (y:sz{SZ.v y < 4})
  (#state_fraction #vector_fraction:perm)
  (#ss:erased (lseq FP.t 4718592)) (#sv:erased (lseq FP.t 128))
  (#old_sums #old_partial:erased (lseq FP.t 4))
  preserves gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
  requires sums |-> old_sums ** partial |-> old_partial
  ensures sums |-> sums_after ss sv (SZ.v offset) (SZ.v column) (SZ.v y) 4
  ensures partial |-> Seq.upd old_partial (SZ.v y) (partial_value ss sv (SZ.v offset) (SZ.v column) (SZ.v y))
{
  sums.(0sz) <- F.zero #FP.t;
  sums.(1sz) <- F.zero #FP.t;
  sums.(2sz) <- F.zero #FP.t;
  sums.(3sz) <- F.zero #FP.t;
  with cleared. assert sums |-> cleared;
  assert pure (Seq.equal cleared (Seq.create 4 (F.zero #FP.t)));
  Seq.lemma_eq_elim cleared (Seq.create 4 (F.zero #FP.t));
  rewrite sums |-> cleared as sums |-> Seq.create 4 (F.zero #FP.t);
  let value = sum_groups state vector sums offset column y;
  partial.(y) <- value;
}

inline_for_extraction noextract
let partials_after (before:lseq FP.t 4) (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (count:nat{count <= 4})
  : GTot (lseq FP.t 4) =
  Seq.init_ghost 4 (fun y -> if y < count then partial_value state vector offset column y else Seq.index before y)

inline_for_extraction noextract
let partials_zero (before:lseq FP.t 4) (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128)
  : Lemma (partials_after before state vector offset column 0 == before) =
  assert (Seq.equal (partials_after before state vector offset column 0) before);
  Seq.lemma_eq_elim (partials_after before state vector offset column 0) before

inline_for_extraction noextract
let partials_step (before:lseq FP.t 4) (state:lseq FP.t 4718592) (vector:lseq FP.t 128)
  (offset:nat{offset + 16384 <= 4718592}) (column:P.index 128) (y:P.index 4)
  : Lemma (
      Seq.upd (partials_after before state vector offset column y) y (partial_value state vector offset column y) ==
      partials_after before state vector offset column (y + 1)) =
  assert (Seq.equal
    (Seq.upd (partials_after before state vector offset column y) y (partial_value state vector offset column y))
    (partials_after before state vector offset column (y + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (partials_after before state vector offset column y) y (partial_value state vector offset column y))
    (partials_after before state vector offset column (y + 1))

inline_for_extraction noextract
fn dot
  (state:larray FP.t 4718592) (vector:larray FP.t 128)
  (sums partials:larray FP.t 4)
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (column:sz{SZ.v column < 128})
  (#state_fraction #vector_fraction:perm)
  (#ss:erased (lseq FP.t 4718592)) (#sv:erased (lseq FP.t 128))
  (#old_sums #old_partial:erased (lseq FP.t 4))
  preserves gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
  requires sums |-> old_sums ** partials |-> old_partial
  returns result:FP.t
  ensures exists* (last_sums:lseq FP.t 4). sums |-> last_sums **
    partials |-> partials_after old_partial ss sv (SZ.v offset) (SZ.v column) 4 **
    pure (result == value ss sv (SZ.v offset) (SZ.v column))
{
  partials_zero old_partial ss sv (SZ.v offset) (SZ.v column);
  let mut y = 0sz;
  while (!y <^ 4sz)
    invariant exists* (current:sz{SZ.v current <= 4}) (last_sums:lseq FP.t 4).
      y |-> current ** sums |-> last_sums **
      partials |-> partials_after old_partial ss sv (SZ.v offset) (SZ.v column) (SZ.v current)
    invariant gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
    decreases (4 - SZ.v !y)
  {
    let current = !y;
    partial state vector sums partials offset column current;
    partials_step old_partial ss sv (SZ.v offset) (SZ.v column) (SZ.v current);
    y := SZ.add current 1sz;
  };
  let a = partials.(0sz);
  let b = partials.(2sz);
  let c = partials.(1sz);
  let d = partials.(3sz);
  F.add (F.add a b) (F.add c d)
}

inline_for_extraction noextract
fn source_dot
  (state:larray FP.t 4718592) (vector:larray FP.t 128)
  (offset:sz{SZ.v offset + 16384 <= 4718592}) (column:sz{SZ.v column < 128})
  (#state_fraction #vector_fraction:perm)
  (#ss:erased (lseq FP.t 4718592)) (#sv:erased (lseq FP.t 128))
  preserves gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
  returns result:FP.t
  ensures pure (result == value ss sv (SZ.v offset) (SZ.v column))
{
  let mut partials : A.array FP.t = [| 4sz |];
  with initial. assert A.pts_to_mask partials initial Init.all_cells;
  let original : erased (Local.snapshot 4) = hide #(Local.snapshot 4) initial;
  Local.written_zero #4 #4 original (partial_value ss sv (SZ.v offset) (SZ.v column));
  let mut y = 0sz;
  while (!y <^ 4sz)
    invariant exists* (current:sz{SZ.v current <= 4}).
      y |-> current ** A.pts_to_mask partials
        (Local.written #4 #4 original (partial_value ss sv (SZ.v offset) (SZ.v column)) (SZ.v current)) Init.all_cells
    invariant gpu ** state |-> Frac state_fraction ss ** vector |-> Frac vector_fraction sv
    decreases (4 - SZ.v !y)
  {
    let current = !y;
    let mut sums : A.array FP.t = [| F.zero #FP.t; 4sz |];
    let partial = sum_groups state vector sums offset column current;
    Local.written_step #4 #4 original (partial_value ss sv (SZ.v offset) (SZ.v column)) (SZ.v current);
    A.mask_write partials current partial;
    y := SZ.add current 1sz;
  };
  let a = A.mask_read partials 0sz;
  let b = A.mask_read partials 2sz;
  let c = A.mask_read partials 1sz;
  let d = A.mask_read partials 3sz;
  F.add (F.add a b) (F.add c d)
}
