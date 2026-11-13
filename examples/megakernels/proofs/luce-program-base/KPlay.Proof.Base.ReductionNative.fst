module KPlay.Proof.Base.ReductionNative

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module P = KPlay.Proof.Shared.Projection
module Norm = KPlay.Proof.Base.Norm
module Prior = KPlay.Proof.Base.NormNative
module Math = KPlay.Proof.Base.AttentionAlgebra
module Approx = Kuiper.Approximates.Base
module S = KPlay.Proof.Shared.Attention

inline_for_extraction noextract
let values (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity) : P.vector FP.t 256 =
  fun i -> if i < capacity then Seq.index before i else F.zero #FP.t

inline_for_extraction noextract
let halved (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (half:pos{2 * half <= capacity}) (count:nat{count <= half}) : GTot (lseq FP.t capacity) =
  Seq.init_ghost capacity (fun i ->
    if i < count then F.add (Seq.index before i) (Seq.index before (i + half))
    else Seq.index before i)

inline_for_extraction noextract
let halved_zero (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (half:pos{2 * half <= capacity})
  : Lemma (halved before half 0 == before) =
  assert (Seq.equal (halved before half 0) before);
  Seq.lemma_eq_elim (halved before half 0) before

inline_for_extraction noextract
let halved_step (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (half:pos{2 * half <= capacity}) (i:nat{i < half})
  : Lemma (
      Seq.index (halved before half i) i == Seq.index before i /\
      Seq.index (halved before half i) (i + half) == Seq.index before (i + half) /\
      Seq.upd (halved before half i) i (F.add (Seq.index before i) (Seq.index before (i + half))) ==
        halved before half (i + 1)) =
  assert (Seq.equal
    (Seq.upd (halved before half i) i (F.add (Seq.index before i) (Seq.index before (i + half))))
    (halved before half (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (halved before half i) i (F.add (Seq.index before i) (Seq.index before (i + half))))
    (halved before half (i + 1))

inline_for_extraction noextract
fn halve
  (#capacity:pos{capacity <= 256})
  (partial:larray FP.t capacity) (half:sz{0 < SZ.v half /\ 2 * SZ.v half <= capacity})
  (#before:erased (lseq FP.t capacity))
  preserves gpu
  requires partial |-> before
  ensures partial |-> halved before (SZ.v half) (SZ.v half)
{
  halved_zero before (SZ.v half);
  let mut i = 0sz;
  while (!i <^ half)
    invariant exists* (j:sz{SZ.v j <= SZ.v half}).
      i |-> j ** partial |-> halved before (SZ.v half) (SZ.v j)
    invariant gpu
    decreases (SZ.v half - SZ.v !i)
  {
    let j = !i;
    halved_step before (SZ.v half) (SZ.v j);
    let left = partial.(j);
    let right = partial.(SZ.add j half);
    partial.(j) <- F.add left right;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
let halved_tree (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (n:pos{Norm.tree_size n /\ 1 < n /\ n <= capacity})
  : Lemma (
      Norm.tree (n / 2) (values (halved before (n / 2) (n / 2))) ==
        Norm.tree n (values before)) =
  Norm.half_size n;
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.tree"]] (Norm.tree n (values before));
  Prior.tree_ext (n / 2) (values (halved before (n / 2) (n / 2)))
    (Norm.paired_float (n / 2) (values before))

inline_for_extraction noextract
let stride_half (n:pos{Norm.tree_size (2 * n)})
  : Lemma (
      Norm.tree_size n /\
      (if n / 2 = 0 then 1 else 2 * (n / 2)) == n /\
      (n / 2 = 0 \/ Norm.tree_size (2 * (n / 2)))) =
  Norm.half_size (2 * n);
  if n > 1 then Norm.half_size n

inline_for_extraction noextract
fn tree
  (#capacity:pos{capacity <= 256})
  (partial:larray FP.t capacity) (size:sz{0 < SZ.v size /\ Norm.tree_size (SZ.v size) /\ SZ.v size <= capacity})
  (#before:erased (lseq FP.t capacity))
  preserves gpu
  requires partial |-> before
  returns result:FP.t
  ensures exists* (after:lseq FP.t capacity).
    partial |-> after ** pure (
      result == Norm.tree (SZ.v size) (values before) /\ result == Seq.index after 0)
{
  let mut stride = size /^ 2sz;
  while (0sz <^ !stride)
    invariant exists* (n:sz{SZ.v n <= SZ.v size / 2 /\
      (SZ.v n = 0 \/ Norm.tree_size (2 * SZ.v n))}) (current:lseq FP.t capacity).
      stride |-> n ** partial |-> current **
      pure (Norm.tree (if SZ.v n = 0 then 1 else 2 * SZ.v n) (values current) ==
        Norm.tree (SZ.v size) (values before))
    invariant gpu
    decreases (SZ.v !stride)
  {
    let n = !stride;
    with current. assert partial |-> current;
    Norm.half_size (2 * SZ.v n);
    halve partial n;
    halved_tree current (2 * SZ.v n);
    stride_half (SZ.v n);
    stride := n /^ 2sz;
  };
  partial.(0sz)
}

inline_for_extraction noextract
let rec linear_value (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (count:pos{count <= capacity}) : GTot FP.t (decreases count) =
  if count = 1 then Seq.index before 0
  else F.add (linear_value before (count - 1)) (Seq.index before (count - 1))

inline_for_extraction noextract
let linear_values (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (count:pos{count <= capacity}) : GTot (lseq FP.t capacity) =
  Seq.upd before 0 (linear_value before count)

inline_for_extraction noextract
let linear_initial (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  : Lemma (linear_values before 1 == before) =
  assert (Seq.equal (linear_values before 1) before);
  Seq.lemma_eq_elim (linear_values before 1) before

inline_for_extraction noextract
let linear_step (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (count:pos{count < capacity})
  : Lemma (
      Seq.index (linear_values before count) count == Seq.index before count /\
      Seq.upd (linear_values before count) 0
        (F.add (linear_value before count) (Seq.index before count)) ==
      linear_values before (count + 1)) =
  assert (Seq.equal
    (Seq.upd (linear_values before count) 0
      (F.add (linear_value before count) (Seq.index before count)))
    (linear_values before (count + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (linear_values before count) 0
      (F.add (linear_value before count) (Seq.index before count)))
    (linear_values before (count + 1))

inline_for_extraction noextract
fn linear
  (#capacity:pos{capacity <= 256})
  (partial:larray FP.t capacity) (size:sz{0 < SZ.v size /\ SZ.v size <= capacity})
  (#before:erased (lseq FP.t capacity))
  preserves gpu
  requires partial |-> before
  returns result:FP.t
  ensures partial |-> linear_values before (SZ.v size)
  ensures pure (result == linear_value before (SZ.v size))
{
  linear_initial before;
  let mut i = 1sz;
  while (!i <^ size)
    invariant exists* (j:sz{1 <= SZ.v j /\ SZ.v j <= SZ.v size}).
      i |-> j ** partial |-> linear_values before (SZ.v j)
    invariant gpu
    decreases (SZ.v size - SZ.v !i)
  {
    let j = !i;
    linear_step before (SZ.v j);
    let old = partial.(0sz);
    let next = partial.(j);
    partial.(0sz) <- F.add old next;
    i := SZ.add j 1sz;
  };
  partial.(0sz)
}

inline_for_extraction noextract
let rec linear_approx (#capacity:pos{capacity <= 256}) (before:lseq FP.t capacity)
  (logical:S.vector 256) (count:pos{count <= capacity})
  : Lemma
      (requires forall (i:nat{i < count}). Approx.v_approximates (Seq.index before i) (logical i))
      (ensures Approx.v_approximates (linear_value before count) (Math.partial_sum count logical))
      (decreases count) =
  if count = 1 then Math.partial_sum_one logical
  else (
    linear_approx before logical (count - 1);
    Math.bounded_sum_step #256 (count - 1) logical;
    Approx.a_add (linear_value before (count - 1)) (Seq.index before (count - 1))
      (Math.partial_sum (count - 1) logical) (logical (count - 1))
  )
