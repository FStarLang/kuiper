module KPlay.Proof.Base.LocalArray

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module FP = Kuiper.Float32
module SZ = Kuiper.SizeT
module P = KPlay.Proof.Shared.Projection
module Init = KPlay.Proof.Base.Initialization
module Norm = KPlay.Proof.Base.Norm
module Reduction = KPlay.Proof.Base.ReductionNative

inline_for_extraction noextract
let snapshot (capacity:pos) = lseq (option FP.t) capacity

inline_for_extraction noextract
let ready (#capacity:pos) (before:snapshot capacity) (count:nat{count <= capacity}) : prop =
  forall (i:nat{i < count}). Some? (Seq.index before i)

inline_for_extraction noextract
let present (value:option FP.t{Some? value}) : FP.t =
  let Some result = value in result

inline_for_extraction noextract
let prefix_values (#capacity:pos) (before:snapshot capacity) (count:pos{count <= capacity})
  (initialized:squash (ready before count)) : GTot (lseq FP.t count) =
  Seq.init_ghost count (fun i -> present (Seq.index before i))

inline_for_extraction noextract
let written (#capacity:pos) (#n:pos{n <= capacity}) (before:snapshot capacity)
  (values:P.vector FP.t n) (count:nat{count <= n}) : GTot (snapshot capacity) =
  Seq.init_ghost capacity (fun i -> if i < count then Some (values i) else Seq.index before i)

inline_for_extraction noextract
let written_zero (#capacity:pos) (#n:pos{n <= capacity}) (before:snapshot capacity) (values:P.vector FP.t n)
  : Lemma (written before values 0 == before) =
  assert (Seq.equal (written before values 0) before);
  Seq.lemma_eq_elim (written before values 0) before

inline_for_extraction noextract
let written_step (#capacity:pos) (#n:pos{n <= capacity}) (before:snapshot capacity)
  (values:P.vector FP.t n) (i:P.index n)
  : Lemma (Seq.upd (written before values i) i (Some (values i)) == written before values (i + 1)) =
  assert (Seq.equal (Seq.upd (written before values i) i (Some (values i))) (written before values (i + 1)));
  Seq.lemma_eq_elim (Seq.upd (written before values i) i (Some (values i))) (written before values (i + 1))

inline_for_extraction noextract
let written_ready (#capacity:pos) (#n:pos{n <= capacity}) (before:snapshot capacity)
  (values:P.vector FP.t n) (count:nat{count <= n})
  : Lemma (ready (written before values count) count) = ()

inline_for_extraction noextract
let after (#capacity:pos) (#n:pos{n <= capacity}) (before:snapshot capacity) (values:lseq FP.t n)
  : GTot (snapshot capacity) =
  Seq.init_ghost capacity (fun i -> if i < n then Some (Seq.index values i) else Seq.index before i)

inline_for_extraction noextract
let outside (count:nat) (i:nat) : prop = count <= i

inline_for_extraction noextract
let view (#capacity:pos) (buffer:larray FP.t capacity) (count:pos{count <= capacity}) : GTot (larray FP.t count) =
  A.gsub buffer 0 count

inline_for_extraction noextract
let view_address (#capacity:pos) (buffer:larray FP.t capacity) (count:pos{count <= capacity})
  : Lemma (A.base_of (view buffer count) == A.base_of buffer /\ A.offset_of (view buffer count) == A.offset_of buffer) =
  A.base_of_gsub buffer 0 count;
  A.offset_of_gsub buffer 0 count

inline_for_extraction noextract
fn borrow
  (#capacity:pos) (buffer:larray FP.t capacity) (count:sz{0 < SZ.v count /\ SZ.v count <= capacity})
  (#before:erased (snapshot capacity)) (#fraction:perm)
  (initialized:squash (ready before (SZ.v count)))
  preserves gpu
  requires A.pts_to_mask buffer #fraction before Init.all_cells
  returns prefix:larray FP.t (SZ.v count)
  ensures rewrites_to prefix (view buffer (SZ.v count))
  ensures A.pts_to_mask buffer #fraction before (outside (SZ.v count))
  ensures prefix |-> Frac fraction (prefix_values before (SZ.v count) initialized)
{
  let prefix = A.sub buffer 0sz (SZ.v count);
  A.mask_mext buffer #fraction #before
    #(fun i -> Init.all_cells i /\ ~(0 <= i /\ i < SZ.v count)) (outside (SZ.v count));
  A.from_mask prefix;
  with values. assert A.pts_to prefix #fraction values;
  A.pts_to_len prefix;
  assert pure (Seq.equal values (prefix_values before (SZ.v count) initialized));
  Seq.lemma_eq_elim values (prefix_values before (SZ.v count) initialized);
  prefix
}

ghost
fn restore
  (#capacity:pos) (buffer:larray FP.t capacity) (count:pos{count <= capacity})
  (#before:erased (snapshot capacity)) (#values:erased (lseq FP.t count)) (#fraction:perm)
  requires A.pts_to_mask buffer #fraction before (outside count)
  requires A.pts_to (view buffer count) #fraction values
  ensures A.pts_to_mask buffer #fraction (after before values) Init.all_cells
{
  A.to_mask (view buffer count);
  with masked. assert A.pts_to_mask (view buffer count) #fraction masked (fun _ -> True);
  let original : erased (Seq.seq (option FP.t)) = hide #(Seq.seq (option FP.t)) (reveal #(snapshot capacity) before);
  A.return_sub buffer #fraction #original #masked #(outside count) #(fun _ -> True) #0 #count;
  with result. assert A.pts_to_mask buffer #fraction result
    (fun i -> outside count i \/ (0 <= i /\ i < count));
  A.mask_mext buffer #fraction #result #(fun i -> outside count i \/ (0 <= i /\ i < count)) Init.all_cells;
  assert pure (Seq.equal result (after before values));
  Seq.lemma_eq_elim result (after before values);
  rewrite A.pts_to_mask buffer #fraction result Init.all_cells
    as A.pts_to_mask buffer #fraction (after before values) Init.all_cells;
}

inline_for_extraction noextract
fn initialized_reduce
  (#capacity:pos{capacity <= 256}) (buffer:larray FP.t capacity)
  (count:sz{0 < SZ.v count /\ SZ.v count <= capacity /\ Norm.tree_size (SZ.v count)})
  (linear:bool) (#before:erased (lseq FP.t capacity))
  preserves gpu
  requires buffer |-> before
  returns value:FP.t
  ensures exists* (values:lseq FP.t capacity). buffer |-> values **
    pure (Seq.index values 0 == value /\
      value == (if linear then Reduction.linear_value before (SZ.v count)
        else Norm.tree (SZ.v count) (Reduction.values before)))
{
  if linear {
    let value = Reduction.linear buffer count;
    value
  } else {
    let value = Reduction.tree buffer count;
    value
  }
}

inline_for_extraction noextract
fn reduce
  (#capacity:pos) (buffer:larray FP.t capacity)
  (count:sz{0 < SZ.v count /\ SZ.v count <= 256 /\ SZ.v count <= capacity /\ Norm.tree_size (SZ.v count)})
  (linear:bool) (#before:erased (snapshot capacity)) (initialized:squash (ready before (SZ.v count)))
  preserves gpu
  requires A.pts_to_mask buffer before Init.all_cells
  returns value:FP.t
  ensures exists* (values:lseq FP.t (SZ.v count)).
    A.pts_to_mask buffer (after before values) Init.all_cells **
    pure (Seq.index values 0 == value /\
      value == (if linear then Reduction.linear_value (prefix_values before (SZ.v count) initialized) (SZ.v count)
        else Norm.tree (SZ.v count) (Reduction.values (prefix_values before (SZ.v count) initialized))))
{
  let prefix = borrow buffer count #before #1.0R initialized;
  let value = initialized_reduce prefix count linear;
  with values. assert prefix |-> values;
  assert pure (Seq.index values 0 == value);
  rewrite A.pts_to prefix values as A.pts_to (view buffer (SZ.v count)) values;
  restore buffer (SZ.v count) #before #values #1.0R;
  value
}
