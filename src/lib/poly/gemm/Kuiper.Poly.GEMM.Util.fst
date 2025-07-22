module Kuiper.Poly.GEMM.Util

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB)
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  let mut k : sz = 0sz;
  let mut sum : et = zero #et #_;

  while (SZ.(!k <^ shared))
    invariant b.
      exists* (vk : SZ.t{vk <= shared}).
        pure (b == (SZ.v vk < shared)) **
        (k |-> vk) **
        (sum |-> MS.__matmul_single eA eB i j vk) **
        (gA |-> Frac fA eA) **
        (gB |-> Frac fB eB) **
        gpu
  {
    let v1 = M.gpu_matrix_read gA i !k;
    let v2 = M.gpu_matrix_read gB !k j;

    sum := !sum `add` mul v1 v2;
    k := SZ.add !k 1sz;

    (**)MS.matmul_single_lemma eA eB i j !k;
    ();
  };
  !sum
}

(* Will only multiply across the minor index. *)
inline_for_extraction noextract
fn matmul_tiled_sub_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols #tile : SZ.t)
  (#lA : mlayout4 rows shared tile tile)
  (#lB : mlayout4 shared cols tile tile)
  {| clayout4 lA, clayout4 lB |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (#eA : ematrix4 et rows shared tile tile)
  (#eB : ematrix4 et shared cols tile tile)
  (bi : szlt rows)
  (bk : szlt shared)
  (bj : szlt cols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  (v0 : et)
  preserves
    gpu **
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB)
  returns
    res : et
  // ensures
  //   pure (res == MS.matmul_single #et #_ #(rows * tile) #(shared * tile) #(cols * tile) eA eB i j shared)
{
  let mut sum = v0;
  let mut k   : sz = 0sz;

  while (let vk = !k; SZ.(vk <^ tile))
    invariant b.
      exists* (vk : SZ.t{vk <= tile}) sumv.
        pure (0 <= tile /\ b == (SZ.v vk < tile) /\ vk <= tile /\ vk >= 0) **
        pts_to k vk **
        pts_to #_ #et sum sumv **
        (gA |-> Frac fA eA) **
        (gB |-> Frac fB eB) **
        gpu
  {
    let vk = !k;
    let s = !sum;
    let v1 = M4.gpu_matrix_read gA bi bk i vk;
    let v2 = M4.gpu_matrix_read gB bk bj vk j;

    sum := s `add` mul v1 v2;
    k := vk +^ 1sz;
  };
  !sum
}

inline_for_extraction noextract
fn matmul_tiled_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols #tile : SZ.t)
  (#lA : mlayout4 rows shared tile tile)
  (#lB : mlayout4 shared cols tile tile)
  {| clayout4 lA, clayout4 lB |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (#eA : ematrix4 et rows shared tile tile)
  (#eB : ematrix4 et shared cols tile tile)
  (bi : szlt rows)
  (bj : szlt cols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu **
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB)
  returns
    res : et
  // ensures
  //   pure (res == MS.matmul_single #et #_ #(rows * tile) #(shared * tile) #(cols * tile) eA eB i j shared)
{
  let mut sum : et = zero #et #_;
  let mut bk  : sz = 0sz;

  while (let vbk = !bk; SZ.(vbk <^ shared))
    invariant b.
      exists* (vbk : SZ.t{vbk <= shared}) sumv.
        pure (0 <= shared /\ b == (SZ.v vbk < shared) /\ vbk <= shared /\ vbk >= 0) **
        pts_to bk vbk **
        pts_to #_ #et sum sumv **
        (gA |-> Frac fA eA) **
        (gB |-> Frac fB eB) **
        gpu
  {
    let vbk = !bk;
    let s = !sum;
    let s' = matmul_tiled_sub_dotprod gA gB bi vbk bj i j s;
    sum := s';
    bk := vbk +^ 1sz;
  };
  !sum
}

(* Used by SHMEM, Blocktiling1D *)
inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 : mlayout tile tile) {| clayout l1 |}
  (#l2 : mlayout tile tile) {| clayout l2 |}
  (m1 : M.gpu_matrix et l1)
  (m2 : M.gpu_matrix et l2)
  (j : szlt tile)
  (#acc0 : erased (seq et))
  (#v1 #v2 : ematrix et tile tile)
  (#f : perm)
  preserves
    gpu **
    (m1 |-> Frac f v1) **
    (m2 |-> Frac f v2)
  requires
    pure (Seq.length acc0 == tile) **
    (acc |-> acc0)
  ensures
    exists* acc'.
      pure (Seq.length acc' == tile) **
      (acc |-> acc')
{
  let mut sk : sz = 0sz;
  while (SZ.(!sk <^ tile))
    invariant b.
      exists* (vsk : SZ.t{vsk <= tile}) (accv : erased (lseq et tile)).
        pure (b == (SZ.v vsk < tile)) **
        (sk |-> vsk) **
        (acc |-> accv) **
        (m1 |-> Frac f v1) **
        (m2 |-> Frac f v2) **
        gpu
  {
    let mut i = 0sz;
    (* We can read v2 out of the inner loop, this is extremely
       important for performance. NVCC may realize this is invariant
       across iterations and hoist it out, but don't rely on it. *)
    let v2 = M.gpu_matrix_read m2 !sk j;
    while (SZ.(!i <^ tile))
      invariant b.
        exists* (vi : SZ.t{vi <= tile}) (accv : erased (lseq et tile))
          (vsk : SZ.t{vsk < tile}).
          pure (b == (SZ.v vi < tile)) **
          (i |-> vi) **
          (sk |-> vsk) **
          (acc |-> accv) **
          (m1 |-> Frac f v1) **
          gpu
    {
      let v1 = M.gpu_matrix_read m1 !i !sk;

      open Pulse.Lib.Array;
      let sum0 = acc.(!i);
      let sum1 = sum0 `add` (v1 `mul` v2);
      acc.(!i) <- sum1;
      i := !i +^ 1sz;
    };
    sk := !sk +^ 1sz;
  }
}
