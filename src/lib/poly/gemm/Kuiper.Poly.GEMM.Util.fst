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
  let mut sum : et = zero;

  while (SZ.(!k <^ shared))
    invariant
      exists* (vk : SZ.t{vk <= shared}).
        (k |-> vk) **
        (sum |-> MS.__matmul_single eA eB i j vk)
  {
    let v1 = M.gpu_matrix_read gA i !k;
    let v2 = M.gpu_matrix_read gB !k j;

    let vsum = !sum;
    sum := vsum `add` mul v1 v2;
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
  let mut k : sz = 0sz;

  while (SZ.(!k <^ tile))
    invariant
      exists* (vk : SZ.t) sumv.
        pure (vk <= tile) **
        (k |-> vk) **
        (sum |-> sumv)
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
  let mut sum : et = zero;
  let mut bk  : sz = 0sz;

  while (SZ.(!bk <^ shared))
    invariant
      exists* (vbk : SZ.t) sumv.
        pure (vbk <= shared) **
        (bk |-> vbk) **
        (sum |-> sumv)
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
    invariant
      exists* (vsk : SZ.t) (accv : erased (lseq et tile)).
        pure (vsk <= tile) **
        (sk |-> vsk) **
        (acc |-> accv)
  {
    let mut i = 0sz;
    (* We can read v2 out of the inner loop, this is extremely
       important for performance. NVCC may realize this is invariant
       across iterations and hoist it out, but don't rely on it. *)
    let v2 = M.gpu_matrix_read m2 !sk j;
    while (SZ.(!i <^ tile))
      invariant
        exists* (vi : SZ.t{vi <= tile}) (accv : erased (lseq et tile)).
          (i |-> vi) **
          (acc |-> accv)
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
