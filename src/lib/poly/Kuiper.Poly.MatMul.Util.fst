module Kuiper.Poly.MatMul.Util

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA |}
  {| clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    M.gpu_matrix_pts_to gA #fA eA **
    M.gpu_matrix_pts_to gB #fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single #et #_ #rows #shared #cols eA eB i j shared)
{
  let mut k : sz = 0sz;
  let mut sum : et = zero #et #_;

  while (let vk = !k; SZ.(vk <^ shared))
    invariant b.
      exists* (vk : SZ.t{vk <= shared}).
        pure (0 <= shared /\ b == (SZ.v vk < shared) /\ vk <= shared /\ vk >= 0) **
        pts_to k vk **
        pts_to #_ #et sum (MS.matmul_single eA eB i j vk) **
        M.gpu_matrix_pts_to gA #fA eA **
        M.gpu_matrix_pts_to gB #fB eB **
        gpu
  {
    let vk = !k;
    let s = !sum;
    let v1 = M.gpu_matrix_read gA i vk;
    let v2 = M.gpu_matrix_read gB vk j;

    sum := s `add` mul v1 v2;
    k := SZ.add vk 1sz;

    (**)MS.matmul_single_lemma eA eB i j (vk + 1);
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
  {| clayout4 lA |}
  {| clayout4 lB |}
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
    m4_pts_to gA #fA eA **
    m4_pts_to gB #fB eB
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
        m4_pts_to gA #fA eA **
        m4_pts_to gB #fB eB **
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
  {| clayout4 lA |}
  {| clayout4 lB |}
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
    m4_pts_to gA #fA eA **
    m4_pts_to gB #fB eB
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
        m4_pts_to gA #fA eA **
        m4_pts_to gB #fB eB **
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
