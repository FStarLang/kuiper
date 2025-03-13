module Kuiper.MatMul.Util

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
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
  (#f : perm)
  (i : szlt rows)
  (j : szlt cols)
  preserves
    gpu **
    M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
    M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB
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
        M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
        M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
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
