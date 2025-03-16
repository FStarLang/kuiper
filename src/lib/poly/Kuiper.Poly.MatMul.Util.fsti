module Kuiper.Poly.MatMul.Util

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix4 { ematrix4 }
open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

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
  (* ^ This takes a v0 and adds the products into it,
  to make sure we compute everything left-nested form,
  and hence have an exact result. *)
  preserves
    gpu **
    m4_pts_to gA #fA eA **
    m4_pts_to gB #fB eB
  returns
    res : et
  // ensures
  //   pure (res == MS.matmul_single #et #_ #(rows * tile) #(shared * tile) #(cols * tile) eA eB i j shared)

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
