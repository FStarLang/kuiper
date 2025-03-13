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
