module Kuiper.Poly.MatMul.Tiled.SHMem

#lang-pulse

open Kuiper

module MS = Kuiper.Spec.MatMul
open Kuiper.EMatrix4
open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  mlayout4,
  clayout4
}

(* TODO: Fit this into the non-tiled types. *)

inline_for_extraction noextract
fn matmul_gpu
  (tile : szp)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : szp)
  (lA : mlayout4 mrows   mshared tile tile)
  (lB : mlayout4 mshared mcols   tile tile)
  (lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (tile * tile <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.matmul eA eB
