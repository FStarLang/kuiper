module Kuiper.Poly.GEMM.OrigBlockTiling1D

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (bm bn bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
