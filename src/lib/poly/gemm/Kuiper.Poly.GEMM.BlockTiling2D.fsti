module Kuiper.Poly.GEMM.BlockTiling2D

#lang-pulse

open Kuiper
open Kuiper.Poly.GEMMGPU.Type
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| strided_row_major lA, strided_row_major lB |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (#eC : ematrix et rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_ : squash (chunk et /? bn))
  (#_ : squash (chunk et /? bk))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /? (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /? (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
