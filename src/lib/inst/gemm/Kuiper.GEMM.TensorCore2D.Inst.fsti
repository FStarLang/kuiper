module Kuiper.GEMM.TensorCore2D.Inst
#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.TensorCore
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

module SZ = FStar.SizeT

inline_for_extraction noextract
fn spec
  // specialize
  (et_ab et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /? bk))
  (#_ : squash (chunk et_ab /? bn))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * 32) -1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * 32) -1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * 32) <= max_threads))

  // do not specialize
  (rows shared cols : szp)
  (gA : gpu_matrix et_ab (row_major rows shared))
  (gB : gpu_matrix et_ab (row_major shared cols))
  (gC : gpu_matrix et_c (row_major rows cols))
  (#eA : ematrix et_ab rows shared)
  (#eB : ematrix et_ab shared cols)
  (#eC : ematrix et_c rows cols)
  (#fA #fB : perm)
  // non of these are are checked because the functions is only
  //  partially applied
  preserves
    cpu **
    // should be checked at runtime
    pure (rows * cols <= max_blocks) **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    gC |-> eC
  ensures
    (exists* eC'. gC |-> eC')
