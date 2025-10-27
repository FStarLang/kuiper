module Kuiper.Poly.GEMM.TensorCore2D

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

open Kuiper.Matrix.Reprs
module R = Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.TensorCore2D.KernelDesc
// ^ Only opened here for `constraints`? If so would be nice
// to factor out.

inline_for_extraction noextract
val mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA, strided_row_major lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB, strided_row_major lB |}
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (#_ : squash (SZ.fits (rows * cols)))
  (#eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (nblk <= max_blocks))
  (#_ : squash (nthr <= max_threads))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** (exists* eC'. gC |-> eC'))
