module Kuiper.Kernel.GEMM.TensorCore

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

open Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = Kuiper.SizeT

inline_for_extraction noextract
val mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA, str_A : strided_row_major lA |}
  (gA : gpu_matrix et_ab lA { is_global gA })
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB, str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : gpu_matrix et_ab lB { is_global gB })
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (row_major rows cols) { is_global gC })
  (#_ : squash (SZ.fits (rows * cols)))
  (#eC : ematrix et_c rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  // WARNING the previous version was wrong, it was assuming that each
  //  thread computes tm*tk results similar to 2D-Blocktiling.
  // There is nothing that catches this.
  // (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  // correct: the amount of tensor core tiles in gC multiplied
  //  by the warp size (each warp computes one tile)
  (nthr : szp{SZ.v nthr == bm/tm*(bn/tn)*warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
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
