module Klas.GEMM.TensorCore.Inst

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.TensorCore

inline_for_extraction noextract
fn specialize_gpu
  // specialize
  (et_ab et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  // should be up here! if part of the precondition, then
  //  the value is not checked for correctness when
  //  the function is only partially applied!
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm * bk)))
  (#_ : squash (SZ.fits (bk * bn)))
  (#_ : squash (bm/tm * bn/tn * warp_size <= max_threads))
  (#_ : squash (SZ.fits (bm*bk + bm/tm * bn/tn * warp_size)))
  (#_ : squash (SZ.fits (bk*bn + bm/tm * bn/tn * warp_size)))

  // do not specialize
  (rows shared cols : szp)
  (gA : gpu_matrix et_ab (row_major rows shared) { is_global_matrix gA })
  (gB : gpu_matrix et_ab (row_major shared cols) { is_global_matrix gB })
  (gC : gpu_matrix et_c (row_major rows cols) { is_global_matrix gC })
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (chunk et_ab * ((bm/tm) * (bn/tn) * warp_sz) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * ((bm/tm) * (bn/tn) * warp_sz) /?+ (bk * bn)))
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
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    on gpu_loc (gC |-> eC)
  ensures
    (exists* eC'. on gpu_loc (gC |-> eC'))
