module Klas.GEMM.BlockTiling2D.Inst

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm,
  c_l2_col_major, c_l2_row_major }

module MS = Kuiper.Spec.GEMM

unfold
let sz_fits (x:int) = 0 <= x /\ x < 4294967296

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  // These are fixed, by this function, to column-major and row-major
  // respectively, which is the more efficient thing to do.
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn /\ (bm/tm * bn/tn <= max_threads)})
  (#_ : squash (sz_fits (bm*bk + (bm/tm * (bn/tn)))))
  (#_ : squash (sz_fits (bk*bn + (bm/tm * (bn/tn)))))
  (et : Type0) {| scalar et, has_vec_cpy et |}
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (alpha beta : et)
  (#rows #shared #cols : szp)
  (gA : array2 et (rm rows shared) { is_global gA })
  (#fA : perm)
  (gB : array2 et (rm shared cols) { is_global gB })
  (#fB : perm)
  (gC : array2 et (rm rows cols) { is_global gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (rows * cols <= max_blocks) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gemm alpha beta eC eA eB)