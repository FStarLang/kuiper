module Klas.GEMM.BlockTiling2D.Inst

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm }

module M = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM

unfold
let sz_fits (x:int) = 0 <= x /\ x < 4294967296

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  // These are fixed, by this function, to column-major and row-major
  // respectively, which is the more efficient thing to do.
  // (slA : full_mlayout bm bk)
  // (slB : full_mlayout bk bn)
  // {| clayout slA, clayout slB |}
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
  (gA : M.gpu_matrix et (rm rows shared) { M.is_global_matrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et (rm shared cols) { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et (rm rows cols) { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (M.core gA)) **
    pure (aligned 16 (M.core gB)) **
    pure (rows * cols <= max_blocks) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gemm alpha beta eC eA eB)