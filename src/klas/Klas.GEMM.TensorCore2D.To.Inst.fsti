module Klas.GEMM.TensorCore2D.To.Inst
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Float.Casts { float_cast }
module MS = Kuiper.Spec.GEMM

module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn spec
  // specialize
  (et_ab et_acc et_cd : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * 32) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * 32) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * 32) - 1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * 32) - 1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * 32) <= max_threads))

  // do not specialize
  (rows shared cols : szp)
  (gA : array2 et_ab (rm rows shared) { is_global gA })
  (gB : array2 et_ab (rm shared cols) { is_global gB })
  (gC : array2 et_cd (rm rows cols) { is_global gC })
  (gD : array2 et_cd (rm rows cols) { is_global gD })
  (alpha beta : et_acc)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#eA : chest2 et_ab rows shared)
  (#eB : chest2 et_ab shared cols)
  (#eC : chest2 et_cd rows cols)
  (#fA #fB #fC : perm)
  preserves
    cpu **
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC)
  requires
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (live gD)
  ensures
    exists* eD'.
      on gpu_loc (gD |-> eD') **
      pure (eD' %~ MS.mmcomb
        (MS.rlincomb (to_real alpha) (to_real beta))
        (to_real_matrix eC)
        (to_real_matrix eA)
        (to_real_matrix eB))

