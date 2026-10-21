module Kuiper.Sparse.SPMM.Compute

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common { (@+) }
open Kuiper.Spec.GEMM
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Common
open Kuiper.Sparse.SPMM.Defs { chest2_tile_prop }
open Kuiper.Array.Vectorized
open Kuiper.Tensor
open Kuiper.Seq.Common { op_At_Bang }


val tile_mm_result
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : nat {  chunk et /? cols })
  (#ly : nat { chunk et /? ly })
  (y0 : erased (lseq et ly))
  (#lA : nat)
  (elems : erased (lseq et lA))
  (row_ind : erased (lseq nat lA) { in_bounds 0 shared row_ind })
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (y : lseq et ly)
  : prop

val tile_mm_lemma0
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : nat {  chunk et /? cols })
  (#ly : nat { chunk et /? ly })
  (y0 : erased (lseq et ly))
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
: Lemma (ensures tile_mm_result y0 (Seq.empty <: lseq et 0) Seq.empty eB j step y0)

val tile_mm_mask_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : nat { chunk et /? cols })
  (#ly : nat { chunk et /? ly })
  (y0 : lseq et ly)
  (nnz : nat)
  (mask_len : natle nnz)
  (elems : erased (lseq et (nnz - mask_len)))
  (row_ind : erased (lseq nat nnz) { in_bounds 0 shared row_ind })
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (y : lseq et ly)
: Lemma
  (requires tile_mm_result y0 (Seq.create mask_len zero @+ elems) row_ind eB j step y)
  (ensures  tile_mm_result y0 elems (Seq.slice row_ind mask_len nnz) eB j step y)

val tile_mm_result_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat { chunk et /? cols })
  (eA : chest2 et rows shared)
  (i : natlt rows)
  (#nnz : nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (#_ : squash (in_bounds 0 shared row_ind /\ sorted row_ind))
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (#ly : nat { chunk et /? ly })
  (y : lseq et ly)
: Lemma
  (requires
    seq_unsparse _ _ elems row_ind == ematrix_row eA i /\
    tile_mm_result
      (Seq.create ly zero)
      elems row_ind
      eB
      j step
      y
  )
  (ensures is_ematrix_tile (matmul eA eB) i j y step)

open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }

inline_for_extraction noextract
fn tile_mm
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : szp { chunk et /? cols } )
  (#ly : sz { chunk et /? ly })
  (y : larray et ly)
  (vy0 : erased (lseq et ly))
  (#vy : erased (lseq et ly))
  (#lA : sz) (elems : larray et lA) (row_ind : larray sz lA) (#fA : perm)
  (#nnz : erased nat)
  (velems : erased (lseq et nnz))
  (vrow_ind : erased (lseq sz nnz) { in_bounds 0 shared (cast_pos vrow_ind) })
  (#lB : layout2 shared cols) {| ctlayout lB, srm : strided_row_major lB |}
  (mB : array2 et lB) (#fB : perm) (#eB : chest2 et shared cols)
  (j : sz { chunk et /? j })
  (step : sz)
  (from to : erased nat { to <= nnz })
  (cant : szle lA { v cant == to - from })
  preserves gpu
  preserves pts_to_slice elems   #fA 0 cant (Seq.slice velems from to <: lseq et cant)
  preserves pts_to_slice row_ind #fA 0 cant (Seq.slice vrow_ind from to <: lseq sz cant)
  preserves mB |-> Frac fB eB
  requires  pure (aligned 16 (core mB) /\ aligned_strided_row_major (chunk et) srm)
  requires  pure (fits (j + ly * step))
  requires
    y |-> vy **
    pure (
      tile_mm_result
        vy0
        (Seq.slice velems 0 from <: lseq et from)
        (Seq.slice (cast_pos vrow_ind) 0 from)
        eB
        j step
        vy
    )
  ensures exists* (vy' : lseq et ly).
    y |-> vy' **
    pure (
      tile_mm_result
        vy0
        (Seq.slice velems 0 to <: lseq et to)
        (Seq.slice (cast_pos vrow_ind) 0 to <: lseq nat to)
        eB j step vy'
    )