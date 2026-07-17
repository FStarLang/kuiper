module Kuiper.Sparse.SPMM.Compute

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common { (@+) }
open Kuiper.Spec.GEMM
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Common
open Kuiper.Sparse.SPMM.Defs { ematrix_tile_prop }
open Kuiper.Array.Vectorized
open Kuiper.Tensor { ctlayout }
module A = Kuiper.Array1
module M = Kuiper.Array2


val tile_vmprod_prop
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (acc : erased (lseq et n1))
  (elems : erased (lseq et m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : ematrix et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (y : lseq et n1)
  : prop

val tile_vmprod_prop_lemma0
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (acc : erased (lseq et n1))
  (elems : erased (lseq et m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : ematrix et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
: Lemma
  (requires m1 == 0)
  (ensures tile_vmprod_prop acc elems row_ind em2 j step acc)

val tile_mask_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat { chunk et /? cols })
  (em1 : ematrix et rows shared)
  (i : natlt rows)
  (#nnz : nat)
  (mask_len : natle nnz)
  (elems : erased (lseq et (nnz - mask_len)))
  (row_ind : erased (lseq nat nnz))
  (#_ : squash (in_bounds 0 shared row_ind /\ sorted row_ind))
  (em2 : ematrix et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (#tlen : nat { chunk et /? tlen })
  (tile0 : lseq et tlen)
  (tile : lseq et tlen)
: Lemma
  (requires
    tile_vmprod_prop
      tile0
      (Seq.create mask_len zero @+ elems) row_ind
      em2 j step tile
  )
  (ensures
    tile_vmprod_prop #_ #_ #_ #solve
      tile0
      elems (Seq.slice row_ind mask_len nnz)
      em2 j step tile
  )

// TODO reemplazar esto con is_ematrix_tile del producto
let tile_result_cell_prop
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat)
  (em1 : ematrix et rows shared)
  (i : natlt rows)
  (em2 : ematrix et shared cols)
  (j : nat)
  (step : nat)
  (#tlen : nat)
  (tile : lseq et tlen)
  (k1 : natlt tlen)
: prop
=
  let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
  k2 < cols ==>
  tile @! k1 == matmul_single em1 em2 i k2

let tile_result_prop
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat)
  (em1 : ematrix et rows shared)
  (i : natlt rows)
  (em2 : ematrix et shared cols)
  (j : nat)
  (step : nat)
  (#tlen : nat)
  (tile : lseq et tlen)
: prop
=
  forall (k1 : natlt tlen).
    tile_result_cell_prop em1 i em2 j step tile k1

val tile_result_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat { chunk et /? cols })
  (em1 : ematrix et rows shared)
  (i : natlt rows)
  (#nnz : nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (#_ : squash (in_bounds 0 shared row_ind /\ sorted row_ind))
  (em2 : ematrix et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (#tlen : nat { chunk et /? tlen })
  (tile : lseq et tlen)
: Lemma
  (requires
    unsparse _ _ elems row_ind == ematrix_row em1 i /\
    tile_vmprod_prop
      (Seq.create tlen zero)
      elems row_ind
      em2
      j step
      tile
  )
  (ensures tile_result_prop #_ #_ #_ #solve em1 i em2 j step tile)

inline_for_extraction noextract
fn tile_vmprod
  (#et : Type0) {| scalar et, sized et, hvc : has_vec_cpy et |}
  (#m1 #n1 : sz { chunk et /? n1 })
  (#ly : A.layout n1) {| ctlayout ly |}
  (y : A.array1 et ly)
  (#vy : erased (lseq et n1))
  (vy0 : erased (lseq et n1))
  (#lx : A.layout m1) {| ctlayout lx |}
  (x : A.array1 et lx)
  (#fx : perm)
  (#nnz : erased nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (to : erased nat { to + m1 <= nnz })
  (#ltm : M.layout m1 n1) {| ctlayout ltm |}
  (tm : M.array2 et ltm)
  (#tem : ematrix et m1 n1)
  (#ftm : perm)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (gem : ematrix et m2 n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (#_ : squash (in_bounds 0 m2 row_ind /\ sorted row_ind))
  norewrite
  preserves gpu
  preserves x  |-> Frac fx (Seq.slice elems to (to + m1) <: lseq et m1)
  preserves tm |-> Frac ftm tem
  requires  pure (ematrix_tile_prop #_ #_ #hvc gem (Seq.slice row_ind to (to + m1)) j step tem)
  requires  y  |-> vy
  requires
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice elems 0 to <: lseq et to) (Seq.slice row_ind 0 to)
        gem
        j step
        vy
    )
  ensures exists* (vy' : lseq et n1).
    y |-> vy' **
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice elems 0 (to + m1) <: lseq et (to + m1)) (Seq.slice row_ind 0 (to + m1))
        gem
        j step
        vy'
    )

open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }

inline_for_extraction noextract
fn tile_load_vmprod
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : sz { chunk et /? n1 })
  (#ly : A.layout n1) {| ctlayout ly |}
  // en realidad y es un larray... por el momento no podemos unificar
  (y : A.array1 et ly)
  (#vy : erased (lseq et n1))
  (vy0 : erased (lseq et n1))
  (#lx  : A.layout m1) {| ctlayout lx |}
  (elems : A.array1 et lx)
  (row_ind : A.array1 sz lx)
  (#fx : perm)
  (#nnz : erased nat)
  (#velems : lseq et nnz)
  (#vrow_ind : lseq sz nnz)
  (#m2 #n2 : szp { chunk et /? n2 })
  (#lm : M.layout m2 n2) {| ctlayout lm, srm : strided_row_major lm |}
  (m : M.array2 et lm)
  (#fm : perm)
  (#em : ematrix et m2 n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (#_ : squash (in_bounds 0 m2 (cast_pos vrow_ind)))
  (from to : erased nat { from + m1 <= nnz })
  (cant : szlt m1 { v cant == to - from })
  preserves gpu
  preserves elems   |-> Frac fx (Seq.slice velems from (from + m1) <: lseq et m1)
  preserves row_ind |-> Frac fx (Seq.slice vrow_ind from (from + m1) <: lseq sz m1)
  preserves m |-> Frac fm em
  requires  pure (aligned 16 (M.core m) /\ aligned_strided_row_major (chunk et) srm)
  requires  pure (fits (j + n1 * step))
  requires  y |-> vy
  requires
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice velems 0 from <: lseq et from)
        (Seq.slice (cast_pos vrow_ind) 0 from)
        em
        j step
        vy
    )
  ensures exists* vy'.
    y |-> vy' **
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice velems 0 to <: lseq et to)
        (Seq.slice (cast_pos vrow_ind) 0 to <: lseq nat to)
        em j step vy'
    )
