module Klas.SPMM.Inst

open Kuiper
module MU = Kuiper.Kernel.GEMM.Util
open Kuiper.Sparse
open Kuiper.Sparse.Approx
open Kuiper.Sparse.SPMM
open Kuiper.Sparse.SPMM.Defs { size_req }
module Math = Kuiper.Sparse.Math
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.EMatrix
open Kuiper.Array.Vectorized
open Kuiper.Array2.Strided
module MS = Kuiper.Spec.GEMM

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper.Kernel.Base

(* Asynchronous, stream-parametric instantiation. See Kuiper.Sparse.SPMM.spmm_on. *)
inline_for_extraction noextract
fn inst_on
  (et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (rows shared cols : szp)
  (gA : csr_matrix et (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et (rm shared cols) { is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#fB : perm)
  (gC : array2 et (rm rows cols) { is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols)
  (#eC : chest2 et rows cols)
  //(#_ : size_req rows shared cols)
  (s : stream_t)
  (#e : epoch_t)
  norewrite
  preserves cpu ** stream_live s
  requires
    pure (eB %~ rB) **
    pure (chunk et /? cols) **
    epoch_live s e **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (live gC) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_done s (epoch_next e)) (on gpu_loc (
      csr_matrix_approx gA #fA eA rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.matmul rA rB)
    ))
{
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;
  Math.prod_divides blockWidth (chunk et) blockItemsX;
  assert pure (chunk et /? cols);
  assert pure (chunk et /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v shared) #(SZ.v cols) (chunk et);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v rows) #(SZ.v cols) (chunk et);

  spmm_on #et
    rows shared cols
    blockItemsK blockItemsX blockWidth
    (blockItemsX /^ blockWidth)
    gA row_indices fri gB gC
    #_ #_ #eA #rA row_perm #eB #rB #eC
    s
}

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (rows shared cols : szp)
  (gA : csr_matrix et (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et (rm shared cols) { is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#fB : perm)
  (gC : array2 et (rm rows cols) { is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols)
  (#eC : chest2 et rows cols)
  //(#_ : size_req rows shared cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (eB %~ rB) **
    pure (chunk et /? cols) **
    on gpu_loc (live gC) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures on gpu_loc (matrix_approx gC (MS.matmul rA rB))
{
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;
  Math.prod_divides blockWidth (chunk et) blockItemsX;
  assert pure (chunk et /? cols);
  assert pure (chunk et /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v shared) #(SZ.v cols) (chunk et);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v rows) #(SZ.v cols) (chunk et);

  spmm #et
    rows shared cols
    blockItemsK blockItemsX blockWidth
    (blockItemsX /^ blockWidth)
    gA row_indices fri gB gC
    #_ #_ #eA #rA
    row_perm
    #eB #rB #eC
}

(* Four-type instantiations: A, B, C, and accumulator, respectively. *)
(* Asynchronous, stream-parametric instantiation. See Kuiper.Sparse.SPMM.spmm_on. *)
inline_for_extraction noextract
fn ginst_on
  (ta tb tc tacc : Type0)
  {| scalar ta, sized ta, has_vec_cpy ta, real_like ta,
     scalar tb, sized tb, has_vec_cpy tb, real_like tb,
     scalar tc, sized tc, has_vec_cpy tc, real_like tc,
     scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (rmapA : real -> real { MU.approx1 mapA rmapA /\ rmapA 0.0R == 0.0R })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (comb : tc -> tacc -> tc)
  (rcomb : real -> real -> real { approx2 comb rcomb })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk ta) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk tc) /? blockItemsX
  }))
  (rows shared cols : szp)
  (gA : csr_matrix ta (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 tb (rm shared cols) { is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#fB : perm)
  (gC : array2 tc (rm rows cols) { is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  // matriz sparse gA
  (#eA : chest2 ta rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 tb shared cols)
  (#rB : chest2 real shared cols)
  (#rC : chest2 real rows cols)
  //(#_ : size_req rows shared cols)
  (s : stream_t)
  (#e : epoch_t)
  norewrite
  preserves cpu ** stream_live s
  requires
    pure (eB %~ rB) **
    pure (chunk tc /? cols) **
    epoch_live s e **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (matrix_approx gC rC) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_done s (epoch_next e)) (on gpu_loc (
      csr_matrix_approx gA #fA eA rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.gmmcomb rmapA rmapB rcomb rC rA rB)
    ))
{
  tensor_pts_to_ref_located gB;
  map_loc gpu_loc
    #(matrix_approx gC rC)
    #(matrix_approx gC rC ** pure (SZ.fits (tlayout_ulen (rm rows cols))))
    fn _ {
      unfold matrix_approx gC rC;
      with actual. assert gC |-> actual;
      tensor_pts_to_ref gC;
      fold matrix_approx gC rC;
    };
  Math.prod_divides blockWidth (chunk tc) blockItemsX;
  assert pure (chunk tc /? cols);
  assert pure (chunk tc /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v shared) #(SZ.v cols) (chunk tc);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v rows) #(SZ.v cols) (chunk tc);

  gspmm_on mapA mapB rmapA rmapB comb rcomb
    rows shared cols
    blockItemsK blockItemsX blockWidth
    (blockItemsX /^ blockWidth)
    gA row_indices fri gB gC
    #_ #_ #eA #rA row_perm #eB #rB #rC
    s
}

inline_for_extraction noextract
fn ginst
  (ta tb tc tacc : Type0)
  {| scalar ta, sized ta, has_vec_cpy ta, real_like ta,
     scalar tb, sized tb, has_vec_cpy tb, real_like tb,
     scalar tc, sized tc, has_vec_cpy tc, real_like tc,
     scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (rmapA : real -> real { MU.approx1 mapA rmapA /\ rmapA 0.0R == 0.0R })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (comb : tc -> tacc -> tc)
  (rcomb : real -> real -> real { approx2 comb rcomb })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk ta) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk tc) /? blockItemsX
  }))
  (rows shared cols : szp)
  (gA : csr_matrix ta (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 tb (rm shared cols) { is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#fB : perm)
  (gC : array2 tc (rm rows cols) { is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  // matriz sparse gA
  (#eA : chest2 ta rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 tb shared cols)
  (#rB : chest2 real shared cols)
  (#rC : chest2 real rows cols)
  //(#_ : size_req rows shared cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (eB %~ rB) **
    pure (chunk tc /? cols) **
    on gpu_loc (matrix_approx gC rC) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures on gpu_loc (matrix_approx gC (MS.gmmcomb rmapA rmapB rcomb rC rA rB))
{
  tensor_pts_to_ref_located gB;
  map_loc gpu_loc
    #(matrix_approx gC rC)
    #(matrix_approx gC rC ** pure (SZ.fits (tlayout_ulen (rm rows cols))))
    fn _ {
      unfold matrix_approx gC rC;
      with actual. assert gC |-> actual;
      tensor_pts_to_ref gC;
      fold matrix_approx gC rC;
    };
  Math.prod_divides blockWidth (chunk tc) blockItemsX;
  assert pure (chunk tc /? cols);
  assert pure (chunk tc /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v shared) #(SZ.v cols) (chunk tc);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v rows) #(SZ.v cols) (chunk tc);

  gspmm mapA mapB rmapA rmapB comb rcomb
    rows shared cols
    blockItemsK blockItemsX blockWidth
    (blockItemsX /^ blockWidth)
    gA row_indices fri gB gC
    #_ #_ #eA #rA
    row_perm
    #eB #rB #rC
}
