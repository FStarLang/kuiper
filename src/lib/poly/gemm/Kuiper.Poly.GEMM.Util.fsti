module Kuiper.Poly.GEMM.Util

#lang-pulse

open Kuiper
open Kuiper.Approximates
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling { ematrix_subtile }

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)

(* Real-valued matrix for specification purposes *)
let ematrix_to_real (#et:Type) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  : GTot (ematrix real rows cols)
  = mkM (fun i j -> to_real (macc em i j))

(* Real-valued matmul_single using real arithmetic (which IS associative) *)
let real_matmul_single
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  (row : natlt rows)
  (col : natlt cols)
  : GTot real
  = MS.__gmatmul_single 0.0R ( *. ) ( +. )
      (ematrix_to_real m1) (ematrix_to_real m2)
      row col shared

(* Real-valued gemm_single: combines initial value with real matmul using comb_r *)
let real_gemm_single
  (comb_r : binop real)
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  (m0 : ematrix et rows cols)
  (row : natlt rows)
  (col : natlt cols)
  : GTot real
  = comb_r (to_real (macc m0 row col)) (real_matmul_single m1 m2 row col)

(* Real-valued GEMM matrix: each cell is real_gemm_single *)
let real_mmcomb
  (comb_r : binop real)
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m0 : ematrix et rows cols)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  : GTot (ematrix real rows cols)
  = mkM (fun i j -> real_gemm_single comb_r m1 m2 m0 i j)

(* Partial real dot product over tiled matrices, summing first `to` elements *)
let __real_matmul_single_tiled
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols #tile : nat)
  (m1 : ematrix et (rows * tile) (shared * tile))
  (m2 : ematrix et (shared * tile) (cols * tile))
  (row : natlt (rows * tile))
  (col : natlt (cols * tile))
  (to : nat{to <= shared * tile})
  : GTot real
  = MS.__gmatmul_single 0.0R ( *. ) ( +. )
      (ematrix_to_real m1) (ematrix_to_real m2)
      row col to

(* Real-valued matmul_single for a subtile *)
let real_matmul_single_subtile
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols #tile : nat)
  (m1 : ematrix et (rows * tile) (shared * tile))
  (m2 : ematrix et (shared * tile) (cols * tile))
  (bi : natlt rows) (bj : natlt cols) (bk : natlt shared)
  (i : natlt tile) (j : natlt tile)
  : GTot real
  = MS.__gmatmul_single 0.0R ( *. ) ( +. )
      (ematrix_to_real (ematrix_subtile m1 tile tile bi bk))
      (ematrix_to_real (ematrix_subtile m2 tile tile bk bj))
      i j tile

(* Splitting partial sum over real matrices:
   sum(0 to base+n) = sum(0 to base) + sum over subtile elements *)
val __gmatmul_single_split
  (#rows #shared #cols : nat)
  (m1 : ematrix real rows shared)
  (m2 : ematrix real shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (base : nat{base <= shared})
  (n : nat{base + n <= shared})
  (#sub_n : nat{n <= sub_n})
  (sub_m1 : ematrix real sub_n sub_n)
  (sub_m2 : ematrix real sub_n sub_n)
  (sub_row : natlt sub_n)
  (sub_col : natlt sub_n)
  : Lemma
    (requires
      (forall (k:nat). k < n ==>
        macc sub_m1 sub_row k == macc m1 row (base + k) /\
        macc sub_m2 k sub_col == macc m2 (base + k) col))
    (ensures
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) m1 m2 row col (base + n)
      ==
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) m1 m2 row col base +.
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) sub_m1 sub_m2 sub_row sub_col n)

(* Stepping the tiled partial sum by one tile block *)
val __real_matmul_single_tiled_step
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (#tile : pos)
  (m1 : ematrix et (rows * tile) (shared * tile))
  (m2 : ematrix et (shared * tile) (cols * tile))
  (bi : natlt rows) (bj : natlt cols) (bk : nat{bk < shared})
  (i : natlt tile) (j : natlt tile)
  : Lemma
    (ensures (
      let row = bi * tile + i in
      let col = bj * tile + j in
      __real_matmul_single_tiled m1 m2 row col ((bk + 1) * tile)
      ==
      __real_matmul_single_tiled m1 m2 row col (bk * tile) +.
      real_matmul_single_subtile m1 m2 bi bj bk i j
    ))

(* Scalar matmul_single of subtile approximates the real version *)
val matmul_single_subtile_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (#tile : pos)
  (m1 : ematrix et (rows * tile) (shared * tile))
  (m2 : ematrix et (shared * tile) (cols * tile))
  (bi : natlt rows) (bj : natlt cols) (bk : natlt shared)
  (i : natlt tile) (j : natlt tile)
  : Lemma
    (ensures (
      MS.matmul_single (ematrix_subtile m1 tile tile bi bk)
                       (ematrix_subtile m2 tile tile bk bj)
                       i j
      %~ real_matmul_single_subtile m1 m2 bi bj bk i j
    ))

(* General matmul approximation: scalar matmul_single approximates real version.
   Generalization of matmul_single_subtile_approx to non-square matrices. *)
val matmul_single_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  (row : natlt rows)
  (col : natlt cols)
  : Lemma
    (ensures (
      MS.matmul_single m1 m2 row col
      %~ real_matmul_single m1 m2 row col
    ))

(* mmcomb approximation: exact mmcomb result approximates real-valued real_mmcomb *)
val mmcomb_approx
  (#et:Type) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real)
  (#rows #shared #cols : nat)
  (eC : ematrix et rows cols)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  : Lemma
    (requires approx2 comb comb_r)
    (ensures ematrix_approximates (MS.mmcomb comb eC eA eB) (real_mmcomb comb_r eC eA eB))

(* Approximation of partial matmul over external real matrices:
   if eA %~ rA and eB %~ rB then
   __gmatmul_single ... eA eB row col n %~ __gmatmul_single ... rA rB row col n *)
val __matmul_single_approx_real
  (#et:Type) {| scalar et |} {| real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (n : nat{n <= shared})
  : Lemma
    (requires eA %~ rA /\ eB %~ rB)
    (ensures
      MS.__gmatmul_single zero mul add eA eB row col n
      %~
      MS.__gmatmul_single #real #real 0.0R Kuiper.Scalars.mul Kuiper.Scalars.add rA rB row col n)

(* mmcomb approximation over external real matrices:
   If eA %~ rA, eB %~ rB, eC %~ rC, and approx2 comb comb_r,
   then mmcomb comb eC eA eB %~ mmcomb comb_r rC rA rB. *)
val mmcomb_approx_real
  (#et:Type) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real)
  (#rows #shared #cols : nat)
  (eC : ematrix et rows cols)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  : Lemma
    (requires approx2 comb comb_r /\
              eA %~ rA /\ eB %~ rB /\ eC %~ rC)
    (ensures MS.mmcomb comb eC eA eB %~ MS.mmcomb comb_r rC rA rB)

(* Version of matmul_tiled_dotprod with approximate postcondition *)
inline_for_extraction noextract
fn matmul_tiled_dotprod'
  (#et : Type0) {| scalar et, real_like et |}
  (#mrows #mshared #mcols #tile : szp)
  (#lA : mlayout (mrows * tile)   (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA #eB : ematrix et _ _)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res %~ real_matmul_single eA eB (bi * tile + i) (bj * tile + j))

(* Version of matmul_tiled_dotprod with external real matrices.
   Proves the result approximates the real-valued dot product over rA, rB. *)
inline_for_extraction noextract
fn matmul_tiled_dotprod_real
  (#et : Type0) {| scalar et, real_like et |}
  (#mrows #mshared #mcols #tile : szp)
  (#lA : mlayout (mrows * tile)   (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA #eB : ematrix et _ _)
  (rA : ematrix real (mrows * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols * tile))
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (eA %~ rA /\ eB %~ rB)
  returns
    res : et
  ensures
    pure (res %~ MS.matmul_single rA rB (bi * tile + i) (bj * tile + j))

(* Used by SHMEM, Blocktiling1D *)
inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 : mlayout tile tile) {| clayout l1 |}
  (#l2 : mlayout tile tile) {| clayout l2 |}
  (m1 : M.gpu_matrix et l1)
  (m2 : M.gpu_matrix et l2)
  (j : szlt tile)
  (#acc0 : erased (seq et))
  (#v1 #v2 : ematrix et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f v1 **
    m2 |-> Frac f v2
  requires
    pure (Seq.length acc0 == tile) **
    acc |-> acc0
  ensures
    exists* acc'.
      pure (Seq.length acc' == tile) **
      (acc |-> acc')
