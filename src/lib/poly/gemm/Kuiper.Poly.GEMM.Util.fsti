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

(* Lemma: __gmatmul_single with non-zero initial value approximates
   the real sum starting from the approximated initial value.
   If x %~ r, then __gmatmul_single x mul add m1 m2 i j n %~
   r +. __gmatmul_single 0.0R ( *. ) ( +. ) (to_real m1) (to_real m2) i j n *)
val gmatmul_single_init_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (x : et) (r : real)
  (m1 : ematrix et rows cols)
  (m2 : ematrix et cols rows)
  (row : natlt rows)
  (col : natlt rows)
  (n : nat{n <= cols})
  (_: squash (x %~ r))
  : Lemma
    (ensures (
      MS.__gmatmul_single x mul add m1 m2 row col n %~
      (r +. MS.__gmatmul_single 0.0R ( *. ) ( +. )
        (ematrix_to_real m1) (ematrix_to_real m2) row col n)))

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
  (#_ : squash (Seq.length acc0 == tile))
  preserves
    gpu **
    m1 |-> Frac f v1 **
    m2 |-> Frac f v2
  requires
    acc |-> acc0
  ensures
    exists* acc'.
      pure (Seq.length acc' == tile /\
        (forall (i:nat{i < tile}).
          Seq.index acc' i == MS.__gmatmul_single (Seq.index acc0 i) mul add v1 v2 i j tile)) **
      (acc |-> acc')
