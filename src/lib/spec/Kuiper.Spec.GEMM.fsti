module Kuiper.Spec.GEMM

(* NOTE: this is for an "exact" matmul, it does not provide
any weak modulo-associativity spec. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Tiling

inline_for_extraction noextract
let comb2 (#et:Type) (x y : et) : et = y

inline_for_extraction noextract
let lincomb
  (#et:Type) {| scalar et |}
  (alpha beta : et)
  (x y : et)
  (* x is the old value, y is the new computed value *)
  : et
  = add (mul beta x) (mul alpha y)

// computes
// sum_{i=0}{to} m1[row][i] * m2[i][col]
// when to=shared, it computes the (row,col) cell of m1*m2
// the sum  is associated to the left, i.e.
// ((zero + m1[row][0] * m2[0][col]) + m1[row][1] * m2[1][col]) + ...
val __matmul_single
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat{to <= shared})
  : GTot et

let matmul_single
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot et
  = __matmul_single m1 m2 row col shared

let gemm_single
  (#et:Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (m0 : ematrix et rows columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot et
  = comb
      (macc m0 row col)
      (matmul_single m1 m2 row col)

val matmul_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row col : nat{row < rows /\ col < columns})
: Lemma
  (ensures (
    __matmul_single m1 m2 row col 0 == zero
  ))
  [SMTPat (__matmul_single m1 m2 row col 0)]

val matmul_single_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat)
: Lemma
  (requires (0 < to /\ to <= shared))
  (ensures (
    __matmul_single m1 m2 row col to ==
    add
      (__matmul_single m1 m2 row col (to - 1))
      (mul (macc m1 row (to-1)) (macc m2 (to-1) col))
  ))
  // [SMTPat (matmul_single m1 m2 row col to)]

val matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns

val matplus
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : ematrix et rows columns)
: ematrix et rows columns

val lemma_matplus_index
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : ematrix et rows columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matplus m1 m2) i j == macc m1 i j `add` macc m2 i j)
        [SMTPat (macc (matplus m1 m2) i j)]

val lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matmul m1 m2) i j == matmul_single m1 m2 i j)
        [SMTPat (matmul_single m1 m2 i j)]

val __matmul_single_tile
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  (to : nat{to <= shared / tk})
  : GTot (ematrix et tm tn)

let matmul_single_tile
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  : GTot (ematrix et tm tn)
  = __matmul_single_tile tm tn tk m1 m2 trow tcol (shared/tk)

val matmul_single_tile_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
: Lemma
  (ensures (
    __matmul_single_tile tm tn tk m1 m2 trow tcol 0 == const_matrix zero
  ))

val matmul_single_tile_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  (to : nat{to <= shared / tk})
: Lemma
  (requires (0 < to /\ to <= shared))
  (ensures (
    __matmul_single_tile tm tn tk m1 m2 trow tcol to ==
    matplus
      (__matmul_single_tile tm tn tk m1 m2 trow tcol (to-1))
      (matmul (ematrix_subtile m1 tm tk trow (to-1))
              (ematrix_subtile m2 tk tn (to-1) tcol)))
  )

let mmcomb
  (#et:Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns
= matrix_comb comb m0 (matmul m1 m2)

val matmul_is_gemm
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  : Lemma (mmcomb comb2 m0 m1 m2 == matmul m1 m2)
          [SMTPat (mmcomb comb2 m0 m1 m2)]

let gemm
  (#et:Type) {| scalar et |}
  (alpha beta : et)
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns
= matrix_comb (lincomb alpha beta) m0 (matmul m1 m2)
