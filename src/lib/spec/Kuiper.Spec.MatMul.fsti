module Kuiper.Spec.MatMul

open Kuiper

[@@erasable]
noeq
type ematrix (et:Type) (rows cols : nat) =
  | M : s:(seq et){ len s == rows * cols } -> ematrix et rows cols

(* Note: row major *)
let macc (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  : GTot et
  = m.s @! (i * cols + j)

// computes
// sum_{i=0}{to} m1[row][i] * m2[i][col]
// when to=shared, it computes the (row,col) cell of m1*m2
// the sum  is associated to the left, i.e.
// ((zero + m1[row][0] * m2[0][col]) + m1[row][1] * m2[1][col]) + ...
let rec matmul_single
  (#et:Type) {| scalar et |}
  (rows shared columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat{to <= shared})
  : GTot et (decreases to)
  =
  if reveal to = 0 then zero
  else (
    // assert ((row + 1) <= rows /\ (row + 1) * shared <= rows * shared);
    add
      (matmul_single rows shared columns m1 m2 row col (to - 1))
      (mul (macc m1 row (to - 1))
           (macc m2 (to - 1) col))
  )

val matmul_single_lemma
  (#et:Type) {| scalar et |}
  (rows shared columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat)
: Lemma
  (requires (0 < to /\ to <= shared))
  (ensures (
    matmul_single rows shared columns m1 m2 row col to ==
    add
      (matmul_single rows shared columns m1 m2 row col (to - 1))
      (mul (macc m1 row (to-1)) (macc m2 (to-1) col))
  ))

val matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: GTot (ematrix et rows columns)

val lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matmul m1 m2) i j == matmul_single rows shared columns m1 m2 i j shared)
