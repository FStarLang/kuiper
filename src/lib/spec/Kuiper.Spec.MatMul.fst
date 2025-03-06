module Kuiper.Spec.MatMul

let matmul_single_lemma
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
    matmul_single m1 m2 row col to ==
    add
      (matmul_single m1 m2 row col (to - 1))
      (mul (macc m1 row (to-1)) (macc m2 (to-1) col))
  ))
  = ()

let matmul_single_at
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (idx : nat{idx < rows * columns})
  : GTot et
=
  let row = idx / columns in
  let col = idx % columns in
  matmul_single m1 m2 row col shared

let matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: GTot (ematrix et rows columns)
= M <| Seq.init_ghost (rows * columns) (matmul_single_at m1 m2)

let lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matmul m1 m2) i j == matmul_single m1 m2 i j shared)
= ()
