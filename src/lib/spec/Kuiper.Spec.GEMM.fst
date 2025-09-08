module Kuiper.Spec.GEMM

let rec __matmul_single
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat{to <= shared})
  : GTot et (decreases to)
  =
  if reveal to = 0 then zero
  else (
    add
      (__matmul_single m1 m2 row col (to - 1))
      (mul (macc m1 row (to - 1))
           (macc m2 (to - 1) col))
  )

let matmul_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row col : nat{row < rows /\ col < columns})
: Lemma
  (ensures (
    __matmul_single m1 m2 row col 0 == zero
  ))
  = ()

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
    __matmul_single m1 m2 row col to ==
    add
      (__matmul_single m1 m2 row col (to - 1))
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
  matmul_single m1 m2 row col

let matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns
= mkM <| fun i j -> matmul_single m1 m2 i j

let matplus
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : ematrix et rows columns)
: ematrix et rows columns
= mkM <| fun i j -> add (macc m1 i j) (macc m2 i j)

let lemma_matplus_index
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : ematrix et rows columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matplus m1 m2) i j == macc m1 i j `add` macc m2 i j)
        [SMTPat (macc (matplus m1 m2) i j)]
= ()

let lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matmul m1 m2) i j == matmul_single m1 m2 i j)
= ()

let matmul_is_gemm
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  : Lemma (mmcomb comb2 m0 m1 m2 == matmul m1 m2)
          [SMTPat (mmcomb comb2 m0 m1 m2)]
  = ematrix_ext (mmcomb comb2 m0 m1 m2) (matmul m1 m2)

let mma
  (#et0 #et1 : Type)
  (#rows #shared #columns : nat)
  (mc : ematrix et1 rows columns)
  (ma : ematrix et0 rows shared)
  (mb : ematrix et0 shared columns)
: ematrix et1 rows columns
= magic()

let lemma_mma_is_matmul_add
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (mc : ematrix et rows columns)
  (ma : ematrix et rows shared)
  (mb : ematrix et shared columns)
: Lemma (mma mc ma mb == matplus mc (matmul ma mb))
= admit()
