module GPU.MatMul.Pure

open FStar.Mul
open Pulse.Lib.Pervasives

open FStar.Seq { seq }

module U64 = FStar.UInt64
unfold let u64 = FStar.UInt64.t

let rec matmul_single
  (rows shared columns: nat)
  (s1: (seq u64){ Seq.length s1 == rows * shared })
  (s2 : (seq u64){ Seq.length s2 == shared * columns })
  (row: nat{row < rows}) (col: nat{col < columns}) (to: nat)
    : GTot u64 (decreases to)
  =
  if reveal to > shared then 0UL else if reveal to = 0 then 0UL else
    (
        assert ((row + 1) <= rows /\ (row + 1) * shared <= rows * shared);
        U64.add_mod (U64.mul_mod (Seq.index s1 (row * shared + (to - 1))) (Seq.index s2 (col + (to - 1) * columns))) (matmul_single rows shared columns s1 s2 row col (to - 1))
    )

val matmul_single_lemma
  (rows shared columns: nat)
  (s1: (seq u64){ Seq.length s1 == rows * shared })
  (s2 : (seq u64){ Seq.length s2 == shared * columns })
  (row: nat{row < rows}) (col: nat{col < columns}) (to: nat)
    : Lemma
      (requires (0 < to /\ to <= shared))
      (ensures (
        assert ((row + 1) <= rows /\ (row + 1) * shared <= rows * shared);
        matmul_single rows shared columns s1 s2 row col to ==
        U64.add_mod (U64.mul_mod (Seq.index s1 (row * shared + (to - 1))) (Seq.index s2 (col + (to - 1) * columns)))
                    (matmul_single rows shared columns s1 s2 row col (to - 1))
      ))

val matmul
  (rows shared columns: nat)
  (s1: (seq u64){ Seq.length s1 == rows * shared })
  (s2 : (seq u64){ Seq.length s2 == shared * columns })
    : GTot (sr:(seq u64){ Seq.length sr == rows * columns })

val lemma_matmul_index
  (rows shared columns: nat)
  (s1: (seq u64){ Seq.length s1 == rows * shared })
  (s2 : (seq u64){ Seq.length s2 == shared * columns })
  (idx: nat{idx < rows * columns})
: Lemma (Seq.index (matmul rows shared columns s1 s2) idx
         == matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared)
