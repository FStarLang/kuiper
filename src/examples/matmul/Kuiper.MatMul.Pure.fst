module Kuiper.MatMul.Pure

#push-options "--fuel 1 --ifuel 1"

open FStar.Mul
module U64 = FStar.UInt64

#push-options "--retry 2" 
let matmul_single_lemma
  (rows shared columns: nat)
  (s1 : seq u64 { Seq.length s1 == rows * shared })
  (s2 : seq u64 { Seq.length s2 == shared * columns })
  (row: nat{row < rows}) (col: nat{col < columns}) (to: nat)
    : Lemma
      (requires (0 < to /\ to <= shared))
      (ensures (
        assert ((row + 1) <= rows /\ (row + 1) * shared <= rows * shared);
        matmul_single rows shared columns s1 s2 row col to) = U64.add_mod (U64.mul_mod (Seq.index s1 (row * shared + (to - 1))) (Seq.index s2 (col + (to - 1) * columns))) (matmul_single rows shared columns s1 s2 row col (to - 1)
      ))
  = ()
#pop-options

private let matmul_single_at
  (rows shared columns: nat)
  (s1 : seq u64 { Seq.length s1 == rows * shared })
  (s2 : seq u64 { Seq.length s2 == shared * columns })
  (idx: nat{idx < rows * columns})
  : GTot u64
=
  let row = idx / columns in
  let col = idx % columns in
  matmul_single rows shared columns s1 s2 row col shared

let matmul
  (rows shared columns: nat)
  (s1 : seq u64 { Seq.length s1 == rows * shared })
  (s2 : seq u64 { Seq.length s2 == shared * columns })
: GTot (sr: seq u64 { Seq.length sr == rows * columns })
=
  Seq.init_ghost (rows * columns) (matmul_single_at rows shared columns s1 s2)

let lemma_matmul_index
  (rows shared columns: nat)
  (s1: (seq u64){ Seq.length s1 == rows * shared })
  (s2 : (seq u64){ Seq.length s2 == shared * columns })
  (idx: nat{idx < rows * columns})
: Lemma (
    Seq.index (matmul rows shared columns s1 s2) idx
    ==
    matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared
  )
= ()

#pop-options
