module GPU.MatMul.Pure

#push-options "--fuel 1 --ifuel 1"

open FStar.Mul
open Pulse.Lib.Pervasives
module U64 = FStar.UInt64

let rec matmul_single
  (rows shared columns: nat)
  (s1: (Seq.Base.seq U64.t){ Seq.Base.length s1 == rows * shared })
  (s2 : (Seq.Base.seq U64.t){ Seq.Base.length s2 == shared * columns })
  (row: nat{row < rows}) (col: nat{col < columns}) (to: nat)
    : GTot U64.t (decreases to)
  =
  if reveal to > shared then 0UL else if reveal to = 0 then 0UL else
    (
        assert ((row + 1) <= rows /\ (row + 1) * shared <= rows * shared);
        U64.add_mod (U64.mul_mod (Seq.Base.index s1 (row * shared + (to - 1))) (Seq.Base.index s2 (col + (to - 1) * columns))) (matmul_single rows shared columns s1 s2 row col (to - 1))
    )

#push-options "--retry 2" 
let matmul_single_lemma
  (rows shared columns: nat)
  (s1: (Seq.Base.seq U64.t){ Seq.Base.length s1 == rows * shared })
  (s2 : (Seq.Base.seq U64.t){ Seq.Base.length s2 == shared * columns })
  (row: nat{row < rows}) (col: nat{col < columns}) (to: nat)
    : Lemma
      (requires (0 < to /\ to <= shared))
      (ensures (
        assert ((row + 1) <= rows /\ (row + 1) * shared <= rows * shared);
        reveal (matmul_single rows shared columns s1 s2 row col to) = U64.add_mod (U64.mul_mod (Seq.Base.index s1 (row * shared + (to - 1))) (Seq.Base.index s2 (col + (to - 1) * columns))) (matmul_single rows shared columns s1 s2 row col (to - 1))
      ))
  = ()
#pop-options

private let matmul_single_at
  (rows shared columns: nat)
  (s1: (Seq.Base.seq U64.t){ Seq.Base.length s1 == rows * shared })
  (s2 : (Seq.Base.seq U64.t){ Seq.Base.length s2 == shared * columns })
  (idx: nat{idx < rows * columns})
    : GTot U64.t
  =
  let row = idx / columns in let col = idx % columns in matmul_single rows shared columns s1 s2 row col shared

let matmul
  (rows shared columns: nat)
  (s1: (Seq.Base.seq U64.t){ Seq.Base.length s1 == rows * shared })
  (s2 : (Seq.Base.seq U64.t){ Seq.Base.length s2 == shared * columns })
    : GTot (sr:(Seq.Base.seq U64.t){ Seq.Base.length sr == rows * columns })
  =
  Seq.Base.init_ghost (rows * columns) (matmul_single_at rows shared columns s1 s2)

let lemma_matmul_index
  (rows shared columns: nat)
  (s1: (Seq.Base.seq U64.t){ Seq.Base.length s1 == rows * shared })
  (s2 : (Seq.Base.seq U64.t){ Seq.Base.length s2 == shared * columns })
  (idx: nat{idx < rows * columns})
    : Lemma
      (ensures (Seq.Base.index (reveal (matmul rows shared columns s1 s2)) idx) == reveal (matmul_single rows shared columns s1 s2 (idx / columns) (idx % columns) shared))
  = ()

#pop-options
