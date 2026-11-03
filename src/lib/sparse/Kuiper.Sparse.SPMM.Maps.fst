module Kuiper.Sparse.SPMM.Maps

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Sparse.Common
open Kuiper.Seq.Common { (@+) }

(* The accumulator's element type does not determine the output geometry. *)
let tile_values_approx
  (#et : Type0) {| scalar et, real_like et |}
  (ch : pos)
  (#rows #cols : nat)
  (rM : chest2 real rows cols)
  (row : natlt rows) (col : nat)
  (#n : nat) (values : lseq et n) (step : nat)
  : prop
  = forall (i : natlt n).
      let c = col + i / ch * step * ch + i % ch in
      c < cols ==> (values @! i) %~ acc2 rM row c

let map_slice (f : real -> real) (s : Seq.seq real)
  (i j : nat { i <= j /\ j <= Seq.length s })
  : Lemma (Seq.map_seq f (Seq.slice s i j) ==
           Seq.slice (Seq.map_seq f s) i j)
    [SMTPat (Seq.map_seq f (Seq.slice s i j))]
  = assert (Seq.equal (Seq.map_seq f (Seq.slice s i j))
             (Seq.slice (Seq.map_seq f s) i j))

let map_zero_prefix (f : real -> real) (n : nat) (s : Seq.seq real)
  : Lemma
    (requires f 0.0R == 0.0R)
    (ensures Seq.map_seq f (Seq.create n 0.0R @+ s) ==
             Seq.create n 0.0R @+ Seq.map_seq f s)
  = assert (Seq.equal (Seq.map_seq f (Seq.create n 0.0R @+ s))
             (Seq.create n 0.0R @+ Seq.map_seq f s))

let map_unsparse
  (f : real -> real { f 0.0R == 0.0R })
  (#nnz : nat) (elems : lseq real nnz) (pos : lseq nat nnz) (n : nat)
  : Lemma (seq_unsparse nnz n (Seq.map_seq f elems) pos ==
           Seq.map_seq f (seq_unsparse nnz n elems pos))
  = assert (Seq.equal (seq_unsparse nnz n (Seq.map_seq f elems) pos)
             (Seq.map_seq f (seq_unsparse nnz n elems pos)))

let map_row (f : real -> real) (#rows #cols : nat)
  (m : chest2 real rows cols) (i : natlt rows)
  : Lemma (ematrix_row (Kuiper.Chest.chest_map f m) i ==
           Seq.map_seq f (ematrix_row m i))
    [SMTPat (ematrix_row (Kuiper.Chest.chest_map f m) i)]
  = assert (Seq.equal (ematrix_row (Kuiper.Chest.chest_map f m) i)
             (Seq.map_seq f (ematrix_row m i)))
