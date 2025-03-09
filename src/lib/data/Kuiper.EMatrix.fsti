module Kuiper.EMatrix
#lang-pulse

(* An "erased" matrix, for specification purposes only *)

open Kuiper
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

[@@erasable]
noeq
type ematrix (et:Type) (rows cols : nat) =
  | M : f:(natlt rows -> natlt cols -> GTot et)
     -> ematrix et rows cols

let macc (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  : GTot et
  = m.f i j

let mupd (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  (v : et)
  : ematrix et rows cols
  = M <| fun i' j' ->
           if i' = i && j' = j
           then v
           else m.f i' j'

let mtranspose (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  : ematrix et cols rows
  = M <| fun i j -> m.f j i

let from_row_major_seq (#et:Type) (#rows #cols : nat)
  (s : lseq et (rows * cols))
  : ematrix et rows cols
  = M (fun i j -> s @! (i * cols + j))

let to_row_major_seq (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  : GTot (lseq et (rows * cols))
  = Seq.init_ghost (rows * cols) (fun ij -> M?.f m (ij/cols) (ij % cols))

// This one requires funext, but we don't seem to need it.... yet
// let row_major_inv (#et:Type) (#rows #cols : nat)
//   (m : ematrix et rows cols)
//   : Lemma (from_row_major_seq (to_row_major_seq m) == m)
//   = ()

let row_major_inv (#et:Type) (#rows #cols : nat)
  (s : lseq et (rows * cols))
  : Lemma (to_row_major_seq (from_row_major_seq s) == s)
          [SMTPat (to_row_major_seq (from_row_major_seq s))]
  = assert (Seq.equal (to_row_major_seq (from_row_major_seq s)) s);
    ()

let row_major_acc (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (macc m i j == to_row_major_seq m @! (i * cols + j))
          [SMTPat (macc m i j)]
  = ()

let row_major_upd (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : natlt rows) (j : natlt cols)
  (v : et)
  : Lemma (to_row_major_seq (mupd m i j v) == Seq.upd (to_row_major_seq m) (i * cols + j) v)
          [SMTPat (mupd m i j v)]
  = assert (Seq.equal (to_row_major_seq (mupd m i j v)) (Seq.upd (to_row_major_seq m) (i * cols + j) v));
    ()

let from_col_major_seq (#et:Type) (#rows #cols : nat)
  (s : lseq et (rows * cols))
  : ematrix et rows cols
  = M (fun i j -> s @! (i + j * rows))

let to_col_major_seq (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  : GTot (lseq et (rows * cols))
  = Seq.init_ghost (rows * cols) (fun ij -> M?.f m (ij % rows) (ij / rows))

let col_major_inv (#et:Type) (#rows #cols : nat)
  (s : lseq et (rows * cols))
  : Lemma (to_col_major_seq (from_col_major_seq s) == s)
          [SMTPat (to_col_major_seq (from_col_major_seq s))]
  = assert (Seq.equal (to_col_major_seq (from_col_major_seq s)) s);
    ()
