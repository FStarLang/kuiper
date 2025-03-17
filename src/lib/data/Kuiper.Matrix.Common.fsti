module Kuiper.Matrix.Common
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.GhostMap
open Kuiper.Matrix.Reprs.Type
module A = Kuiper.ArrayView

let from_seq (#et:Type) (#rows #cols : _)
  (l : mlayout rows cols)
  (s : lseq et (rows * cols))
  : ematrix et rows cols
  = mkM fun i j -> s @! l.bij.ff (i,j)

let to_seq (#et:Type) (#rows #cols : _)
  (l : mlayout rows cols)
  (m : ematrix et rows cols)
  : GTot (lseq et (rows * cols))
  = Seq.init_ghost (rows * cols) (fun i -> m.f (l.bij.gg i))

let ematrix_is_ghost_map
  (et:Type) (#rows #cols : nat)
  : is_ghost_map (ematrix et rows cols) (natlt rows & natlt cols) et
= {
    bij = {
      ff = (fun m -> M?.f m);
      gg = (fun f -> M f);
      ff_gg = ez;
      gg_ff = ez;
    };
    acc = (fun m (i, j) -> macc m i j);
    upd = (fun m (i, j) x -> mupd m i j x);
    l1 = ez;
    l2 = ez;
  }

let aview_from_mlayout
  (et : Type) (#rows #cols : erased nat)
  (l : mlayout rows cols)
  : A.aview et (rows * cols) (ematrix et rows cols) =
  {
    it = natlt rows & natlt cols;
    igm = ematrix_is_ghost_map et;
    ibij = l.bij;
  }

let from_seq_rel (#et #rows #cols : _) (l : mlayout rows cols)
  (s : lseq et (rows * cols))
  : Lemma (from_seq l s == A.from_seq (aview_from_mlayout et l) s)
  = assert (Kuiper.EMatrix.equal (from_seq l s) (A.from_seq (aview_from_mlayout et l) s))

let to_seq_rel (#et #rows #cols : _) (l : mlayout rows cols)
  (s : ematrix et rows cols)
  : Lemma (to_seq l s == A.to_seq (aview_from_mlayout et l) s)
  = assert (Seq.equal (to_seq l s) (A.to_seq (aview_from_mlayout et l) s))

let to_from (#et #rows #cols : _)
  (l : mlayout rows cols)
  (s : lseq et (mlayout_size l))
  : Lemma (ensures to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]
  = assert (Seq.equal (to_seq l (from_seq l s)) s)
