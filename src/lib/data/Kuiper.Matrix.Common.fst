module Kuiper.Matrix.Common
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.GhostMap
open Kuiper.Matrix.Reprs.Type
module V = Kuiper.View
module F = FStar.FunctionalExtensionality

let from_seq (#et:Type) (#rows #cols : nat)
  (l : full_mlayout rows cols)
  (s : lseq et (rows * cols))
  : ematrix et rows cols
  = mkM fun i j -> s @! l.map.f (i,j)

let to_seq (#et:Type) (#rows #cols : nat)
  (l : full_mlayout rows cols)
  (m : ematrix et rows cols)
  : GTot (lseq et (rows * cols))
  = Seq.init_ghost (rows * cols) (fun i -> m.f (Kuiper.Injection.inverse_f l.map i))

instance ematrix_is_ghost_map
  (et:Type) (#rows #cols : nat)
  : is_ghost_map (ematrix et rows cols) (natlt rows & natlt cols) et
= {
    bij = {
      ff = (fun m -> M?.f m);
      gg = (fun f -> M f);
      ff_gg = ez;
      gg_ff = ez;
    };
    acc = (fun m (i, j) -> (macc m i j));
    upd = (fun m (i, j) x -> mupd m i j x);
    l1 = ez;
    l2 = ez;
  }

let aview_from_mlayout
  (et : Type) (#rows #cols : nat)
  (l : mlayout rows cols)
  : V.aview et (ematrix et rows cols) =
let open Kuiper.Bijection in
{
  iview = {
    len = l.len;
    ait = natlt rows & natlt cols;
    step = {
      imap = l.map;
    };
  };
  igm = ematrix_is_ghost_map et;
}

let from_seq_rel (#et #rows #cols : _) (l : mlayout rows cols {is_full_layout l})
  (s : lseq et (rows * cols))
  : Lemma (from_seq l s == V.from_seq (aview_from_mlayout et l) s)
  = let vw = aview_from_mlayout et l in
    let aux (i : natlt rows) (j : natlt cols)
      : Lemma (macc (from_seq l s) i j == macc (V.from_seq vw s) i j)
    = calc (==) {
        macc (from_seq l s) i j;
        == { _ by (Tactics.compute ()) } // weird
        s @! l.map.f (i, j);
        == {}
        s @! V.it_to_nat vw (i, j);
        == {}
        (F.on_g vw.iview.ait <| fun i -> s @! V.it_to_nat vw i) (i, j);
        == {}
        macc (vw.igm.bij.gg (F.on_g vw.iview.ait <| fun i -> s @! V.it_to_nat vw i)) i j;
        == { _ by (Tactics.compute ()) } // weird
        macc (V.from_seq vw s) i j;
      }
    in
    Classical.forall_intro_2 aux;
    assert (Kuiper.EMatrix.equal (from_seq l s) (V.from_seq (aview_from_mlayout et l) s))

let to_seq_rel (#et #rows #cols : _) (l : mlayout rows cols{is_full_layout l})
  (s : ematrix et rows cols)
  : Lemma (to_seq l s == V.to_seq (aview_from_mlayout et l) s)
  = assert (Seq.equal (to_seq l s) (V.to_seq (aview_from_mlayout et l) s))

let to_from (#et #rows #cols : _)
  (l : mlayout rows cols { is_full_layout l })
  (s : lseq et (mlayout_size l))
  : Lemma (ensures to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]
  = from_seq_rel l s;
    to_seq_rel l (from_seq l s);
    ()
