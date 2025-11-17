module Kuiper.Matrix.Common
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Container
open Kuiper.Matrix.Reprs.Type
module V = Kuiper.View

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

instance ematrix_is_container
  (et:Type) (#rows #cols : nat)
  : container (ematrix et rows cols) (natlt rows & natlt cols) et
= {
    acc = (fun m (r,c) -> macc m r c);
    upd = (fun m (i, j) x -> mupd m i j x);
    l1 = ez;
    l2 = ez;
    ext = (fun c1 c2 _ -> assert (Kuiper.EMatrix.equal c1 c2));
    from_fun = (fun f -> mkM fun i j -> f (i, j));
    from_fun_ok = ez;
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
  ctn = ematrix_is_container et;
}

let from_seq_rel (#et #rows #cols : _) (l : mlayout rows cols {is_full_layout l})
  (s : lseq et (rows * cols))
  : Lemma (from_seq l s == V.from_seq (aview_from_mlayout et l) s)
  = assert (Kuiper.EMatrix.equal (from_seq l s) (V.from_seq (aview_from_mlayout et l) s))

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
