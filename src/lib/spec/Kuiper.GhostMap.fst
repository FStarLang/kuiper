module Kuiper.GhostMap

open Kuiper.Common
open Kuiper.Bijection
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

let ghost_map_acc
  (#mt:Type) (#it:Type) (#et:Type)
  (gm : is_ghost_map mt it et)
  (i : it) (m : mt)
  : Lemma (gm.bij.ff m i == gm.acc m i)
          [SMTPatOr [[SMTPat (gm.bij.ff m i)];
                     [SMTPat (gm.acc m i)]]]
  = gm.l1 i m

let ghost_map_upd
  (#mt:Type) (#it:Type) (#et:Type)
  (gm : is_ghost_map mt it et)
  (i : it) (m : mt) (e : et)
  : Lemma (gm.bij.ff (gm.upd m i e) == oplus (gm.bij.ff m) i e)
          [SMTPatOr [[SMTPat (gm.bij.ff (gm.upd m i e))];
                     [SMTPat (oplus (gm.bij.ff m) i e)]]]
  = gm.l2 i m e

let prod_ff
  (#et : Type)
  (#ma #ia : Type)
  (#mb #ib : Type)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  : (ma & mb) -> (either ia ib ^->> et)
  = fun m ->
      F.on_g _ <| fun (i : either ia ib) ->
      match i with
      | Inl i1 -> gm1.bij.ff (fst m) i1
      | Inr i2 -> gm2.bij.ff (snd m) i2

let prod_gg
  (#et : Type)
  (#ma #ia : Type)
  (#mb #ib : Type)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  : ((either ia ib ^->> et) -> GTot (ma & mb))
  = fun (gm : (either ia ib) ^->> et) ->
    (gm1.bij.gg (F.on_g _ <| gm `oo` Inl),
     gm2.bij.gg (F.on_g _ <| gm `oo` Inr))

let prod_ff_gg
  (#et : Type)
  (#ma #ia : Type)
  (#mb #ib : Type)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  (m : either ia ib ^->> et)
  : squash (prod_ff gm1 gm2 (prod_gg gm1 gm2 m) == m)
  = assert (F.feq_g (prod_ff gm1 gm2 (prod_gg gm1 gm2 m)) m)

let prod_gg_ff
  (#et : Type)
  (#ma #ia : Type)
  (#mb #ib : Type)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  (m : ma & mb)
  : squash (prod_gg gm1 gm2 (prod_ff gm1 gm2 m) == m)
  = let m1, m2 = m in
    let m1', m2' = prod_gg gm1 gm2 (prod_ff gm1 gm2 m) in
    assert (F.feq_g (prod_ff gm1 gm2 m `oo` Inl)
                    (gm1.bij.ff m1));
    assert (F.feq_g (prod_ff gm1 gm2 m `oo` Inr)
                    (gm2.bij.ff m2));
    gm1.bij.gg_ff m1;
    gm2.bij.gg_ff m2;
    ()

let prod_bij
  (#et : Type)
  (#ma #ia : Type)
  (#mb #ib : Type)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  : GTot (bijection ((ma & mb)) (either ia ib ^->> et))
= Mkbijection
    #((ma & mb))
    #(either ia ib ^->> et) // Some terrible inference here, forced me to give these parameters explicitly
    (prod_ff gm1 gm2)
    (prod_gg gm1 gm2)
    (prod_ff_gg gm1 gm2)
    (prod_gg_ff gm1 gm2)

instance is_ghost_map_prod
  (#et : Type0)
  (#ma #ia : Type0)
  (#mb #ib : Type0)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  : is_ghost_map (ma & mb) (either ia ib) et
  = {
    bij = prod_bij gm1 gm2;
    acc = (fun (m : (ma & mb)) (i : either ia ib) ->
      match i with
      | Inl i1 -> gm1.acc (fst m) i1
      | Inr i2 -> gm2.acc (snd m) i2)
    ;
    upd = (fun (m : (ma & mb)) (i : either ia ib) (e : et) ->
      match i with
      | Inl i1 -> hide (reveal (gm1.upd (fst m) i1 e), snd m)
      | Inr i2 -> hide (fst m, reveal (gm2.upd (snd m) i2 e)))
    ;

    l1 = (fun (i : either ia ib) (m : ma & mb) ->
      match i with
      | Inl i1 -> gm1.l1 i1 (fst m)
      | Inr i2 -> gm2.l1 i2 (snd m))
    ;
    l2 = (fun (i : either ia ib) (m : ma & mb) (e : et) ->
      match i with
      | Inl i1 -> gm1.l2 i1 (fst m) e
      | Inr i2 -> gm2.l2 i2 (snd m) e);
  }

let lemma_is_ghost_map_prod_acc
  (#et:Type)
  (#ma #ia : Type)
  (#mb #ib : Type)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  (v : (ma & mb))
  (i : either ia ib)
  : Lemma (ensures (is_ghost_map_prod gm1 gm2).acc v i == (match i with
          | Inl i1 -> gm1.acc (fst v) i1
          | Inr i2 -> gm2.acc (snd v) i2))
          [SMTPat ((is_ghost_map_prod gm1 gm2).acc v i)]
  = ()
