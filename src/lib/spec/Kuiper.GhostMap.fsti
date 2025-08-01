module Kuiper.GhostMap

open Kuiper
open Kuiper.Bijection
open FStar.FunctionalExtensionality { (^->>), (^->) }
open FStar.Tactics.Typeclasses
module F = FStar.FunctionalExtensionality

(* If this axiom causes some eyebrows to raise, we can always
   change the l2 statement to avoid it. *)
let oplus (#a #b : Type) (f : a -> GTot b) (x : a) (y : b) : (a ^->> b) =
  F.on_g _ fun x' ->
    if FStar.StrongExcludedMiddle.strong_excluded_middle (x == x')
    then y
    else f x'

(* This type shows that mt is essentially a ghost map from it to et. *)
(* Q: Why even provide acc and upd if we have the bijection?
   A: Because they make the specs look reasonable for a user,
      they get to choose the shape they want. *)
[@@erasable]
class is_ghost_map (mt : Type) (it : Type) (et : Type) = {
  [@@@no_method]
  bij : erased mt =~ (it ^->> et);
  [@@@no_method]
  acc : mt -> it -> erased et;
  [@@@no_method]
  upd : mt -> it -> et -> erased mt;

  [@@@no_method]
  l1 : (i:it -> m:mt ->
         squash (hide (bij.ff m i) == acc m i));
  [@@@no_method]
  l2 : (i:it -> m:mt -> e:et ->
         squash (bij.ff (upd m i e) `F.feq_g` oplus (bij.ff m) i e));
}

val ghost_map_acc
  (#mt:Type) (#it:Type) (#et:Type)
  (gm : is_ghost_map mt it et)
  (i : it) (m : erased mt)
  : Lemma (hide (gm.bij.ff m i) == gm.acc m i)
          [SMTPatOr [[SMTPat (gm.bij.ff m i)];
                     [SMTPat (gm.acc m i)]]]

val ghost_map_upd
  (#mt:Type) (#it:Type) (#et:Type)
  (gm : is_ghost_map mt it et)
  (i : it) (m : erased mt) (e : et)
  : Lemma (gm.bij.ff (gm.upd m i e) == oplus (gm.bij.ff m) i e)
          [SMTPatOr [[SMTPat (gm.bij.ff (gm.upd m i e))];
                     [SMTPat (oplus (gm.bij.ff m) i e)]]]

instance val is_ghost_map_prod
  (#et : Type0)
  (#ma #ia : Type0)
  (#mb #ib : Type0)
  (gm1 : is_ghost_map ma ia et)
  (gm2 : is_ghost_map mb ib et)
  : is_ghost_map (ma & mb) (either ia ib) et

val lemma_is_ghost_map_prod_acc
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

open FStar.Ghost
open FStar.Seq

let lseq_to_ghost_map (#et:Type) (#len:nat) : erased (lseq et len) -> (natlt len ^->> et) =
  fun v -> F.on_g _ fun (i:natlt len) -> Seq.index (reveal v) i <: et

let lseq_from_ghost_map (#et:Type) (#len:nat) : (natlt len ^->> et) -> erased (lseq et len) =
  fun f -> hide (Seq.init_ghost len f)

noextract
let bij_lseq_ghost_map (et:Type) (len:nat) : bijection (erased (lseq et len)) (natlt len ^->> et) = {
  ff = lseq_to_ghost_map;
  gg = lseq_from_ghost_map;
  ff_gg = (fun f ->
    assert (F.feq_g (lseq_to_ghost_map (lseq_from_ghost_map f)) f);
    ()
  );
  (* We need to state this, otherwise we get a coercion that messes things up. *)
  gg_ff = (fun (es : erased (lseq et len)) ->
    (* ARGH! without stating erased above, we seem to implicly get a reveal coercion that messes things up. *)
    assert (Seq.equal (lseq_from_ghost_map (lseq_to_ghost_map es)) es);
    ()
  );
}

noextract
instance lseq_is_ghost_map (et:Type) (len:nat) : is_ghost_map (lseq et len) (natlt len) et = {
  bij = bij_lseq_ghost_map et len;
  acc = (fun v i -> v @! i);
  upd = (fun v i x -> Seq.upd (reveal v) i x);
  l1 = ez;
  l2 = ez;
}

instance ghost_map_is_ghost_map #a #b : is_ghost_map (a ^->> b) a b =
  {
    bij = 
      (Mkbijection #(erased (a ^->> b)) #(a ^->> b)
        reveal
        hide
        ez
        ez
      );
    acc = (fun m i -> m i);
    upd = (fun m i e -> oplus m i e);
    l1 = ez;
    l2 = ez;
  }
