module Kuiper.GhostMap

open Kuiper.Bijection
open FStar.FunctionalExtensionality { (^->>) }

let oplus (#a #b : Type) (f : a -> GTot b) (x : a) (y : b) : a -> GTot b =
  fun x' ->
    if FStar.StrongExcludedMiddle.strong_excluded_middle (x == x')
    then y
    else f x'

(* This type shows that mt is essentially a ghost map from it to et. *)
[@@erasable]
noeq
type is_ghost_map (mt : Type) (it : Type) (et : Type) = {
  bij : mt =~ (it ^->> et);
  acc : mt -> it -> GTot et;
  upd : mt -> it -> et -> GTot mt;

  l1 : (i:it -> m:mt ->
         squash (bij.ff m i == acc m i));
  i2 : (i:it -> m:mt -> e:et ->
         squash (bij.ff (upd m i e) == oplus (bij.ff m) i e));
}

val ghost_map_acc
  (#mt:Type) (#it:Type) (#et:Type)
  (gm : is_ghost_map mt it et)
  (i : it) (m : mt)
  : Lemma (gm.bij.ff m i == gm.acc m i)
          [SMTPatOr [[SMTPat (gm.bij.ff m i)];
                     [SMTPat (gm.acc m i)]]]

val ghost_map_upd
  (#mt:Type) (#it:Type) (#et:Type)
  (gm : is_ghost_map mt it et)
  (i : it) (m : mt) (e : et)
  : Lemma (gm.bij.ff (gm.upd m i e) == oplus (gm.bij.ff m) i e)
          [SMTPatOr [[SMTPat (gm.bij.ff (gm.upd m i e))];
                     [SMTPat (oplus (gm.bij.ff m) i e)]]]
