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
  bij : mt =~ (it ^->> et);
  [@@@no_method]
  acc : mt -> it -> GTot et;
  [@@@no_method]
  upd : mt -> it -> et -> GTot mt;

  [@@@no_method]
  l1 : (i:it -> m:mt ->
         squash (bij.ff m i == acc m i));
  [@@@no_method]
  l2 : (i:it -> m:mt -> e:et ->
         squash (bij.ff (upd m i e) `F.feq_g` oplus (bij.ff m) i e));
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

let lseq_to_ghost_map (#et:Type) (#len:nat) : (lseq et len) -> GTot (natlt len ^->> et) =
  fun v -> F.on_g _ fun (i:natlt len) -> Seq.index (reveal v) i <: et

let lseq_from_ghost_map (#et:Type) (#len:nat) : (natlt len ^->> et) -> GTot (lseq et len) =
  fun f -> hide (Seq.init_ghost len f)

noextract
let bij_lseq_ghost_map (et:Type) (len:nat) : bijection (lseq et len) (natlt len ^->> et) = {
  ff = lseq_to_ghost_map;
  gg = lseq_from_ghost_map;
  ff_gg = (fun f ->
    assert (F.feq_g (lseq_to_ghost_map (lseq_from_ghost_map f)) f);
    ()
  );
  (* We need to state this, otherwise we get a coercion that messes things up. *)
  gg_ff = (fun es ->
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
    bij = bij_self (a ^->> b);
    acc = (fun m i -> m i);
    upd = (fun m i e -> oplus m i e);
    l1 = ez;
    l2 = ez;
  }

let ghost_map_fun_ff
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  : (idx ^->> mt) -> ((x:idx & i x) ^->> e)
=
  fun (f : (idx -> GTot mt)) ->
    F.on_g _ <|
    fun (xy : (x:idx & i x)) ->
      let (| x, y |) = xy in
      (sub x).bij.ff (f x) y

let ghost_map_fun_gg
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  : ((x:idx & i x) ^->> e) -> (idx ^->> mt)
=
  fun (g : (x:idx & i x) ^->> e) ->
    F.on_g _ <|
    fun (x : idx) ->
      (sub x).bij.gg (F.on_g _ <| fun y -> g (| x, y |))

let lemma_ghost_map_fun_ff_gg
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  (g : (x:idx & i x) ^->> e)
  : Lemma (ghost_map_fun_ff idx mt i e (ghost_map_fun_gg idx mt i e g) == g)
          [SMTPat (ghost_map_fun_ff idx mt i e (ghost_map_fun_gg idx mt i e g))]
=
  let aux (ix : (x:idx & i x))
    : Lemma (ghost_map_fun_ff idx mt i e (ghost_map_fun_gg idx mt i e g) ix == g ix)
  =
    let (| j, x |) = ix in
    calc (==) {
      ghost_map_fun_ff idx mt i e (ghost_map_fun_gg idx mt i e g) ix;
      == {}
      (sub j).bij.ff (ghost_map_fun_gg idx mt i e g j) x;
      == {}
      (sub j).bij.ff ((sub j).bij.gg (F.on_g _ <| fun y -> g (| j, y |))) x;
      == {}
      (sub j).bij.ff ((sub j).bij.gg (F.on_g _ <| fun y -> g (| j, y |))) x;
      == { (sub j).bij.ff_gg (F.on_g _ <| fun y -> g (| j, y |)) }
      (F.on_g _ <| fun y -> g (| j, y |)) x;
      == {}
      g (| j, x |);
    };
    ()
  in
  Classical.forall_intro aux;
  assert (ghost_map_fun_ff idx mt i e (ghost_map_fun_gg idx mt i e g) `F.feq_g` g)

let lemma_ghost_map_fun_gg_ff
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  (f : (idx ^->> mt))
  : Lemma (ghost_map_fun_gg idx mt i e (ghost_map_fun_ff idx mt i e f) == f)
          [SMTPat (ghost_map_fun_gg idx mt i e (ghost_map_fun_ff idx mt i e f))]
=
  let aux (x : idx)
    : Lemma (ghost_map_fun_gg idx mt i e (ghost_map_fun_ff idx mt i e f) x == f x)
  =
    calc (==) {
      ghost_map_fun_gg idx mt i e (ghost_map_fun_ff idx mt i e f) x;
      == { _ by (Tactics.compute ()) } // weird
      (F.on_g _ <| fun (x : idx) ->
        (sub x).bij.gg (F.on_g _ <| fun y -> ghost_map_fun_ff idx mt i e f (| x, y |))) x;
      == {}
      (sub x).bij.gg (F.on_g _ <| fun y -> ghost_map_fun_ff idx mt i e f (| x, y |));
      == { _ by (Tactics.compute ()) } // weird
      (sub x).bij.gg (F.on_g _ <| fun y -> (sub x).bij.ff (f x) y);
      == { assert ((F.on_g _ <| fun y -> (sub x).bij.ff (f x) y) `F.feq_g` (sub x).bij.ff (f x));
           () }
      (sub x).bij.gg ((sub x).bij.ff (f x));
      == { (sub x).bij.gg_ff (f x) }
      f x;
    };
    ()
  in
  Classical.forall_intro aux;
  assert (ghost_map_fun_gg idx mt i e (ghost_map_fun_ff idx mt i e f) `F.feq_g` f)

let ghost_map_fun_bij
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  : ((idx ^->> mt) =~ ((x:idx & i x) ^->> e))
= {
    ff = ghost_map_fun_ff idx mt i e;
    gg = ghost_map_fun_gg idx mt i e;
    ff_gg = ez;
    gg_ff = (fun f -> lemma_ghost_map_fun_gg_ff idx mt i e f);
}

let ghost_map_fun_acc
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  (m : (idx ^->> mt))
  (ix : (x:idx & i x))
  : GTot e
=
  let (| x, y |) = ix in
  (sub x).acc (m x) y

let ghost_map_fun_upd
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  (m : (idx ^->> mt))
  (ix : (x:idx & i x))
  (ev : e)
  : GTot (idx ^->> mt)
=
  let (| x, y |) = ix in
  let m' : (idx ^->> mt) =
    F.on_g _
    fun x' ->
      if x = x'
      then (sub x).upd (m x) y ev
      else m x'
  in
  m'

let lemma_ghost_map_fun_l2
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  (it : (x:idx & i x))
  (m : (idx ^->> mt))
  (ev : e)
  : squash (
      (ghost_map_fun_bij idx mt i e).ff (ghost_map_fun_upd idx mt i e m it ev)
      ==
      oplus ((ghost_map_fun_bij idx mt i e).ff m) it ev)
=
  let (| x0, y0 |) = it in
  let aux (xy : (x:idx & i x))
    : Lemma (
        (ghost_map_fun_bij idx mt i e).ff (ghost_map_fun_upd idx mt i e m it ev) xy
        ==
        oplus ((ghost_map_fun_bij idx mt i e).ff m) it ev xy)
  =
    let (| x, y |) = xy in
    (* Trivial if we're in a different component. *)
    if x <> x0 then
      ()
    else
      calc (==) {
        (ghost_map_fun_bij idx mt i e).ff (ghost_map_fun_upd idx mt i e m it ev) xy;
        == {}
        (ghost_map_fun_bij idx mt i e).ff (F.on_g _ fun x' -> (sub x).upd (m x) y0 ev) xy;
        == {}
        ghost_map_fun_ff idx mt i e (F.on_g _ fun x' -> (sub x).upd (m x) y0 ev) xy;
        == {}
        (sub x).bij.ff ((sub x).upd (m x) y0 ev) y;
        == {}
        oplus ((ghost_map_fun_bij idx mt i e).ff m) it ev xy;
      };
      ()
  in
  Classical.forall_intro aux;
  assert (
    (ghost_map_fun_bij idx mt i e).ff (ghost_map_fun_upd idx mt i e m it ev)
    `F.feq_g`
    oplus ((ghost_map_fun_bij idx mt i e).ff m) it ev)

instance ghost_map_fun
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> is_ghost_map mt (i x) e) |}
  : is_ghost_map (idx ^->> mt) (x:idx & i x) e =
{
  bij = ghost_map_fun_bij idx mt i e;
  acc = ghost_map_fun_acc idx mt i e;
  upd = ghost_map_fun_upd idx mt i e;
  l1  = ez;
  l2  = lemma_ghost_map_fun_l2 idx mt i e;
}
