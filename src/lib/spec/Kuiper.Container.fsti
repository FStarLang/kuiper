module Kuiper.Container

open FStar.FunctionalExtensionality { (^->>), (^->) }
open FStar.Ghost
open FStar.Seq
open Kuiper
open Kuiper.Bijection
open FStar.Tactics.Typeclasses { fundeps }

module F = FStar.FunctionalExtensionality

// noeq
// type lens (c it vt : Type) = {
//   view   : c -> it -> GTot vt;
//   upd : c -> it -> vt -> GTot c;
// }

[@@fundeps [1;2]; erasable]
class container (ct it vt : Type) = {
  acc : ct -> it -> GTot vt;
  upd : ct -> it -> vt -> GTot ct;

  #[Tactics.Easy.easy_fill()]
  l1 : c:ct -> i:it -> v:vt ->
    squash (acc (upd c i v) i == v);

  #[Tactics.Easy.easy_fill()]
  l2 : c:ct -> i1:it -> i2:it{i1 =!= i2} -> v:vt ->
    squash (acc (upd c i2 v) i1 == acc c i1);

  ext : c1:ct -> c2:ct ->
    squash (forall (i:it). acc c1 i == acc c2 i) ->
    squash (c1 == c2);

  from_fun : f:(it -> GTot vt) -> GTot ct;

  #[Tactics.Easy.easy_fill()]
  from_fun_ok : f:(it -> GTot vt) -> (i:it) -> squash (acc (from_fun f) i == f i);
}

(* If this axiom causes some eyebrows to raise, we can always
   change the l2 statement to avoid it. *)
let oplus (#a #b : Type) (f : a -> GTot b) (x : a) (y : b) : (a ^->> b) =
  F.on_g _ fun x' ->
    if t2b (x == x')
    then y
    else f x'

let container_prod_acc
  (#et : Type0)
  (#ma #ia : Type0)
  (#mb #ib : Type0)
  (gm1 : container ma ia et)
  (gm2 : container mb ib et)
  (v : ma & mb)
  (i : either ia ib)
  : GTot et
  =
    match i with
    | Inl i1 -> gm1.acc (fst v) i1
    | Inr i2 -> gm2.acc (snd v) i2

instance container_prod
  (#et : Type0)
  (#ma #ia : Type0)
  (#mb #ib : Type0)
  (gm1 : container ma ia et)
  (gm2 : container mb ib et)
  : container (ma & mb) (either ia ib) et
  = {
    acc = container_prod_acc gm1 gm2;
    upd = (fun (v : ma & mb) (i : either ia ib) (x : et) ->
      match i with
      | Inl i1 -> (gm1.upd (fst v) i1 x, snd v)
      | Inr i2 -> (fst v, gm2.upd (snd v) i2 x)
    );
    l1 = (fun c i v ->
      match i with
      | Inl i1 -> gm1.l1 (fst c) i1 v
      | Inr i2 -> gm2.l1 (snd c) i2 v
    );
    l2 = (fun c i1 i2 v ->
      match (i1, i2) with
      | (Inl j1, Inl j2) -> gm1.l2 (fst c) j1 j2 v
      | (Inr j1, Inr j2) -> gm2.l2 (snd c) j1 j2 v
      | _ -> ()
    );
    ext = (fun c1 c2 _ ->
      assert (forall (x:ia). gm1.acc (fst c1) x == container_prod_acc gm1 gm2 c2 (Inl x));
      assert (forall (x:ib). gm2.acc (snd c1) x == container_prod_acc gm1 gm2 c2 (Inr x));
      let _ = gm1.ext (fst c1) (fst c2) () in
      let _ = gm2.ext (snd c1) (snd c2) () in
      ()
    );
    from_fun = (fun f ->
      (gm1.from_fun (fun i1 -> f (Inl i1)),
       gm2.from_fun (fun i2 -> f (Inr i2)))
    );
    from_fun_ok = (fun f i ->
      match i with
      | Inl i1 -> gm1.from_fun_ok (fun i1' -> f (Inl i1')) i1
      | Inr i2 -> gm2.from_fun_ok (fun i2' -> f (Inr i2')) i2
    );
  }

noextract
instance lseq_container (et:Type) (len:nat) : container (lseq et len) (natlt len) et = {
  acc   = (fun (v : lseq et len) (i : natlt len) -> v @! i);
  upd = (fun v i x -> Seq.upd (reveal v) i x);
  ext = (fun c1 c2 _ -> assert (Seq.equal c1 c2));
  from_fun = (fun f -> Seq.init_ghost len f);
}

instance ghost_map_container #a #b : container (a ^->> b) a b =
  {
    acc = (fun (m : (a ^->> b)) i -> m i);
    upd = (fun m i e -> oplus m i e <: a ^->> b);
    ext = (fun c1 c2 _ -> assert (F.feq_g c1 c2));
    from_fun = (fun f -> F.on_g _ f);
  }

instance dep_ghost_map_container
  (idx : eqtype)
  (mt : Type)
  (i : idx -> Type)
  (e : Type)
  {| sub : (x:idx -> container mt (i x) e) |}
  : container (idx ^->> mt) (x:idx & i x) e
=
  let acc
    (m : idx ^->> mt)
    (idx0 : (x:idx & i x))
    : GTot e =
    let (|idx, it|) = idx0 in
    (sub idx).acc (m idx) it
  in
  let upd
    (m : idx ^->> mt)
    (idx0 : (x:idx & i x))
    (v : e)
    : GTot (idx ^->> mt) =
    let (|idx, it|) = idx0 in
    let m' : mt = (sub idx).upd (m idx) it v in
    oplus m idx m'
  in
  let l1
    (c : idx ^->> mt)
    (idx0 : (x:idx & i x))
    (v : e)
    : squash (acc (upd c idx0 v) idx0 == v) =
    let (| idx, it |) = idx0 in
    (sub idx).l1 (c idx) it v
  in
  let l2
    (c : idx ^->> mt)
    (i1 : (x:idx & i x))
    (i2 : (x:idx & i x){i1 =!= i2})
    (v : e)
    : squash (acc (upd c i2 v) i1 == acc c i1) =
    let (| idx1, it1 |) = i1 in
    let (| idx2, it2 |) = i2 in
    if t2b (idx1 == idx2)
    then
      (sub idx1).l2 (c idx1) it1 it2 v
  in
  let from_fun
    (f : (x:idx & i x) -> GTot e)
    : GTot (idx ^->> mt) =
    F.on_g _
    fun idx ->
      (sub idx).from_fun (fun it -> f (| idx, it |))
  in
  let from_fun_ok
    (f : (x:idx & i x) -> GTot e)
    (idx0 : (x:idx & i x))
    : squash (acc (from_fun f) idx0 == f idx0) =
    let (| idx, it |) = idx0 in
    (sub idx).from_fun_ok (fun it' -> f (| idx, it' |)) it
  in
  let ext
    (c1 : idx ^->> mt)
    (c2 : idx ^->> mt)
    (_ : squash (forall (idx0 : (x:idx & i x)). acc c1 idx0 == acc c2 idx0))
    : squash (c1 == c2) =
    let aux (idx : idx) : Lemma (c1 idx == c2 idx) =
      assert (forall (i : i idx).
                (sub idx).acc (c1 idx) i == acc c1 (| idx, i |));
      assert (forall (i : i idx).
                (sub idx).acc (c2 idx) i == acc c2 (| idx, i |));
      (sub idx).ext (c1 idx) (c2 idx) ()
    in
    Classical.forall_intro aux;
    assert (F.feq_g c1 c2)
  in
  // Bad inference with record notation
  Mkcontainer #(idx ^->> mt) #(x:idx & i x) #e
    acc upd #l1 #l2 ext from_fun #from_fun_ok

val oplus_lemma
  (#ct #it #vt : Type)
  {| gm : container ct it vt |}
  (c : ct)
  (i : it)
  (v : vt)
  : Lemma (acc (upd c i v)
           `F.feq_g`
            oplus (acc c) i v)
