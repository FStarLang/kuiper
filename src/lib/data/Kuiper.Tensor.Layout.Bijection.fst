module Kuiper.Tensor.Layout.Bijection

(* Bijective layouts and specialized variants like folding dimensions. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Bijection
open Kuiper.Shape
open Kuiper.Chest
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT


inline_for_extraction noextract
let ctlayout_bij_cimap
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : tlayout d1) {| c: ctlayout l |}
  (idx: conc d2)
  : Tot (x : szlt l.ulen{SZ.v x == l.imap.f ((f.gg) (up idx))})  =
  fconc_correct idx;
  c.cimap (fconc idx)

inline_for_extraction noextract
instance ctlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2 { all_fit d2 })
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : tlayout d1) {| c: ctlayout l |}
  : ctlayout #r2 #d2 (tlayout_bij f l) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx: conc d2) ->
              fconc_correct idx;
              c.cimap (fconc idx));
  }

ghost
fn tensor_apply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    a |-> Frac fp m
  ensures
    from_array (tlayout_bij f l) (core a) |-> Frac fp (mk d2 (fun a -> acc m (a <~| f)))
{
  sizeof_bijection f;
  assert pure (tlayout_size l == tlayout_size (tlayout_bij f l));
  tensor_concr a;
  tensor_abs' (tlayout_bij f l) (core a);
  assert pure (from_seq (tlayout_bij f l) (to_seq l m) `Kuiper.Chest.equal`
               mk d2 (fun a -> acc m (a <~| f)));
  ()
}

ghost
fn tensor_unapply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (#fp : perm) (#m' : chest d2 et)
  requires
    from_array (tlayout_bij f l) (core a) |-> Frac fp m'
  ensures
    a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i)))
{
  let fa = from_array (tlayout_bij f l) (core a);
  assert rewrites_to fa (from_array (tlayout_bij f l) (core a));
  sizeof_bijection f;
  tensor_concr fa;
  tensor_abs' l (core fa);
  assert pure (
    from_seq l (to_seq (tlayout_bij f l) m')
    `Kuiper.Chest.equal`
    mk d1 (fun i -> acc m' (f.ff i)));
  rewrite each
    from_seq l (to_seq (tlayout_bij f l) m')
  as
    mk d1 (fun i -> acc m' (f.ff i));
  assert pure (from_array l (core fa) == a);
  rewrite each from_array l (core fa) as a;
}

ghost
fn tensor_apply_bij_st_core
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    a |-> Frac fp m
  ensures
    from_array (tlayout_bij f l) (core a)
      |-> Frac fp (mk d2 (fun i -> acc m (i <~| f))) **
    (forall* (m' : chest d2 et).
      from_array (tlayout_bij f l) (core a) |-> Frac fp m' @==>
      a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i))))
{
  let fa = from_array (tlayout_bij f l) (core a);
  assert rewrites_to fa (from_array (tlayout_bij f l) (core a));
  tensor_apply_bij f a;

  ghost
  fn restore (m' : chest d2 et)
    ensures
      fa |-> Frac fp m' @==>
      a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i)))
  {
    ghost
    fn make_trade ()
      ensures
        fa |-> Frac fp m' @==>
        a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i)))
    {
      Pulse.Lib.Trade.intro_trade
        (fa |-> Frac fp m')
        (a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i))))
        emp
        fn _ {
          tensor_unapply_bij f a;
        };
    };
    make_trade ();
  };
  Pulse.Lib.Forall.intro_forall _ restore;
}

ghost
fn tensor_apply_bij_ro_core
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    a |-> Frac fp m
  ensures
    factored
      (from_array (tlayout_bij f l) (core a)
        |-> Frac fp (mk d2 (fun i -> acc m (i <~| f))))
      (a |-> Frac fp m)
{
  tensor_apply_bij_st_core f a;
  Pulse.Lib.Forall.elim_forall (mk d2 (fun i -> acc m (i <~| f)));
  assert pure (
    mk d1
      (fun i -> acc (mk d2 (fun j -> acc m (j <~| f))) (f.ff i))
    `Kuiper.Chest.equal` m);
  rewrite each
    mk d1
      (fun i -> acc (mk d2 (fun j -> acc m (j <~| f))) (f.ff i))
  as m;
}

ghost
fn tensor_apply_bij_st_located_core
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (#loc: loc_id)
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    on loc (a |-> Frac fp m)
  ensures
    on loc
      (from_array (tlayout_bij f l) (core a)
        |-> Frac fp (mk d2 (fun i -> acc m (i <~| f)))) **
    (forall* (m' : chest d2 et).
      on loc (from_array (tlayout_bij f l) (core a) |-> Frac fp m') @==>
      on loc (a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i)))))
{
  let fa = from_array (tlayout_bij f l) (core a);
  assert rewrites_to fa (from_array (tlayout_bij f l) (core a));
  map_loc loc
    #(a |-> Frac fp m)
    #(fa |-> Frac fp (mk d2 (fun i -> acc m (i <~| f))))
    fn () {
      tensor_apply_bij f a;
    };

  ghost
  fn restore (m' : chest d2 et)
    ensures
      on loc (fa |-> Frac fp m') @==>
      on loc (a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i))))
  {
    ghost
    fn make_trade ()
      ensures
        on loc (fa |-> Frac fp m') @==>
        on loc (a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i))))
    {
      Pulse.Lib.Trade.intro_trade
        (on loc (fa |-> Frac fp m'))
        (on loc (a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i)))))
        emp
        fn _ {
          map_loc loc
            #(fa |-> Frac fp m')
            #(a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i))))
            fn () {
              tensor_unapply_bij f a;
            };
        };
    };
    make_trade ();
  };
  Pulse.Lib.Forall.intro_forall _ restore;
}

ghost
fn tensor_apply_bij_ro_located_core
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (#loc: loc_id)
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    on loc (a |-> Frac fp m)
  ensures
    factored
      (on loc
        (from_array (tlayout_bij f l) (core a)
          |-> Frac fp (mk d2 (fun i -> acc m (i <~| f)))))
      (on loc (a |-> Frac fp m))
{
  tensor_apply_bij_st_located_core f a;
  Pulse.Lib.Forall.elim_forall (mk d2 (fun i -> acc m (i <~| f)));
  assert pure (
    mk d1
      (fun i -> acc (mk d2 (fun j -> acc m (j <~| f))) (f.ff i))
    `Kuiper.Chest.equal` m);
  rewrite each
    mk d1
      (fun i -> acc (mk d2 (fun j -> acc m (j <~| f))) (f.ff i))
  as m;
}

fn tensor_apply_bij_ro
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    a |-> Frac fp m
  returns
    fa: tensor et (tlayout_bij f l)
  ensures
    rewrites_to fa (from_array (tlayout_bij f l) (core a)) **
    factored
      (fa |-> Frac fp (mk d2 (fun i -> acc m (i <~| f))))
      (a |-> Frac fp m)
{
  let fa = from_array (tlayout_bij f l) (core a);
  assert rewrites_to fa (from_array (tlayout_bij f l) (core a));
  tensor_apply_bij_ro_core f a;
  fa
}

fn tensor_apply_bij_ro_located
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (#loc: loc_id)
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    on loc (a |-> Frac fp m)
  returns
    fa: tensor et (tlayout_bij f l)
  ensures
    rewrites_to fa (from_array (tlayout_bij f l) (core a)) **
    factored
      (on loc (fa |-> Frac fp (mk d2 (fun i -> acc m (i <~| f)))))
      (on loc (a |-> Frac fp m))
{
  let fa = from_array (tlayout_bij f l) (core a);
  assert rewrites_to fa (from_array (tlayout_bij f l) (core a));
  tensor_apply_bij_ro_located_core f a;
  fa
}

fn tensor_apply_bij_st
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    a |-> Frac fp m
  returns
    fa: tensor et (tlayout_bij f l)
  ensures
    rewrites_to fa (from_array (tlayout_bij f l) (core a)) **
    fa |-> Frac fp (mk d2 (fun i -> acc m (i <~| f))) **
    (forall* (m' : chest d2 et).
      fa |-> Frac fp m' @==>
      a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i))))
{
  let fa = from_array (tlayout_bij f l) (core a);
  assert rewrites_to fa (from_array (tlayout_bij f l) (core a));
  tensor_apply_bij_st_core f a;
  fa
}

fn tensor_apply_bij_st_located
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (#loc: loc_id)
  (a : tensor et l)
  (#fp : perm) (#m : chest d1 et)
  requires
    on loc (a |-> Frac fp m)
  returns
    fa: tensor et (tlayout_bij f l)
  ensures
    rewrites_to fa (from_array (tlayout_bij f l) (core a)) **
    on loc (fa |-> Frac fp (mk d2 (fun i -> acc m (i <~| f)))) **
    (forall* (m' : chest d2 et).
      on loc (fa |-> Frac fp m') @==>
      on loc (a |-> Frac fp (mk d1 (fun i -> acc m' (f.ff i)))))
{
  let fa = from_array (tlayout_bij f l) (core a);
  assert rewrites_to fa (from_array (tlayout_bij f l) (core a));
  tensor_apply_bij_st_located_core f a;
  fa
}

(* ---------------------- Folding outer dimensions ---------------------- *)

let fold_bij (#r: nat {r > 1}) (#d: shape r): (abs d =~ abs (fold_outer d)) = {
  ff = fold_index;
  gg = unfold_index;
  ff_gg = ez;
  gg_ff = ez;
}

let fold_bij_gg (#r: nat {r > 1}) (#d: shape r)
  (x : abs (fold_outer d))
  : Lemma ((fold_bij #r #d).gg x == unfold_index x)
  = ()

inline_for_extraction noextract
let unfold_index_conc
  (#r: erased nat {r > 1})
  (#d: shape r { all_fit d }) {| cs: concrete_sz (desc_top2 d)._2 |}
  (i : conc (fold_outer d)): Tot (conc d) =
  let i : szlt (head d * head (tail d)) & conc (tail (tail d)) = i in
  let (ih, it) = i in
  let ih1: szlt (head d) = ih /^ (concr' cs) in
  let ih2: szlt (head (tail d)) = ih %^ (concr' cs) in
  (ih1, (ih2, it))

let all_fit_fold_outer (#r: nat {r > 1}) (#d: shape r { all_fit d }) (#top2_fits: SZ.fits ((desc_top2 d)._1 * (desc_top2 d)._2)):
  Lemma (all_fit (fold_outer d)) = ()

inline_for_extraction noextract
instance ctlayout_fold_outer
  (#r : nat {r > 1}) (#d : shape r { all_fit d })
  (#top2_fits: SZ.fits ((desc_top2 d)._1 * (desc_top2 d)._2))
  (l : tlayout d) {| c: ctlayout l, cs: concrete_sz (desc_top2 d)._2 |}
  : ctlayout #_ #(fold_outer d) (tlayout_fold_outer l) =
  ctlayout_bij (fold_bij #r #d)
    (unfold_index_conc #r #d #cs)
    (fun (x: conc (fold_outer d)) ->
       (() <: squash (up (unfold_index_conc #r #d #cs x) == unfold_index (up x))))
    l

ghost
fn tensor_fold_outer
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d)
  (a : tensor et l)
  (#f : perm) (#m : chest d et)
  requires
    a |-> Frac f m
  ensures
    from_array (tlayout_fold_outer l) (core a) |-> Frac f (fold_chest m)
{
  tensor_pts_to_ref a;
  tensor_explode a;
  (* forall+ (i : abs d). Cell a i |-> Frac f (acc m i) *)

  forevery_iso fold_bij
    (fun (i : abs d) -> Cell a i |-> Frac f (acc m i));
  (* forall+ (j : abs (fold_outer d)).
        Cell a (fold_bij.gg j) |-> Frac f (acc m (fold_bij.gg j)) *)

  forevery_map
    (fun (j : abs (fold_outer d)) ->
      Cell a (fold_bij.gg j) |-> Frac f (acc m (fold_bij.gg j)))
    (fun (j : abs (fold_outer d)) ->
      Cell (from_array (tlayout_fold_outer l) (core a)) j
        |-> Frac f (acc (fold_chest m) j))
    fn j {
      tensor_pts_to_cell_eq a (fold_bij.gg j) f (acc m (fold_bij.gg j));
      tensor_pts_to_cell_eq (from_array (tlayout_fold_outer l) (core a)) j f
        (acc (fold_chest m) j);
      rewrite
        Cell a (fold_bij.gg j) |-> Frac f (acc m (fold_bij.gg j))
      as
        Cell (from_array (tlayout_fold_outer l) (core a)) j
          |-> Frac f (acc (fold_chest m) j);
    };

  tensor_implode (from_array (tlayout_fold_outer l) (core a));
}

ghost
fn tensor_unfold_outer
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d)
  (a : tensor et (tlayout_fold_outer l))
  (#f: perm) (#m : chest (fold_outer d) et)
  requires
    a |-> Frac f m
  ensures
    from_array l (core a) |-> Frac f (unfold_chest m)
{
  tensor_pts_to_ref a;
  tensor_explode a;
  (* forall+ (j : abs (fold_outer d)). Cell a j |-> Frac f (acc m j) *)

  forevery_iso (bij_sym fold_bij)
    (fun (j : abs (fold_outer d)) -> Cell a j |-> Frac f (acc m j));
  (* forall+ (i : abs d).
        Cell a (fold_bij.ff i) |-> Frac f (acc m (fold_bij.ff i)) *)

  forevery_map
    (fun (i : abs d) ->
      Cell a (fold_bij.ff i) |-> Frac f (acc m (fold_bij.ff i)))
    (fun (i : abs d) ->
      Cell (from_array l (core a)) i
        |-> Frac f (acc (unfold_chest m) i))
    fn i {
      tensor_pts_to_cell_eq a (fold_bij.ff i) f (acc m (fold_bij.ff i));
      tensor_pts_to_cell_eq (from_array l (core a)) i f
        (acc (unfold_chest m) i);
      rewrite
        Cell a (fold_bij.ff i) |-> Frac f (acc m (fold_bij.ff i))
      as
        Cell (from_array l (core a)) i
          |-> Frac f (acc (unfold_chest m) i);
    };

  tensor_implode (from_array l (core a));
}

inline_for_extraction noextract
fn tensor_fold_ro
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d { is_full l })
  (a : tensor et l)
  (#f : perm) (#m : chest d et)
  requires
    a |-> Frac f m
  returns
    fa: tensor et (tlayout_fold_outer l)
  ensures
    rewrites_to fa (from_array (tlayout_fold_outer l) (core a)) **
    factored
      (fa |-> Frac f (fold_chest m))
      (a |-> Frac f m)
{
  tensor_apply_bij_ro fold_bij a
}

inline_for_extraction noextract
fn tensor_fold_ro_located
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d { is_full l })
  (#loc: loc_id)
  (a : tensor et l)
  (#f : perm) (#m : chest d et)
  requires
    on loc (a |-> Frac f m)
  returns
    fa: tensor et (tlayout_fold_outer l)
  ensures
    rewrites_to fa (from_array (tlayout_fold_outer l) (core a)) **
    factored
      (on loc (fa |-> Frac f (fold_chest m)))
      (on loc (a |-> Frac f m))
{
  tensor_apply_bij_ro_located fold_bij a
}

inline_for_extraction noextract
fn tensor_fold_st
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d { is_full l })
  (a : tensor et l)
  (#f : perm) (#m : chest d et)
  requires
    a |-> Frac f m
  returns
    fa: tensor et (tlayout_fold_outer l)
  ensures
    rewrites_to fa (from_array (tlayout_fold_outer l) (core a)) **
    fa |-> Frac f (fold_chest m) **
    (forall* (m' : chest (fold_outer d) et).
      fa |-> Frac f m' @==>
      a |-> Frac f (unfold_chest m'))
{
  tensor_apply_bij_st fold_bij a
}

inline_for_extraction noextract
fn tensor_fold_st_located
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d { is_full l })
  (#loc: loc_id)
  (a : tensor et l)
  (#f : perm) (#m : chest d et)
  requires
    on loc (a |-> Frac f m)
  returns
    fa: tensor et (tlayout_fold_outer l)
  ensures
    rewrites_to fa (from_array (tlayout_fold_outer l) (core a)) **
    (on loc (fa |-> Frac f (fold_chest m))) **
    (forall* (m' : chest (fold_outer d) et).
      (on loc (fa |-> Frac f m')) @==>
      (on loc (a |-> Frac f (unfold_chest m'))))
{
  tensor_apply_bij_st_located fold_bij a
}

let unfold_fold_chest_id (#et:Type0) (#r:nat{r>1}) (#d:shape r) (m : chest d et)
  : Lemma (unfold_chest #et #r #d (fold_chest #et #r #d m) == m)
  = let ICons h1 (ICons h2 ts) = d in
    introduce forall (i : abs d).
      acc (unfold_chest #et #r #d (fold_chest #et #r #d m)) i == acc m i
    with (
      let (i1,(i2,it)) : (natlt h1 & (natlt h2 & abs ts)) = i in
      FStar.Math.Lemmas.lemma_div_plus i2 i1 h2;
      FStar.Math.Lemmas.lemma_mod_plus i2 i1 h2;
      FStar.Math.Lemmas.small_div i2 h2;
      FStar.Math.Lemmas.small_mod i2 h2;
      assert (unfold_index #r #d (fold_index #r #d i) == i)
    );
    Kuiper.Chest.lemma_equal_intro (unfold_chest #et #r #d (fold_chest #et #r #d m)) m;
    Kuiper.Chest.ext (unfold_chest #et #r #d (fold_chest #et #r #d m)) m

let fold_unfold_chest_id (#et:Type0) (#r:nat {r>1}) (#d:shape r)
  (m : chest (fold_outer d) et)
  : Lemma (fold_chest (unfold_chest m) == m) =
  let lhs = fold_chest (unfold_chest m) in
  introduce forall (i : abs (fold_outer d)).
    acc lhs i == acc m i
  with (
    assert (fold_index #r #d (unfold_index #r #d i) == i)
  );
  Kuiper.Chest.lemma_equal_intro lhs m;
  Kuiper.Chest.ext lhs m