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


(* ---------------------- Folding outer dimensions ---------------------- *)

let fold_bij (#r: nat {r > 1}) (#d: shape r): (abs d =~ abs (fold_outer d)) = {
  ff = fold_index;
  gg = unfold_index;
  ff_gg = ez;
  gg_ff = ez;
}

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