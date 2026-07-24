module Kuiper.Tensor.Layout.Bijection

(* Bijective layouts and specialized variants like folding dimensions. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Bijection
open Kuiper.Shape
open Kuiper.Chest
open Pulse.Lib.Trade { (@==>) }

module SZ = Kuiper.SizeT

let tlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (l : tlayout d1)
  : tlayout d2
  = {
      ulen = l.ulen;
      imap = inj_bij' f `Kuiper.Injection.inj_comp` l.imap;
  }

inline_for_extraction noextract
instance val ctlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2 { all_fit d2 })
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : tlayout d1) {| c: ctlayout l |}
  : ctlayout #r2 #d2 (tlayout_bij f l)

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
    from_array (tlayout_bij f l) (core a) |-> Frac fp (mk d2 (fun i -> acc m (i <~| f)))


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

(* ---------------------- Folding outer dimensions ---------------------- *)

let unfold_index (#r: nat {r > 1}) (#d: shape r) (i : abs (fold_outer d)): GTot (abs d) =
  let ICons h1 (ICons h2 ts) = d in
  let i : natlt (h1 * h2) & abs ts = i in
  let (ih, it) = i in
  (((ih / h2 <: natlt h1), ((ih % h2 <: natlt h2), it)))

let fold_index (#r: nat {r > 1}) (#d: shape r) (i : abs d): GTot (abs (fold_outer d)) =
  let ICons h1 (ICons h2 ts) = d in
  let i : natlt h1 & (natlt h2 & abs ts) = i in
  let (ih1, (ih2, it)) = i in
  let ih12 : natlt (h1 * h2) = ih1 * h2 + ih2 in
  (ih12, it)

[@@erasable] // avoid silly warning
val fold_bij (#r: nat {r > 1}) (#d: shape r): abs d =~ abs (fold_outer d)

let fold_chest (#et : Type0) (#r: nat {r > 1}) (#d: shape r) (m : chest d et): GTot (chest (fold_outer d) et) =
  mk (fold_outer d) (fun i -> acc m (unfold_index i))

let unfold_chest (#et : Type0) (#r: nat {r > 1}) (#d: shape r) (m : chest (fold_outer d) et): GTot (chest d et) =
  mk d (fun i -> acc m (fold_index i))

unfold let tlayout_fold_outer
  (#r : nat {r > 1}) (#d : shape r)
  (l : tlayout d) = tlayout_bij fold_bij l

unfold
let desc_top2 (#r: nat {r > 1}) (d: shape r): GTot (nat & nat & shape (r-2)) = (head d, head (tail d), tail (tail d))

inline_for_extraction noextract
instance val ctlayout_fold_outer
  (#r : nat {r > 1}) (#d : shape r { all_fit d })
  (#top2_fits: SZ.fits ((desc_top2 d)._1 * (desc_top2 d)._2))
  (l : tlayout d) {| c: ctlayout l, cs: concrete_sz (desc_top2 d)._2 |}
  : ctlayout #_ #(fold_outer d) (tlayout_fold_outer l)

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

val unfold_fold_chest_id (#et:Type0) (#r:nat{r>1}) (#d:shape r) (m : chest d et)
  : Lemma (unfold_chest #et #r #d (fold_chest #et #r #d m) == m)
  [SMTPat (unfold_chest (fold_chest m))]

val fold_unfold_chest_id (#et:Type0) (#r:nat{r>1}) (#d:shape r) (m : chest (fold_outer d) et)
  : Lemma (fold_chest (unfold_chest m) == m)
  [SMTPat (fold_chest (unfold_chest m))]