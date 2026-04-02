module Kuiper.Array3
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let desc (d0 d1 d2 : nat) : idesc 3 =
  d0 @| d1 @| d2 @| INil

let adapt_idx (#d0 #d1 #d2 : nat) (idx : abs (desc d0 d1 d2)) : natlt d0 & natlt d1 & natlt d2 =
  match idx with
  | (i, (j, (k, ()))) -> (i, j, k)

let adapt_idx_back (#d0 #d1 #d2 : nat) (idx : natlt d0 & natlt d1 & natlt d2) : abs (desc d0 d1 d2) =
  match idx with
  | (i, j, k) -> (i, (j, (k, ())))

#push-options "--ifuel 3" // sigh
let tr_layout (#d0 #d1 #d2 : nat) (l : layout d0 d1 d2) : tlayout (desc d0 d1 d2) = {
  ulen = l.ulen;
  imap = mk_injection (fun idx -> l.imap.f (adapt_idx idx)) ez;
}
#pop-options

#push-options "--ifuel 3" // sigh
let abs_bij (#d0 #d1 #d2 : nat) : (abs (desc d0 d1 d2) =~ (natlt d0 & natlt d1 & natlt d2)) =
  {
    ff = (fun (i, (j, (k, ()))) -> (i, j, k));
    gg = (fun (i, j, k) -> (i, (j, (k, ()))));
    ff_gg = ez;
    gg_ff = ez;
  }
#pop-options

let tr_val (#et : Type) (#d0 #d1 #d2 : nat) (s : EMatrix3.t et d0 d1 d2)
  : chest (desc d0 d1 d2) et
  = Chest.mk (desc d0 d1 d2) (fun (i, (j, (k, ()))) -> EMatrix3.macc s i j k)

inline_for_extraction noextract
instance clayout_to_tlayout (#d0 #d1 #d2 : erased nat) (#l : layout d0 d1 d2)
  (c : clayout l)
  : ctlayout (tr_layout l) =
{
  culen   = c.culen;
  all_fit = ();
  cimap   = (fun (idx : conc (desc d0 d1 d2)) ->
              match idx with
              | (i, (j, (k, ()))) -> c.cimap i j k);
}

let t (et : Type0) (#d0 #d1 #d2 : nat) (l : layout d0 d1 d2) : Type0 =
  T.tensor et (tr_layout l)

let is_global (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : prop =
  T.is_global_tensor a

let from_array
  (#et : Type0) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (a : gpu_array et (layout_size l))
  : t et l
  = T.from_array _ a

let core
  (#et : Type0) (#d0 #d1 #d2 : erased nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : gpu_array et (layout_size l)
  = T.core a

let lem_core_from_array
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
   = ()

let lem_from_array_core
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let pts_to
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : EMatrix3.t et d0 d1 d2)
  : slprop
  = T.tensor_pts_to a #f (tr_val s)

instance is_send_across_global
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l { is_global a })
  (#f : perm) (s : EMatrix3.t et d0 d1 d2)
  : is_send_across gpu_of (pts_to a #f s)
  = solve

ghost
fn pts_to_ref
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm) (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))
{
  unfold pts_to a #f s;
  T.tensor_pts_to_ref a;
  fold pts_to a #f s;
}

ghost
fn share_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold pts_to a #f s;
  T.tensor_share_n a k;
  forevery_map
    (fun (i:natlt k) -> T.tensor_pts_to a #(f /. k) (tr_val s))
    (fun (i:natlt k) -> pts_to a #(f /. k) s)
    fn i { fold pts_to a #(f /. k) s };
}

ghost
fn gather_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i:natlt k) -> pts_to a #(f /. k) s)
    (fun (i:natlt k) -> T.tensor_pts_to a #(f /. k) (tr_val s))
    fn i { unfold pts_to a #(f /. k) s };
  T.tensor_gather_n a k;
  fold pts_to a #f s;
}

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| clayout l |}
  (a : t et l)
  (i : szlt d0)
  (j : szlt d1)
  (k : szlt d2)
  (#f : perm)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == EMatrix3.macc s i j k)
{
  unfold pts_to a #f s;
  let v = T.tensor_read a (i, (j, (k, ())));
  fold pts_to a #f s;
  v
}

#push-options "--ifuel 3" // sigh
inline_for_extraction noextract
fn write
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| clayout l |}
  (a : t et l)
  (i : szlt d0)
  (j : szlt d1)
  (k : szlt d2)
  (v : et)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  requires a |-> s
  ensures  a |-> EMatrix3.mupd s i j k v
{
  unfold pts_to a s;
  T.tensor_write a (i, (j, (k, ()))) v;
  with cs'. assert T.tensor_pts_to a cs';
  assert pure (Chest.equal cs' (tr_val (EMatrix3.mupd s i j k v)));
  fold pts_to a (EMatrix3.mupd s i j k v);
  ()
}
#pop-options

let pts_to_cell
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ijk : natlt d0 & natlt d1 & natlt d2)
  (v : et)
  : slprop
  = T.tensor_pts_to_cell a #f (adapt_idx_back ijk) v

let pts_to_cell_eq
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (ijk : natlt d0 & natlt d1 & natlt d2) (f : perm) (v : et)
  : Lemma (pts_to_cell a #f ijk v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f ijk) v)
  = T.tensor_pts_to_cell_eq a (adapt_idx_back ijk) f v

ghost
fn explode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires a |-> Frac f s
  ensures
    forall+ (ijk : natlt d0 & natlt d1 & natlt d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))
{
  unfold pts_to a #f s;
  T.tensor_explode a;
  forevery_iso abs_bij _;
  forevery_ext _ (fun (ijk : natlt d0 & natlt d1 & natlt d2) -> Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)));
  ()
}

ghost
fn implode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ijk : natlt d0 & natlt d1 & natlt d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))
  ensures
    a |-> Frac f s
{
  forevery_iso (bij_sym abs_bij) _;
  forevery_ext _ (fun (i : abs (desc d0 d1 d2)) -> Cell a i |-> Frac f (acc (tr_val s) i));
  T.tensor_implode a;
  fold pts_to a #f s;
}
