module Kuiper.Array4
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.EMatrix4
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

inline_for_extraction noextract
let adapt_cit_back (d0 d1 d2 d3 : erased nat) (idx : raw_cit{cit_fits d0 d1 d2 d3 idx}) : conc (desc d0 d1 d2 d3) =
  match idx with
  | (i, j, k, l) -> (i, (j, (k, (l, ()))))

#push-options "--ifuel 4" // sigh
let abs_bij (#d0 #d1 #d2 #d3 : nat) : (abs (desc d0 d1 d2 d3) =~ (ait d0 d1 d2 d3)) =
  {
    ff = (fun (i, (j, (k, (l, ())))) -> (i, j, k, l));
    gg = (fun (i, j, k, l) -> (i, (j, (k, (l, ())))));
    ff_gg = ez;
    gg_ff = ez;
  }
#pop-options

let tr_val (#et : Type) (#d0 #d1 #d2 #d3 : nat) (s : EMatrix4.t et d0 d1 d2 d3)
  : chest (desc d0 d1 d2 d3) et
  = Chest.mk (desc d0 d1 d2 d3) (fun (i, (j, (k, (l, ())))) -> EMatrix4.macc s i j k l)

let t (et : Type0) (#d0 #d1 #d2 #d3 : nat) (l : layout d0 d1 d2 d3) : Type0 =
  T.tensor et l

let is_global (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : prop =
  T.is_global a

let from_array
  (#et : Type0) (#d0 #d1 #d2 #d3 : erased nat)
  (l : layout d0 d1 d2 d3)
  (a : gpu_array et (layout_size l))
  : t et l
  = T.from_array _ a

let core
  (#et : Type0) (#d0 #d1 #d2 #d3 : erased nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : gpu_array et (layout_size l)
  = T.core a

let lem_core_from_array
  (#et : Type) (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
   = ()

let lem_from_array_core
  (#et : Type) (#d0 #d1 #d2 #d3 : erased nat)
  (l : layout d0 d1 d2 d3)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat)
  (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let pts_to
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : EMatrix4.t et d0 d1 d2 d3)
  : slprop
  = T.tensor_pts_to a #f (tr_val s)

instance is_send_across_global
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l { is_global a })
  (#f : perm) (s : EMatrix4.t et d0 d1 d2 d3)
  : is_send_across gpu_of (pts_to a #f s)
  = solve

ghost
fn pts_to_ref
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  (#f : perm) (#s : erased (EMatrix4.t et d0 d1 d2 d3))
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
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix4.t et d0 d1 d2 d3)
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
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix4.t et d0 d1 d2 d3)
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
  (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3) {| ctlayout l |}
  (a : t et l)
  (idx : raw_cit{cit_fits d0 d1 d2 d3 idx})
  (#f : perm)
  (#s : erased (EMatrix4.t et d0 d1 d2 d3))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == EMatrix4.macc s (pi_4_0 idx) (pi_4_1 idx) (pi_4_2 idx) (pi_4_3 idx))
{
  unfold pts_to a #f s;
  let v = T.tensor_read a (adapt_cit_back d0 d1 d2 d3 idx);
  fold pts_to a #f s;
  v
}

#push-options "--ifuel 4" // sigh
inline_for_extraction noextract
fn write
  (#et : Type0)
  (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3) {| ctlayout l |}
  (a : t et l)
  (idx : raw_cit{cit_fits d0 d1 d2 d3 idx})
  (v : et)
  (#s : erased (EMatrix4.t et d0 d1 d2 d3))
  requires
    a |-> s
  ensures
    a |-> EMatrix4.mupd s (pi_4_0 idx) (pi_4_1 idx) (pi_4_2 idx) (pi_4_3 idx) v
{
  unfold pts_to a s;
  T.tensor_write a (adapt_cit_back d0 d1 d2 d3 idx) v;
  with cs'. assert T.tensor_pts_to a cs';
  assert pure (Chest.equal cs' (tr_val (EMatrix4.mupd s (pi_4_0 idx) (pi_4_1 idx) (pi_4_2 idx) (pi_4_3 idx) v)));
  fold pts_to a (EMatrix4.mupd s (pi_4_0 idx) (pi_4_1 idx) (pi_4_2 idx) (pi_4_3 idx) v);
  ()
}
#pop-options

let pts_to_cell
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ijkl : ait d0 d1 d2 d3)
  (v : et)
  : slprop
  = T.tensor_pts_to_cell a #f (adapt_idx_back ijkl) v

let pts_to_cell_eq
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l) (ijkl : ait d0 d1 d2 d3) (f : perm) (v : et)
  : Lemma (pts_to_cell a #f ijkl v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f (adapt_idx_back ijkl)) v)
  = T.tensor_pts_to_cell_eq a (adapt_idx_back ijkl) f v

ghost
fn explode
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix4.t et d0 d1 d2 d3)
  requires a |-> Frac f s
  ensures
    forall+ (ijkl : ait d0 d1 d2 d3).
      Cell a ijkl |-> Frac f (EMatrix4.macc s (pi_4_0 ijkl) (pi_4_1 ijkl) (pi_4_2 ijkl) (pi_4_3 ijkl))
{
  unfold pts_to a #f s;
  T.tensor_explode a;
  forevery_iso abs_bij _;
  forevery_ext _ (fun (ijkl : ait d0 d1 d2 d3) -> Cell a ijkl |-> Frac f (EMatrix4.macc s (pi_4_0 ijkl) (pi_4_1 ijkl) (pi_4_2 ijkl) (pi_4_3 ijkl)));
  ()
}

ghost
fn implode
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix4.t et d0 d1 d2 d3)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ijkl : ait d0 d1 d2 d3).
      Cell a ijkl |-> Frac f (EMatrix4.macc s (pi_4_0 ijkl) (pi_4_1 ijkl) (pi_4_2 ijkl) (pi_4_3 ijkl))
  ensures
    a |-> Frac f s
{
  forevery_iso (bij_sym abs_bij) _;
  forevery_ext _ (fun (i : abs (desc d0 d1 d2 d3)) -> Cell a i |-> Frac f (acc (tr_val s) i));
  T.tensor_implode a;
  fold pts_to a #f s;
}
