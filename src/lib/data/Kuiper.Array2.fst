module Kuiper.Array2
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.EMatrix
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let tr_layout (#rows #cols : nat) (l : layout rows cols) : tlayout (ICons rows (ICons cols INil)) = {
  ulen = l.ulen;
  imap = mk_injection (fun (i, (j, ())) -> l.imap.f (i, j)) ez;
}

let abs_bij (#rows #cols : nat) : (abs (ICons rows (ICons cols INil)) =~ (natlt rows & natlt cols)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
    ff_gg = ez;
    gg_ff = ez;
  }

let tr_val (#et : Type) (#rows #cols : nat) (s : ematrix et rows cols) : chest (ICons rows (ICons cols INil)) et =
  Chest.mk (ICons rows (ICons cols INil)) (fun (i, (j, ())) -> macc s i j)

inline_for_extraction noextract
instance clayout_to_tlayout (#rows #cols : erased nat) (#l : layout rows cols)
  (c : clayout l)
  : ctlayout (tr_layout l) =
{
  culen   = c.culen;
  all_fit = ();
  cimap   = (fun (idx : conc (ICons rows (ICons cols INil))) ->
              match idx with
              | (i, (j, ())) -> c.cimap i j);
}

let t (et : Type0) (#rows #cols : nat) (l : layout rows cols) : Type0 =
  T.tensor et (tr_layout l)

let is_global (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  : prop =
  T.is_global_tensor a

let from_array
  (#et : Type0) (#rows #cols : erased nat)
  (l : layout rows cols)
  (a : gpu_array et (layout_size l))
  : t et l
  = T.from_array _ a

let core
  (#et : Type0) (#rows #cols : erased nat) (#l : layout rows cols)
  (a : t et l)
  : gpu_array et (layout_size l)
  = T.core a

let lem_core_from_array
  (#et : Type) (#rows #cols : erased nat)
  (#l : layout rows cols)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
   = ()

let lem_from_array_core
  (#et : Type) (#rows #cols : erased nat)
  (l : layout rows cols)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let pts_to
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : ematrix et rows cols)
  : slprop
  = T.tensor_pts_to a #f (tr_val s)

instance is_send_across_global
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l { is_global a })
  (#f : perm) (s : ematrix et rows cols)
  : is_send_across gpu_of (pts_to a #f s)
  = solve

ghost
fn pts_to_ref
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm) (#s : erased (ematrix et rows cols))
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
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (k : pos)
  (#f : perm) (#s : ematrix et rows cols)
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
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (k : pos)
  (#f : perm) (#s : ematrix et rows cols)
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
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| clayout l |}
  (a : t et l)
  (i : szlt rows)
  (j : szlt cols)
  (#f : perm)
  (#s : erased (ematrix et rows cols))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == macc s i j)
{
  unfold pts_to a #f s;
  let v = T.tensor_read a (i, (j, ()));
  fold pts_to a #f s;
  v
}

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| clayout l |}
  (a : t et l)
  (i : szlt rows)
  (j : szlt cols)
  (v : et)
  (#s : erased (ematrix et rows cols))
  requires a |-> s
  ensures  a |-> (mupd s i j v <: ematrix et rows cols)
{
  unfold pts_to a s;
  T.tensor_write a (i, (j, ())) v;
  with cs'. assert T.tensor_pts_to a cs';
  assert pure (Chest.equal cs' (tr_val (mupd s i j v)));
  fold pts_to a (mupd s i j v);
  ()
}

let pts_to_cell
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ij : natlt rows & natlt cols)
  (v : et)
  : slprop
  = T.tensor_pts_to_cell a #f (fst ij, (snd ij, ())) v

let pts_to_cell_eq
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (ij : natlt rows & natlt cols) (f : perm) (v : et)
  : Lemma (pts_to_cell a #f ij v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f ij) v)
  = T.tensor_pts_to_cell_eq a (fst ij, (snd ij, ())) f v

ghost
fn explode
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires a |-> Frac f s
  ensures
    forall+ (ij : natlt rows & natlt cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))
{
  unfold pts_to a #f s;
  T.tensor_explode a;
  forevery_iso abs_bij _;
  forevery_ext _ (fun (ij : natlt rows & natlt cols) -> Cell a ij |-> Frac f (macc s (fst ij) (snd ij)));
  ()
}

ghost
fn implode
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ij : natlt rows & natlt cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))
  ensures
    a |-> Frac f s
{
  forevery_iso (bij_sym abs_bij) _;
  forevery_ext _ (fun (i : abs (ICons rows (ICons cols INil))) -> Cell a i |-> Frac f (acc (tr_val s) i));
  T.tensor_implode a;
  fold pts_to a #f s;
}
