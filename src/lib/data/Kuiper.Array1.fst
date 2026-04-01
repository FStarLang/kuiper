module Kuiper.Array1
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let tr_layout (#len : nat) (l : layout len) : tlayout (ICons len INil) = {
  ulen = l.ulen;
  imap = mk_injection (fun (i, ()) -> l.imap.f i) ez;
}

let abs_bij (#len : nat) : (abs (ICons len INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
    ff_gg = ez;
    gg_ff = ez;
  }

let tr_val (#et : Type) (#len : nat) (s : lseq et len) : chest (ICons len INil) et =
  Chest.mk (ICons len INil) (fun (i, ()) -> s @! i)

inline_for_extraction noextract
instance clayout_to_tlayout (#len : erased nat) (#l : layout len)
  (c : clayout l)
  : ctlayout (tr_layout l) =
{
  culen   = c.culen;
  all_fit = ();
  cimap   = (fun (idx : conc (ICons len INil)) ->
              match idx with
              | (i, ()) -> c.cimap i);
}

let t (et : Type0) (#len : nat) (l : layout len) : Type0 =
  T.tensor et (tr_layout l)

let is_global (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  : prop =
  T.is_global_tensor a

let from_array
  (#et : Type0) (#len : erased nat)
  (l : layout len)
  (a : gpu_array et (layout_size l))
  : t et l
  = T.from_array _ a

let core
  (#et : Type0) (#len : erased nat) (#l : layout len)
  (a : t et l)
  : gpu_array et (layout_size l)
  = T.core a

let lem_core_from_array
  (#et : Type) (#len : erased nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
   = ()

let lem_from_array_core
  (#et : Type) (#len : erased nat)
  (l : layout len)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#len : nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let pts_to
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : lseq et len)
  : slprop
  = T.tensor_pts_to a #f (tr_val s)

instance is_send_across_global
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l { is_global a })
  (#f : perm) (s : lseq et len)
  : is_send_across gpu_of (pts_to a #f s)
  = solve

ghost
fn pts_to_ref
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm) (#s : erased (lseq et len))
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
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
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
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
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
  (#len : erased nat)
  (#l : layout len) {| clayout l |}
  (a : t et l)
  (i : szlt len)
  (#f : perm)
  (#s : erased (lseq et len))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == Seq.index s i)
{
  unfold pts_to a #f s;
  let idx = (i, ());
  let v = T.tensor_read a idx;
  fold pts_to a #f s;
  v
}

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| clayout l |}
  (a : t et l)
  (i : szlt len)
  (v : et)
  (#s : erased (lseq et len))
  requires a |-> s
  ensures  a |-> (Seq.upd s i v <: lseq et len)
{
  unfold pts_to a s;
  T.tensor_write a (i, ()) v;
  with cs'. assert T.tensor_pts_to a cs';
  assert pure (Chest.equal cs' (tr_val (Seq.upd s i v)));
  fold pts_to a (Seq.upd s i v);
  ()
}

let pts_to_cell
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] i : natlt len)
  (v : et)
  : slprop
  = T.tensor_pts_to_cell a #f (i, ()) v

let pts_to_cell_eq
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l) (i : natlt len) (f : perm) (v : et)
  : Lemma (pts_to_cell a #f i v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f i) v)
  = T.tensor_pts_to_cell_eq a (i, ()) f v

ghost
fn explode
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm)
  (#s : lseq et len)
  requires a |-> Frac f s
  ensures
    forall+ (i : natlt len).
      Cell a i |-> Frac f (Seq.index s i)
{
  unfold pts_to a #f s;
  T.tensor_explode a;
  forevery_iso abs_bij _;
  forevery_ext _ (fun (i : natlt len) -> Cell a i |-> Frac f (Seq.index s i));
  ()
}

ghost
fn implode
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm)
  (#s : lseq et len)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (i : natlt len).
      Cell a i |-> Frac f (Seq.index s i)
  ensures
    a |-> Frac f s
{
  forevery_iso (bij_sym abs_bij) _;
  forevery_ext _ (fun (i : abs (ICons len INil)) -> Cell a i |-> Frac f (acc (tr_val s) i));
  T.tensor_implode a;
  fold pts_to a #f s;
}
