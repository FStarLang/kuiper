module Kuiper.Array2
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.EMatrix
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

inline_for_extraction noextract
let adapt_cit_back (rows cols : erased nat) (idx : raw_cit{cit_fits rows cols idx}) : conc (desc rows cols) =
  match idx with
  | (i, j) -> (i, (j, ()))

let abs_bij (#rows #cols : nat) : (abs (desc rows cols) =~ (ait rows cols)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
    ff_gg = ez;
    gg_ff = ez;
  }

let tr_val (#et : Type) (#rows #cols : nat) (s : ematrix et rows cols)
  : chest (desc rows cols) et
  = Chest.mk (desc rows cols) (fun (i, (j, ())) -> EMatrix.macc s i j)

let t (et : Type0) (#rows #cols : nat) (l : layout rows cols) : Type0 =
  T.tensor et l

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
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (idx : raw_cit{cit_fits rows cols idx})
  (#f : perm)
  (#s : erased (ematrix et rows cols))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == macc s (pi_2_0 idx) (pi_2_1 idx))
{
  unfold pts_to a #f s;
  let v = T.tensor_read a (adapt_cit_back rows cols idx);
  fold pts_to a #f s;
  v
}

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (idx : raw_cit{cit_fits rows cols idx})
  (v : et)
  (#s : erased (ematrix et rows cols))
  requires
    a |-> s
  ensures
    a |-> mupd s (pi_2_0 idx) (pi_2_1 idx) v
{
  unfold pts_to a s;
  T.tensor_write a (adapt_cit_back rows cols idx) v;
  with cs'. assert T.tensor_pts_to a cs';
  assert pure (Chest.equal cs' (tr_val (mupd s (pi_2_0 idx) (pi_2_1 idx) v)));
  fold pts_to a (mupd s (pi_2_0 idx) (pi_2_1 idx) v);
  ()
}

let pts_to_cell
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ij : ait rows cols)
  (v : et)
  : slprop
  = T.tensor_pts_to_cell a #f (adapt_idx_back ij) v

let pts_to_cell_eq
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (ij : ait rows cols) (f : perm) (v : et)
  : Lemma (pts_to_cell a #f ij v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f (adapt_idx_back ij)) v)
  = T.tensor_pts_to_cell_eq a (adapt_idx_back ij) f v

ghost
fn explode
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires a |-> Frac f s
  ensures
    forall+ (ij : ait rows cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))
{
  unfold pts_to a #f s;
  T.tensor_explode a;
  forevery_iso abs_bij _;
  forevery_ext _ (fun (ij : ait rows cols) -> Cell a ij |-> Frac f (macc s (fst ij) (snd ij)));
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
    forall+ (ij : ait rows cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))
  ensures
    a |-> Frac f s
{
  forevery_iso (bij_sym abs_bij) _;
  forevery_ext _ (fun (i : abs (desc rows cols)) -> Cell a i |-> Frac f (acc (tr_val s) i));
  T.tensor_implode a;
  fold pts_to a #f s;
}
