module Kuiper.Example.TensorRO

(* Zero-copy Tensor -> ROTensor flow with two broadcast dimensions, followed
   by recovery of the original writable tensor. *)

#lang-pulse
open Kuiper
open Kuiper.TensorRO
open Kuiper.Tensor.Layout.Alg
module KT = Kuiper.Tensor
module SZ = Kuiper.SizeT

(* Concrete instance crutch (mirrors Kuiper.Example.Tensor). *)
inline_for_extraction noextract
instance _crutch_vec (m : erased nat{SZ.fits m})
  : Kuiper.Tensor.Layout.ctlayout (l1_forward m)
  = c_l1_forward m

(* Arithmetic cimap avoids an extracted index struct; the zero-weighted
   broadcast coordinates fold away, leaving [t[i]]. *)
inline_for_extraction noextract
let bcast2_cimap
  (m : erased nat{SZ.fits m}) (e1 : erased nat{SZ.fits e1}) (e2 : erased nat{SZ.fits e2})
  (i : conc (e2 @| e1 @| (m @| INil)))
  : r : SZ.t { SZ.v r ==
      (extended_layout (extended_layout (vtlayout_of_tlayout (l1_forward m)) e1) e2).imap (up i) }
  = let (a, (b, rest0)) = i in
    let base : SZ.t = (c_l1_forward m).cimap rest0 in
    base `SZ.add` (0sz `SZ.mul` a) `SZ.add` (0sz `SZ.mul` b)

inline_for_extraction noextract
instance cvt_bcast2
  (m : erased nat{SZ.fits m}) (e1 : erased nat{SZ.fits e1}) (e2 : erased nat{SZ.fits e2})
  : cvtlayout (extended_layout (extended_layout (vtlayout_of_tlayout (l1_forward m)) e1) e2)
  = {
      ulen_fits = ();
      all_fit = ();
      cimap = bcast2_cimap m e1 e2;
    }

(* Read through two broadcast dimensions, remove them in reverse order, and
   restore the original writable tensor. *)
inline_for_extraction noextract
fn ex_broadcast_two_read
  (#m : erased nat{SZ.fits m})
  (#e1 : (erased nat){e1 > 0 /\ SZ.fits e1})
  (#e2 : (erased nat){e2 > 0 /\ SZ.fits e2})
  (t : KT.tensor u32 (l1_forward m))
  (i : szlt m)
  (j1 : szlt e1)
  (j2 : szlt e2)
  (#f : perm)
  (#s : chest1 u32 m)
  requires
    KT.tensor_pts_to t #f s ** pure (is_full (l1_forward m))
  returns
    v : u32
  ensures
    KT.tensor_pts_to t #f s ** pure (v == acc1 s i)
{
  let ro = tensor_to_rotensor t;
  let b1 = rotensor_add_dim_view ro e1;
  let b2 = rotensor_add_dim_view b1 e2;
  let v = tensor_read #_ #_ #_ #_ #(cvt_bcast2 m e1 e2) b2 (j2, (j1, (i, ())));
  rotensor_remove_dim b1 e2;
  rotensor_remove_dim ro e1;
  rotensor_to_tensor t ro;
  v
}

(* Concrete extraction witness: generated addressing is [return t[i];]. *)
fn ex_read_vec10_bcast
  (t : KT.tensor u32 (l1_forward 10))
  (i : szlt 10)
  (j1 : szlt 4)
  (j2 : szlt 7)
  (#f : perm)
  (#s : chest1 u32 10)
  requires
    KT.tensor_pts_to t #f s ** pure (is_full (l1_forward 10))
  returns
    v : u32
  ensures
    KT.tensor_pts_to t #f s ** pure (v == acc1 s i)
{
  ex_broadcast_two_read #10 #4 #7 t i j1 j2
}
