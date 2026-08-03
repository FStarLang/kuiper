module Kuiper.Example.TensorRO

(* Demonstrates the read-only tensor borrow: view a writable vector as a
   read-only tensor (zero-copy), read an element, and recover the original
   writable vector once the borrow is no longer used.

   The borrow is done with [tensor_to_rotensor], which *returns* the read-only
   view, and undone with the named [rotensor_to_tensor]; neither the backing
   array expression ([from_array .. (core t)]) nor the generic trade eliminator
   ever shows up in client code. *)

#lang-pulse
open Kuiper
open Kuiper.TensorRO
open Kuiper.Tensor.Layout.Alg
open Kuiper.Shape
open Kuiper.Bijection
module KT = Kuiper.Tensor
module Ch = Kuiper.Chest
module SZ = Kuiper.SizeT

let swap2 (#a #b : nat)
  : (abs (a @| b @| INil) =~ abs (b @| a @| INil))
  = {
      ff = (function (i, (j, ())) -> (j, (i, ())));
      gg = (function (j, (i, ())) -> (i, (j, ())));
    }

(* Concrete instance crutch (mirrors Kuiper.Example.Tensor). *)
inline_for_extraction noextract
instance _crutch_vec (m : erased nat{SZ.fits m})
  : Kuiper.Tensor.Layout.ctlayout (l1_forward m)
  = c_l1_forward m

(* Borrow the writable vector [t] as a read-only tensor, read element [i]
   (== the vector's [i]-th entry), then recover [t] unchanged. *)
inline_for_extraction noextract
fn ex_borrow_read
  (#m : erased nat{SZ.fits m})
  (t : KT.tensor u32 (l1_forward m))
  (i : szlt m)
  (#f : perm)
  (#s : Ch.chest1 u32 m)
  requires
    KT.tensor_pts_to t #f s ** pure (is_full (l1_forward m))
  returns
    v : u32
  ensures
    KT.tensor_pts_to t #f s ** pure (v == Ch.acc1 s (SZ.v i))
{
  let ro = tensor_to_rotensor t;
  let v = tensor_read ro (i, ());
  rotensor_to_tensor t ro;
  v
}

(* Add an outer dimension without copying: every row aliases the same vector.
   Removing the view recovers the original writable tensor. *)
inline_for_extraction noextract
fn ex_broadcast_rows
  (#m : erased nat{SZ.fits m})
  (#n : (erased nat){n > 0 /\ SZ.fits n})
  (t : KT.tensor u32 (l1_forward m))
  (#f : perm)
  (#s : Ch.chest1 u32 m)
  requires
    KT.tensor_pts_to t #f s ** pure (is_full (l1_forward m))
  ensures
    KT.tensor_pts_to t #f s
{
  let ro = tensor_to_rotensor t;
  rotensor_add_dim ro n;
  rotensor_remove_dim ro n;
  rotensor_to_tensor t ro;
}

(* Transposing the repeated-row view gives repeated columns, still backed by
   the original vector and still recoverable without a copy. *)
inline_for_extraction noextract
fn ex_broadcast_columns
  (#m : erased nat{SZ.fits m})
  (#n : (erased nat){n > 0 /\ SZ.fits n})
  (t : KT.tensor u32 (l1_forward m))
  (#f : perm)
  (#s : Ch.chest1 u32 m)
  requires
    KT.tensor_pts_to t #f s ** pure (is_full (l1_forward m))
  ensures
    KT.tensor_pts_to t #f s
{
  let ro = tensor_to_rotensor t;
  let rows = rotensor_add_dim_view ro n;
  tensor_apply_bij (swap2 #n #m) rows;
  tensor_unapply_bij (swap2 #n #m) rows;
  rotensor_remove_dim ro n;
  rotensor_to_tensor t ro;
}

(* An extracted instantiation of the borrow, to witness that it really is
   zero-cost: the generated code is a plain load from the original pointer,
   with no copy, no allocation and no struct in sight. *)
fn ex_read_vec10
  (t : KT.tensor u32 (l1_forward 10))
  (i : szlt 10)
  (#f : perm)
  (#s : Ch.chest1 u32 10)
  requires
    KT.tensor_pts_to t #f s ** pure (is_full (l1_forward 10))
  returns
    v : u32
  ensures
    KT.tensor_pts_to t #f s ** pure (v == Ch.acc1 s (SZ.v i))
{
  ex_borrow_read t i
}
