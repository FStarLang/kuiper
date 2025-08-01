module Kuiper.ArrayView.Test.EvenOdds3

(* Splitting an array into two varrays, of the even and odd
positions in it.

This even simpler version defines a strided view to capture both even and odd. *)
#lang-pulse

open Kuiper
open Kuiper.View
open Kuiper.VArray
open Kuiper.Bijection
open Kuiper.Injection
module IView = Kuiper.IView
module SZ    = FStar.SizeT

// Can we use divup here? It seems much harder on Z3.
noextract
let strided_view et (len : nat) (stride : nat) (offset : natlt stride) :
  aview et len (lseq et ((len + stride - 1 - offset) / stride))
= {
  iview = {
    ait = natlt ((len + stride - 1 - offset) / stride);
    ait_enum = solve;
    imap = {
      f = (fun (i : natlt ((len + stride - 1 - offset) / stride)) -> i * stride + offset <: natlt len);
      is_inj = ez;
    }
  };
  igm = solve;
}

let even_view et len : aview et len _ = strided_view et len 2 0
let odd_view  et len : aview et len _ = strided_view et len 2 1

inline_for_extraction noextract
instance _cview_strided
   (#et : Type) (#len : erased nat{SZ.fits len})
   (stride : sz) (offset : szlt stride)
: IView.cview (strided_view et len stride offset).iview
= {
  fits = ();
  cit  = szlt ((len + stride - 1 - offset) / stride);
  bij  = fin_size_t_bij _;
  imap = {
    f = (fun (i : szlt ((len + stride - 1 - offset) / stride)) -> i `SZ.mul` stride `SZ.add` offset <: szlt len);
    is_inj = ez;
  };
  compat = ez;
}

inline_for_extraction noextract
instance _cview_even #et (#len : erased nat{SZ.fits len}) : IView.cview (even_view et len).iview =
  _cview_strided #et #len 2sz 0sz

inline_for_extraction noextract
instance _cview_odd #et (#len : erased nat{SZ.fits len}) : IView.cview (odd_view et len).iview =
  _cview_strided #et #len 2sz 1sz

fn foo_even (a : varray (even_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  requires a |-> v0
  returns u32
  ensures  a |-> v0
{
  varray_read a 10sz;
}

fn foo_odd (a : varray (odd_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  requires a |-> v0
  returns u32
  ensures  a |-> v0
{
  varray_read a 10sz;
}

let vw = sum_aview (even_view u32 100) (odd_view u32 100)

fn test_simpler (a : gpu_array u32 100)
  (#v0 : erased (lseq u32 100))
  preserves gpu
  requires a |-> v0
  returns u32
  ensures exists* v1. a |-> v1
{
  varray_abs' vw a;
  let va = from_array vw a;

  let vl, vr = varray_split2
    (even_view u32 100)
    (odd_view u32 100)
    (from_array vw a)
    #_
    #(from_seq vw v0) // ARGH, why do I have to provide this!?!??! terrible error otherwise
    ;
  // Note: that doesn't happen if we use split2_, the ghost version

  let x = foo_even vl;
  let y = foo_odd vr;

  let res = x `UInt32.add_mod` y;

  let va = varray_join2 vl vr;

  varray_concr va;

  rewrite each core va as a;

  res
}
