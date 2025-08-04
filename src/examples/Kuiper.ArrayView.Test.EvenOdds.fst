module Kuiper.ArrayView.Test.EvenOdds

(* Splitting an array into two varrays, of the even and odd
positions in it. *)
#lang-pulse

open Kuiper
open Kuiper.View
open Kuiper.VArray
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.EvenOdds
module IView = Kuiper.IView
module SZ    = FStar.SizeT

noextract
let even_view et len : aview et len (lseq et ((len + 1) / 2)) = {
  iview = {
    ait = natlt ((len + 1) / 2);
    ait_enum = solve;
    imap = {
      f = (fun (i : natlt ((len + 1)/2)) -> i * 2 <: natlt len);
      is_inj = ez;
    }
  };
  igm = solve;
}

noextract
let odd_view et len : aview et len (lseq et (len / 2)) = {
  iview = {
    ait = natlt (len / 2);
    ait_enum = solve;
    imap = {
      f = (fun (i : natlt (len/2)) -> 1 + i * 2 <: natlt len);
      is_inj = ez;
    }
  };
  igm = solve;
}

inline_for_extraction noextract
instance _cview_even #et (#len : erased nat{SZ.fits len}) : IView.cview (even_view et len).iview = {
  fits = ();
  cit  = szlt ((len + 1) / 2);
  bij  = fin_size_t_bij _;
  imap = {
    f = (fun (i : szlt ((len + 1) / 2)) -> i `SZ.mul` 2sz <: szlt len);
    is_inj = ez;
  };
  compat = ez;
}

inline_for_extraction noextract
instance _cview_odd #et (#len : erased nat{SZ.fits len}) : IView.cview (odd_view et len).iview = {
  fits = ();
  cit  = szlt (len / 2);
  bij  = fin_size_t_bij _;
  imap = {
    f = (fun (i : szlt (len / 2)) -> 1sz `SZ.add` (i `SZ.mul` 2sz) <: szlt len);
    is_inj = ez;
  };
  compat = ez;
}

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

fn test (a : gpu_array u32 100)
  (#v0 : erased (lseq u32 100))
  preserves gpu
  requires a |-> v0
  returns  u32
  ensures  a |-> v0
{
  let va = varray_begin a;
  let va = varray_reindex (bij_nat_interleave #100) va;
  let va = varray_review  (bij_seq_interleave #u32 #100) va;
  let va = varray_view_equiv va (sum_aview (even_view u32 100) (odd_view u32 100));

  let vl, vr = varray_split2 _ _ va;

  let x = foo_even vl;
  let y = foo_odd vr;

  let res = x `UInt32.add_mod` y;

  let va = varray_join2 vl vr;

  // assume (pure (is_full_view (sum_aview (even_view u32 100) (odd_view u32 100))));
  varray_concr va;

  assume (pure (core va == a));
  rewrite each core va as a;

  // All the bijection stuff is making this hard.
  with v1. assert (a |-> v1);
  // assert (pure (Seq.equal v0 v1));
  assume (pure (Seq.equal v0 v1));

  res
}
