module Kuiper.Example.ArrayView.Test.EvenOdds

(* Splitting an array into two varrays, of the even and odd
positions in it. *)
#lang-pulse

#set-options "--z3rlimit 15"

open Kuiper
open Kuiper.View
open Kuiper.VArray
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.Example.EvenOdds
module IView = Kuiper.IView
module SZ    = FStar.SizeT

noextract
let even_view et (len : nat) : aview et (lseq et ((len + 1) / 2)) = {
  iview = {
    len;
    ait = natlt ((len + 1) / 2);
    step = {
      imap = {
        f = (fun (i : natlt ((len + 1)/2)) -> i * 2 <: natlt len);
        is_inj = ez;
      };
    }
  };
  ctn = solve;
}

noextract
let odd_view et (len : nat) : aview et (lseq et (len / 2)) = {
  iview = {
    len;
    ait = natlt (len / 2);
    step = {
      imap = {
        f = (fun (i : natlt (len/2)) -> 1 + i * 2 <: natlt len);
        is_inj = ez;
      };
    };
  };
  ctn = solve;
}

inline_for_extraction noextract
instance _cview_even #et (len : erased nat{SZ.fits len}) : IView.ciview (even_view et len).iview = {
  len_fits = ();
  sch = {
    cit  = szlt ((len + 1) / 2);
    bij  = natural;
  };
  step = {
    cimap = mk_cinj (fun (i : szlt ((len + 1) / 2)) -> i `SZ.mul` 2sz <: szlt len);
    compat = ez;
  };
}

inline_for_extraction noextract
instance _cview_odd #et (len : erased nat{SZ.fits len}) : IView.ciview (odd_view et len).iview = {
  len_fits = ();
  sch = {
    cit  = szlt (len / 2);
    bij  = natural;
  };
  step = {
    cimap = mk_cinj (fun (i : szlt (len / 2)) -> 1sz +^ i `SZ.mul` 2sz <: szlt len);
    compat = ez;
  };
}

fn foo_even (a : varray (even_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  preserves a |-> v0
  returns   u32
{
  // Actually reads index 20 (see generated code)
  varray_read #_ #_ #_ #(_cview_even #_ _) a 10sz;
  // Bad tc resolution due to the different shape
  // of the lengths in lseq. The one for a gets simplified
  // to 50, which does not unify with (?u+1)/2 in the instance.
}

fn foo_odd (a : varray (odd_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  requires a |-> v0
  returns u32
  ensures  a |-> v0
{
  // Actually reads index 21 (see generated code)
  varray_read #_ #_ #_ #(_cview_odd #_ _) a 10sz;
}

fn foo_odd_modify (a : varray (odd_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  requires a |-> v0
  ensures  a |-> (Seq.upd v0 10 42ul <: lseq u32 50)
{
  // Actually writes into index 21 (see generated code)
  varray_write #_ #_ #_ #(_cview_odd #_ _) a 10sz 42ul;
}
