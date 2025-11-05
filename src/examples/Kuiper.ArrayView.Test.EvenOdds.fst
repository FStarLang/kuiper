module Kuiper.ArrayView.Test.EvenOdds

(* Splitting an array into two varrays, of the even and odd
positions in it. *)
#lang-pulse

#set-options "--z3rlimit 15"

open Kuiper
open Kuiper.View
open Kuiper.VArray
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.EvenOdds
module IView = Kuiper.IView
module SZ    = FStar.SizeT

noextract
let even_view et (len : nat) : aview et (lseq et ((len + 1) / 2)) = {
  iview = {
    len;
    sch = {
      ait = natlt ((len + 1) / 2);
    };
    step = {
      imap = {
        f = (fun (i : natlt ((len + 1)/2)) -> i * 2 <: natlt len);
        is_inj = ez;
      };
    }
  };
  igm = solve;
}

noextract
let odd_view et (len : nat) : aview et (lseq et (len / 2)) = {
  iview = {
    len;
    sch = {
      ait = natlt (len / 2);
    };
    step = {
      imap = {
        f = (fun (i : natlt (len/2)) -> 1 + i * 2 <: natlt len);
        is_inj = ez;
      };
    };
  };
  igm = solve;
}

(* Somehow generate these automatically for constants? *)
inline_for_extraction noextract
instance concrete_sz_100 : concrete_sz 100 = { x = 100sz }

inline_for_extraction noextract
instance _cview_even #et (len : erased nat) (sz_len : concrete_sz len) : IView.ciview (even_view et len).iview = {
  clen = concr' sz_len;
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
instance _cview_odd #et (len : erased nat) (sz_len : concrete_sz len) : IView.ciview (odd_view et len).iview = {
  clen = concr' sz_len;
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
  varray_read #_ #_ #_ #(_cview_even #_ _ solve) a 10sz;
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
  varray_read #_ #_ #_ #(_cview_odd #_ _ solve) a 10sz;
}

#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"
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
  assert pure (no_overlap (even_view u32 100).iview.step.imap.f (odd_view u32 100).iview.step.imap.f);
  // ^ This seems pretty important for performance, or we have to keep
  // proving that it is well-formed (i.e. the no_overlap precondition).
  let vw = sum_aview (even_view u32 100) (odd_view u32 100);
  let va = varray_view_equiv va vw;

  // Using underscores here, instead of the explicit views, seems
  // to cause an SMT query with uvars (at least that's what
  // it looks like in F* terms), which then fails.
  let vl, vr = varray_split2 (even_view u32 100) (odd_view u32 100) va;

  let x = foo_even vl;
  let y = foo_odd vr;

  let res = x `UInt32.add_mod` y;

  let va = varray_join2 vl vr;

  // assume (pure (is_full_view (sum_aview (even_view u32 100) (odd_view u32 100))));
  IView.full_iff_cardinal vw.iview #(solve <: enumerable (either (natlt 50) (natlt 50)));
  varray_concr va;

  with l1 v1. assert gpu_pts_to_slice (core va) 0 l1 v1;
  assume (pure (Seq.equal v0 v1)); // All the bijection stuff is making this hard.
  rewrite gpu_pts_to_slice (core va) 0 l1 v1 as a |-> v0;

  res
}
#pop-options
