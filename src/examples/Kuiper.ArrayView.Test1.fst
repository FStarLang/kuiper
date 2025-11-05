module Kuiper.ArrayView.Test1
inline_for_extraction noextract let _ = 1
#lang-pulse

open Kuiper
open Kuiper.VArray
open Kuiper.GhostMap
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.View
open FStar.FunctionalExtensionality { (^->>) }
module SZ = Kuiper.SizeT
module IView = Kuiper.IView

inline_for_extraction noextract
let inj_nat_rev (len : nat) : (natlt len @~> natlt len) = {
  f = (fun (i : natlt len) -> len - 1 - i <: natlt len);
  is_inj = ez;
}

inline_for_extraction noextract
let inj_sz_rev (len : sz) : (szlt len @~> szlt len) = {
  f = (fun (i : szlt len) -> len -^ 1sz -^ i <: szlt len);
  is_inj = ez;
}

inline_for_extraction noextract
let base_view (et : Type) (len : nat) : aview et (lseq et len) = {
  iview = {
    len;
    ait = natlt len;
    step = {
      imap     = inj_id;
    };
  };
  igm = solve;
}

inline_for_extraction noextract
let r_base_view (et : Type) (len : nat) : aview et (lseq et len) = {
  iview = {
    len;
    ait = natlt len;
    step = {
      imap     = inj_nat_rev len;
    };
  };
  igm = solve;
}

noeq
inline_for_extraction noextract
type _normal et len = | N of lseq et len

inline_for_extraction noextract
let bij__normal (et len : _) : (lseq et len =~ _normal et len) = {
  ff = N;
  gg = N?._0;
  ff_gg = ez;
  gg_ff = ez;
}

inline_for_extraction noextract
let normal_view (et:Type) (len:nat) : aview et (_normal et len) =
  review_view (base_view et len) (bij__normal et len)

noeq
inline_for_extraction noextract
type _reverse et len = | R : lseq et len -> _reverse et len

inline_for_extraction noextract
let bij__reverse (et len : _) : (lseq et len =~ _reverse et len) = {
  ff = R;
  gg = R?._0;
  ff_gg = ez;
  gg_ff = ez;
}

inline_for_extraction noextract
let reverse_view (et:Type) (len:nat) : aview et (_reverse et len) =
  review_view (r_base_view et len) (bij__reverse et len)

inline_for_extraction noextract
instance cnormal_view et (len : erased nat{SZ.fits len}) {| sz_len : concrete_sz len |} : IView.ciview (normal_view et len).iview = {
  clen = concr' sz_len;
  sch = {
    cit      = szlt len;
    bij      = natural;
  };
  step = {
    cimap    = cinj_id;
    compat   = ez;
  };
}

inline_for_extraction noextract
instance creverse_view et (len : erased nat{SZ.fits len}) {| sz_len : concrete_sz len |} : IView.ciview (reverse_view et len).iview = {
  clen = concr' sz_len;
  sch = {
    cit      = szlt len;
    bij      = natural;
  };
  step = {
    cimap    =
      // Can't use imap = inj_sz_rev (SZ.uint_to_t len) for stupid reasons,
      // a type inside a refinement does not match exactly.
      mk_cinj (fun (i : szlt len) -> concr' sz_len -^ 1sz -^ i <: szlt len);
    compat   = ez;
  };
}

(* Why does this work without the instance below? *)
fn test (a : varray (normal_view u32 50))
  preserves gpu
  requires a |-> N 's
  returns  u32
  ensures  a |-> N 's
{ varray_read #_ #_ #_ #solve_debug a 0sz; }

inline_for_extraction noextract
instance _ : concrete_sz 50 = { x = 50sz; }

fn test2 (a : varray (reverse_view u32 50sz))
  preserves gpu
  requires a |-> R 's
  returns  u32
  ensures  a |-> R 's
{ varray_read #_ #_ #_ #(creverse_view _ _ #_) a 0sz; }

fn write1 (a : varray (normal_view u32 50sz))
  preserves gpu
  requires  a |-> N 's
  ensures   a |-> N (Seq.upd 's 0 123ul)
{
  varray_write a 0sz 123ul;
}

fn write2 (a : varray (reverse_view u32 50sz))
  (#s : erased (lseq u32 50))
  preserves gpu
  requires a |-> R s
  ensures  a |-> R (Seq.upd s 0 123ul)
{
  varray_write a 0sz 123ul;
  // varray_write #_ #_ #_ #_ #(creverse_view _ _) a 0sz 123ul;
  // assert (pure (Seq.equal
  //                (R?._0 ((reverse_view u32 50).igm.upd (R s) (ci_to_ai (reverse_view u32 50) #(creverse_view u32 50sz) 0sz) 123ul))
  //                (Seq.upd s 0 123ul)));
  ();
}

let seq_rev (#a:Type) (s:seq a) : seq a =
  Seq.init (Seq.length s) (fun i -> Seq.index s (Seq.length s - 1 - i))

(* awkward, we should be able to start from a random array (not "core a")
   and use abs on it. *)
(* fixed! but could be nicer. *)
#push-options "--z3rlimit 30"
fn write3
  (p : gpu_array u32 50)
  (#s : erased (lseq u32 50))
  preserves gpu
  requires p |-> s
  ensures  p |-> Seq.upd s 49 123ul
{
  IView.full_iff_cardinal (reverse_view u32 50).iview #_;
  varray_abs' (reverse_view u32 50) p;
  let a' = from_array (reverse_view u32 50) p;
  assert (pure (Seq.equal (to_seq (reverse_view u32 50) (from_seq (reverse_view u32 50) s)) s));
  ();
  assert from_array (reverse_view u32 50) p |-> from_seq (reverse_view u32 50) s;
  rewrite
    from_array (reverse_view u32 50) p |-> from_seq (reverse_view u32 50) s
  as
    a' |-> from_seq (reverse_view u32 50) s;
  assert (pure (Seq.equal (R?._0 (from_seq (reverse_view u32 50) s))
               (seq_rev s)));
  ();
  rewrite
    a' |-> from_seq (reverse_view u32 50) s
  as
    a' |-> R (seq_rev s);
  write2 a';
  assert a' |-> R (Seq.upd (seq_rev s) 0 123ul);
  assert (pure (Seq.equal (Seq.upd (seq_rev s) 0 123ul) (seq_rev (Seq.upd s 49 123ul))));
  assert a' |-> R (seq_rev (Seq.upd s 49 123ul));
  varray_concr a';
  rewrite
    core a' |-> to_seq (reverse_view u32 50) (R (seq_rev (Seq.upd s 49 123ul)))
  as
    p |-> to_seq (reverse_view u32 50) (R (seq_rev (Seq.upd s 49 123ul)));
  assert (pure (Seq.equal (to_seq (reverse_view u32 50) (R (seq_rev (Seq.upd s 49 123ul)))) (Seq.upd s 49 123ul)));
  ();
}
#pop-options
