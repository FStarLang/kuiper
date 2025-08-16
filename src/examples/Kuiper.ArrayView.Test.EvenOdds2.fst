module Kuiper.ArrayView.Test.EvenOdds2

(* Splitting an array into two varrays, of the even and odd
positions in it. This simpler version does not transform
the view and just jumps into it. *)
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
let even_view et (len : nat) : aview et (lseq et ((len + 1) / 2)) = {
  iview = {
    len;
    sch = {
      ait = natlt ((len + 1) / 2);
      ait_enum = solve;
    };
    step = {
      imap = {
        f = (fun (i : natlt ((len + 1)/2)) -> i * 2 <: natlt len);
        is_inj = ez;
      };
    };
  };
  igm = solve;
}

noextract
let odd_view et (len : nat) : aview et (lseq et (len / 2)) = {
  iview = {
    len;
    sch = {
      ait = natlt (len / 2);
      ait_enum = solve;
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

inline_for_extraction noextract
instance _cview_even #et
  (#len : erased nat{SZ.fits len})
  (sz_len : concrete_sz len)
: IView.ciview (even_view et len).iview = {
  clen = concr' sz_len;
  sch = {
    cit    = szlt ((len + 1) / 2);
    bij    = natural;
  };
  step = {
    cimap = {
      f = (fun (i : szlt ((len + 1) / 2)) -> i `SZ.mul` 2sz <: szlt len);
      is_inj = ez;
    };
    compat = ez;
  };
}

inline_for_extraction noextract
instance _cview_odd #et
  (#len : erased nat{SZ.fits len})
  (sz_len : concrete_sz len)
: IView.ciview (odd_view et len).iview = {
  clen = concr' sz_len;
  sch = {
    cit  = szlt (len / 2);
    bij  = natural;
  };
  step = {
    cimap = {
      f = (fun (i : szlt (len / 2)) -> 1sz `SZ.add` (i `SZ.mul` 2sz) <: szlt len);
      is_inj = ez;
    };
    compat = ez;
  };
}

let _sanity1 (#len : nat{SZ.fits len}) (_ : concrete_sz len) (x : szlt ((len + 1) / 2)) : Lemma (ci_to_ai (even_view u32 len) x == SZ.v x)
  = ()

let _sanity2 (#len : nat{SZ.fits len}) (_ : concrete_sz len)(x : szlt (len / 2)) : Lemma (ci_to_ai (odd_view u32 len) x == SZ.v x)
  = ()

inline_for_extraction noextract
instance _ : concrete_sz 100 = { x = 100sz; }

fn foo_even (a : varray (even_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  requires a |-> v0
  returns u32
  ensures  a |-> v0
{
  varray_read #_ #_ #_ #(_cview_even solve) a 10sz;
}

fn foo_odd (a : varray (odd_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  requires a |-> v0
  returns u32
  ensures  a |-> v0
{
  varray_read #_ #_ #_ #(_cview_odd solve) a 10sz;
}

fn write_even (a : varray (even_view u32 100))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  requires a |-> v0
  ensures  a |-> (Seq.upd v0 10 42ul <: lseq u32 50)
{
  varray_write #_ #_ #_ #(_cview_even solve) a 10sz 42ul;
}

let vw = sum_aview (even_view u32 100) (odd_view u32 100)

fn test_simpler (a : gpu_array u32 100)
  (#v0 : erased (lseq u32 100))
  preserves gpu
  requires a |-> v0
  returns u32
  ensures  a |-> v0
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

let merge_lemma #et (#len:nat) (sl : lseq et ((len + 1) / 2)) (sr : lseq et (len / 2))
  : Lemma (
            to_seq (sum_aview (even_view et len) (odd_view et len)) (sl, sr)
            ==
            seq_interleave sl sr
  )
  [SMTPat (to_seq (sum_aview (even_view et len) (odd_view et len)) (sl, sr))]
= let aux (i : natlt len)
      : Lemma (to_seq (sum_aview (even_view et len) (odd_view et len)) (sl, sr) @! i
               ==
               seq_interleave sl sr @! i)
  = admit () // flaky
  in
  Classical.forall_intro (Classical.move_requires aux);
  assert (Seq.equal
              (to_seq (sum_aview (even_view et len) (odd_view et len)) (sl, sr))
              (seq_interleave sl sr))


let split_lemma #et (#len:nat) (s : lseq et len)
  : Lemma (
            from_seq (sum_aview (even_view et len) (odd_view et len)) s
            ==
            (seq_evens s, seq_odds s)
  )
  [SMTPat (from_seq (sum_aview (even_view et len) (odd_view et len)) s)]
(* Very easy proof: map each side to a sequence, they are trivially equal by
   SMT, the bijection then gives us our result. *)
= assert (Seq.equal
            (to_seq (sum_aview (even_view et len) (odd_view et len)) (seq_evens s, seq_odds s))
            s)

#push-options "--z3rlimit 30"
fn test_write (a : gpu_array u32 100)
    (#v0 : erased (lseq u32 100))
    preserves gpu
    requires a |-> v0
    ensures  a |-> Seq.upd (Seq.upd v0 20 42ul) 41 43ul
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

  varray_write #_ #_ #_ #(_cview_even solve) vl 10sz 42ul;
  varray_write #_ #_ #_ #(_cview_odd  solve) vr 20sz 43ul;

  let va = varray_join2 vl vr;

  varray_concr va;

  rewrite each core va as a;

  with v1.
    assert (a |-> v1);
    assert (pure (Seq.equal v1 (Seq.upd (Seq.upd v0 20 42ul) 41 43ul))); // use extensionality

  ()
}
#pop-options
