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
    ait = natlt ((len + 1) / 2);
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
    ait = natlt (len / 2);
    step = {
      imap = {
        f = (fun (i : natlt (len/2)) -> 1 + i * 2 <: natlt len);
        is_inj = ez;
      };
    };
  };
  igm = solve;
}

let lem_no_overlap #et (len : nat)
  : Lemma (no_overlap (even_view et len).iview.step.imap.f (odd_view et len).iview.step.imap.f)
          [SMTPat (no_overlap (even_view et len).iview.step.imap.f (odd_view et len).iview.step.imap.f)]
  = ()

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
    cimap = mk_cinj (fun (i : szlt ((len + 1) / 2)) -> i `SZ.mul` 2sz <: szlt len);
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
    cimap = mk_cinj (fun (i : szlt (len / 2)) -> 1sz `SZ.add` (i `SZ.mul` 2sz) <: szlt len);
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
  IView.full_iff_cardinal vw.iview #_;
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

  with v1 l1. rewrite
    gpu_pts_to_slice (core va) 0 l1 v1
  as
    a |-> v0;

  res
}

#push-options "--split_queries always"
let surj_lemma' #et (#len:nat) (i : natlt len)
  : Lemma (exists (j : either (natlt ((len + 1) / 2)) (natlt (len / 2))).
             it_to_nat (sum_aview (even_view et len) (odd_view et len)) j == i)
= let vw = sum_aview (even_view et len) (odd_view et len) in
  if i % 2 = 0
  then assert (it_to_nat vw (Inl #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2)) == i)
  else assert (it_to_nat vw (Inr #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2)) == i)
#pop-options

let surj_lemma #et (len:nat)
  : Lemma (is_surj (it_to_nat (sum_aview (even_view et len) (odd_view et len))))
          [SMTPat (it_to_nat (sum_aview (even_view et len) (odd_view et len)))]
  = Classical.forall_intro (surj_lemma' #et #len)

let is_full (et:Type) (len:nat)
  : Lemma (is_full_view (sum_aview (even_view et len) (odd_view et len)))
          [SMTPat (is_full_view (sum_aview (even_view et len) (odd_view et len)))]
  = IView.full_iff_cardinal (sum_aview (even_view et len) (odd_view et len)).iview #(solve <: enumerable _)

#push-options "" // "--split_queries always"
#restart-solver
let lem_idx1 #et (#len : nat) (i : natlt len{i % 2 = 0})
  (#_ : squash (in_image (it_to_nat (sum_aview (even_view et len) (odd_view et len))) i)) // should come from surj_lemma
  : Lemma (it_of_nat (sum_aview (even_view et len) (odd_view et len)) i == Inl #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2))
= lem_no_overlap #et len;
  calc (==) {
    it_to_nat (sum_aview (even_view et len) (odd_view et len)) (Inl #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2));
    == {}
    it_to_nat (even_view et len) (i / 2);
    == {}
    (i / 2) * 2;
    == {(* i is even *) }
    i;
  };
  it_nat_rel (sum_aview (even_view et len) (odd_view et len)) (Inl #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2)) i;
  assert (it_to_nat (sum_aview (even_view et len) (odd_view et len)) (Inl #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2)) == i);
  assert (Inl #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2) == it_of_nat (sum_aview (even_view et len) (odd_view et len)) i);
  ()
#pop-options

#push-options "--z3rlimit 20"
let lem_idx2 #et (#len : nat) (i : natlt len{i % 2 = 1})
  (#_ : squash (in_image (it_to_nat (sum_aview (even_view et len) (odd_view et len))) i)) // should come from surj_lemma
  : Lemma (it_of_nat (sum_aview (even_view et len) (odd_view et len)) i == Inr #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2))
= assert (it_to_nat (sum_aview (even_view et len) (odd_view et len)) (Inr #(natlt ((len + 1)/ 2)) #(natlt (len / 2)) (i / 2)) == i);
  ()
#pop-options

#push-options "--split_queries always --z3rlimit 10"
let merge_lemma #et (#len:nat) (sl : lseq et ((len + 1) / 2)) (sr : lseq et (len / 2))
  : Lemma (
            to_seq (sum_aview (even_view et len) (odd_view et len)) (sl, sr)
            ==
            seq_interleave sl sr
  )
  [SMTPat (to_seq (sum_aview (even_view et len) (odd_view et len)) (sl, sr))]
= let vw = sum_aview (even_view et len) (odd_view et len) in
  surj_lemma #et len;
  let aux (i : natlt len)
      : Lemma (to_seq vw (sl, sr) @! i == seq_interleave sl sr @! i)
  = if i % 2 = 0 then lem_idx1 #et #len i #() else lem_idx2 #et #len i #()
  in
  Classical.forall_intro (Classical.move_requires aux);
  assert (to_seq vw (sl, sr) `Seq.equal` seq_interleave sl sr)
#pop-options

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
            s
            (to_seq (sum_aview (even_view et len) (odd_view et len)) (seq_evens s, seq_odds s)))

#push-options "--z3rlimit 60"
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

  with vret. assert pure (vret == Seq.upd (Seq.upd v0 20 42ul) 41 43ul);
  with l1 v1. assert gpu_pts_to_slice (core va) 0 l1 v1;
  assert pure (Seq.equal v1 vret);
  rewrite
    gpu_pts_to_slice (core va) 0 l1 v1
  as
    a |-> vret;
}
#pop-options
