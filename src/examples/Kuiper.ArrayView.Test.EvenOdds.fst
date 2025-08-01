module Kuiper.ArrayView.Test.EvenOdds

(* Splitting an array into two varrays, of the even and odd
positions in it. *)
#lang-pulse

open Kuiper
open Kuiper.View
open Kuiper.VArray
open Kuiper.Bijection
open Kuiper.Injection
module IView = Kuiper.IView
module SZ    = FStar.SizeT

noextract
let bij_nat_interleave (#len:nat)
  : Tot (natlt len =~ either (natlt ((len + 1) / 2)) (natlt (len / 2))) =
  mk_bijection #(natlt len) #(either (natlt ((len + 1) / 2)) (natlt (len / 2)))
    (fun i -> if i % 2 = 0 then Inl (i / 2) else Inr (i / 2))
    (function
     | Inl i -> i * 2
     | Inr i -> 1 + i * 2)
    ez ez

noextract
let seq_evens #a (#n : nat)
  (s : lseq a n)
  : Tot (lseq a ((n+1)/2))
=
  Seq.init ((n+1)/2) fun i -> Seq.index s (2 * i)

noextract
let seq_odds #a (#n : nat)
  (s : lseq a n)
  : Tot (lseq a (n/2))
=
  Seq.init (n/2) fun i -> Seq.index s (2 * i + 1)

noextract
let seq_interleave #a (#n : nat)
  (s1 : lseq a ((n + 1) / 2))
  (s2 : lseq a (n / 2))
  : Tot (lseq a n)
=
  Seq.init n fun i ->
    if i % 2 = 0 then
      Seq.index s1 (i / 2)
    else
      Seq.index s2 (i / 2)

let interleave_lemma1 #a #n (s : lseq a n)
  : Lemma (ensures seq_interleave (seq_evens s) (seq_odds s) == s)
          [SMTPat (seq_interleave (seq_evens s) (seq_odds s))]
= assert (Seq.equal (seq_interleave (seq_evens s) (seq_odds s)) s);
  ()

let interleave_lemma1_nopat #a #n (s : lseq a n)
  : Lemma (ensures seq_interleave (seq_evens s) (seq_odds s) == s)
= ()

let interleave_lemma2 #a (#n:nat) (s1 : lseq a ((n + 1) / 2)) (s2 : lseq a (n / 2))
  : Lemma (ensures seq_evens (seq_interleave s1 s2) == s1 /\ seq_odds (seq_interleave s1 s2) == s2)
          [SMTPat (seq_evens (seq_interleave s1 s2)); SMTPat (seq_odds (seq_interleave s1 s2))]
= assert (Seq.equal (seq_evens (seq_interleave s1 s2)) s1);
  assert (Seq.equal (seq_odds (seq_interleave s1 s2)) s2);
  ()

noextract
let bij_seq_interleave (#et:Type) (#len:nat)
  : Tot (lseq et len =~ (lseq et ((len + 1) / 2)) & (lseq et (len / 2))) =
  mk_bijection #(lseq et len) #((lseq et ((len + 1) / 2)) & (lseq et (len / 2)))
    (fun i -> (seq_evens i, seq_odds i))
    (fun (i, j) -> seq_interleave i j)
    ez ez

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
  returns u32
  ensures exists* v1. a |-> v1
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

  assume (pure (is_full_view (sum_aview (even_view u32 100) (odd_view u32 100))));
  varray_concr va;

  assume (pure (core va == a));
  rewrite each core va as a;

  res
}
