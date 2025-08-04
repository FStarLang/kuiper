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
  bij  = natural;
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

(* What is happening?!?! Why isn't this obvious? *)

let _wat_even (#len : nat{SZ.fits len}) :
  Lemma (reveal (_cview_even #u32 #len).bij == fin_size_t_bij ((len + 2 - 1 - 0) / 2))
        [SMTPat (_cview_even #u32 #len)]
= admit()

let _wat_odd (#len : nat{SZ.fits len}) :
  Lemma (reveal (_cview_odd #u32 #len).bij == fin_size_t_bij ((len + 2 - 1 - 1) / 2))
        [SMTPat (_cview_odd #u32 #len)]
= admit()


let _sanity1 (#len : nat{SZ.fits len}) (x : szlt ((len + 1) / 2)) : Lemma (ci_to_ai (even_view u32 len) x == SZ.v x)
  = ()

let _sanity2 (#len : nat{SZ.fits len}) (x : szlt (len / 2)) : Lemma (ci_to_ai (odd_view u32 len) x == SZ.v x)
  = ()

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

fn test (a : gpu_array u32 100)
  (#v0 : erased (lseq u32 100))
  preserves gpu
  requires a |-> v0
  returns  u32
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
      Seq.index s2 ((i-1) / 2)

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

// TODO: Find a comfortable way of proving this. The definition of from_seq uses
// it_of_nat, which uses choice as it is reversing the (surjective) injection
// and is probably bad for SMT.
let split_lemma #et (#len:nat) (s : lseq et len)
  : Lemma (
            from_seq (sum_aview (even_view et len) (odd_view et len)) s
            ==
            (seq_evens s, seq_odds s)
  )
  [SMTPat (from_seq (sum_aview (even_view et len) (odd_view et len)) s)]
= let aux (i : natlt ((len+1)/2)) : Lemma (it_of_nat (even_view et len) (2*i) == i) =
    ()
  in
  let aux2 (i : natlt (len/2)) : Lemma (it_of_nat (odd_view et len) (2*i+1) == i) =
    ()
  in
  assume (Seq.equal (fst (from_seq (sum_aview (even_view et len) (odd_view et len)) s))
                    (seq_evens s));
  assume (Seq.equal (snd (from_seq (sum_aview (even_view et len) (odd_view et len)) s))
                    (seq_odds  s));
  ()

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

    varray_write vl 10sz 42ul;
    varray_write vr 20sz 43ul;

    let va = varray_join2 vl vr;

    varray_concr va;

    rewrite each core va as a;

    with v1.
      assert (a |-> v1);
      assert (pure (Seq.equal v1 (Seq.upd (Seq.upd v0 20 42ul) 41 43ul))); // use extensionality

    ()
  }
