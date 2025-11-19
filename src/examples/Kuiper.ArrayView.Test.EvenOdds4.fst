module Kuiper.ArrayView.Test.EvenOdds4

(* Splitting an array into two varrays, of the even and odd positions in it.

This even simpler version defines a strided view to capture both even and odd.

We chain this view with the base one instead of defining a view in one go. This
is just kicking the tires, it would be nicer to compose with the reverse view or
split into even/odds twice. *)
#lang-pulse

let x = 1ul

open Kuiper
open Kuiper.VArray
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.EvenOdds
module IView = Kuiper.IView
module SZ    = FStar.SizeT
open Kuiper.IView
open Kuiper.View

noextract
let strided_ait (len : nat) (stride : nat) (offset : natlt stride) : Type =
  natlt ((len + stride - 1 - offset) / stride)

let strided_step (len : nat) (stride : nat) (offset : natlt stride) :
  aiview_step
    (natlt ((len + stride - 1 - offset) / stride))
    (natlt len)
= {
    imap = {
      f = (fun (i : natlt ((len + stride - 1 - offset) / stride)) -> i * stride + offset <: natlt len);
      is_inj = ez;
    };
}

let strided_view #et (#len : nat) (stride : nat) (offset : natlt stride)
  (* Any base view with abstract indices being natlt can be strided. *)
  (base : aview et (lseq et len) { base.iview.len == len /\ base.iview.ait == natlt len })
 : aview et (lseq et ((base.iview.len + stride - 1 - offset) / stride))
= {
    iview = {
      len = base.iview.len;
      ait = strided_ait base.iview.len stride offset;
      step = IView.compose_astep (strided_step len stride offset) base.iview.step;
    };
    ctn = solve;
}

let even_view #et #len
  (base : aview et (lseq et len) { base.iview.len == len /\ base.iview.ait == natlt len})
  : aview et (lseq et ((len + 1) / 2))
  = strided_view 2 0 base

let odd_view #et #len
  (base : aview et (lseq et len) { base.iview.len == len /\ base.iview.ait == natlt len})
  : aview et (lseq et (len / 2))
  = strided_view 2 1 base

let strided_cischema (len : nat{SZ.fits len}) (stride : sz) (offset : szlt stride)
  : ciview_schema (strided_ait len stride offset) =
{
  cit  = szlt ((len + stride - 1 - offset) / stride);
  bij  = natural;
}

inline_for_extraction noextract
let strided_cistep (len : erased nat{SZ.fits len}) (stride : sz) (offset : szlt stride)
: IView.ciview_step
    (strided_cischema len stride offset)
    (raw_ciview_schema len)
    (strided_step len stride offset)
= {
  cimap = mk_cinj
    (fun (i : szlt ((len + stride - 1 - offset) / stride)) -> (i `SZ.mul` stride `SZ.add` offset) <: szlt len);
  compat = ez;
}

// FIXME! I broke this somehow.

(*
inline_for_extraction noextract
instance _cview_strided
  (#et : Type) (#len : erased nat{SZ.fits len})
  (stride : sz) (offset : szlt stride)
  (#base : aview et (lseq et len) { base.iview.len == reveal len /\ base.iview.sch == raw_aiview_schema len })
  (c : IView.ciview base.iview)
  (#_ : squash (c.sch == (raw_ciview_schema len)))
: IView.ciview (strided_view stride offset base).iview
= {
  clen = c.clen;
  sch  = strided_cischema base.iview.len stride offset;
  step = IView.compose_cstep (strided_cistep len stride offset) c.step;
}

inline_for_extraction noextract
instance _cview_even #et #len
  (vw : aview et len (lseq et len))
  (#_ : squash (vw.iview.sch == raw_aiview_schema len))
  (c : cview vw { c.sch == raw_ciview_schema len })
  : IView.ciview (even_view vw).iview
=
  _cview_strided #et #len 2sz 0sz c #()

inline_for_extraction noextract
instance _cview_odd #et #len
  (vw : aview et len (lseq et len))
  (#_ : squash (vw.iview.sch == raw_aiview_schema len))
  (c : cview vw { c.sch == raw_ciview_schema len })
  : IView.ciview (odd_view vw).iview
=
  _cview_strided #et #len 2sz 1sz c #()


let _sanity1 (#len : nat{SZ.fits len}) (x : szlt ((len + 1) / 2))
  : Lemma (IView.ci_to_ai (IView.raw_view #len) x == SZ.v x)
  = ()

let _sanity2 #et (#len : nat{SZ.fits len}) (x : szlt ((len + 1) / 2))
  : Lemma (IView.ci_to_ai (even_view (Kuiper.View.raw_view #et #len)).iview
                    x
                      == SZ.v x)
  = ()


let _sanity3 #et (#len : nat{SZ.fits len}) (x : szlt (len / 2))
  : Lemma (IView.ci_to_ai (even_view (Kuiper.View.raw_view #et #len)).iview
                    x
                      == SZ.v x)
  = ()

fn foo_even
  (a : varray (even_view (raw_view #u32 #100)))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  preserves a |-> v0
  returns   u32
{ varray_read a 10sz; }

fn foo_odd
  (a : varray (odd_view (raw_view #u32 #100)))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  preserves a |-> v0
  returns   u32
{ varray_read a 10sz; }

inline_for_extraction noextract // view polymorphic
fn foo_even'
  (#vw : aview u32 100 (lseq u32 100))
  (#_ : squash (is_full_view vw /\ vw.iview.sch == (raw_aiview_schema 100)))
  {| c : cview vw |}
  (#_ : squash (c.sch == raw_ciview_schema 100))
  (a : varray (even_view vw))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  preserves a |-> v0
  returns   u32
{
  varray_read a 10sz;
}

inline_for_extraction noextract // view polymorphic
fn foo_odd'
  (#vw : aview u32 100 (lseq u32 100))
  (#_ : squash (is_full_view vw /\ vw.iview.sch == (raw_aiview_schema 100)))
  {| c : cview vw |}
  (#_ : squash (c.sch == raw_ciview_schema 100))
  (a : varray (odd_view vw))
  (#v0 : erased (lseq u32 50))
  preserves gpu
  preserves a |-> v0
  returns   u32
{
  varray_read a 10sz;
}

let split_view (vw : aview 'et 'len (lseq 'et 'len) { vw.iview.ait == (raw_aiview_schema 'len).ait }) =
  sum_aview (even_view vw) (odd_view vw)

inline_for_extraction noextract // view polymorphic
fn test'
  (#vw : aview u32 100 (lseq u32 100))
  (#_ : squash (is_full_view vw /\ vw.iview.sch == (raw_aiview_schema 100)))
  {| c : cview vw |}
  (#_ : squash (c.sch == raw_ciview_schema 100))
  (a : varray vw)
  (#v0 : erased (lseq u32 100))
  preserves gpu
  requires a |-> v0
  returns  u32
  ensures  a |-> v0
{
  (* silly, we should have varray_reabs or similar. And not require fullness. *)
  varray_concr a;
  varray_abs' (split_view vw) (core a);
  let a' = from_array (split_view vw) (core a);
  rewrite each from_array (split_view vw) (core a) as a';

  let vl, vr = varray_split2
    (even_view vw)
    (odd_view vw)
    a'
    #_
    #(from_seq (split_view vw) (to_seq vw v0)) // ARGH, why do I have to provide this!?!??! terrible error otherwise
    ;
  // Note: that doesn't happen if we use split2_, the ghost version

  let x = foo_even' vl;
  let y = foo_odd'  vr;

  let res = x `UInt32.add_mod` y;

  let va = varray_join2 vl vr;

  varray_concr va;
  varray_abs' vw (core va);

  rewrite each from_array vw (core va) as a;

  res
}

let foo_even_over_raw a #v0 = foo_even' #raw_view #() #solve #() a #v0
let foo_odd_over_raw  a #v0 = foo_odd'  #raw_view #() #solve #() a #v0
let test_over_raw     a #v0 = test'     #raw_view #() #solve #() a #v0
