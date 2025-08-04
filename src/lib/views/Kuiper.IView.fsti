module Kuiper.IView

(* Indexing views *)

#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.Injection
open FStar.Tactics.Typeclasses { no_method }
module SZ = FStar.SizeT

[@@erasable]
noeq
type aiview (len : nat) = {
  (* abstract index type *)
  ait       : Type0;

  (* The index type must be enumerable. This is mostly so we can
     use forall+, the bijection inside here is not used much elsewhere. *)
  ait_enum  : Enumerable.enumerable ait;
  // Note: we have an instance for this field below, instead
  // of using @@@tcinstance here. This is so we can mark it
  // `unfold`.

  (* Index translation *)
  imap : ait @~> natlt len;
}

unfold
instance enumerable_aiview_ait (#len:nat) (vw : aiview len)
  : Enumerable.enumerable vw.ait
= vw.ait_enum

let is_full_view (#len : nat) (avw : aiview len) : prop =
  is_surj avw.imap.f

(* Will this be useful? *)
val full_iff_cardinal
  (#len : nat)
  (vw : aiview len)
  : Lemma (is_full_view vw <==> vw.ait_enum._cardinal == len)
          [SMTPat (is_full_view vw)]

(* Nothing fancy here. *)
let raw_view (#len:nat) : aiview len = {
  ait      = natlt len;

  ait_enum = solve;

  imap     = inj_id;
}

(* What it means for an indexing view to be concretizable, i.e. executable. *)
inline_for_extraction noextract
class ciview (#len : erased nat) (avw : aiview len) =
{
  [@@@no_method]
  fits  : squash (SZ.fits len);

  (* The concrete index type *)
  [@@@no_method]
  cit   : Type0;

  (* A bijection from the abstract indices to the concrete indices 
     Need not be executable. *)
  [@@@no_method]
  bij   : erased (avw.ait =~ cit);

  (* Concrete mapping. *)
  [@@@no_method]
  imap  : cit @~> szlt len;

  (* The mappings are compatible. I.e. the following diagram commutes:

     avw.it  --avw.imap-> natlt len
       |                     |
       |                     |
    cvw.bij              uint_to_t
       |                     |
       v                     v
      cit    --cvw.imap-> szlt len
   *)
  [@@@no_method]
  compat : 
    ai : avw.ait ->
      squash (imap.f (bij.ff ai) == SZ.uint_to_t (avw.imap.f ai));
}

let inj_bij (#a #b : Type) (bij : a =~ b) : (a @~> b) =
  {
    f = bij.ff;
    is_inj = ez;
  }

let inj_bij' (#a #b : Type) (bij : a =~ b) : (b @~> a) =
  {
    f = bij.gg;
    is_inj = ez;
  }

let reindex_view (#len : nat)
  (vw : aiview len)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.ait =~ ait')
  : aiview len = {
  ait      = ait';
  ait_enum = solve;

  imap     = inj_bij' bij `inj_comp` vw.imap;
}

let concrete_raw_view (#len:nat{SZ.fits len}) : ciview (raw_view #len) = {
  fits  = ();
  cit   = szlt len;
  bij   = natural;
  imap  = inj_id;
  compat = ez;
}

let it_to_nat
  (#len:nat)
  (vw : aiview len)
  (i : vw.ait)
  : GTot (natlt len)
  = i |~> vw.imap

let it_of_nat
  (#len:nat)
  (vw : aiview len)
  (i: natlt len{FStar.Functions.in_image vw.imap.f i})
  : GTot vw.ait
  = i <~| vw.imap

let ci_to_ai
  (#len:nat)
  (vw : aiview len) {| cw : ciview vw |}
  (i : cw.cit)
  : GTot vw.ait
  = let open Kuiper.Bijection in
    i <~| cw.bij

let ai_to_ci
  (#len:nat)
  (vw : aiview len) {| cw : ciview vw |}
  (i : vw.ait)
  : GTot cw.cit
  = let open Kuiper.Bijection in
    i |~> cw.bij

let sum_aiview (#len : nat) 
  (vw1 vw2 : aiview len)
  (#_ : squash (no_overlap vw1.imap.f vw2.imap.f))
  : aiview len = {
  ait      = either vw1.ait vw2.ait;
  ait_enum = solve;
  imap     = {
    f      = merge_either vw1.imap.f vw2.imap.f;
    is_inj = ez;
  };
}

[@@erasable]
noeq
type iview_transform (#len : erased nat) (vw : aiview len) = {
  ait : Type0;
  ait_enum : Enumerable.enumerable ait;

  imap : ait =~ vw.ait;
}

class ciview_transform (#len : erased nat) (#vw : aiview len) (cw : ciview vw) (t : iview_transform vw) = {
  [@@@no_method]
  cit : Type0;

  [@@@no_method]
  bij : erased (t.ait =~ cit);

  [@@@no_method]
  cimap : cit @~> cw.cit;
  (* ^ This will in fact be a bijection. *)

  [@@@no_method]
  compat :
    ai : t.ait ->
      squash (cimap.f (bij.ff ai) == cw.bij.ff (t.imap.ff ai));
}

let apply_itransform (#len : erased nat) (vw : aiview len) (t : iview_transform vw)
  : aiview len =
{
  ait      = t.ait;
  ait_enum = t.ait_enum;

  imap     = inj_bij t.imap `inj_comp` vw.imap;
}

let apply_ctransform
  (#len : erased nat)
  (#vw : aiview len)
  (#cw : ciview vw)
  (#at : iview_transform vw)
  (ct : ciview_transform cw at)
  : ciview (apply_itransform vw at) =
{
  fits = ();
  cit  = ct.cit;
  bij  = ct.bij;
  imap = ct.cimap `inj_comp` cw.imap;
  compat = (fun ait ->
    ct.compat ait;
    cw.compat (at.imap.ff ait)
  );
}
