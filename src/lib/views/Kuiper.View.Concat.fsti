module Kuiper.View.Concat

(* Concatenating two views into a view of the two
   things, separately. *)

#lang-pulse

open Kuiper
open Kuiper.View
open Kuiper.Bijection
open Kuiper.Injection
module SZ = FStar.SizeT
module IView = Kuiper.IView

let aview_concat
  (#a : Type)
  (#len1 : nat) (#st1 : Type)
  (#len2 : nat) (#st2 : Type)
  (vw1 : aview a len1 st1)
  (vw2 : aview a len2 st2)
  : aview a (len1 + len2) (st1 & st2) =
{
  iview = {
    sch = {
      ait      = either vw1.iview.sch.ait vw2.iview.sch.ait;
      ait_enum = solve;
    };
    step = {
      imap     = inj_either vw1.iview.step.imap vw2.iview.step.imap `inj_comp` inj_nat_sum _ _;
    };
  };
  igm = solve;
}

// Note: maybe len1/len2 should be nats, and the relevant size_t
// should come from the cview of each. I.e. having a cview
// for an array implies you have a concrete value for its length.
inline_for_extraction noextract
instance cview_concat
  (#a:Type)
  (#len1 : sz) (#st1 : Type)
  (#len2 : sz) (#st2 : Type)
  (vw1 : aview a len1 st1)
  (cw1 : cview vw1)
  (vw2 : aview a len2 st2)
  (cw2 : cview vw2)
  (_ : squash (SZ.fits (len1 + len2)))
  : IView.ciview (aview_concat vw1 vw2).iview =
{
  fits = ();

  sch = {
    cit  = either cw1.sch.cit cw2.sch.cit;
    bij  = bij_either cw1.sch.bij cw2.sch.bij;
  };

  step = {
    cimap = inj_either cw1.step.cimap cw2.step.cimap `inj_comp` inj_sz_sum len1 len2;

    // Why is the cast needed?
    compat = (fun i ->
      let i : either vw1.iview.sch.ait vw2.iview.sch.ait = i in
      match i with
      | Inl x -> cw1.step.compat x
      | Inr y -> cw2.step.compat y);
  };
}
