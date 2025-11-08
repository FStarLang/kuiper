module Kuiper.View.Concat

(* Concatenating two views into a view of the two
   things, separately. *)

#lang-pulse

open Kuiper
open Kuiper.View
open Kuiper.Bijection
open Kuiper.Injection
module SZ = Kuiper.SizeT
module IView = Kuiper.IView

let aview_concat
  (#a : Type)
  (#st1 : Type)
  (#st2 : Type)
  (vw1 : aview a st1)
  (vw2 : aview a st2)
  : aview a (st1 & st2) =
{
  iview = {
    len = vw1.iview.len + vw2.iview.len;
    ait = either vw1.iview.ait vw2.iview.ait;
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
  (#st1 : Type)
  (#st2 : Type)
  (vw1 : aview a st1)
  (cw1 : cview vw1)
  (vw2 : aview a st2)
  (cw2 : cview vw2)
  (_ : squash (SZ.fits (len vw1 + len vw2)))
  : IView.ciview (aview_concat vw1 vw2).iview =
{
  clen = cw1.clen +^ cw2.clen;

  sch = {
    cit  = either cw1.sch.cit cw2.sch.cit;
    bij  = bij_either cw1.sch.bij cw2.sch.bij;
  };

  step = {
    cimap = (
      assert (SZ.v cw1.clen == vw1.iview.len);
      assert (SZ.v cw2.clen == vw2.iview.len);
      cinj_either cw1.step.cimap cw2.step.cimap
              `cinj_comp` cinj_sz_sum (len vw1) (len vw2) cw1.clen
    );


    // Why is the cast needed?
    compat = (fun i ->
      let i : either vw1.iview.ait vw2.iview.ait = i in
      match i with
      | Inl x -> cw1.step.compat x
      | Inr y -> cw2.step.compat y);
  };
}
