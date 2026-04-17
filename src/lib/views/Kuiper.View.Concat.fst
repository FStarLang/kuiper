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
  ctn = solve;
}

inline_for_extraction noextract
instance cview_concat
  (#a:Type)
  (#st1 : Type)
  (#st2 : Type)
  (#vw1 : aview a st1) (cw1 : cview vw1)
  (#vw2 : aview a st2) (cw2 : cview vw2)
  (* We need two extra things:
     1. That the combined length fits a size_t
     2. A concrete value for the left length, to implement the shift. *)
  (_ : squash (SZ.fits (len vw1 + len vw2)))
  {| Kuiper.Concrete.concrete_sz (len vw1) |}
  : IView.ciview (aview_concat vw1 vw2).iview =
{
  len_fits = ();

  sch = {
    cit  = either cw1.sch.cit cw2.sch.cit;
    bij  = bij_either cw1.sch.bij cw2.sch.bij;
  };

  step = {
    cimap = (
      cinj_either cw1.step.cimap cw2.step.cimap
              `cinj_comp` cinj_sz_sum (len vw1) (len vw2) (concr (len vw1))
    );


    compat = (function
              | Inl x -> cw1.step.compat x
              | Inr y -> cw2.step.compat y);
  };
}
