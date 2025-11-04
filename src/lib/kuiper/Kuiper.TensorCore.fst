module Kuiper.TensorCore

#lang-pulse

open Kuiper
open Pulse.Lib.Trade
include Kuiper.TensorCore.Base

ghost
fn array_fragment_pts_to_ref
  (#et : Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  ([@@@mkey] farr: array (fragment et knd m n k l))
  (#f : perm)
  (#ems : seq (value_for et knd m n k))
  preserves array_fragment_pts_to farr #f ems
  ensures   pure (Seq.length ems == Pulse.Lib.Array.length farr)
{
  unfold array_fragment_pts_to farr #f ems;
  Pulse.Lib.Array.pts_to_len farr;
  fold array_fragment_pts_to farr #f ems;
}

ghost
fn array_fragment_extract
  (#et:Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  (farr: array (fragment et knd m n k l))
  // (#f : perm) // Assuming 1.0R for now
  (#ems : seq (value_for et knd m n k))
  (i : natlt (Seq.length ems))
  requires
    array_fragment_pts_to farr ems
  ensures
    exists* (s : (lseq (fragment et knd m n k l) (Seq.length ems))).
      farr |-> s **
      (s @! i) |-> (ems @! i) **
      (forall* (em' : value_for et knd m n k).
        (farr |-> s **
         (s @! i) |-> em') @==>
          array_fragment_pts_to farr (Seq.upd ems i em'))
{
  unfold array_fragment_pts_to;
  // Why "cannot find typeclass"?
  // with s. assert farr |-> Frac f s;
  with s. assert Pulse.Lib.Array.pts_to farr s;

  forevery_extract_if_eqtype i
    (fun (x : natlt (Seq.length s)) -> (s @! x) |-> (ems @! x));

  ghost
  fn f_elim (em' : value_for et knd m n k)
    requires
      (forall+ (j : natlt (Seq.length ems)).
        if op_Equality #(natlt (Seq.length ems)) j i then emp
        else (s @! j) |-> (ems @! j))
    ensures
      (farr |-> s ** (s @! i) |-> em')
      @==> array_fragment_pts_to farr (Seq.upd ems i em')
  {
    ghost
    fn f_elim2 ()
      requires
        (forall+ (j : natlt (Seq.length ems)).
          if op_Equality #(natlt (Seq.length ems)) j i then emp
          else (s @! j) |-> (ems @! j)) **
        (farr |-> s ** (s @! i) |-> em')
      ensures
        (array_fragment_pts_to farr (Seq.upd ems i em'))
    {
      let ems' = Seq.upd ems i em';
      forevery_ext
        (fun (j : natlt (Seq.length ems)) ->
          if op_Equality #(natlt (Seq.length ems)) j i then emp
          else (s @! j) |-> (ems @! j))
        (fun (j : natlt (Seq.length ems')) ->
          if op_Equality #(natlt (Seq.length ems)) j i then emp
          else (s @! j) |-> (ems' @! j));
      forevery_unextract_if_eqtype #(natlt (Seq.length ems)) i
        (fun (j : natlt (Seq.length ems')) -> (s @! j) |-> (ems' @! j));
      forevery_rw_type
        (natlt (Seq.length ems))
        (natlt (Seq.length ems'))
        (fun (j : natlt (Seq.length ems')) -> (s @! j) |-> (ems' @! j));
      fold array_fragment_pts_to farr ems';
    };
    intro_trade _ _ _ f_elim2;
  };
  intro_forall (forevery _ _) f_elim;
  ();
}

ghost
fn array_fragment_extract_ro
  (#et:Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  (farr: array (fragment et knd m n k l))
  (#ems : seq (value_for et knd m n k))
  (#f : perm)
  (i : natlt (Seq.length ems))
  requires
    array_fragment_pts_to farr #f ems
  ensures
    exists* (s : (lseq (fragment et knd m n k l) (Seq.length ems))).
      factored
        (farr |-> Frac f s ** (s @! i) |-> Frac f (ems @! i))
        (array_fragment_pts_to farr #f ems)
{
  assume rewrites_to f 1.0R;
  array_fragment_extract farr i; with s. _;
  Pulse.Lib.Forall.elim_forall (ems @! i);
  rewrite each Seq.Base.upd ems i (Seq.Base.index ems i) as ems;
}
