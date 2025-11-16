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
  (#f : perm)
  (#ems : seq (value_for et knd m n k))
  (i : natlt (Seq.length ems))
  requires
    array_fragment_pts_to farr #f ems
  ensures
    exists* (s : (lseq (fragment et knd m n k l) (Seq.length ems))).
      farr |-> Frac f s **
      (s @! i) |-> (ems @! i) **
      (forall* (em' : value_for et knd m n k).
        (farr |-> Frac f s **
         (s @! i) |-> em') @==>
          array_fragment_pts_to farr #f (Seq.upd ems i em'))
{
  unfold array_fragment_pts_to;
  with s. assert Pulse.Lib.Array.pts_to farr #f s;

  forevery_extract_if_eqtype i
    (fun (x : natlt (Seq.length s)) -> (s @! x) |-> (ems @! x));

  ghost
  fn f_elim (em' : value_for et knd m n k)
    requires
      (forall+ (j : natlt (Seq.length ems)).
        if op_Equality #(natlt (Seq.length ems)) j i then emp
        else (s @! j) |-> (ems @! j))
    ensures
      (farr |-> Frac f s ** (s @! i) |-> em')
      @==> array_fragment_pts_to farr #f (Seq.upd ems i em')
  {
    ghost
    fn f_elim2 ()
      requires
        (forall+ (j : natlt (Seq.length ems)).
          if op_Equality #(natlt (Seq.length ems)) j i then emp
          else (s @! j) |-> (ems @! j)) **
        (farr |-> Frac f s ** (s @! i) |-> em')
      ensures
        array_fragment_pts_to farr #f (Seq.upd ems i em')
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
      fold array_fragment_pts_to farr #f ems';
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
  array_fragment_extract farr i;
  elim_forall (ems @! i);
  rewrite each Seq.upd ems i (ems @! i) as ems;
  ();
}
