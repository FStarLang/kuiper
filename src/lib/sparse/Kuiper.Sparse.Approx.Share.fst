module Kuiper.Sparse.Approx.Share

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Sparse.Matrix
open Kuiper.Sparse.Approx


ghost
fn smatrix_share_approx_n
  (#et:Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : chest2 et rows cols)
  (#rm : chest2 real rows cols)
  requires
    smatrix_approx m #f em rm
  ensures
    forall+ (_ : natlt k).
      smatrix_approx m #(f /. k) em rm
{
  unfold smatrix_approx m #f em rm;
  with v_elems v_col_ind v_row_off relems.
    unfold smatrix_model m #f v_elems v_col_ind v_row_off relems em rm;
  smatrix_share_approx_n' m #f _ _ _ relems rm k;

  forevery_map #(natlt k)
    (fun _ -> smatrix_approx' m #(f /. k) v_elems v_col_ind v_row_off relems rm)
    (fun _ -> smatrix_approx m #(f /. k) em rm)
    fn _ {
      fold smatrix_model m #(f /. k) v_elems v_col_ind v_row_off relems em rm;
      fold smatrix_approx m #(f /. k) em rm;
    };

}

ghost
fn smatrix_gather_approx_n
  (#et:Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : chest2 et rows cols)
  (#rm : chest2 real rows cols)
  requires
    forall+ (_ : natlt k). smatrix_approx m #(f /. k) em rm
  ensures
    smatrix_approx m #f em rm
{
  forevery_natlt_pop k _;
  unfold smatrix_approx m #(f /. k) em rm;
  with v_elems v_col_ind v_row_off relems.
    unfold smatrix_model m #(f /. k) v_elems v_col_ind v_row_off relems em rm;

  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    preserves
      pts_to m.elems   #(f /. k) v_elems **
      pts_to m.col_ind #(f /. k) v_col_ind **
      pts_to m.row_off #(f /. k) v_row_off
    requires
      smatrix_approx m #(f /. k) em rm
    ensures
      pts_to m.elems   #(f /. k) v_elems **
      pts_to m.col_ind #(f /. k) v_col_ind **
      pts_to m.row_off #(f /. k) v_row_off
  {
    open Pulse.Lib.Array;
    unfold smatrix_approx m #(f /. k) em rm;
    with es cs rs real_es. unfold smatrix_model m #(f /. k) es cs rs real_es em rm;

    pts_to_injective_eq m.elems;
    pts_to_injective_eq m.col_ind;
    pts_to_injective_eq m.row_off;
    ()
  };

  forevery_map_extra _ _ _ aux;
  forevery_natlt_push k _;

  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;

  Kuiper.Array.Extra.array_gather m.elems   k;
  Kuiper.Array.Extra.array_gather m.col_ind k;
  Kuiper.Array.Extra.array_gather m.row_off k;

  fold smatrix_model m #f v_elems v_col_ind v_row_off relems em rm;
  fold smatrix_approx m #f em rm;
}

(* Array ownership, including the pure real model, stays on the same device. *)
instance is_send_across_pts_to #a r #p n
: is_send_across (visibility_of r) (Pulse.Lib.Array.PtsTo.pts_to #a r #p n)
=
  let i s = is_send_across_pts_to_mask r p s (fun i -> True) in
  Tactics.Typeclasses.solve

instance is_send_across_smatrix_approx
  #et {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (vis : visibility)
  (#_ : squash (visibility_of m.elems == vis))
  (#_ : squash (visibility_of m.col_ind == vis))
  (#_ : squash (visibility_of m.row_off == vis))
  (#f : perm)
  (v : chest2 et rows cols) (rv : chest2 real rows cols)
  : is_send_across vis (smatrix_approx m #f v rv)
=
  let open Pulse.Lib.Array.PtsTo in
  let i_elems elems :
    is_send_across vis (pts_to m.elems #f elems) =
    is_send_across_pts_to m.elems #f elems  in
  let i_col_ind col_ind :
    is_send_across vis (pts_to m.col_ind #f col_ind) =
    is_send_across_pts_to m.col_ind #f col_ind  in
  let i_row_off row_off :
    is_send_across vis (pts_to m.row_off #f row_off) =
    is_send_across_pts_to m.row_off #f row_off  in
  solve
