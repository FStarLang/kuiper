module Kuiper.Sparse.Matrix

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
open Kuiper.Chest
open Kuiper.EMatrix
module SZ = Kuiper.SizeT

// TODO por que no está definido?
instance is_send_across_pts_to #a r #p n
: is_send_across (visibility_of r) (Pulse.Lib.Array.PtsTo.pts_to #a r #p n)
=
  let i s = is_send_across_pts_to_mask r p s (fun i -> True) in
  Tactics.Typeclasses.solve

instance is_send_across_smatrix
  #et {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (vis : visibility)
  (#_ : squash (visibility_of m.elems == vis))
  (#_ : squash (visibility_of m.col_ind == vis))
  (#_ : squash (visibility_of m.row_off == vis))
  (#f : perm)
  (v : chest2 et rows cols)
  : is_send_across vis (smatrix_pts_to m #f v)
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

ghost
fn smatrix_share_n'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (em : chest2 et rows cols)
  (k : pos)
  requires smatrix_pts_to' m #f v_elems v_col_ind v_row_off em
  ensures forall+ (_ : natlt k).
    smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off  em
{
  Kuiper.Array.Extra.array_share m.elems k;
  Kuiper.Array.Extra.array_share m.col_ind k;
  Kuiper.Array.Extra.array_share m.row_off k;

  forevery_zip
    (fun _ -> pts_to m.col_ind #(f /. k) v_col_ind)
    (fun _ -> pts_to m.row_off #(f /. k) v_row_off);
  forevery_zip
    (fun _ -> pts_to m.elems #(f /.k) v_elems) _;

  forevery_map #(natlt k)
    (fun _ ->
      pts_to m.elems   #(f /. k) v_elems **
      pts_to m.col_ind #(f /. k) v_col_ind **
      pts_to m.row_off #(f /. k) v_row_off
    )
    (fun _ -> smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off em)
    fn _ {};
}

ghost
fn smatrix_share_n
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : chest2 et rows cols)
  requires
    smatrix_pts_to m #f em
  ensures
    forall+ (_ : natlt k).
      smatrix_pts_to m #(f /. k) em
{
  unfold smatrix_pts_to m #f em;
  with v_elems.
    assert pts_to m.elems #f v_elems;
  with v_col_ind.
    assert pts_to m.col_ind #f v_col_ind;
  with v_row_off.
    assert pts_to m.row_off #f v_row_off;

  smatrix_share_n' m #f _ _ _ em k;

  forevery_map #(natlt k)
    (fun _ -> smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off em)
    (fun _ -> smatrix_pts_to m #(f /. k) em)
    fn _ { fold smatrix_pts_to m #(f /. k) em; };

}


let forall_natlt_elim (n : pos) (p : prop)
: Lemma (requires forall (_ : natlt n). p) (ensures p)
= eliminate forall (_ : natlt n). p with 0

ghost
fn forevery_natlt_elim
  (n : pos) (p : prop)
  requires forall+ (_ : natlt n). pure p
  ensures pure p
{
  forevery_extract_pure #(natlt n)
    (fun _ -> pure p) (fun _ -> p) fn _ {};

  forall_natlt_elim n p;

  forevery_map #(natlt n) (fun _ -> pure p) (fun _ -> emp) fn _ {};
  forevery_emp_elim _;

}

ghost
fn smatrix_gather_n'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (em : chest2 et rows cols)
  (k : pos)
  requires forall+ (_ : natlt k).
    smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off  em
  ensures smatrix_pts_to' m #f v_elems v_col_ind v_row_off em
{
  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;

  Kuiper.Array.Extra.array_gather m.elems   k;
  Kuiper.Array.Extra.array_gather m.col_ind k;
  Kuiper.Array.Extra.array_gather m.row_off k;

  forevery_natlt_elim k _;

  ();
}

// Para escribir esto en terminos de smatrix_gather_n'
// tendriamos que probar smatrix_pts_to_eq'
ghost
fn smatrix_gather_n
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : chest2 et rows cols)
  requires
    forall+ (_ : natlt k). smatrix_pts_to m #(f /. k) em
  ensures
    smatrix_pts_to m #f em
{
  forevery_natlt_pop k _;
  unfold smatrix_pts_to m #(f /. k) em;
  with v_elems.   assert pts_to m.elems   #(f /. k) v_elems;
  with v_col_ind. assert pts_to m.col_ind #(f /. k) v_col_ind;
  with v_row_off. assert pts_to m.row_off #(f /. k) v_row_off;

  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    preserves
      pts_to m.elems   #(f /. k) v_elems **
      pts_to m.col_ind #(f /. k) v_col_ind **
      pts_to m.row_off #(f /. k) v_row_off
    requires
      smatrix_pts_to m #(f /. k) em
    ensures
      pts_to m.elems   #(f /. k) v_elems **
      pts_to m.col_ind #(f /. k) v_col_ind **
      pts_to m.row_off #(f /. k) v_row_off
  {
    open Pulse.Lib.Array;
    unfold smatrix_pts_to m #(f /. k) em;

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

  fold smatrix_pts_to m #f em;
}

let rec mem_slice_lemma
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Lemma
    (ensures mem_slice x s a b <==> Seq.mem x (Seq.slice s a b))
    (decreases (b - a))
    [SMTPatOr
      [[SMTPat (mem_slice x s a b)];
       [SMTPat (Seq.mem x (Seq.slice s a b))]]]
=
  if a < b && s @! a <> x
    then mem_slice_lemma x s (a + 1) b
    else ()

let rec index_mem_slice_lemma
  (#et : eqtype)
  (x : et) (s : seq et)
  (a b : nat {a <= b /\ b <= len s})
  : Lemma
    (requires mem_slice x s a b /\ Seq.mem x (Seq.slice s a b))
    (ensures
      s @! index_mem_slice x s a b ==
      Seq.slice s a b @! Seq.index_mem x (Seq.slice s a b)
    )
    (decreases (b - a))
=
  if a < b && s @! a <> x
    then index_mem_slice_lemma x s (a + 1) b
    else ()


let unsparse_row_lemma
  (#et:Type0) {| scalar et |}
  (#nnz rows cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  (i : natlt rows)
  : Lemma
    (requires valid_smatrix rows cols col_ind row_off)
    (ensures
      ematrix_row (matrix_unsparse rows cols elems col_ind row_off) i ==
      seq_unsparse
        ((row_off @! i + 1) - (row_off @! i)) cols
        (Seq.slice elems (row_off @! i) (row_off @! i + 1))
        (Seq.slice col_ind (row_off @! i) (row_off @! i + 1))
    )
=
  let m = matrix_unsparse rows cols elems col_ind row_off in
  let row = ematrix_row m i in

  let ri = row_off @! i in
  let re = row_off @! (i + 1) in

  let selems = Seq.slice elems ri re in
  let spos = Seq.slice col_ind ri re in
  let s = seq_unsparse (re - ri) cols selems spos in

  introduce forall (j : natlt cols).
    row @! j == s @! j
  with (
    if mem_slice j col_ind ri re
      then (
        mem_slice_lemma j col_ind ri re;
        index_mem_slice_lemma j col_ind ri re;
        ()
      )
      else ()
  );
  assert row `Seq.equal` s