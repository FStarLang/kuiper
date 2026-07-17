module Kuiper.Ghost.TensorTranspose

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg
open Kuiper.Injection

// Not great that we need these helpers.
#push-options "--fuel 2 --ifuel 2 --z3rlimit 80"
let lem_imap_swap_rc (#rows #cols : nat) (rc : natlt rows & (natlt cols & unit))
  : Lemma ((l2_row_major rows cols).imap.f rc ==
           (l2_col_major cols rows).imap.f (rc._2._1, (rc._1, ())))
          [SMTPat ((l2_row_major rows cols).imap.f rc)]
  = ()

let lem_imap_swap_cr (#rows #cols : nat) (rc : natlt rows & (natlt cols & unit))
  : Lemma ((l2_col_major rows cols).imap.f rc ==
           (l2_row_major cols rows).imap.f (rc._2._1, (rc._1, ())))
          [SMTPat ((l2_col_major rows cols).imap.f rc)]
  = ()
#pop-options

#push-options "--z3rlimit 40 --retry 3"
ghost
fn ghost_transpose1
  (#et:Type)
  (#rows #cols : nat)
  (gA : tensor et (l2_row_major rows cols))
  (#m : chest2 et rows cols)
  requires
    gA |-> m
  ensures
    row2col gA |-> mtranspose m
{
  tensor_concr gA;
  assert (pure (Seq.equal
                  (to_seq (l2_row_major rows cols) m)
                  (to_seq (l2_col_major cols rows) (mtranspose m))));
  rewrite core gA |-> to_seq (l2_row_major rows cols) m
       as core gA |-> to_seq (l2_col_major cols rows) (mtranspose m);
  tensor_abs (l2_col_major cols rows) (core gA);
}
#pop-options

#push-options "--z3rlimit 40 --retry 3"
ghost
fn ghost_transpose2
  (#et:Type)
  (#rows #cols : nat)
  (gA : tensor et (l2_col_major rows cols))
  (#m : chest2 et rows cols)
  requires
    gA |-> m
  ensures
    col2row gA |-> mtranspose m
{
  tensor_concr gA;
  assert (pure (Seq.equal
                  (to_seq (l2_col_major rows cols) m)
                  (to_seq (l2_row_major cols rows) (mtranspose m))));
  rewrite core gA |-> to_seq (l2_col_major rows cols) m
       as core gA |-> to_seq (l2_row_major cols rows) (mtranspose m);
  tensor_abs (l2_row_major cols rows) (core gA);
}
#pop-options

ghost
fn ghost_transpose1_back
  (#et:Type)
  (#rows #cols : nat)
  (gA : tensor et (l2_row_major rows cols))
  (#m : chest2 et cols rows)
  requires
    row2col gA |-> m
  ensures
    gA |-> mtranspose m
{
  ghost_transpose2 (row2col gA);
  rewrite
    col2row (row2col gA) |-> mtranspose m
  as
    gA |-> mtranspose m;
  ()
}

ghost
fn ghost_transpose2_back
  (#et:Type)
  (#rows #cols : nat)
  (gA : tensor et (l2_col_major rows cols))
  (#m : chest2 et cols rows)
  requires
    col2row gA |-> m
  ensures
    gA |-> mtranspose m
{
  ghost_transpose1 (col2row gA);
  rewrite
    row2col (col2row gA) |-> mtranspose m
  as
    gA |-> mtranspose m;
  ()
}
