module Kuiper.Sparse.SPMM.Barrier

(* Interface for the SPMM barrier proof. *)

#lang-pulse

open Kuiper
module B = Kuiper.Barrier
open Kuiper.Sparse
open Kuiper.Math { even, odd }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Bijection { ( |~> ) }

(* --- Barrier slprop definitions --- *)

let barrier_p_odd
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (k : natlt (p.blockItemsK /^ p.blockWidth))
  : slprop
  =
  let off = ri + idx * p.blockItemsK in
  exists* (x : et) (c : sz).
    pts_to_cell elems_tile (k * p.blockWidth + tid) x **
    pts_to_cell col_ind_tile (k * p.blockWidth + tid) c **
    pure (
      off + k * p.blockWidth + tid < re ==>
        x == elems   @! off + k * p.blockWidth + tid /\
        c == col_ind @! off + k * p.blockWidth + tid
    )

let barrier_p
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  : B.barrier_side p.blockWidth
  =
  fun it tid ->
    let trow = brow p bid |~> row_perm in
    let ri = row_off @! trow in
    let re = row_off @! trow + 1 in
    let off = ri + (it / 2) * p.blockItemsK in
    if off > re then emp else
    if even it then
      (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
      (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
    else
      forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
        barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) tid k

let barrier_q_even
  (#et : Type0)
  (p : parameters)
  (nnz : sz)
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (k : natlt(p.blockItemsK /^ p.blockWidth))
  : slprop
  =
  array_live_cell elems_tile (k * p.blockWidth + tid) **
  array_live_cell col_ind_tile (k * p.blockWidth + tid)

let barrier_q_odd
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (k : natlt p.blockItemsK)
  : slprop
  =
  let off = ri + idx * p.blockItemsK in
  exists* (x : et) (c : sz).
    pts_to_cell elems_tile #(1.0R /. p.blockWidth) k x **
    pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c **
    pure (
      off + k < re ==>
        x == elems   @! off + k /\
        c == col_ind @! off + k
    )

let barrier_q
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  : B.barrier_side p.blockWidth
  =
  fun it tid ->
    let trow = brow p bid |~> row_perm in
    let ri = row_off @! trow in
    let re = row_off @! trow + 1 in
    let off = ri + (it / 2) * p.blockItemsK in
    if off > re then emp else
      if even it then
        forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
          barrier_q_even p nnz elems_tile col_ind_tile ri re (it / 2) tid k
      else if off + p.blockItemsK <= re
        then
          elems_tile |-> Frac (1.0R /. p.blockWidth)
            (Seq.slice elems off (off + p.blockItemsK)) **
          col_ind_tile |-> Frac (1.0R /. p.blockWidth)
            (Seq.slice col_ind off (off + p.blockItemsK))
        else
          forall+ (k : natlt p.blockItemsK).
            barrier_q_odd p elems col_ind elems_tile col_ind_tile
              ri re (it / 2) k

let barrier_contract
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  : B.contract p.blockWidth =
  {
    rin  = barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
    rout = barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
  }

(* --- Utility --- *)

ghost
fn forevery_prod_to_flat
  (#n : nat) (#bw : pos)
  (p : (natlt n & natlt bw) -> slprop)
  requires forall+ (xy : natlt n & natlt bw). p xy
  ensures forall+ (i : natlt (n * bw)). p (Kuiper.Bijection.prod_gg n bw i)

(* --- Fold/unfold helpers --- *)

ghost
fn barrier_p_fold_even
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  requires
    pure (ri + idx * p.blockItemsK <= re) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures barrier_p p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid

ghost
fn barrier_p_fold_odd
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  requires
    forall+ (k: natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  ensures
    barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid (idx * 2 + 1) tid

ghost
fn barrier_q_unfold_even
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  requires
    barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid (idx * 2) tid
  ensures
    forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_q_even p nnz elems_tile col_ind_tile ri re idx tid k

ghost
fn barrier_q_unfold_odd
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK <= re))
  requires
    barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid (idx * 2 + 1) tid
  ensures
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice elems
        (ri + idx * p.blockItemsK)
        (ri + idx * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind
        (ri + idx * p.blockItemsK)
        (ri + idx * p.blockItemsK + p.blockItemsK))

ghost
fn barrier_q_unfold_odd_residue
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK > re))
  requires
    barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid (idx * 2 + 1) tid
  ensures forall+ (k : natlt p.blockItemsK).
    barrier_q_odd p elems col_ind elems_tile col_ind_tile ri re idx k

(* --- Main barrier transform --- *)

ghost
fn barrier_p_to_q_transform
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (it : nat)
  requires
    forall+ (tid : natlt p.blockWidth).
      barrier_p p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid it tid
  ensures
    forall+ (tid : natlt p.blockWidth).
      barrier_q p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid it tid
