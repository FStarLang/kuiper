module Kuiper.Sparse.SPMM.Barrier

(* Barrier proof for SPMM kernel. Proves that barrier_p transforms to barrier_q. *)

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Bijection { ( |~> ) }

#set-options "--z3rlimit 20"

#restart-solver
#push-options "--z3rlimit 80 --fuel 0 --ifuel 0"
ghost
fn forevery_prod_to_flat
  (#n : nat) (#bw : pos)
  (p : (natlt n & natlt bw) -> slprop)
  requires forall+ (xy : natlt n & natlt bw). p xy
  ensures forall+ (i : natlt (n * bw)). p (Kuiper.Bijection.prod_gg n bw i)
{
  assert pure (n * bw == n * bw);
  forevery_iso (Kuiper.Bijection.bij_nat_prod #n #bw) p;
  forevery_ext
    (fun (i : natlt (n * bw)) -> p ((Kuiper.Bijection.bij_nat_prod #n #bw).gg i))
    (fun (i : natlt (n * bw)) -> p (Kuiper.Bijection.prod_gg n bw i));
}

ghost
fn forevery_flat_to_prod
  (#n : nat) (#bw : pos)
  (f : natlt (n * bw) -> slprop)
  requires forall+ (i : natlt (n * bw)). f i
  ensures forall+ (xy : natlt n & natlt bw). f (Kuiper.Bijection.prod_ff n bw xy)
{
  assert pure (n * bw == n * bw);
  forevery_iso (Kuiper.Bijection.bij_sym (Kuiper.Bijection.bij_nat_prod #n #bw)) f;
  forevery_ext
    (fun (xy : natlt n & natlt bw) -> f ((Kuiper.Bijection.bij_sym (Kuiper.Bijection.bij_nat_prod #n #bw)).gg xy))
    (fun (xy : natlt n & natlt bw) -> f (Kuiper.Bijection.prod_ff n bw xy));
}
#pop-options

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
{
  rewrite (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
          (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
       as barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid (idx * 2) tid;
  ();
}

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
{
  assert rewrites_to ri (row_off @! (brow p bid |~> row_perm));
  assert rewrites_to re (row_off @! (brow p bid |~> row_perm) + 1);

  let it = idx * 2 + 1;

  rewrite each idx as (it / 2);
  rewrite forall+ (k: natlt (p.blockItemsK /^ p.blockWidth)).
    barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) tid k
    as (
      let off = ri + (it / 2) * p.blockItemsK in
      if off > re then emp else
      if even it then
        (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
        (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
      else
        forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
          barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) tid k
    );

  fold barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;

  rewrite each it as (idx * 2 + 1);

  ();
}

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
{
  assert rewrites_to ri (row_off @! (brow p bid |~> row_perm));
  assert rewrites_to re (row_off @! (brow p bid |~> row_perm) + 1);

  let it = idx * 2;

  rewrite each (idx * 2) as it;
  unfold barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;

  rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
  rewrite each (even it) as true;

  rewrite each it as (idx * 2);
  rewrite each (idx * 2 / 2) as idx;

  ();

}

#push-options "--z3rlimit 40"
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
{
  assert rewrites_to ri (row_off @! (brow p bid |~> row_perm));
  assert rewrites_to re (row_off @! (brow p bid |~> row_perm) + 1);

  let it = idx * 2 + 1;

  rewrite each (idx * 2 + 1) as it;
  unfold barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;

  FStar.Math.Lemmas.lemma_div_mod_plus 1 idx 2;
  assert pure (ri + (it / 2) * p.blockItemsK + p.blockItemsK <= re);
  rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
  rewrite each (even it) as false;
  rewrite each (ri + (it / 2) * p.blockItemsK + p.blockItemsK <= re) as true;

  rewrite each it as (idx * 2 + 1);
  rewrite each ((idx * 2 + 1) / 2) as idx;

  ();
}
#pop-options

#push-options "--z3rlimit 20"
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
{
  assert rewrites_to ri (row_off @! (brow p bid |~> row_perm));
  assert rewrites_to re (row_off @! (brow p bid |~> row_perm) + 1);

  let it = idx * 2 + 1;

  rewrite each (idx * 2 + 1) as it;
  unfold barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;

  FStar.Math.Lemmas.lemma_div_mod_plus 1 idx 2;
  rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
  rewrite each (even it) as false;
  rewrite each (ri + (it / 2) * p.blockItemsK + p.blockItemsK <= re) as false;

  rewrite each it as (idx * 2 + 1);
  rewrite each ((idx * 2 + 1) / 2) as idx;

  ();
}
#pop-options

#push-options "--z3rlimit 400 --fuel 4 --ifuel 4"
ghost
fn barrier_q_fold_even
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
  (ri : nat{ri == row_off @! (brow p bid |~> row_perm)})
  (re : nat{re == row_off @! (brow p bid |~> row_perm) + 1})
  (it : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (even it))
  (#_ : squash (ri + (it / 2) * p.blockItemsK <= re))
  requires
    forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_q_even p nnz elems_tile col_ind_tile ri re (it / 2) tid k
  ensures
    barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid
{
  rewrite each ri as (row_off @! (brow p bid |~> row_perm));
  rewrite each re as (row_off @! (brow p bid |~> row_perm) + 1);

  rewrite
    forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_q_even p nnz elems_tile col_ind_tile (row_off @! (brow p bid |~> row_perm)) (row_off @! (brow p bid |~> row_perm) + 1) (it / 2) tid k
  as barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;
}
#pop-options

#push-options "--z3rlimit 80 --fuel 2 --ifuel 2"
ghost
fn barrier_q_fold_odd
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
  (ri : nat{ri == row_off @! (brow p bid |~> row_perm)})
  (re : nat{re == row_off @! (brow p bid |~> row_perm) + 1})
  (it : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (odd it))
  (#_ : squash (ri + (it / 2) * p.blockItemsK <= re))
  (#_ : squash (ri + (it / 2) * p.blockItemsK + p.blockItemsK <= re))
  requires
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice (reveal elems) (ri + (it / 2) * p.blockItemsK) (ri + (it / 2) * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice (reveal col_ind) (ri + (it / 2) * p.blockItemsK) (ri + (it / 2) * p.blockItemsK + p.blockItemsK))
  ensures
    barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid
{
  rewrite each ri as (row_off @! (brow p bid |~> row_perm));

  rewrite
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice (reveal elems) ((row_off @! (brow p bid |~> row_perm)) + (it / 2) * p.blockItemsK) ((row_off @! (brow p bid |~> row_perm)) + (it / 2) * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice (reveal col_ind) ((row_off @! (brow p bid |~> row_perm)) + (it / 2) * p.blockItemsK) ((row_off @! (brow p bid |~> row_perm)) + (it / 2) * p.blockItemsK + p.blockItemsK))
  as barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;
}
#pop-options

#push-options "--z3rlimit 80 --fuel 2 --ifuel 2"
ghost
fn barrier_q_fold_odd_residue
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
  (ri : nat{ri == row_off @! (brow p bid |~> row_perm)})
  (re : nat{re == row_off @! (brow p bid |~> row_perm) + 1})
  (it : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (odd it))
  (#_ : squash (ri + (it / 2) * p.blockItemsK <= re))
  (#_ : squash (ri + (it / 2) * p.blockItemsK + p.blockItemsK > re))
  requires
    forall+ (k : natlt p.blockItemsK).
      barrier_q_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) k
  ensures
    barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid
{
  rewrite each ri as (row_off @! (brow p bid |~> row_perm));
  rewrite each re as (row_off @! (brow p bid |~> row_perm) + 1);

  rewrite
    forall+ (k : natlt p.blockItemsK).
      barrier_q_odd p elems col_ind elems_tile col_ind_tile (row_off @! (brow p bid |~> row_perm)) (row_off @! (brow p bid |~> row_perm) + 1) (it / 2) k
  as barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;
}
#pop-options

open Kuiper.Bijection

(* Helper: reindex barrier_p_odd from (tid, k) pairs to flat cell indexing.
   Shared by both odd_full and odd_residue proofs. *)

#push-options "--z3rlimit 60"
ghost
fn barrier_p_odd_to_cells
  (#et : Type0)
  (p : parameters { size_req p })
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  requires
    forall+ (tid : natlt p.blockWidth).
      forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
        barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  ensures
    forall+ (i : natlt p.blockItemsK).
      exists* (x : et) (c : sz).
        pts_to_cell elems_tile i x **
        pts_to_cell col_ind_tile i c **
        pure ((ri + idx * p.blockItemsK) + i < re ==>
          x == elems @! ((ri + idx * p.blockItemsK) + i) /\
          c == col_ind @! ((ri + idx * p.blockItemsK) + i))
{
  // Step 1: Commute (tid, k) → (k, tid)
  forevery_commute
    (fun (tid : natlt p.blockWidth) (k : natlt (p.blockItemsK /^ p.blockWidth)) ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k);

  // Step 2: Flatten (k, tid) → pair
  forevery_flatten
    (fun (k : natlt (p.blockItemsK /^ p.blockWidth)) (tid : natlt p.blockWidth) ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k);

  // Step 3: Pair → flat via prod_to_flat
  assert pure ((p.blockItemsK /^ p.blockWidth) * p.blockWidth == p.blockItemsK);

  forevery_prod_to_flat
    (fun (xy : natlt (p.blockItemsK /^ p.blockWidth) & natlt p.blockWidth) ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx xy._2 xy._1);

  // Step 4: Unfold barrier_p_odd and simplify i/bw*bw + i%bw → i
  forevery_map
    (fun (i : natlt ((p.blockItemsK /^ p.blockWidth) * p.blockWidth)) ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx
        (prod_gg (p.blockItemsK /^ p.blockWidth) p.blockWidth i)._2
        (prod_gg (p.blockItemsK /^ p.blockWidth) p.blockWidth i)._1)
    (fun (i : natlt ((p.blockItemsK /^ p.blockWidth) * p.blockWidth)) ->
      exists* (x : et) (c : sz).
        pts_to_cell elems_tile i x **
        pts_to_cell col_ind_tile i c **
        pure ((ri + idx * p.blockItemsK) + i < re ==>
          x == elems @! ((ri + idx * p.blockItemsK) + i) /\
          c == col_ind @! ((ri + idx * p.blockItemsK) + i)))
    fn i {
      unfold barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx
        (i % p.blockWidth) (i / p.blockWidth);
      assert pure ((i / p.blockWidth) * p.blockWidth + i % p.blockWidth == i);
      rewrite each ((i / p.blockWidth) * p.blockWidth + i % p.blockWidth) as i;
    };

  // Step 5: rw_size
  forevery_rw_size ((p.blockItemsK /^ p.blockWidth) * p.blockWidth) p.blockItemsK;
}
#pop-options

#push-options "--z3rlimit 200"
ghost
fn even_barrier_p_to_q
  (#et : Type0)
  (p : parameters { size_req p })
  (#nnz : sz)
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  requires
    forall+ (tid : natlt p.blockWidth).
      (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
      (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures
    forall+ (tid : natlt p.blockWidth).
      forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
        barrier_q_even p nnz elems_tile col_ind_tile ri re idx tid k
{
  // Step 1: Separate elems and col_ind existentials
  forevery_unzip
    (fun (tid : natlt p.blockWidth) ->
      exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s)
    (fun (tid : natlt p.blockWidth) ->
      exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s);

  // Step 2: Gather elems fractions
  array_gather_underspec elems_tile p.blockWidth;
  // Step 3: Gather col_ind fractions
  array_gather_underspec col_ind_tile p.blockWidth;

  // Step 4: Slice into per-cell ownership
  with ve. assert pts_to elems_tile   ve;
  with vc. assert pts_to col_ind_tile vc;
  Pulse.Lib.Array.pts_to_len elems_tile;
  Pulse.Lib.Array.pts_to_len col_ind_tile;

  array_slice_1 elems_tile;
  array_slice_1 col_ind_tile;

  // Step 5: Zip and wrap cells into array_live_cell
  forevery_zip
    (fun (i : natlt p.blockItemsK) ->
      pts_to_cell elems_tile i (ve @! i))
    (fun (i : natlt p.blockItemsK) ->
      pts_to_cell col_ind_tile i (vc @! i));

  forevery_map
    (fun (i : natlt p.blockItemsK) ->
      pts_to_cell elems_tile i (ve @! i) **
      pts_to_cell col_ind_tile i (vc @! i))
    (fun (i : natlt p.blockItemsK) ->
      array_live_cell elems_tile i ** array_live_cell col_ind_tile i)
    fn i {
      fold array_live_cell elems_tile i;
      fold array_live_cell col_ind_tile i;
    };

  // Step 7: Flat→pair reindex
  assert pure ((p.blockItemsK /^ p.blockWidth) * p.blockWidth == p.blockItemsK);

  forevery_rw_size p.blockItemsK ((p.blockItemsK /^ p.blockWidth) * p.blockWidth);

  forevery_flat_to_prod
    (fun (i : natlt ((p.blockItemsK /^ p.blockWidth) * p.blockWidth)) ->
      array_live_cell elems_tile i ** array_live_cell col_ind_tile i);

  // Step 8: Unflatten to forall+ k. forall+ tid.
  forevery_unflatten
    (fun (k : natlt (p.blockItemsK /^ p.blockWidth)) (tid : natlt p.blockWidth) ->
      array_live_cell elems_tile (k * p.blockWidth + tid) **
      array_live_cell col_ind_tile (k * p.blockWidth + tid));

  // Step 9: Commute to forall+ tid. forall+ k.
  forevery_commute
    (fun (k : natlt (p.blockItemsK /^ p.blockWidth)) (tid : natlt p.blockWidth) ->
      array_live_cell elems_tile (k * p.blockWidth + tid) **
      array_live_cell col_ind_tile (k * p.blockWidth + tid));

  assert
    forall+ (tid : natlt p.blockWidth).
      forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
        barrier_q_even p nnz elems_tile col_ind_tile ri re idx tid k;
}

#pop-options

(* --- Odd case, full block: per-thread cells → shared array with known content --- *)

#push-options "--z3rlimit 200"
ghost
fn odd_full_barrier_p_to_q
  (#et : Type0)
  (p : parameters { size_req p })
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK <= re))
  requires
    forall+ (tid : natlt p.blockWidth).
      forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
        barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  ensures
    forall+ (tid : natlt p.blockWidth).
      elems_tile |-> Frac (1.0R /. p.blockWidth)
        (Seq.slice (reveal elems) (ri + idx * p.blockItemsK)
                                  (ri + idx * p.blockItemsK + p.blockItemsK)) **
      col_ind_tile |-> Frac (1.0R /. p.blockWidth)
        (Seq.slice (reveal col_ind) (ri + idx * p.blockItemsK)
                                    (ri + idx * p.blockItemsK + p.blockItemsK))
{
  let off = ri + idx * p.blockItemsK;

  // Reindex from (tid, k) to flat cells
  barrier_p_odd_to_cells p elems col_ind elems_tile col_ind_tile ri re idx;

  // Specialize cells and rewrite indices for unslice
  forevery_map
    (fun (i : natlt p.blockItemsK) ->
      exists* (x : et) (c : sz).
        pts_to_cell elems_tile i x **
        pts_to_cell col_ind_tile i c **
        pure ((ri + idx * p.blockItemsK) + i < re ==>
          x == elems @! ((ri + idx * p.blockItemsK) + i) /\
          c == col_ind @! ((ri + idx * p.blockItemsK) + i)))
    (fun (i : natlt p.blockItemsK) ->
      pts_to_cell elems_tile i ((Seq.slice (reveal elems) off (off + p.blockItemsK)) @! i) **
      pts_to_cell col_ind_tile i ((Seq.slice (reveal col_ind) off (off + p.blockItemsK)) @! i))
    fn i {
      with x c. assert (
        pts_to_cell elems_tile i x **
        pts_to_cell col_ind_tile i c **
        pure ((ri + idx * p.blockItemsK) + i < re ==>
          x == elems @! ((ri + idx * p.blockItemsK) + i) /\
          c == col_ind @! ((ri + idx * p.blockItemsK) + i)));
      assert pure (off + i < re);
      Seq.lemma_index_slice (reveal elems) off (off + p.blockItemsK) i;
      Seq.lemma_index_slice (reveal col_ind) off (off + p.blockItemsK) i;
      rewrite each x as ((Seq.slice (reveal elems) off (off + p.blockItemsK)) @! i);
      rewrite each c as ((Seq.slice (reveal col_ind) off (off + p.blockItemsK)) @! i);
      drop_ (pure ((ri + idx * p.blockItemsK) + i < re ==>
        (Seq.slice (reveal elems) off (off + p.blockItemsK)) @! i ==
          elems @! ((ri + idx * p.blockItemsK) + i) /\
        (Seq.slice (reveal col_ind) off (off + p.blockItemsK)) @! i ==
          col_ind @! ((ri + idx * p.blockItemsK) + i)));
    };

  // Unzip, unslice, share, zip
  forevery_unzip
    (fun (i : natlt p.blockItemsK) ->
      pts_to_cell elems_tile i ((Seq.slice (reveal elems) off (off + p.blockItemsK)) @! i))
    (fun (i : natlt p.blockItemsK) ->
      pts_to_cell col_ind_tile i ((Seq.slice (reveal col_ind) off (off + p.blockItemsK)) @! i));

  array_unslice_1 elems_tile;
  array_unslice_1 col_ind_tile;

  Kuiper.Array.Extra.array_share elems_tile   p.blockWidth;
  Kuiper.Array.Extra.array_share col_ind_tile p.blockWidth;

  forevery_zip
    (fun (_ : natlt p.blockWidth) ->
      elems_tile |-> Frac (1.0R /. p.blockWidth)
        (Seq.slice (reveal elems) off (off + p.blockItemsK)))
    (fun (_ : natlt p.blockWidth) ->
      col_ind_tile |-> Frac (1.0R /. p.blockWidth)
        (Seq.slice (reveal col_ind) off (off + p.blockItemsK)));

  // Normalize off in ensures
  rewrite each off as (ri + idx * p.blockItemsK);
}
#pop-options

(* --- Odd case, residue: per-thread cells → shared fractional cells --- *)

#push-options "--z3rlimit 60"
ghost
fn odd_residue_barrier_p_to_q
  (#et : Type0)
  (p : parameters { size_req p })
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK > re))
  requires
    forall+ (tid : natlt p.blockWidth).
      forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
        barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  ensures
    forall+ (tid : natlt p.blockWidth).
      forall+ (k : natlt p.blockItemsK).
        barrier_q_odd p elems col_ind elems_tile col_ind_tile ri re idx k
{
  // Reindex from (tid, k) to flat cells
  barrier_p_odd_to_cells p elems col_ind elems_tile col_ind_tile ri re idx;

  // Share each cell p.blockWidth ways
  forevery_map
    (fun (i : natlt p.blockItemsK) ->
      exists* (x : et) (c : sz).
        pts_to_cell elems_tile i x **
        pts_to_cell col_ind_tile i c **
        pure ((ri + idx * p.blockItemsK) + i < re ==>
          x == elems @! ((ri + idx * p.blockItemsK) + i) /\
          c == col_ind @! ((ri + idx * p.blockItemsK) + i)))
    (fun (i : natlt p.blockItemsK) ->
      forall+ (_ : natlt p.blockWidth).
        exists* (x : et) (c : sz).
          pts_to_cell elems_tile #(1.0R /. p.blockWidth) i x **
          pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) i c **
          pure ((ri + idx * p.blockItemsK) + i < re ==>
            x == elems @! ((ri + idx * p.blockItemsK) + i) /\
            c == col_ind @! ((ri + idx * p.blockItemsK) + i)))
    fn i {
      with x c. assert (
        pts_to_cell elems_tile i x **
        pts_to_cell col_ind_tile i c **
        pure ((ri + idx * p.blockItemsK) + i < re ==>
          x == elems @! ((ri + idx * p.blockItemsK) + i) /\
          c == col_ind @! ((ri + idx * p.blockItemsK) + i)));

      // Unfold cells to slices for sharing
      unfold pts_to_cell elems_tile #1.0R i x;
      unfold pts_to_cell col_ind_tile #1.0R i c;

      // Share each slice p.blockWidth ways
      slice_share elems_tile i (i + 1) p.blockWidth;
      slice_share col_ind_tile i (i + 1) p.blockWidth;

      // Introduce duplicated pure fact and zip all three
      forevery_intro_pure
        (fun (_ : natlt p.blockWidth) ->
          (ri + idx * p.blockItemsK) + i < re ==>
            x == elems @! ((ri + idx * p.blockItemsK) + i) /\
            c == col_ind @! ((ri + idx * p.blockItemsK) + i));

      forevery_zip3
        (fun (_ : natlt p.blockWidth) ->
          pts_to_slice elems_tile #(1.0R /. p.blockWidth) i (i + 1) (seq![x]))
        (fun (_ : natlt p.blockWidth) ->
          pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) i (i + 1) (seq![c]))
        (fun (_ : natlt p.blockWidth) ->
          pure ((ri + idx * p.blockItemsK) + i < re ==>
            x == elems @! ((ri + idx * p.blockItemsK) + i) /\
            c == col_ind @! ((ri + idx * p.blockItemsK) + i)));

      // Fold cells back and package existentials
      forevery_map
        (fun (_ : natlt p.blockWidth) ->
          pts_to_slice elems_tile #(1.0R /. p.blockWidth) i (i + 1) (seq![x]) **
          pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) i (i + 1) (seq![c]) **
          pure ((ri + idx * p.blockItemsK) + i < re ==>
            x == elems @! ((ri + idx * p.blockItemsK) + i) /\
            c == col_ind @! ((ri + idx * p.blockItemsK) + i)))
        (fun (_ : natlt p.blockWidth) ->
          exists* (x' : et) (c' : sz).
            pts_to_cell elems_tile #(1.0R /. p.blockWidth) i x' **
            pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) i c' **
            pure ((ri + idx * p.blockItemsK) + i < re ==>
              x' == elems @! ((ri + idx * p.blockItemsK) + i) /\
              c' == col_ind @! ((ri + idx * p.blockItemsK) + i)))
        fn _ {
          fold (pts_to_cell elems_tile #(1.0R /. p.blockWidth) i x);
          fold (pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) i c);
        };
    };

  // Step 4: Commute: forall+ i. forall+ _:p.blockWidth → forall+ _:p.blockWidth. forall+ i
  forevery_commute
    (fun (i : natlt p.blockItemsK) (_ : natlt p.blockWidth) ->
      barrier_q_odd p elems col_ind elems_tile col_ind_tile ri re idx i);
}
#pop-options

(* --- Main barrier transform --- *)

#push-options "--z3rlimit 200"
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
{
  let trow = brow p bid;
  let ri : nat = row_off @! (trow |~> row_perm);
  let re : nat = row_off @! (trow |~> row_perm) + 1;
  let off : nat = ri + (it / 2) * p.blockItemsK;
  let bik : nat = p.blockItemsK;

  if (off > re) {
    // Case 1: out of bounds — both sides are emp
    forevery_ext
      (fun (tid : natlt p.blockWidth) ->
        barrier_p p row_perm elems col_ind row_off
          elems_tile col_ind_tile bid it tid)
      (fun (tid : natlt p.blockWidth) ->
        barrier_q p row_perm elems col_ind row_off
          elems_tile col_ind_tile bid it tid);
  } else {
    let ev = even it;
    if ev {
      // Case 2: even — shared fractions → per-thread cells
      assert pure (even it);
      assert pure (off <= re);
      let idx = it / 2;

      // Unfold barrier_p into its even expansion (inlined)
      forevery_map
        (fun (tid : natlt p.blockWidth) ->
          barrier_p p row_perm elems col_ind row_off
            elems_tile col_ind_tile bid it tid)
        (fun (tid : natlt p.blockWidth) ->
          (exists* (s : seq et).
            elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
          (exists* (s : seq sz).
            col_ind_tile |-> Frac (1.0R /. p.blockWidth) s))
        fn tid {
          assert rewrites_to ri (row_off @! (brow p bid |~> row_perm));
          assert rewrites_to re (row_off @! (brow p bid |~> row_perm) + 1);
          unfold barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;
          rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
          rewrite each (even it) as true;
          ()
        };

      even_barrier_p_to_q #et p #nnz elems_tile col_ind_tile ri re idx;

      // Fold barrier_q from its even expansion
      rewrite each idx as (it / 2);
      forevery_map
        (fun (tid : natlt p.blockWidth) ->
          forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
            barrier_q_even p nnz elems_tile col_ind_tile ri re (it / 2) tid k)
        (fun (tid : natlt p.blockWidth) ->
          barrier_q p row_perm elems col_ind row_off
            elems_tile col_ind_tile bid it tid)
        fn tid {
          barrier_q_fold_even p row_perm elems col_ind row_off
            elems_tile col_ind_tile bid ri re it tid;
        };
    } else {
      // Case 3/4: odd — per-thread cells → shared
      assert pure (odd it);
      assert pure (off <= re);
      let idx = it / 2;

      // Unfold barrier_p into its odd expansion (inlined)
      forevery_map
        (fun (tid : natlt p.blockWidth) ->
          barrier_p p row_perm elems col_ind row_off
            elems_tile col_ind_tile bid it tid)
        (fun (tid : natlt p.blockWidth) ->
          forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
            barrier_p_odd p elems col_ind elems_tile col_ind_tile
              ri re idx tid k)
        fn tid {
          assert rewrites_to ri (row_off @! (brow p bid |~> row_perm));
          assert rewrites_to re (row_off @! (brow p bid |~> row_perm) + 1);
          unfold barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid;
          rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
          rewrite each (even it) as false;
          rewrite each (it / 2) as idx;
        };

      if (off + bik <= re) {
        // Case 3: full block
        odd_full_barrier_p_to_q p elems col_ind
          elems_tile col_ind_tile ri re idx;

        // Fold barrier_q from its odd-full expansion
        rewrite each (ri + idx * p.blockItemsK) as off;
        forevery_map
          (fun (tid : natlt p.blockWidth) ->
            elems_tile |-> Frac (1.0R /. p.blockWidth)
              (Seq.slice (reveal elems) off (off + p.blockItemsK)) **
            col_ind_tile |-> Frac (1.0R /. p.blockWidth)
              (Seq.slice (reveal col_ind) off (off + p.blockItemsK)))
          (fun (tid : natlt p.blockWidth) ->
            barrier_q p row_perm elems col_ind row_off
              elems_tile col_ind_tile bid it tid)
          fn tid {
            barrier_q_fold_odd p row_perm elems col_ind row_off
              elems_tile col_ind_tile bid ri re it tid;
          };
      } else {
        // Case 4: residue
        odd_residue_barrier_p_to_q p elems col_ind
          elems_tile col_ind_tile ri re idx;

        // Fold barrier_q from its odd-residue expansion
        rewrite each idx as (it / 2);
        forevery_map
          (fun (tid : natlt p.blockWidth) ->
            forall+ (k : natlt p.blockItemsK).
              barrier_q_odd p elems col_ind elems_tile col_ind_tile
                ri re (it / 2) k)
          (fun (tid : natlt p.blockWidth) ->
            barrier_q p row_perm elems col_ind row_off
              elems_tile col_ind_tile bid it tid)
          fn tid {
            barrier_q_fold_odd_residue p row_perm elems col_ind row_off
              elems_tile col_ind_tile bid ri re it tid;
          };
      }
    }
  }
}
#pop-options
