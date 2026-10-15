module Kuiper.Sparse.SPMM.Barrier

(* Barrier ownership transformations for SpMM. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.Sparse
open Kuiper.Array.Vectorized
open Kuiper.Math { even }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Bijection { ( |~> ) }

#set-options "--z3rlimit 20"

(* Distribute writable scalar cells while retaining read shares of the tail. *)
ghost
fn array_to_partial
  (#et : Type0) (#n : pos)
  (x : larray et n) (len : natle n) (nthr : pos)
  requires live x
  ensures forall+ (tid : natlt nthr).
    thread_slice_live x 0 len nthr tid **
    slice_live x #(1.0R /. nthr) len n
{
  array_to_slice x;
  drop_ (is_full_slice x n);
  with s. assert pts_to_slice x 0 n s;
  Kuiper.Seq.Common.lem_append_slice s 0 len n;
  rewrite pts_to_slice x 0 n s as
    pts_to_slice x 0 n (Seq.append (Seq.slice s 0 len) (Seq.slice s len n));
  slice_split x 0 len n;
  fold slice_live x 0 len;
  thread_slice_share x 0 len #0 nthr;
  slice_share x len n nthr #1.0R;
  forevery_map
    (fun (_ : natlt nthr) -> pts_to_slice x #(1.0R /. nthr) len n (Seq.slice s len n))
    (fun _ -> slice_live x #(1.0R /. nthr) len n)
    fn _ { fold slice_live x #(1.0R /. nthr) len n; };
  forevery_zip
    (fun tid -> thread_slice_live x 0 len nthr tid)
    (fun (_ : natlt nthr) -> slice_live x #(1.0R /. nthr) len n);
}

(* Gather initialized cells, then share them for every thread's computation.
   The retained tail also supplies allocation identity when len = 0. *)
ghost
fn partial_to_shared
  (#et : Type0) (#n : pos) (#m : nat)
  (x : larray et n) (s : lseq et m)
  (off : nat) (len : natle n { off + len <= m }) (nthr : pos)
  requires forall+ (tid : natlt nthr).
    thread_slice_pts_to x 0 len s off nthr tid **
    slice_live x #(1.0R /. nthr) len n
  ensures forall+ (_ : natlt nthr).
    pts_to_slice x #(1.0R /. nthr) 0 len (Seq.slice s off (off + len)) **
    slice_live x #(1.0R /. nthr) len n
{
  forevery_unzip
    (fun tid -> thread_slice_pts_to x 0 len s off nthr tid)
    (fun (_ : natlt nthr) -> slice_live x #(1.0R /. nthr) len n);
  forevery_natlt_pop nthr (fun _ -> slice_live x #(1.0R /. nthr) len n);
  unfold slice_live x #(1.0R /. nthr) len n;
  slice_array_exists x len n;
  fold slice_live x #(1.0R /. nthr) len n;
  forevery_natlt_push nthr (fun _ -> slice_live x #(1.0R /. nthr) len n);
  thread_slice_gather x 0 len s off nthr;
  rewrite each (off + (len - 0)) as (off + len);
  slice_share x 0 len nthr #1.0R;
  forevery_zip
    (fun (_ : natlt nthr) -> pts_to_slice x #(1.0R /. nthr) 0 len (Seq.slice s off (off + len)))
    (fun (_ : natlt nthr) -> slice_live x #(1.0R /. nthr) len n);
}

ghost
fn chunks_to_mask
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : pos) (#m : nat)
  (x : larray et n) (s : lseq et m)
  (off : nat { off + n <= m }) (len : natle n) (nthr : pos)
  (#_ : squash ((nthr * chunk et) /? n))
  requires forall+ (tid : natlt nthr). thread_pts_to_chunks x s off nthr tid
  ensures forall+ (tid : natlt nthr).
    thread_slice_live x 0 len nthr tid **
    pts_to_slice x #(1.0R /. nthr) len n (Seq.slice s (off + len) (off + n))
{
  thread_gather_chunks x s off nthr;
  array_to_slice x;
  drop_ (is_full_slice x n);
  Kuiper.Seq.Common.lem_append_slice s off (off + len) (off + n);
  rewrite pts_to_slice x 0 n (Seq.slice s off (off + n)) as
    pts_to_slice x 0 n (Seq.append (Seq.slice s off (off + len)) (Seq.slice s (off + len) (off + n)));
  slice_split x 0 len n;
  fold slice_live x 0 len;
  thread_slice_share x 0 len #0 nthr;
  slice_share x len n nthr #1.0R;
  forevery_zip
    (fun tid -> thread_slice_live x 0 len nthr tid)
    (fun (_ : natlt nthr) -> pts_to_slice x #(1.0R /. nthr) len n (Seq.slice s (off + len) (off + n)));
}

ghost
fn mask_to_shared
  (#et : Type0) (#n : pos)
  (x : larray et n) (len : natle n)
  (z : et) (s : lseq et (n - len)) (nthr : pos)
  requires forall+ (tid : natlt nthr).
    thread_slice_pts_to_value x 0 len z nthr tid **
    pts_to_slice x #(1.0R /. nthr) len n s
  ensures forall+ (_ : natlt nthr).
    x |-> Frac (1.0R /. nthr) (Seq.append (Seq.create len z) s)
{
  forevery_unzip
    (fun tid -> thread_slice_pts_to_value x 0 len z nthr tid)
    (fun (_ : natlt nthr) -> pts_to_slice x #(1.0R /. nthr) len n s);
  slice_gather x len n nthr #1.0R;
  slice_array_exists x len n;
  thread_slice_gather_value x 0 len z nthr;
  rewrite each (len - 0) as len;
  slice_concat x 0 len n;
  slice_to_array_full x;
  Kuiper.Array.Extra.array_share x nthr;
}


(* --- Fold/unfold helpers --- *)

ghost
fn barrier_in_fold_mask_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri_ : sz{ri_ == row_off @! (brow p bid |~> row_perm)})
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) ri_})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + p.blockItemsK <= re))
  requires
    thread_pts_to_chunks elems_tile elems ri p.blockWidth tid **
    thread_pts_to_chunks col_ind_tile col_ind ri p.blockWidth tid
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid
{
  rewrite (thread_pts_to_chunks elems_tile elems ri p.blockWidth tid **
    thread_pts_to_chunks col_ind_tile col_ind ri p.blockWidth tid)
    as (barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid);
}

ghost
fn barrier_in_fold_mask_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri_ : sz{ri_ == row_off @! (brow p bid |~> row_perm)})
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) ri_})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + p.blockItemsK <= re))
  requires
    thread_slice_pts_to_value elems_tile 0 (ri_ - ri) zero
      p.blockWidth tid **
    pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri_ - ri) p.blockItemsK
      (Seq.slice elems ri_ (ri + p.blockItemsK))
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid
{
  rewrite (thread_slice_pts_to_value elems_tile 0 (ri_ - ri) zero
      p.blockWidth tid **
    pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri_ - ri) p.blockItemsK
      (Seq.slice elems ri_ (ri + p.blockItemsK)))
    as (barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid);
}

ghost
fn barrier_out_unfold_mask_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri_ : sz{ri_ == row_off @! (brow p bid |~> row_perm)})
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) ri_})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + p.blockItemsK <= re))
  requires barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid
  ensures
    thread_slice_live elems_tile 0 (ri_ - ri) p.blockWidth tid **
    pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri_ - ri) p.blockItemsK
      (Seq.slice elems ri_ (ri + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind ri (ri + p.blockItemsK))
{
  rewrite (barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid)
    as (thread_slice_live elems_tile 0 (ri_ - ri) p.blockWidth tid **
    pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri_ - ri) p.blockItemsK
      (Seq.slice elems ri_ (ri + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind ri (ri + p.blockItemsK)));
}

ghost
fn barrier_out_unfold_mask_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri_ : sz{ri_ == row_off @! (brow p bid |~> row_perm)})
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) ri_})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + p.blockItemsK <= re))
  requires barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid
  ensures
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.append
        (Seq.create (ri_ - ri) zero)
        (Seq.slice elems ri_ (ri  + p.blockItemsK))
      )
{
  rewrite (barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid)
    as (elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.append
        (Seq.create (ri_ - ri) zero)
        (Seq.slice elems ri_ (ri  + p.blockItemsK))
      ));
}

ghost
fn barrier_in_fold_main_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat { idx > 0 })
  (tid : natlt p.blockWidth)
  requires
    pure (ri + idx * p.blockItemsK + p.blockItemsK <= re) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid
{
  Kuiper.Math.div_mod_of_mul_add 2 idx 0;
  rewrite ((exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s))
    as (barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid);
}

ghost
fn barrier_in_fold_main_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat { idx > 0 })
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK <= re))
  requires
    thread_pts_to_chunks elems_tile elems (ri + idx * p.blockItemsK) p.blockWidth tid **
    thread_pts_to_chunks col_ind_tile col_ind (ri + idx * p.blockItemsK) p.blockWidth tid
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
{
  Kuiper.Math.div_mod_of_mul_add 2 idx 1;
  rewrite (thread_pts_to_chunks elems_tile elems (ri + idx * p.blockItemsK) p.blockWidth tid **
    thread_pts_to_chunks col_ind_tile col_ind (ri + idx * p.blockItemsK) p.blockWidth tid)
    as (barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid);
}

ghost
fn barrier_out_unfold_main_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat { idx > 0 })
  (tid : natlt p.blockWidth)
  requires barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid **
    pure (ri + idx * p.blockItemsK + p.blockItemsK <= re)
  ensures
    thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid
{
  Kuiper.Math.div_mod_of_mul_add 2 idx 0;
  rewrite (barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid)
    as (thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid);
}

ghost
fn barrier_out_unfold_main_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : nat { idx > 0 })
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK <= re))
  requires barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
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
  Kuiper.Math.div_mod_of_mul_add 2 idx 1;
  rewrite (barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid)
    as (elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice elems
        (ri + idx * p.blockItemsK)
        (ri + idx * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind
        (ri + idx * p.blockItemsK)
        (ri + idx * p.blockItemsK + p.blockItemsK)));
}

ghost
fn barrier_in_fold_residue0_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  requires
    pure (re - ri < p.blockItemsK) **
    thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid
{
  rewrite (thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid)
    as (barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid);
}

ghost
fn barrier_in_fold_residue_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  requires
    pure (re - ri >= p.blockItemsK) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ((re - ri) / p.blockItemsK * 2) tid
{
  FStar.Math.Lemmas.euclidean_division_definition (re - ri) p.blockItemsK;
  Kuiper.Math.div_mod_of_mul_add 2 ((re - ri) / p.blockItemsK) 0;
  rewrite ((exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s))
    as (barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ((re - ri) / p.blockItemsK * 2) tid);
}

(* Relate the final scalar load to the aligned full-tile loop. *)
let residue_phase_bounds
  (block : pos) (ri ri' re : nat { ri' <= ri /\ ri <= re })
  (idx residue : nat { residue_pred block ri ri' re idx residue })
  : Lemma
      (ri' + idx * block <= re /\ re < ri' + idx * block + block /\
       (if idx = 0 then residue == re - ri else residue == re - (ri' + idx * block)))
  = FStar.Math.Lemmas.euclidean_division_definition (re - ri') block;
    if re - ri' < block then
      FStar.Math.Lemmas.small_division_lemma_1 (re - ri') block

ghost
fn barrier_in_fold_residue_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  (idx residue : nat { residue_pred p.blockItemsK ri ri' re idx residue })
  requires
    thread_slice_pts_to elems_tile 0 residue
      elems (re - residue) p.blockWidth tid **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    thread_slice_pts_to col_ind_tile 0 residue
      col_ind (re - residue) p.blockWidth tid **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
{
  residue_phase_bounds p.blockItemsK ri ri' re idx residue;
  Kuiper.Math.div_mod_of_mul_add 2 idx 1;
  rewrite (thread_slice_pts_to elems_tile 0 residue
      elems (re - residue) p.blockWidth tid **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    thread_slice_pts_to col_ind_tile 0 residue
      col_ind (re - residue) p.blockWidth tid **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK)
    as (barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid);
}

ghost
fn barrier_out_unfold_residue_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  (idx residue : nat { residue_pred p.blockItemsK ri ri' re idx residue })
  requires barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid
  ensures
    thread_slice_live elems_tile 0 residue p.blockWidth tid **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    thread_slice_live col_ind_tile 0 residue p.blockWidth tid **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK
{
  residue_phase_bounds p.blockItemsK ri ri' re idx residue;
  Kuiper.Math.div_mod_of_mul_add 2 idx 0;
  rewrite (barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid)
    as (thread_slice_live elems_tile 0 residue p.blockWidth tid **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    thread_slice_live col_ind_tile 0 residue p.blockWidth tid **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK);
}

ghost
fn barrier_out_unfold_residue_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  (ri : sz{SZ.v ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : natlt p.blockWidth)
  (idx residue : nat { residue_pred p.blockItemsK ri ri' re idx residue })
  requires barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
  ensures
    pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice elems (re - residue) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice col_ind (re - residue) re) **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK
{
  residue_phase_bounds p.blockItemsK ri ri' re idx residue;
  Kuiper.Math.div_mod_of_mul_add 2 idx 1;
  rewrite (barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid)
    as (pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice elems (re - residue) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice col_ind (re - residue) re) **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK);
}

(* --- Main barrier transform --- *)

ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
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
      barrier_in p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid it tid
  ensures
    forall+ (tid : natlt p.blockWidth).
      barrier_out p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid it tid
{
  let trow = brow p bid |~> row_perm;
  let ri = SZ.v (row_off @! trow);
  let re = SZ.v (row_off @! trow + 1);
  let ri' = round2 (max (chunk et) (chunk sz)) ri;
  row_prefix_bound p ri;
  assert pure (ri <= re /\ re <= nnz);
  let off = ri' + (it / 2) * p.blockItemsK;
  let full = off + p.blockItemsK <= re;
  let ev = even it;
  if full {
    // The first full tile is masked before it is read.
    if (it = 0) {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> thread_pts_to_chunks elems_tile elems off p.blockWidth tid ** thread_pts_to_chunks
            col_ind_tile col_ind off p.blockWidth tid)
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            (thread_pts_to_chunks elems_tile elems off p.blockWidth tid ** thread_pts_to_chunks col_ind_tile
            col_ind off p.blockWidth tid); };
      forevery_unzip (fun (tid : natlt p.blockWidth) -> thread_pts_to_chunks elems_tile elems off p.blockWidth
          tid) (fun tid -> thread_pts_to_chunks col_ind_tile col_ind off p.blockWidth tid);
      chunks_to_mask elems_tile elems off (ri - ri') p.blockWidth;
      rewrite each (off + (ri - ri')) as ri;
      rewrite each (off + p.blockItemsK) as (ri' + p.blockItemsK);
      thread_gather_chunks col_ind_tile col_ind off p.blockWidth;
      Kuiper.Array.Extra.array_share col_ind_tile p.blockWidth;
      forevery_zip (fun (tid : natlt p.blockWidth) -> (thread_slice_live elems_tile 0 (ri - ri') p.blockWidth
          tid ** pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri - ri') p.blockItemsK (Seq.slice elems ri
          (ri' + p.blockItemsK)))) (fun tid -> col_ind_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice col_ind
          off (off + p.blockItemsK)));
      forevery_map
        (fun (tid : natlt p.blockWidth) -> (thread_slice_live elems_tile 0 (ri - ri') p.blockWidth tid **
            pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri - ri') p.blockItemsK (Seq.slice elems ri (ri'
            + p.blockItemsK))) ** col_ind_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice col_ind off (off +
            p.blockItemsK)))
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (thread_slice_live elems_tile 0 (ri - ri') p.blockWidth tid ** pts_to_slice
            elems_tile #(1.0R /. p.blockWidth) (ri - ri') p.blockItemsK (Seq.slice elems ri (ri' +
            p.blockItemsK)) ** col_ind_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice col_ind off (off +
            p.blockItemsK))) as barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it
            tid; };
    } else if (it = 1) {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> thread_slice_pts_to_value elems_tile 0 (ri - ri') zero p.blockWidth tid ** pts_to_slice
            elems_tile #(1.0R /. p.blockWidth) (ri - ri') p.blockItemsK (Seq.slice elems ri (ri' +
            p.blockItemsK)))
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            (thread_slice_pts_to_value elems_tile 0 (ri - ri') zero p.blockWidth tid ** pts_to_slice
            elems_tile #(1.0R /. p.blockWidth) (ri - ri') p.blockItemsK (Seq.slice elems ri (ri' +
            p.blockItemsK))); };
      mask_to_shared elems_tile (ri - ri') zero (Seq.slice elems ri (ri' + p.blockItemsK)) p.blockWidth;
      forevery_map
        (fun (tid : natlt p.blockWidth) -> elems_tile |-> Frac (1.0R /. p.blockWidth) (Seq.append (Seq.create
            (ri - ri') zero) (Seq.slice elems ri (ri' + p.blockItemsK))))
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (elems_tile |-> Frac (1.0R /. p.blockWidth) (Seq.append (Seq.create (ri - ri') zero)
            (Seq.slice elems ri (ri' + p.blockItemsK)))) as barrier_out p row_perm elems col_ind row_off
            elems_tile col_ind_tile bid it tid; };
    } else if ev {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) ** (exists* (s : seq
            sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s))
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            ((exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) ** (exists* (s : seq sz).
            col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)); };
      forevery_unzip (fun (tid : natlt p.blockWidth) -> (exists* (s : seq et). elems_tile |-> Frac (1.0R /.
          p.blockWidth) s)) (fun tid -> (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth)
          s));
      array_gather_underspec elems_tile #1.0R p.blockWidth;
      array_gather_underspec col_ind_tile #1.0R p.blockWidth;
      thread_share_chunks elems_tile p.blockWidth;
      thread_share_chunks col_ind_tile p.blockWidth;
      forevery_zip (fun (tid : natlt p.blockWidth) -> thread_live_chunks elems_tile p.blockWidth tid) (fun tid
          -> thread_live_chunks col_ind_tile p.blockWidth tid);
      forevery_map
        (fun (tid : natlt p.blockWidth) -> thread_live_chunks elems_tile p.blockWidth tid **
            thread_live_chunks col_ind_tile p.blockWidth tid)
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (thread_live_chunks elems_tile p.blockWidth tid ** thread_live_chunks col_ind_tile
            p.blockWidth tid) as barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it
            tid; };
    } else {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> thread_pts_to_chunks elems_tile elems off p.blockWidth tid ** thread_pts_to_chunks
            col_ind_tile col_ind off p.blockWidth tid)
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            (thread_pts_to_chunks elems_tile elems off p.blockWidth tid ** thread_pts_to_chunks col_ind_tile
            col_ind off p.blockWidth tid); };
      forevery_unzip (fun (tid : natlt p.blockWidth) -> thread_pts_to_chunks elems_tile elems off p.blockWidth
          tid) (fun tid -> thread_pts_to_chunks col_ind_tile col_ind off p.blockWidth tid);
      thread_gather_chunks elems_tile elems off p.blockWidth;
      thread_gather_chunks col_ind_tile col_ind off p.blockWidth;
      Kuiper.Array.Extra.array_share elems_tile p.blockWidth;
      Kuiper.Array.Extra.array_share col_ind_tile p.blockWidth;
      forevery_zip (fun (tid : natlt p.blockWidth) -> elems_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice
          elems off (off + p.blockItemsK))) (fun tid -> col_ind_tile |-> Frac (1.0R /. p.blockWidth)
          (Seq.slice col_ind off (off + p.blockItemsK)));
      forevery_map
        (fun (tid : natlt p.blockWidth) -> elems_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice elems off
            (off + p.blockItemsK)) ** col_ind_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice col_ind off (off
            + p.blockItemsK)))
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (elems_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice elems off (off +
            p.blockItemsK)) ** col_ind_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice col_ind off (off +
            p.blockItemsK))) as barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it
            tid; };
    }
  } else if (off <= re) {
    // First partial tiles load from ri; later ones start at off.
    if (it = 0) {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> thread_live_chunks elems_tile p.blockWidth tid ** thread_live_chunks col_ind_tile
            p.blockWidth tid)
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            (thread_live_chunks elems_tile p.blockWidth tid ** thread_live_chunks col_ind_tile p.blockWidth
            tid); };
      forevery_unzip (fun (tid : natlt p.blockWidth) -> thread_live_chunks elems_tile p.blockWidth tid) (fun
          tid -> thread_live_chunks col_ind_tile p.blockWidth tid);
      thread_gather_live_chunks elems_tile p.blockWidth;
      thread_gather_live_chunks col_ind_tile p.blockWidth;
      array_to_partial elems_tile (re - ri) p.blockWidth;
      array_to_partial col_ind_tile (re - ri) p.blockWidth;
      forevery_zip (fun (tid : natlt p.blockWidth) -> (thread_slice_live elems_tile 0 (re - ri) p.blockWidth
          tid ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK)) (fun tid ->
          (thread_slice_live col_ind_tile 0 (re - ri) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /.
          p.blockWidth) (re - ri) p.blockItemsK));
      forevery_map
        (fun (tid : natlt p.blockWidth) -> (thread_slice_live elems_tile 0 (re - ri) p.blockWidth tid **
            slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK) ** (thread_slice_live
            col_ind_tile 0 (re - ri) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re -
            ri) p.blockItemsK))
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (thread_slice_live elems_tile 0 (re - ri) p.blockWidth tid ** slice_live elems_tile
            #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK ** thread_slice_live col_ind_tile 0 (re - ri)
            p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK) as
            barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid; };
    } else if (it = 1) {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> (thread_slice_pts_to elems_tile 0 (re - ri) elems (ri) p.blockWidth tid ** slice_live
            elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK) ** (thread_slice_pts_to col_ind_tile 0
            (re - ri) col_ind (ri) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re -
            ri) p.blockItemsK))
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            (thread_slice_pts_to elems_tile 0 (re - ri) elems (ri) p.blockWidth tid ** slice_live elems_tile
            #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK ** thread_slice_pts_to col_ind_tile 0 (re - ri)
            col_ind (ri) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - ri)
            p.blockItemsK); };
      forevery_unzip (fun (tid : natlt p.blockWidth) -> (thread_slice_pts_to elems_tile 0 (re - ri) elems (ri)
          p.blockWidth tid ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK)) (fun tid
          -> (thread_slice_pts_to col_ind_tile 0 (re - ri) col_ind (ri) p.blockWidth tid ** slice_live
          col_ind_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK));
      partial_to_shared elems_tile elems (ri) (re - ri) p.blockWidth;
      partial_to_shared col_ind_tile col_ind (ri) (re - ri) p.blockWidth;
      rewrite each ((ri) + (re - ri)) as re;
      forevery_zip (fun (tid : natlt p.blockWidth) -> (pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 (re -
          ri) (Seq.slice elems (ri) re) ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri)
          p.blockItemsK)) (fun tid -> (pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 (re - ri)
          (Seq.slice col_ind (ri) re) ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - ri)
          p.blockItemsK));
      forevery_map
        (fun (tid : natlt p.blockWidth) -> (pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 (re - ri)
            (Seq.slice elems (ri) re) ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri)
            p.blockItemsK) ** (pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 (re - ri) (Seq.slice
            col_ind (ri) re) ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK))
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 (re - ri) (Seq.slice elems (ri)
            re) ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK ** pts_to_slice
            col_ind_tile #(1.0R /. p.blockWidth) 0 (re - ri) (Seq.slice col_ind (ri) re) ** slice_live
            col_ind_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK) as barrier_out p row_perm elems
            col_ind row_off elems_tile col_ind_tile bid it tid; };
    } else if ev {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) ** (exists* (s : seq
            sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s))
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            ((exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) ** (exists* (s : seq sz).
            col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)); };
      forevery_unzip (fun (tid : natlt p.blockWidth) -> (exists* (s : seq et). elems_tile |-> Frac (1.0R /.
          p.blockWidth) s)) (fun tid -> (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth)
          s));
      array_gather_underspec elems_tile #1.0R p.blockWidth;
      array_gather_underspec col_ind_tile #1.0R p.blockWidth;
      array_to_partial elems_tile (re - off) p.blockWidth;
      array_to_partial col_ind_tile (re - off) p.blockWidth;
      forevery_zip (fun (tid : natlt p.blockWidth) -> (thread_slice_live elems_tile 0 (re - off) p.blockWidth
          tid ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK)) (fun tid ->
          (thread_slice_live col_ind_tile 0 (re - off) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /.
          p.blockWidth) (re - off) p.blockItemsK));
      forevery_map
        (fun (tid : natlt p.blockWidth) -> (thread_slice_live elems_tile 0 (re - off) p.blockWidth tid **
            slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK) ** (thread_slice_live
            col_ind_tile 0 (re - off) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re
            - off) p.blockItemsK))
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (thread_slice_live elems_tile 0 (re - off) p.blockWidth tid ** slice_live elems_tile
            #(1.0R /. p.blockWidth) (re - off) p.blockItemsK ** thread_slice_live col_ind_tile 0 (re - off)
            p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK) as
            barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid; };
    } else {
      forevery_map
        (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        (fun tid -> (thread_slice_pts_to elems_tile 0 (re - off) elems (off) p.blockWidth tid ** slice_live
            elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK) ** (thread_slice_pts_to col_ind_tile
            0 (re - off) col_ind (off) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re
            - off) p.blockItemsK))
        fn tid { rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid as
            (thread_slice_pts_to elems_tile 0 (re - off) elems (off) p.blockWidth tid ** slice_live elems_tile
            #(1.0R /. p.blockWidth) (re - off) p.blockItemsK ** thread_slice_pts_to col_ind_tile 0 (re - off)
            col_ind (off) p.blockWidth tid ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off)
            p.blockItemsK); };
      forevery_unzip (fun (tid : natlt p.blockWidth) -> (thread_slice_pts_to elems_tile 0 (re - off) elems
          (off) p.blockWidth tid ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK))
          (fun tid -> (thread_slice_pts_to col_ind_tile 0 (re - off) col_ind (off) p.blockWidth tid **
          slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK));
      partial_to_shared elems_tile elems (off) (re - off) p.blockWidth;
      partial_to_shared col_ind_tile col_ind (off) (re - off) p.blockWidth;
      rewrite each ((off) + (re - off)) as re;
      forevery_zip (fun (tid : natlt p.blockWidth) -> (pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 (re -
          off) (Seq.slice elems (off) re) ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - off)
          p.blockItemsK)) (fun tid -> (pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 (re - off)
          (Seq.slice col_ind (off) re) ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off)
          p.blockItemsK));
      forevery_map
        (fun (tid : natlt p.blockWidth) -> (pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 (re - off)
            (Seq.slice elems (off) re) ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - off)
            p.blockItemsK) ** (pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 (re - off) (Seq.slice
            col_ind (off) re) ** slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK))
        (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
        fn tid { rewrite (pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 (re - off) (Seq.slice elems (off)
            re) ** slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK ** pts_to_slice
            col_ind_tile #(1.0R /. p.blockWidth) 0 (re - off) (Seq.slice col_ind (off) re) ** slice_live
            col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK) as barrier_out p row_perm elems
            col_ind row_off elems_tile col_ind_tile bid it tid; };
    }
  } else {
    forevery_map
      (fun tid -> barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
      (fun tid -> barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid it tid)
      fn tid {
        rewrite barrier_in p row_perm elems col_ind row_off
          elems_tile col_ind_tile bid it tid as emp;
        rewrite emp as barrier_out p row_perm elems col_ind row_off
          elems_tile col_ind_tile bid it tid;
      };
  }
}
