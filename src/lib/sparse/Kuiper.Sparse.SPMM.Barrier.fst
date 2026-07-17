module Kuiper.Sparse.SPMM.Barrier

(* Barrier proof for SPMM kernel. Proves that barrier_p transforms to barrier_q. *)

// TODO

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Bijection { ( |~> ) }

#set-options "--z3rlimit 20"

(* --- Helpful lemmas --- *)

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
  admit();
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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
    gpu_pts_to_slice elems_tile (ri_ - ri) p.blockItemsK
      (Seq.slice elems ri_ (ri + p.blockItemsK))
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid
{
  admit();
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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
    gpu_pts_to_slice elems_tile (ri_ - ri) p.blockItemsK
      (Seq.slice elems ri_ (ri + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind ri (ri + p.blockItemsK))
{
  admit();
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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
  admit()
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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
  admit();
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
  admit();
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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
  admit();
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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
  admit();
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
  admit();
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
  admit();
}

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
  admit();
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
  admit();
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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
    gpu_pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice elems (re - residue) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    gpu_pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice col_ind (re - residue) re) **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK
{
  admit();
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
  admit()
}
