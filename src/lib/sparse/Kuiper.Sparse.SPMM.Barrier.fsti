module Kuiper.Sparse.SPMM.Barrier

(* Interface for the SPMM barrier proof. *)

#lang-pulse

open Kuiper
module B = Kuiper.Barrier
module SZ = Kuiper.SizeT
open Kuiper.Sparse
open Kuiper.Math { even, odd }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Array.Vectorized

(* --- Barrier slprop definitions --- *)

let barrier_in
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
  : B.barrier_side p.blockWidth
  =
  fun it tid ->
    let trow = brow p bid |~> row_perm in
    let ri = row_off @! trow in
    let re = row_off @! trow + 1 in
    let ri' = round2 (max (chunk et) (chunk sz)) ri in
    let off = ri' + (it / 2) * p.blockItemsK in
    if off + p.blockItemsK <= re then (
      // MASK
      // Pre: pasamos de ownership vectorial a escalar
      if it = 0 then
        thread_pts_to_chunks elems_tile elems off p.blockWidth tid **
        thread_pts_to_chunks col_ind_tile col_ind off p.blockWidth tid
      // Post: gather
      else if it = 1 then
        thread_slice_pts_to_value elems_tile 0 (ri - ri') zero
          p.blockWidth tid **
        gpu_pts_to_slice elems_tile (ri - ri') p.blockItemsK
          (Seq.slice elems ri (ri' + p.blockItemsK))
      // MAIN
      // Pre: share
      else if even it then
        (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
        (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
      // Post: gather
      else
        thread_pts_to_chunks elems_tile elems off p.blockWidth tid **
        thread_pts_to_chunks col_ind_tile col_ind off p.blockWidth tid
    )
    else if off < re then (
      // RESIDUE0 (sin computo previo)
      // Pre: share
      if it = 0 then
        thread_live_chunks elems_tile p.blockWidth tid **
        thread_live_chunks col_ind_tile p.blockWidth tid
      // Post: gather
      else if it = 1 then
        thread_slice_pts_to elems_tile 0 (re - ri)
          elems off p.blockWidth tid **
        // aca podriamos obviar el resto pero prob sea mas comodo así
        slice_live elems_tile (re - ri) p.blockItemsK **
        thread_slice_pts_to col_ind_tile 0 (re - ri)
          col_ind off p.blockWidth tid **
        slice_live col_ind_tile (re - ri) p.blockItemsK
      // RESIDUE
      // Pre: share
      // else if even it then
      else if it = (re - ri') / p.blockItemsK * 2 then
        (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
        (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
      // Post: gather
      else
        thread_slice_pts_to elems_tile 0 (re - off)
          elems off p.blockWidth tid **
        // aca podriamos obviar el resto pero prob sea mas comodo así
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK **
        thread_slice_pts_to col_ind_tile 0 (re - off)
          col_ind off p.blockWidth tid **
        slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK
    )
      // DONE
    else emp

let barrier_out
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
  : B.barrier_side p.blockWidth
  =
  fun it tid ->
    let trow = brow p bid |~> row_perm in
    let ri = row_off @! trow in
    let re = row_off @! trow + 1 in
    let ri' = round2 (max (chunk et) (chunk sz)) ri in
    let off = ri' + (it / 2) * p.blockItemsK in
    if off + p.blockItemsK <= re then (
      // MASK
      // Pre: pasamos de ownership vectorial a escalar
      if it = 0 then
        thread_slice_live elems_tile 0 (ri - ri') p.blockWidth tid **
        gpu_pts_to_slice elems_tile (ri - ri') p.blockItemsK
          (Seq.slice elems ri (ri' + p.blockItemsK)) **
        col_ind_tile |-> Frac (1.0R /. p.blockWidth)
          (Seq.slice col_ind off (off + p.blockItemsK))
      // Post: gather
      else if it = 1 then
        // TODO será la mejor manera de escribirlo?
        elems_tile |-> Frac (1.0R /. p.blockWidth)
          (Seq.append
            (Seq.create (ri - ri') zero)
            (Seq.slice elems
              (ri + it / 2 * p.blockItemsK)
              (off  + p.blockItemsK)))
      // MAIN
      // Pre: share
      else if even it then
        thread_live_chunks elems_tile p.blockWidth tid **
        thread_live_chunks col_ind_tile p.blockWidth tid
      // Post: gather
      else
        elems_tile |-> Frac (1.0R /. p.blockWidth)
          (Seq.slice elems off (off + p.blockItemsK)) **
        col_ind_tile |-> Frac (1.0R /. p.blockWidth)
          (Seq.slice col_ind off (off + p.blockItemsK))
    )
    else if off < re then (
      // RESIDUE0 (sin computo previo)
      // Pre: share
      if it = 0 then
        thread_slice_live elems_tile 0 (re - ri) p.blockWidth tid **
        slice_live elems_tile (re - ri) p.blockItemsK **
        thread_slice_live col_ind_tile 0 (re - ri) p.blockWidth tid **
        slice_live col_ind_tile (re - ri) p.blockItemsK
      // Post: gather
      else if it = 1 then
        gpu_pts_to_slice elems_tile 0 (re - ri) (Seq.slice elems ri re) **
        // aca podriamos obviar el resto pero prob sea mas comodo así
        slice_live elems_tile (re - ri) p.blockItemsK **
        gpu_pts_to_slice col_ind_tile 0 (re - ri) (Seq.slice col_ind ri re) **
        slice_live col_ind_tile (re - ri) p.blockItemsK
      // RESIDUE
      // Pre: share
      else if it = (re - ri') / p.blockItemsK * 2 then
        thread_slice_live elems_tile 0 (re - off) p.blockWidth tid **
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK **
        thread_slice_live col_ind_tile 0 (re - off) p.blockWidth tid **
        slice_live col_ind_tile #(1.0R /. p.blockWidth)(re - off) p.blockItemsK
      // Post: gather
      else
        gpu_pts_to_slice elems_tile #(1.0R /. p.blockWidth)
          0 (re - off) (Seq.slice elems off re) **
        // aca podriamos obviar el resto pero prob sea mas comodo así
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK **
        gpu_pts_to_slice col_ind_tile #(1.0R /. p.blockWidth)
          0 (re - off) (Seq.slice col_ind off re) **
        slice_live col_ind_tile #(1.0R /. p.blockWidth)(re - off) p.blockItemsK
    )
      // DONE
    else emp

let barrier_contract
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
  : B.contract p.blockWidth =
  {
    rin  = barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
    rout = barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
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
fn barrier_in_fold_mask_pre
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
    thread_pts_to_chunks elems_tile elems ri p.blockWidth tid **
    thread_pts_to_chunks col_ind_tile col_ind ri p.blockWidth tid
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid

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

ghost
fn barrier_in_fold_main_post
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
  requires
    thread_pts_to_chunks elems_tile elems (ri + idx * p.blockItemsK) p.blockWidth tid **
    thread_pts_to_chunks col_ind_tile col_ind (ri + idx * p.blockItemsK) p.blockWidth tid
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid

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

ghost
fn barrier_in_fold_residue0_pre
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
  (tid : natlt p.blockWidth)
  requires
    pure (re - ri < p.blockItemsK) **
    thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid

ghost
fn barrier_in_fold_residue_pre
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
  (tid : natlt p.blockWidth)
  requires
    pure (re - ri >= p.blockItemsK) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ((re - ri) / p.blockItemsK * 2) tid


let _residue_pred0
  (blockItemsK : nat)
  (ri ri' re : nat)
  (idx : nat)
  (nnz : nat)
: prop
=
  re - ri' < blockItemsK  /\
  idx == 0 /\
  nnz == re - ri

let _residue_pred
  (blockItemsK : pos)
  (ri ri' re : nat)
  (idx : nat)
  (residue : nat)
: prop
=
  re - ri' >= blockItemsK  /\
  idx == (re - ri') / blockItemsK /\
  residue == (re - ri') % blockItemsK

let residue_pred
  (blockItemsK : pos)
  (ri ri' re : nat)
  (idx : nat)
  (residue : nat)
: prop
=
  _residue_pred0 blockItemsK ri ri' re idx residue \/
  _residue_pred blockItemsK ri ri' re idx residue

ghost
fn barrier_in_fold_residue_post
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
  requires
    thread_slice_pts_to elems_tile 0 residue
      elems (re - residue) p.blockWidth tid **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    thread_slice_pts_to col_ind_tile 0 residue
      col_ind (re - residue) p.blockWidth tid **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK
  ensures barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid

ghost
fn barrier_out_unfold_residue_pre
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
    elems_tile col_ind_tile bid (idx * 2) tid
  ensures
    thread_slice_live elems_tile 0 residue p.blockWidth tid **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    thread_slice_live col_ind_tile 0 residue p.blockWidth tid **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK

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
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
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
