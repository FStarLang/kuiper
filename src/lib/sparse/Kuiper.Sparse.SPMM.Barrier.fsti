module Kuiper.Sparse.SPMM.Barrier

(* Interface for the SPMM barrier proof. *)

#lang-pulse

open Kuiper
module B = Kuiper.Barrier
module SZ = Kuiper.SizeT
open Kuiper.Sparse
open Kuiper.Sparse.Math { round2 }
open Kuiper.Math { even, odd }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Array.Vectorized

(* Type-level barrier iteration expressions must be natural.  This explicit
   triggered fact survives F*'s split proof obligations. *)
private let div_times_two_nonneg_pat (a : nat) (b : pos)
  : Lemma (a / b * 2 >= 0) [SMTPat (a / b * 2)]
  = FStar.Math.Lemmas.nat_over_pos_is_nat a b;
    FStar.Math.Lemmas.nat_times_nat_is_nat (a / b) 2

(* The barrier offset [ri' + (it / 2) * blockItemsK] is natural (and equal
   to [ri'] in the first two iterations); the product is nonlinear, so supply
   these facts as a triggered lemma. *)
private let half_times_nonneg_pat (a : nat) (b : nat)
  : Lemma (a / 2 * b >= 0 /\ (a < 2 ==> a / 2 * b == 0)) [SMTPat (a / 2 * b)]
  = FStar.Math.Lemmas.nat_times_nat_is_nat (a / 2) b;
    if a < 2 then FStar.Math.Lemmas.small_div a 2

(* [round2 k n] rounds [n] down to a multiple of [k]: [n - k < round2 k n <= n].
   Nonlinear (Euclidean division), needed to typecheck the slice bounds in
   [barrier_in]/[barrier_out] below; supplied as a triggered fact. *)
private let round2_bounds_pat (k : pos) (n : nat)
  : Lemma (0 <= round2 k n /\ round2 k n <= n /\ n - round2 k n < k) [SMTPat (round2 k n)]
  = FStar.Math.Lemmas.euclidean_division_definition n k;
    FStar.Math.Lemmas.lemma_mod_lt n k;
    FStar.Math.Lemmas.nat_times_nat_is_nat (n / k) k

(* Every chunk size is at most [blockItemsK] (since [blockWidth * chunk]
   divides it); together with [round2_bounds_pat] this bounds the slices
   [ri .. ri' + blockItemsK] below. *)
private let chunk_le_blockItemsK_pat
  (#et : Type0) {| sized et, has_vec_cpy et |} (p : parameters et)
  : Lemma (chunk et <= p.blockItemsK /\ chunk sz <= p.blockItemsK)
          [SMTPat p.blockItemsK]
  = Kuiper.Divides.lemma_divides_le (p.blockWidth * chunk et) p.blockItemsK;
    Kuiper.Divides.lemma_divides_le (p.blockWidth * chunk sz) p.blockItemsK;
    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v (chunk et)) 1 (SZ.v p.blockWidth);
    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v (chunk sz)) 1 (SZ.v p.blockWidth)

(* Offsets [ri + idx * blockItemsK] are natural: nonlinear sign fact, supplied
   as a triggered lemma on the product. *)
private let times_blockItemsK_nonneg_pat
  (#et : Type0) {| sized et, has_vec_cpy et |} (p : parameters et) (a : nat)
  : Lemma (a * p.blockItemsK >= 0) [SMTPat (a * p.blockItemsK)]
  = FStar.Math.Lemmas.nat_times_nat_is_nat a (SZ.v p.blockItemsK)

(* Alignment padding fits strictly inside the first vector tile. *)
let row_prefix_bound
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et) (ri : nat)
  : Lemma (ri - round2 (max (chunk et) (chunk sz)) ri < p.blockItemsK)
  = Kuiper.Sparse.Math.prod_divides p.blockWidth (chunk et) p.blockItemsK;
    Kuiper.Sparse.Math.prod_divides p.blockWidth (chunk sz) p.blockItemsK;
    Kuiper.Divides.lemma_divides_le (chunk et) p.blockItemsK;
    Kuiper.Divides.lemma_divides_le (chunk sz) p.blockItemsK;
    FStar.Math.Lemmas.euclidean_division_definition ri (max (chunk et) (chunk sz))

(* --- Barrier slprop definitions --- *)

(* Whole slices retained by every thread carry only 1/blockWidth permission.
   The partial phase includes off = re: the kernel still executes its two
   barriers for an empty remainder, and must retain the shared allocation.
   A first partial tile starts at the actual row start, without masking. *)

let barrier_in
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
  : B.barrier_side p.blockWidth
  =
  fun it tid ->
    let trow = brow p bid |~> row_perm in
    let ri = row_off @! trow in
    let re = row_off @! trow + 1 in
    let _ = row_prefix_bound p ri in
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
        pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri - ri') p.blockItemsK
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
    else if off <= re then (
      // RESIDUE0 (sin computo previo)
      // Pre: share
      if it = 0 then
        thread_live_chunks elems_tile p.blockWidth tid **
        thread_live_chunks col_ind_tile p.blockWidth tid
      // Post: gather
      else if it = 1 then
        thread_slice_pts_to elems_tile 0 (re - ri)
          elems ri p.blockWidth tid **
        // aca podriamos obviar el resto pero prob sea mas comodo así
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK **
        thread_slice_pts_to col_ind_tile 0 (re - ri)
          col_ind ri p.blockWidth tid **
        slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK
      // RESIDUE
      // Pre: share
      else if even it then
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
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
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
    let _ = row_prefix_bound p ri in
    let ri' = round2 (max (chunk et) (chunk sz)) ri in
    let off = ri' + (it / 2) * p.blockItemsK in
    if off + p.blockItemsK <= re then (
      // MASK
      // Pre: pasamos de ownership vectorial a escalar
      if it = 0 then
        thread_slice_live elems_tile 0 (ri - ri') p.blockWidth tid **
        pts_to_slice elems_tile #(1.0R /. p.blockWidth) (ri - ri') p.blockItemsK
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
    else if off <= re then (
      // RESIDUE0 (sin computo previo)
      // Pre: share
      if it = 0 then
        thread_slice_live elems_tile 0 (re - ri) p.blockWidth tid **
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK **
        thread_slice_live col_ind_tile 0 (re - ri) p.blockWidth tid **
        slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK
      // Post: gather
      else if it = 1 then
        pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 (re - ri) (Seq.slice elems ri re) **
        // aca podriamos obviar el resto pero prob sea mas comodo así
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK **
        pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 (re - ri) (Seq.slice col_ind ri re) **
        slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - ri) p.blockItemsK
      // RESIDUE
      // Pre: share
      else if even it then
        thread_slice_live elems_tile 0 (re - off) p.blockWidth tid **
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK **
        thread_slice_live col_ind_tile 0 (re - off) p.blockWidth tid **
        slice_live col_ind_tile #(1.0R /. p.blockWidth)(re - off) p.blockItemsK
      // Post: gather
      else
        pts_to_slice elems_tile #(1.0R /. p.blockWidth)
          0 (re - off) (Seq.slice elems off re) **
        // aca podriamos obviar el resto pero prob sea mas comodo así
        slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK **
        pts_to_slice col_ind_tile #(1.0R /. p.blockWidth)
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
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (bid : natlt (nblocks p))
  : B.contract p.blockWidth =
  {
    rin  = barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
    rout = barrier_out p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
  }

(* --- Fold/unfold helpers --- *)

#push-options "--z3rlimit 20"

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


let _residue_pred0
  (blockItemsK : nat)
  (ri ri' re : nat)
  (idx nnz : nat)
: prop
=
  re - ri' < blockItemsK  /\
  idx == 0 /\
  nnz == re - ri

let _residue_pred
  (blockItemsK : pos)
  (ri ri' re : nat)
  (idx residue : nat)
: prop
=
  re - ri' >= blockItemsK  /\
  idx == (re - ri') / blockItemsK /\
  residue == (re - ri') % blockItemsK

let residue_pred
  (blockItemsK : pos)
  (ri ri' re : nat)
  (idx residue : nat)
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


#pop-options

(* --- Main barrier transform --- *)

#push-options "--z3rlimit_factor 2"
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

#pop-options
