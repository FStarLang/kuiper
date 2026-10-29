module Kuiper.Sparse.SPMM.Main

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.Approx
open Kuiper.Sparse.LoadStore
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Sparse.SPMM.Barrier

open Kuiper.Tensor
open Kuiper.Array.Vectorized
open Pulse.Lib.Pledge
open Kuiper.Kernel.Base
open Kuiper.EMatrix
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Array2.Strided

module Math = Kuiper.Sparse.Math
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module Compute = Kuiper.Sparse.SPMM.Compute

open Kuiper.Sparse.SPMM.Ownership

open Kuiper.Sparse.SPMM.Head
open Kuiper.Seq.Common { (@+) }

#push-options "--z3rlimit 60 --using_facts_from '* -Kuiper.Sparse.Common.sorted_slice'"
let spmm_idx_succ_bound
  (idx : nat)
  (block_items : pos)
  (row_start row_end : nat)
: Lemma
  (requires row_end - row_start - idx * block_items >= block_items)
  (ensures  idx + 1 <= (row_end - row_start) / block_items)
= ()

(* Name the dependent prefix specification so loop-invariant well-formedness
   does not repeatedly reconstruct its slice and output-length refinements. *)
let tile_mm_prefix_result
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (out0 : lseq real (threadItemsX p))
  (#n : nat)
  (elems : lseq real n)
  (row_ind : lseq sz n)
  (rB : chest2 real p.shared p.cols)
  (n_idx : nat { chunk et /? n_idx })
  (stop : nat)
  (v_out : seq et)
  : prop
  = stop <= n /\
    in_bounds 0 p.shared (cast_pos row_ind) /\
    len v_out == threadItemsX p /\
    Compute.tile_mm_result
      #et #_ #_ #_ #_ #p.shared #p.cols #(threadItemsX p)
      out0 #stop
      (Seq.slice elems 0 stop)
      (Seq.slice (cast_pos row_ind) 0 stop)
      rB n_idx p.blockWidth v_out

inline_for_extraction noextract
fn kf_main
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#lB : layout2 p.shared p.cols) {| ctlB : ctlayout lB, srmB : strided_row_major lB |}
  (gB : array2 et lB)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#relems : erased (lseq real gA.nnz))
  (rA : chest2 real p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  // TODO creo tid : natlt p.blockWidth esta bien
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == tcol p bid tid })
  (nnz idx : ref sz) // realmente no necesito una referencia a idx
  norewrite
  preserves gpu
  preserves smatrix_approx' gA #fA elems col_ind row_off relems rA
  requires  pure (aligned 16 gA.elems)
  requires  pure (aligned 16 gA.col_ind)
  preserves gB |-> Frac (fB /. allthreads p) eB
  requires  pure (aligned 16 (core gB))
  requires  pure (aligned_strided_row_major (chunk et) srmB)
  preserves
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    )
  preserves thread_id p.blockWidth tid
  requires  thread_live_chunks elems_tile p.blockWidth tid
  requires  thread_live_chunks col_ind_tile p.blockWidth tid
  requires  B.barrier_state 0
  requires  out |-> Seq.create (p.blockItemsX / p.blockWidth) d.zero
  requires  live idx
  requires  nnz |-> (re -^ ri')
  ensures exists* (v_idx :sz) (residue : szle (re - ri)) vout.
    idx |-> v_idx **
    nnz |-> (residue <: sz) **
    pure (residue_pred p.blockItemsK ri ri' re v_idx residue) **
    B.barrier_state (v_idx * 2) **
    barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (v_idx * 2) tid **
    out |-> (vout <: seq et) **
    pure (
      Compute.tile_mm_result #_ #_ #_ #hvc #_
        (Seq.create (threadItemsX p) 0.0R <: lseq real (threadItemsX p))
        (Seq.slice relems ri (re - residue))
        (lslice' (cast_pos col_ind) ri (re - residue))
        rB n_idx p.blockWidth vout
    )

{
  // Reuse fraction positivity when checking the preserved array frames.
  assert pure (1.0R /. p.blockWidth >. 0.0R);
  assert pure (fB /. allthreads p >. 0.0R);
  tcol_fits p bid tid;
  row_prefix_bound p (SZ.v ri);
  assert pure (ri' <= ri /\ ri <= re /\ re <= gA.nnz);
  assert pure (ri - ri' < p.blockItemsK);

  let out0 : erased (lseq et (threadItemsX p)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;
  let rout0 : erased (lseq real (threadItemsX p)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  if (!nnz >=^ p.blockItemsK)
  {
    assert pure (ri <= ri' + p.blockItemsK /\ ri' + p.blockItemsK <= re);
    let row_elems_ : erased (lseq et (re - ri)) = hide (Seq.slice elems ri re);
    let row_elems : erased (lseq et (re - ri')) = masked_row ri' ri re row_elems_;
    let row_relems_ : erased (lseq real (re - ri)) = hide (Seq.slice relems ri re);
    let row_relems : erased (lseq real (re - ri')) = masked_row ri' ri re row_relems_;
    seq_mask_approx (ri - ri') row_elems_ row_relems_;
    assert pure (row_elems %~ row_relems);
    assert rewrites_to row_elems (seq_mask (ri - ri') row_elems_);

    // let row_ind : erased (lseq nat (re - ri')) = hide (Seq.slice (cast_pos col_ind) ri' re);
    let row_ind_ : erased (lseq sz (re - ri')) = hide (Seq.slice col_ind ri' re);

    kf_head
      p row_perm
      gA gB #_ #_ #_ #relems
      rA #_ #rB
      out
      elems_tile col_ind_tile
      bid
      ri ri' re
      tid n_idx;

    idx := 1sz;
    nnz := !nnz -^ p.blockItemsK;

    seq_mask_prefix
      (ri - ri') row_elems_
      (ri' + p.blockItemsK - ri);
    seq_mask_prefix
      (ri - ri') row_relems_
      (ri' + p.blockItemsK - ri);
    Seq.slice_slice elems ri re 0 (ri' + p.blockItemsK - ri);
    Seq.slice_slice relems ri re 0 (ri' + p.blockItemsK - ri);

    assert pure (
      Seq.equal
        (Seq.create (ri - ri') zero @+ Seq.slice elems ri (ri' + p.blockItemsK))
        (Seq.slice row_elems 0 (!idx * p.blockItemsK))
    );
    assert pure (
      Seq.equal
        (Seq.create (ri - ri') zero @+ Seq.slice relems ri (ri' + p.blockItemsK))
        (Seq.slice row_relems 0 (!idx * p.blockItemsK))
    );
    assert pure (
      Seq.equal
        (lslice (cast_pos col_ind) ri' p.blockItemsK)
        (Seq.slice (cast_pos row_ind_) 0 (!idx * p.blockItemsK))
    );

    with v_out. assert out |-> v_out;
    assert pure (
      Compute.tile_mm_result
        #et #_ #_ #_ #_
        #p.shared #p.cols
        #(threadItemsX p)
        (rout0 <: lseq real (threadItemsX p))
        #(!idx * p.blockItemsK)
        (Seq.slice row_relems 0 (!idx * p.blockItemsK <: lseq real (!idx * p.blockItemsK)))
        (Seq.slice (cast_pos row_ind_) 0 (!idx * p.blockItemsK))
        rB n_idx p.blockWidth v_out
    );

    #set-options "--z3refresh" {
    while (!nnz >=^ p.blockItemsK)
      invariant
        (exists* v_out.
          out |-> v_out **
          live idx **
          live nnz **
          B.barrier_state (!idx * 2) **
          (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
          (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
          pure (
            !idx > 0 /\ !idx <= (re - ri') / p.blockItemsK /\
            SZ.v !nnz == re - ri' - !idx * p.blockItemsK /\
            tile_mm_prefix_result p rout0 row_relems row_ind_
              rB n_idx (!idx * p.blockItemsK) v_out
          )
        )
      decreases SZ.v !nnz
    {
      sparse_load_main p row_perm gA #_ #_ #_ #_ #relems #rA
        elems_tile col_ind_tile bid ri' re !idx tid;

      with v_idx. assert idx |-> v_idx;
      FStar.Math.Lemmas.lemma_mult_le_right (SZ.v p.blockItemsK) 1 (SZ.v v_idx);
      assert pure (ri - ri' <= v_idx * p.blockItemsK);

      Pulse.Lib.Array.pts_to_len out;
      with v_out. assert out |-> v_out;
      assert pure (len v_out == p.blockItemsX / p.blockWidth);

      Seq.slice_slice col_ind ri' re
        (!idx * p.blockItemsK)
        (!idx * p.blockItemsK + p.blockItemsK);

      assert pure (
        Seq.equal
          (Seq.slice
            row_ind_
            (!idx * p.blockItemsK) (!idx * p.blockItemsK + p.blockItemsK))
          (Seq.slice col_ind
              (ri' + !idx * p.blockItemsK)
              (ri' + !idx * p.blockItemsK + p.blockItemsK))
      );

      assert pure (row_elems == seq_mask (ri - ri') row_elems_);
      seq_mask_source_slice elems ri' ri re row_elems_ row_elems
        (v_idx * p.blockItemsK)
        (v_idx * p.blockItemsK + p.blockItemsK);
      seq_mask_source_slice relems ri' ri re row_relems_ row_relems
        (v_idx * p.blockItemsK)
        (v_idx * p.blockItemsK + p.blockItemsK);
      // TODO
      assert pure (chunk et /? p.cols);
      assert pure (in_bounds 0 p.shared (cast_pos row_ind_));
      assert pure (!idx * p.blockItemsK + p.blockItemsK <= re - ri');
      Seq.lemma_len_slice
        row_elems 0 (!idx * p.blockItemsK + p.blockItemsK);
      Seq.lemma_len_slice
        row_ind_ 0 (!idx * p.blockItemsK + p.blockItemsK);

      let next_end : erased nat = (SZ.v v_idx + 1) * SZ.v p.blockItemsK;

      FStar.Math.Lemmas.distributivity_add_left
        (SZ.v !idx) 1 (SZ.v p.blockItemsK);

      // Reuse this source bound when checking tile_mm's loaded-slice frame.
      assert pure (ri' + !idx * p.blockItemsK + p.blockItemsK <= gA.nnz);

      #set-options "--z3rlimit 60" {
        Compute.tile_mm
          out
          rout0
          elems_tile col_ind_tile #(1.0R /. p.blockWidth)
          #(re - ri')
          row_elems row_ind_ row_relems
          gB #_ #_ #rB
          n_idx p.blockWidth
          (!idx * p.blockItemsK) next_end
          p.blockItemsK;
      };

      assert pure (
        SZ.v (!nnz -^ p.blockItemsK) == re - ri' - next_end
      );

      spmm_idx_succ_fits (SZ.v !idx) (SZ.v p.blockItemsK)
        (SZ.v ri') (SZ.v re);
      spmm_idx_succ_bound (SZ.v !idx) (SZ.v p.blockItemsK)
        (SZ.v ri') (SZ.v re);
      idx := !idx +^ 1sz;
      nnz := !nnz -^ p.blockItemsK;

      assert pure (!idx <= (re - ri') / p.blockItemsK);
      #set-options "--z3rlimit 120" { () };
    };
    };


    with v_out. assert out |-> v_out;
    with v_idx. assert idx |-> v_idx;
    with v_nnz. assert nnz |-> v_nnz;
    FStar.Math.Lemmas.lemma_mult_le_right
      (SZ.v p.blockItemsK) 1 (SZ.v v_idx);
    assert pure (
      ri <= ri' + v_idx * p.blockItemsK /\
      ri' + v_idx * p.blockItemsK <= re
    );

    assert pure (
      (ri - ri') + (ri' + v_idx * p.blockItemsK - ri)
        == v_idx * p.blockItemsK
    );
    seq_mask_lslice_prefix
      (ri - ri') row_elems_ row_elems
      (ri' + v_idx * p.blockItemsK - ri)
      (v_idx * p.blockItemsK) ();
    seq_mask_lslice_prefix
      (ri - ri') row_relems_ row_relems
      (ri' + v_idx * p.blockItemsK - ri)
      (v_idx * p.blockItemsK) ();

    Compute.tile_mm_mask_lemma
      #et #_ #_ #_ #_
      #p.shared #p.cols
      #(threadItemsX p) rout0
      (v_idx * p.blockItemsK)
      (ri - ri')
      (lslice row_relems_ 0 (ri' + v_idx * p.blockItemsK - ri))
      (Seq.slice (cast_pos row_ind_) 0 (v_idx * p.blockItemsK))
      rB n_idx p.blockWidth
      v_out;

    assert pure (
      Seq.equal
        (lslice row_elems_ 0 (ri' + v_idx * p.blockItemsK - ri))
        (Seq.slice elems ri (re - v_nnz))
    );
    assert pure (
      Seq.equal
        (lslice row_relems_ 0 (ri' + v_idx * p.blockItemsK - ri))
        (Seq.slice relems ri (re - v_nnz))
    );
    cast_pos_slice col_ind ri' re;
    Seq.slice_slice
      (cast_pos row_ind_)
      0 (v_idx * p.blockItemsK)
      (ri - ri') (v_idx * p.blockItemsK);
    Seq.slice_slice
      (cast_pos col_ind)
      ri' re
      (ri - ri') (v_idx * p.blockItemsK);
    assert pure (
      Seq.equal
        (Seq.slice
          (lslice (cast_pos row_ind_) 0 (v_idx * p.blockItemsK))
          (ri - ri') (v_idx * p.blockItemsK)
        )
        (lslice' (cast_pos col_ind) ri (re - v_nnz))
    );

    barrier_in_fold_residue_pre p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid ri' re tid;

    // Loop exit: re - ri' == idx * blockItemsK + nnz with nnz < blockItemsK, so
    // idx is exactly the quotient. Spelled out as Euclidean division because
    // the solver no longer gets there unaided.
    FStar.Math.Lemmas.small_division_lemma_1 (SZ.v v_nnz) (SZ.v p.blockItemsK);
    FStar.Math.Lemmas.lemma_div_plus (SZ.v v_nnz) (SZ.v v_idx) (SZ.v p.blockItemsK);
    assert pure (SZ.v v_idx == (re - ri') / p.blockItemsK);
    lem_div2 (SZ.v v_idx) (SZ.v p.blockItemsK) (SZ.v v_nnz);
    assert pure (SZ.v v_nnz == (re - ri') % p.blockItemsK);
    assert pure (residue_pred p.blockItemsK ri ri' re v_idx v_nnz);
    rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid ((re - ri') / p.blockItemsK * 2) tid
    as barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid (v_idx * 2) tid;

    #set-options "--z3refresh" { () };
    ();
  }
  else {
    idx := 0sz;
    nnz := re -^ ri;

    Compute.tile_mm_lemma0
      rout0
      rB n_idx p.blockWidth out0;

    assert pure (
      Seq.equal
        Seq.empty
        (Seq.slice elems ri (re - !nnz))
    );
    assert pure (
      Seq.equal
        Seq.empty
        (Seq.slice relems ri (re - !nnz))
    );
    assert pure (
      Seq.equal
        Seq.empty
        (lslice' (cast_pos col_ind) ri (re - !nnz))
    );

    barrier_in_fold_residue0_pre p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid ri' re tid;

    rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid 0 tid
    as barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid (!idx * 2) tid;
    ();
  };
}

#pop-options
