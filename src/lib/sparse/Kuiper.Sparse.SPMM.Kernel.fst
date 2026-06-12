module Kuiper.Sparse.SPMM.Kernel

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module Compute = Kuiper.Sparse.SPMM.Compute
module Array2 = Kuiper.Array2
open Kuiper.Sparse
open Kuiper.Sparse.Load
open Kuiper.EMatrix
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Sparse.SPMM.Barrier
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Array.Vectorized

unfold
let block_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et {size_req p})
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. allthreads p)
    elems col_ind row_off eA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> matrix_live_cell
        gC (brow p bid |~> row_perm) (bcol p bid + k * p.blockWidth + tid))


unfold
let block_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. allthreads p)
    elems col_ind row_off eA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> Array2.pts_to_cell gC
        (brow p bid |~> row_perm,
         bcol p bid + k * p.blockWidth + tid)
        (MS.matmul_single eA eB
          (brow p bid |~> row_perm)
          (bcol p bid + k * p.blockWidth + tid)))

unfold
let kpre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  // TODO no se si puedo hacer eso
  (sh : c_shmems (shmems_desc p))
  (#_ : squash (aligned 16 (fst sh) /\ aligned 16 (fst (snd sh))))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  block_pre
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB
    bid tid **
  thread_live_chunks (fst sh) p.blockWidth tid **
  thread_live_chunks (fst (snd sh)) p.blockWidth tid

unfold
let kpost
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  block_post
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB
    bid tid **
  live (fst sh) #(1.0R /. p.blockWidth) **
  live (fst (snd sh)) #(1.0R /. p.blockWidth)

noextract
let barrier_count
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : nat)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (bid : natlt (nblocks p))
: Ghost nat
  (requires
    valid_smatrix p.rows p.shared (cast_pos col_ind) (cast_pos row_off)
  )
  (ensures fun _ -> true)
=
  let ri = row_off @! (brow p bid |~> row_perm) in
  let re = row_off @! (brow p bid |~> row_perm) + 1 in
  ((re - ri) / p.blockItemsK + 1) * 2

noextract inline_for_extraction
let align_sz (size : szp) (x : sz) : sz =
  (x /^ size) *^ size

noextract inline_for_extraction
let align_offset
  (et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (off : sz)
: Pure sz
  (requires true)
  (ensures fun r -> SZ.v r == round2 (max (chunk et) (chunk sz)) off)
  (* chunk sz y chunk et son potencias de dos, así que alinear
     a la mayor es alinear a ambas. *)
= if chunk sz <^ chunk et
    then align_sz (chunk et) off
    else align_sz (chunk sz) off


inline_for_extraction noextract
fn process_first
  (#et : Type0) {| d : scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lb : Array2.layout p.shared p.cols)
  {| ctlayout lb |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (gB : Array2.t et lb)
  // matriz sparse ga
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matriz densa gb
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri_ : sz{ri_ == row_off @! (brow p bid |~> row_perm)})
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) ri_})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == bcol p bid })
  (#_ : squash (ri + p.blockItemsK <= re))
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac (fB /. allthreads p) eB **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid **
    out |-> Seq.create (p.blockItemsX / p.blockWidth) d.zero **
    B.barrier_state 0
  ensures
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
    B.barrier_state 2 **
    out |->
      Compute.compute_result
        p.blockWidth p.blockItemsX #(ri + p.blockItemsK - ri_)
        (Seq.slice elems ri_ (ri + p.blockItemsK))
        (Seq.slice (cast_pos col_ind) ri_ (ri + p.blockItemsK))
        eB
        (Seq.create (p.blockItemsX / p.blockWidth) zero)
        tid (bcol p bid)
{
  admit()
}

inline_for_extraction noextract
fn sparse_load_main
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  // matriz sparse gA
  (#row_off : lseq sz (p.rows + 1))
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#eA : ematrix et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : sz { idx > 0 })
  (tid : szlt p.blockWidth)
  (#_ : squash(ri + idx * p.blockItemsK + p.blockItemsK <= re))
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures
    B.barrier_state ((idx + 1) * 2) **
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice elems
        (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind
        (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK))
{
  admit();
}

// noextract
// let step_result
//   (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
//   (p : parameters et { size_req p })
//   (#nnz : nat)
//   (elems : lseq et nnz)
//   // (col_ind : lseq nat nnz { in_bounds 0 p.shared col_ind })
//   (col_ind : lseq nat nnz)
//   (eB : ematrix et p.shared p.cols)
//   // (i j k : natle nnz { i <= j /\ j <= i + p.blockItemsK /\ i + p.blockItemsK <= k })
//   (i j k : nat)
//   (n_idx : natlt p.cols)
//   (tid : natlt p.blockWidth)
//   // (idx : nat { 0 < idx /\ idx <= (k - i) / p.blockItemsK })
//   (idx : nat)
// : lseq et (p.blockItemsX / p.blockWidth)
// =
//   admit();
//   Compute.compute_result
//     p.blockWidth p.blockItemsX #(i - j + idx * p.blockItemsK)
//     (Seq.slice elems j (i + idx * p.blockItemsK))
//     (Seq.slice col_ind j (i + idx * p.blockItemsK))
//     eB (Seq.create (p.blockItemsX / p.blockWidth) zero)
//     tid n_idx

// #push-options "--debug SMTFail --split_queries always"
// inline_for_extraction noextract
// fn kf_main_step
//   (#et : Type0) {| d : scalar et, sized et, has_vec_cpy et |}
//   (p : parameters et { size_req p })
//   (row_perm : permutation (natlt p.rows))
//   (#lb : Array2.layout p.shared p.cols)
//   {| ctlayout lb |}
//   (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
//   // (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
//   (gB : Array2.t et lb)
//   // matriz sparse ga
//   (#elems : lseq et gA.nnz)
//   (#col_ind : lseq sz gA.nnz)
//   (#row_off : lseq sz (p.rows + 1))
//   (#eA : ematrix et p.rows p.shared)
//   // matriz densa gb
//   (#eB : ematrix et p.shared p.cols)
//   (#fA #fB : perm)
//   (#_ : squash (well_formed p col_ind row_off))
//   // salida
//   (out : larray et (p.blockItemsX /^ p.blockWidth))
//   // shmem
//   // (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
//   // (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
//   (elems_tile : gpu_array et p.blockItemsK)
//   (col_ind_tile : gpu_array sz p.blockItemsK)
//   (bid : szlt (nblocks p))
//   (ri_ : sz{ri_ == row_off @! (brow p bid |~> row_perm)})
//   (ri : sz{ri <= ri_ /\ ri_ <= ri + p.blockItemsK})
//   (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
//   (tid : szlt p.blockWidth)
//   (n_idx : sz {SZ.v n_idx == bcol p bid })
//   (idx : sz { 0 < idx /\ (idx + 1) <= (re - ri) / p.blockItemsK })
//   norewrite
//   preserves
//     gpu **
//     smatrix_pts_to' gA #fA elems col_ind row_off eA **
//     gB |-> Frac (fB /. allthreads p) eB **
//     B.barrier_tok (
//       barrier_contract p row_perm elems col_ind row_off
//         elems_tile col_ind_tile bid
//     ) **
//     // thread_id p.blockWidth tid **
//     (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
//     (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
//   requires
//     // out |-> step_result p elems (cast_pos col_ind) eB ri ri_ re n_idx tid idx **
//     B.barrier_state (idx * 2)
//   ensures
//     // out |-> step_result p elems (cast_pos col_ind) eB ri ri_ re n_idx tid (idx + 1) **
//     B.barrier_state ((idx + 1) * 2)



#push-options "--z3rlimit 20"
#push-options "--debug SMTFail --split_queries always"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lb : Array2.layout p.shared p.cols)
  (#lc : Array2.layout p.rows p.cols)
  {| ctlayout lb, ctlayout lc |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lb)
  (gC : Array2.t et lc)
  // matriz sparse ga
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matriz densa gb
  (#eB : ematrix et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (#_ : squash (aligned 16 (fst sh) /\ aligned 16 (fst (snd sh))))
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  ()
  norewrite
  requires
    gpu **
    kpre
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      eA eB
      fA fri fB
      sh
      bid tid **
    thread_id p.blockWidth tid **
    block_id (nblocks p) bid **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        (fst sh) (fst (snd sh)) bid
    ) **
    B.barrier_state 0
  ensures
    gpu **
    kpost
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      eA eB
      fA fri fB
      sh
      bid tid **
    thread_id p.blockWidth tid **
    block_id (nblocks p) bid **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        (fst sh) (fst (snd sh)) bid
    ) **
    B.barrier_state (barrier_count p row_perm col_ind row_off bid)
{
  let m_idx = gpu_array_read row_indices (brow_ p bid);
  assert rewrites_to m_idx (SZ.uint_to_t (brow p bid |~> row_perm));
  let n_idx = bcol_ p bid;

  let (elems_tile0, (col_ind_tile0, _)) = sh;

  (* This incantation here improves the generated code by actually defining
  these variables at this point. *)
  let elems_tile   = elems_tile0;     assert rewrites_to elems_tile   elems_tile0;
  let col_ind_tile = col_ind_tile0;   assert rewrites_to col_ind_tile col_ind_tile0;

  assert rewrites_to elems_tile (fst sh);
  assert rewrites_to col_ind_tile (fst (snd sh));

  let ri_ = gpu_array_read gA.row_off m_idx;
  let re = gpu_array_read gA.row_off (m_idx +^ 1sz);

  let ri = align_offset et ri_;

  let mut nnz : sz = re -^ ri;
  let mut idx = 0sz;

  let mut out = [| zero #et #_; blockChunks |];
  let out0 : lseq et (p.blockItemsX / p.blockWidth) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  let row_elems : lseq et (re - ri_) = hide (Seq.slice elems ri_ re);
  let row_pos : lseq nat (re - ri_) = hide (Seq.slice (cast_pos col_ind) ri_ re);

  if (!nnz >=^ p.blockItemsK)
  {
    process_first
      p row_perm blockChunks
      gA gB #_ #_ #_ #eA
      out
      elems_tile col_ind_tile
      bid
      ri_ ri re
      tid n_idx;

    idx := 1sz;
    nnz := !nnz -^ p.blockItemsK;

    // let ri_off : erased nat = ri_ - ri;

    assert pure (
      Seq.equal
        (Seq.slice elems ri_ (ri + p.blockItemsK))
        // (Seq.slice row_elems 0 (ri_off + !idx * p.blockItemsK))
        (Seq.slice row_elems 0 (ri - ri_ + !idx * p.blockItemsK))
    );
    assert pure (
      Seq.equal
        (Seq.slice (cast_pos col_ind) ri_ (ri + p.blockItemsK))
        // (Seq.slice row_pos 0 (ri_off + !idx * p.blockItemsK))
        (Seq.slice row_pos 0 (ri - ri_ + !idx * p.blockItemsK))
    );

    assert out |->
      Compute.compute_result
        p.blockWidth p.blockItemsX #(ri - ri_ + !idx * p.blockItemsK)
        (Seq.slice row_elems 0 (ri - ri_ + !idx * p.blockItemsK))
        (Seq.slice row_pos 0 (ri - ri_ + !idx * p.blockItemsK))
        eB out0 tid n_idx;

    assert pure (ri + p.blockItemsK > ri_);
    // assert pure (!idx <= (re - ri) / p.blockItemsK);
    while (!nnz >=^ p.blockItemsK)
      invariant
        (exists* v_out.
          out |-> v_out **
          live idx **
        // out |->
        //   Compute.compute_result
        //     p.blockWidth p.blockItemsX #(ri - ri_ + !idx * p.blockItemsK)
        //     (Seq.slice row_elems 0 (ri - ri_ + !idx * p.blockItemsK))
        //     (Seq.slice row_pos 0 (ri - ri_ + !idx * p.blockItemsK))
        //     eB out0 tid n_idx **
          live nnz **
          B.barrier_state (!idx * 2) **
          (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
          (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
          pure (
            !idx > 0 /\ !idx <= (re - ri) / p.blockItemsK /\
            SZ.v !nnz == re - ri - !idx * p.blockItemsK /\
            v_out ==
            Compute.compute_result
              p.blockWidth p.blockItemsX #(ri - ri_ + !idx * p.blockItemsK)
              (Seq.slice row_elems 0 (ri - ri_ + !idx * p.blockItemsK))
              (Seq.slice row_pos 0 (ri - ri_ + !idx * p.blockItemsK))
              eB out0 tid n_idx
          )
        )
    {
      sparse_load_main p row_perm gA #_ #_ #_ #_ #eA
        elems_tile col_ind_tile bid ri re !idx tid;

      assert pure (
        Seq.equal
          (Seq.slice elems
            (ri + !idx * p.blockItemsK)
            (ri + (!idx + 1) * p.blockItemsK))
          (Seq.slice row_elems
            (ri - ri_ + !idx * p.blockItemsK)
            (ri - ri_ + (!idx + 1) * p.blockItemsK))
      );
      assert pure (
        Seq.equal
          (cast_pos #p.blockItemsK (
            Seq.slice col_ind
              (ri + !idx * p.blockItemsK)
              (ri + (!idx + 1) * p.blockItemsK)
          ))
          (Seq.slice row_pos
            (ri - ri_ + !idx * p.blockItemsK)
            (ri - ri_ + (!idx + 1) * p.blockItemsK))
      );
      Compute.compute
        p.blockWidth p.blockItemsK p.blockItemsX
        elems_tile col_ind_tile p.blockItemsK gB out tid n_idx;

      Compute.compute_step
        p.blockWidth p.blockItemsX
        row_elems row_pos eB out0 tid n_idx
        (ri - ri_ + !idx * p.blockItemsK)
        (ri - ri_ + (!idx + 1) * p.blockItemsK);

      idx := !idx +^ 1sz;
      nnz := !nnz -^ p.blockItemsK;

      // admit();
    };

    admit();
  };
  admit();
}
