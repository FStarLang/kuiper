module Kuiper.Sparse.SPMM

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

open Kuiper.Sparse.SPMM.Main

#push-options "--z3rlimit 60 --using_facts_from '* -Kuiper.Sparse.Common.sorted_slice'"
inline_for_extraction noextract
fn kf_residue
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols) {| ctlayout lB, srm : strided_row_major lB |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
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
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == tcol p bid tid })
  (idx : sz)
  (residue : sz {
    residue <= re - ri /\ residue_pred p.blockItemsK ri ri' re idx residue
  })
  norewrite
  preserves gpu
  preserves smatrix_approx' gA #fA elems col_ind row_off relems rA
  requires  pure (aligned 16 gA.elems)
  requires  pure (aligned 16 gA.col_ind)
  preserves gB |-> Frac (fB /. allthreads p) eB
  requires  pure (aligned 16 (core gB))
  requires  pure (aligned_strided_row_major (chunk et) srm)
  preserves
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    )
  preserves thread_id p.blockWidth tid
  requires B.barrier_state (idx * 2)
  requires
    barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (idx * 2) tid
  // Keep the erased output's lseq type explicit at the tile_mm boundary.
  requires exists* (vout : lseq et (threadItemsX p)).
    out |-> (vout <: seq et) **
    pure (
      Compute.tile_mm_result #_ #_ #_ #hvc #_
        #p.shared #p.cols
        #(threadItemsX p)
        (Seq.create (p.blockItemsX / p.blockWidth) 0.0R)
        #(re - residue - ri)
        (Seq.slice relems ri (re - residue))
        (lslice' (cast_pos col_ind) ri (re - residue))
        rB n_idx p.blockWidth vout
    )
  ensures B.barrier_state ((idx + 1) * 2)
  ensures exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures exists* (vout : lseq et (threadItemsX p)).
    out |-> (vout <: seq et) **
    pure (
      Compute.tile_mm_result #_ #_ #_ #hvc #_
        #p.shared #p.cols
        #(threadItemsX p)
        (Seq.create (p.blockItemsX / p.blockWidth) 0.0R)
        #(re - ri)
        (Seq.slice relems ri re)
        (lslice' (cast_pos col_ind) ri re)
        rB n_idx p.blockWidth vout
    )
{
  // Reuse fraction positivity when checking the preserved array frames.
  assert pure (1.0R /. p.blockWidth >. 0.0R);
  assert pure (fB /. allthreads p >. 0.0R);
  assert pure (ri' <= ri /\ ri <= re /\ re <= gA.nnz);
  FStar.Math.Lemmas.lemma_mod_lt (SZ.v re - SZ.v ri') (SZ.v p.blockItemsK);
  assert pure (residue < p.blockItemsK);
  let out0 : erased (lseq et (threadItemsX p)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;
  let rout0 : erased (lseq real (threadItemsX p)) =
    Seq.create (p.blockItemsX / p.blockWidth) 0.0R;
  let row_elems : erased (lseq et (re - ri)) = hide (Seq.slice elems ri re);
  let row_relems : erased (lseq real (re - ri)) = hide (Seq.slice relems ri re);
  assert pure (row_elems %~ row_relems);
  let row_ind : erased (lseq sz (re - ri)) = hide (Seq.slice col_ind ri re);

  assert pure (
    Seq.equal
      (Seq.slice elems ri (re - residue))
      (lslice' row_elems 0 ((re - ri) - residue))
  );
  assert pure (
    Seq.equal
      (Seq.slice relems ri (re - residue))
      (lslice' row_relems 0 ((re - ri) - residue))
  );
  assert pure (
    Seq.equal
      (lslice' (cast_pos col_ind) ri (re - residue))
      (lslice' (cast_pos row_ind) 0 ((re - ri) - residue))
  );

  Seq.lemma_eq_elim
    (Seq.slice relems ri (re - residue))
    (lslice' row_relems 0 (re - ri - residue));
  Seq.lemma_eq_elim
    (lslice' (cast_pos col_ind) ri (re - residue))
    (lslice' (cast_pos row_ind) 0 (re - ri - residue));

  with vout. assert out |-> vout;
  assert pure (
    Compute.tile_mm_result
      rout0
      #(re - ri - residue)
      (Seq.slice row_relems 0 (re - ri - residue))
      (Seq.slice (cast_pos row_ind) 0 (re - ri - residue))
      rB n_idx p.blockWidth vout
  );

  sparse_load_residue p row_perm gA #_ #row_off #elems #col_ind #relems #rA #fA
    elems_tile col_ind_tile bid ri ri' re tid idx residue;

  Seq.slice_slice elems   ri re ((re - ri) - residue) (re - ri);
    Seq.slice_slice relems   ri re ((re - ri) - residue) (re - ri);
  Seq.slice_slice col_ind ri re ((re - ri) - residue) (re - ri);

  assert pure (
    Seq.equal
      (Seq.slice elems (re - residue) re)
      (Seq.slice row_elems ((re - ri) - residue) (re - ri))
  );
  assert pure (
    Seq.equal
      (Seq.slice relems (re - residue) re)
      (Seq.slice row_relems ((re - ri) - residue) (re - ri))
  );
  assert pure (
    Seq.equal
      (Seq.slice col_ind (re - residue) re)
      (Seq.slice row_ind ((re - ri) - residue) (re - ri))
  );

  assert pure (threadItemsX p == p.blockItemsX / p.blockWidth);
  tcol_fits p bid tid;

  Compute.tile_mm
    out rout0
    #vout #p.blockItemsK elems_tile col_ind_tile
    #(1.0R /. p.blockWidth) #(re - ri)
    row_elems row_ind row_relems
    gB #_ #_ #rB n_idx p.blockWidth
    ((re - ri) - residue) (re - ri) residue;

  assert pure (
    Seq.equal
      (Seq.slice row_elems 0 (re - ri))
      (Seq.slice elems ri re)
  );
  assert pure (
    Seq.equal
      (Seq.slice row_relems 0 (re - ri))
      (Seq.slice relems ri re)
  );
  assert pure (
    Seq.equal
      (Seq.slice (cast_pos row_ind) 0 (re - ri))
      (lslice' (cast_pos col_ind) ri re)
  );

  unfold slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK;
  slice_concat elems_tile #(1.0R /. p.blockWidth)
    0 residue p.blockItemsK;

  unfold slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK;
  slice_concat col_ind_tile #(1.0R /. p.blockWidth)
    0 residue p.blockItemsK;

  add_full_slice elems_tile 0 p.blockItemsK p.blockItemsK;
  slice_to_array elems_tile;
  add_full_slice col_ind_tile 0 p.blockItemsK p.blockItemsK;
  slice_to_array col_ind_tile;
  ();
}
#pop-options

#push-options "--z3rlimit 15"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v p.blockItemsX / SZ.v p.blockWidth
  }) // Ver nota abajo
  (#lB : layout2 p.shared p.cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 p.rows p.cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  // TODO esto tiene que estar acá? podria estar en block_pre?
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  // TODO esto tiene que estar acá? podria estar en block_pre?
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (gC : array2 et lC)
  // TODO esto tiene que estar acá? podria estar en block_pre?
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#relems : erased (lseq real gA.nnz))
  (#rA : chest2 real p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (sh : c_shmems (shmems_desc p))
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  ()
  norewrite
  preserves
    gpu
  requires
    pure (c_shmems_inv sh) **
    kpre
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      #relems rA eB rB
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
    kpost
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      #relems rA eB rB
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
  assert pure (1.0R /. p.blockWidth >. 0.0R);
  assert pure (fA /. allthreads p >. 0.0R /\
               fri /. allthreads p >. 0.0R /\
               fB /. allthreads p >. 0.0R);
  let m_idx = Pulse.Lib.Array.(row_indices.(brow_ p bid));
  assert rewrites_to m_idx (SZ.uint_to_t (brow p bid |~> row_perm));
  // let n_idx = bcol_ p bid;
  let n_idx = tcol_ p bid tid;
  assert rewrites_to n_idx (SZ.uint_to_t (tcol p bid tid));
  tcol_fits p bid tid;

  let (elems_tile0, (col_ind_tile0, _)) = sh;

  // pts_to_len elems_tile0;
  // pts_to_len col_ind_tile0;

  (* This incantation here improves the generated code by actually defining
  these variables at this point. *)
  let elems_tile   = elems_tile0;     assert rewrites_to elems_tile   elems_tile0;
  let col_ind_tile = col_ind_tile0;   assert rewrites_to col_ind_tile col_ind_tile0;

  assert rewrites_to elems_tile (fst sh);
  assert rewrites_to col_ind_tile (fst (snd sh));

  let ri = Pulse.Lib.Array.(gA.row_off.(m_idx));
  let re = Pulse.Lib.Array.(gA.row_off.(m_idx +^ 1sz));

  let ri' = align_offset et ri;

  let mut nnz : sz = re -^ ri';
  let mut idx = 0sz;

  (* GM: Nota: no podemos tener una expresión de división como la
  longitud del array, porque sería un VLA. Por eso agregué un argumento
  al kernel (blockChunks) que tiene un refinamiento que asegura que
  es igual (p.blockItemsK / p.blockWidth). *)
  let mut out = [| zero #et #_; blockChunks |];
  assume pure (aligned 16 out);

  let out0 : erased (lseq et (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;
  let rout0 : erased (lseq real (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  //------------------main-----------------------------------------

  kf_main
    p row_perm
    gA gB #_ #_ #_ #relems
    rA #_ #rB
    out
    elems_tile col_ind_tile
    bid
    ri ri' re
    tid n_idx
    nnz idx;

  //------------------residue-----------------------------------------

  residue_index (SZ.v p.blockItemsK) (SZ.v ri) (SZ.v ri') (SZ.v re)
    (SZ.v !idx) (SZ.v !nnz);
  assert pure ((SZ.v !idx + 1) * 2 ==
    barrier_count p row_perm col_ind row_off bid);

  kf_residue
    p row_perm
    gA gB #_ #_ #_ #relems rA #_ #rB
    out
    elems_tile col_ind_tile
    bid ri ri' re tid n_idx !idx !nnz;

  //------------------dense store-----------------------------------------

  with v_out. assert out |-> v_out;
  Pulse.Lib.Array.pts_to_len out;

  unsparse_row_lemma
    p.rows p.shared
    relems (cast_pos col_ind) (cast_pos row_off) m_idx;

  assert pure (
    Seq.equal
      (cast_pos #(re - ri) (Seq.slice col_ind ri re))
      (Seq.slice (cast_pos col_ind) ri re)
  );

  Compute.tile_mm_result_lemma
    rA m_idx
    #(re - ri)
    (Seq.slice relems ri re) (Seq.slice (cast_pos col_ind) ri re)
    rB n_idx p.blockWidth #(threadItemsX p) v_out;

  assert pure ((brow p (SizeT.v bid) |~> row_perm) == SZ.v m_idx);

  // no esta funcionando bien el rewrites_to m_idx acá; lo hacemos a mano
  rewrite each (brow p bid |~> row_perm)
  as (SizeT.v (Seq.Base.index (ordering row_perm) (SizeT.v (brow_ p bid))));

  // TODO renombrar y usar modulo calificado
  matrix_store_tile_vec_approx
    gC m_idx n_idx (Kuiper.Spec.GEMM.matmul rA rB)
    (p.blockItemsX /^ p.blockWidth) out p.blockWidth;

  rewrite each (SizeT.v (Seq.Base.index (ordering row_perm) (SizeT.v (brow_ p bid))))
  as (brow p bid |~> row_perm);


  //------------------exit-----------------------------------------

  slice_to_array row_indices;

  Pulse.Lib.Array.PtsTo.pts_to_len elems_tile;
  assert is_full_slice elems_tile p.blockItemsK;
  slice_to_array elems_tile;

  Pulse.Lib.Array.PtsTo.pts_to_len col_ind_tile;
  assert is_full_slice col_ind_tile p.blockItemsK;
  slice_to_array col_ind_tile;

  ()
}
#pop-options


inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v p.blockItemsX / SZ.v p.blockWidth
  }) // Ver nota abajo
  (#lB : layout2 p.shared p.cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 p.rows p.cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB {is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (gC : array2 et lC {is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (elems : erased (lseq et gA.nnz))
  (col_ind : erased (lseq sz gA.nnz))
  (row_off : erased (lseq sz (p.rows + 1)))
  (#relems : erased (lseq real gA.nnz))
  (rA : chest2 real p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  : kernel_desc
    (
      smatrix_approx' gA #fA elems col_ind row_off relems rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      live gC
    )
    (
      smatrix_approx' gA #fA elems col_ind row_off relems rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.matmul rA rB)
    )
= {
  nblk = nblocks p;
  nthr = p.blockWidth;

  barrier_contract = (fun bid ptrs ->
    barrier_contract p row_perm elems col_ind row_off
      (fst ptrs) (fst (snd ptrs)) bid);
  barrier_count = (fun bid -> barrier_count p row_perm col_ind row_off bid);
  barrier_ok = (fun bid ptrs ->
    barrier_p_to_q_transform p
      row_perm
      elems col_ind row_off
      (fst ptrs) (fst (snd ptrs)) bid);

  shmems_desc = shmems_desc p;

  frame = emp;

  block_pre  = (fun bid -> forall+ (tid : natlt p.blockWidth).
    block_pre
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      #relems rA eB rB
      fA fri fB
      bid tid
  );
  block_post = (fun bid -> forall+ (tid : natlt p.blockWidth).
    block_post
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      #relems rA eB rB
      fA fri fB
      bid tid
  );
  setup = setup
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems #_ #_ #rB #fA;
  teardown = teardown
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems #_ #_ #rB #fA;

  block_frame    = (fun _ar _bid -> emp);
  block_setup = block_setup
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems #rA #eB #rB
    #fA #fri #fB;
  block_teardown = block_teardown
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems #rA #eB #rB
    #fA #fri #fB;

  kpre  = kpre
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems rA eB rB
    fA fri fB;
  kpost = kpost
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems rA eB rB
    fA fri fB;

  f =
    kf
      p row_perm blockChunks
      gA
      row_indices gB gC;

  block_pre_sendable=magic();
  block_post_sendable=magic();
  kpre_sendable=magic();
  kpost_sendable=magic();
}

(* Keep model unpacking behind a named predicate: Pulse's assertion elaborator
   cannot purify a located model while its existential witnesses are unsolved. *)
let located_smatrix_model
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #shared : nat)
  (gA : smatrix et rows shared) (#fA : perm)
  (elems : lseq et gA.nnz) (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1)) (relems : lseq real gA.nnz)
  (eA : chest2 et rows shared)
  (rA : chest2 real rows shared)
  : slprop
  = on gpu_loc (smatrix_model gA #fA elems col_ind row_off relems eA rA)

inline_for_extraction noextract
fn spmm_on
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (rows shared cols : szp { chunk et /? cols })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v blockItemsX / SZ.v blockWidth
  }) // Ver nota abajo
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (#fB : perm)
  (gC : array2 et lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols)
  (#eC : chest2 et rows cols)
  (s : stream_t)
  (#e : epoch_t)
  norewrite
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    on gpu_loc (smatrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (live gC) **
    pure (eB %~ rB) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_done s (epoch_next e)) (on gpu_loc (
      smatrix_approx gA #fA eA rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.matmul rA rB)
    ))
{
  Math.prod_cancel_divides blockItemsX (chunk et) blockWidth;
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v blockItemsK) #(SZ.v blockChunks) (chunk et);

  map_loc gpu_loc
    #(smatrix_approx gA #fA eA rA)
    #(exists* elems col_ind row_off relems. smatrix_model gA #fA elems col_ind row_off relems eA rA)
    fn _ {
      unfold smatrix_approx gA #fA eA rA;
    };
  with elems col_ind row_off relems.
    fold located_smatrix_model gA #fA elems col_ind row_off relems eA rA;
  unfold located_smatrix_model gA #fA elems col_ind row_off relems eA rA;
  map_loc gpu_loc
    #(smatrix_model gA #fA elems col_ind row_off relems eA rA)
    #(smatrix_approx' gA #fA elems col_ind row_off relems rA **
      pure (valid_smatrix rows shared (cast_pos col_ind) (cast_pos row_off) /\
        eA == matrix_unsparse rows shared elems (cast_pos col_ind) (cast_pos row_off)))
    fn _ { unfold smatrix_model gA #fA elems col_ind row_off relems eA rA; };

  launch (
    kdesc #et #_
      ({ rows; shared; cols; blockItemsK; blockItemsX; blockWidth; sz_req = () })
      row_perm blockChunks #lB #_ #_ #lC
      gA row_indices gB gC elems col_ind row_off #relems rA
      #eB #rB #fA #fri #fB
  ) s;

  rewrite_pledge #emp_inames #(epoch_done s (epoch_next e))
    (on gpu_loc (
      smatrix_approx' gA #fA elems col_ind row_off relems rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.matmul rA rB)
    ))
    (on gpu_loc (
      smatrix_approx gA #fA eA rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.matmul rA rB)
    ))
    #emp_inames
    fn _ {
      map_loc gpu_loc
        #(smatrix_approx' gA #fA elems col_ind row_off relems rA)
        #(smatrix_approx gA #fA eA rA)
        fn _ {
          fold smatrix_model gA #fA elems col_ind row_off relems eA rA;
          fold smatrix_approx gA #fA eA rA;
        };
    };
}

(* Same structure as Kuiper.Kernel.launch_kernel_full_sync, one level up. *)
inline_for_extraction noextract
fn spmm
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (rows shared cols : szp { chunk et /? cols })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v blockItemsX / SZ.v blockWidth
  }) // Ver nota abajo
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (#fB : perm)
  (gC : array2 et lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols)
  (#eC : chest2 et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (smatrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    on gpu_loc (live gC) **
    pure (eB %~ rB) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures on gpu_loc (matrix_approx gC (MS.matmul rA rB))
{
  let s = fresh_stream ();
  init_epoch s ();
  spmm_on
    rows shared cols
    blockItemsK blockItemsX blockWidth
    blockChunks
    gA row_indices fri gB gC
    #_ #_ #eA #rA row_perm #eB #rB #eC
    s;
  sync_stream s;
  redeem_pledge emp_inames (epoch_done s _) (on gpu_loc (
      smatrix_approx gA #fA eA rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.matmul rA rB)
  ));
  drop_ (epoch_done s _);
  drop_ (epoch_live s _);
  destroy_stream s;
}
