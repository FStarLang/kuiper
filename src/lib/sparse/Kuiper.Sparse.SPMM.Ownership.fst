module Kuiper.Sparse.SPMM.Ownership

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

let prod_divides_pat (a b c : pos)
: Lemma (requires (a * b) /? c) (ensures a /? c /\ b /? c) [SMTPat ((a * b) /? c)]
= Math.prod_divides a b c

unfold
let block_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (rA : chest2 real p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (rB : chest2 real p.shared p.cols { eB %~ rB })
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_approx' gA #(fA /. allthreads p)
    elems col_ind row_off relems rA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  thread_live_tile_vec
    gC
    (brow p bid |~> row_perm)
    (tcol p bid tid)
    (p.blockItemsX / p.blockWidth)
    p.blockWidth

unfold
let block_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (rA : chest2 real p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (rB : chest2 real p.shared p.cols { eB %~ rB })
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_approx' gA #(fA /. allthreads p)
    elems col_ind row_off relems rA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  thread_tile_approx
    gC
    (brow p bid |~> row_perm)
    (tcol p bid tid)
    (MS.matmul rA rB)
    (p.blockItemsX / p.blockWidth)
    p.blockWidth

(* Projecting [rin]/[rout] out of the record literal built by [barrier_contract]
   is no longer reduced for the SMT solver, so we discharge those slprop
   equalities by normalization instead. *)
let unfold_barrier_contract () : FStar.Tactics.V2.Tac unit =
  FStar.Tactics.V2.norm [delta_only [`%barrier_contract]; iota; primops];
  Pulse.Lib.Core.slprop_equiv_norm ()

let barrier_phase_succ (idx : nat)
  : Lemma (idx * 2 + 1 + 1 == (idx + 1) * 2)
  = ()

let spmm_idx_succ_fits
  (idx : nat)
  (block_items : pos)
  (row_start row_end : nat)
  : Lemma
      (requires
        idx * block_items + block_items <= row_end - row_start /\
        FStar.SizeT.fits row_end)
      (ensures FStar.SizeT.fits (idx + 1))
  = FStar.SizeT.fits_lte (idx + 1) row_end

unfold
let kpre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (rA : chest2 real p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (rB : chest2 real p.shared p.cols { eB %~ rB })
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  let sh_e : larray et p.blockItemsK = fst sh in
  let sh_i : larray sz p.blockItemsK = fst (snd sh) in
  block_pre
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems rA eB rB
    fA fri fB
    bid tid **
  thread_live_chunks sh_e p.blockWidth tid **
  thread_live_chunks sh_i p.blockWidth tid **
  pure (
    aligned 16 sh_e /\
    aligned 16 sh_i
  )

unfold
let kpost
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (rA : chest2 real p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (rB : chest2 real p.shared p.cols { eB %~ rB })
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  let sh_e : larray et p.blockItemsK = fst sh in
  let sh_i : larray sz p.blockItemsK = fst (snd sh) in
  block_post
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #relems rA eB rB
    fA fri fB
    bid tid **
  live sh_e #(1.0R /. p.blockWidth) **
  live sh_i #(1.0R /. p.blockWidth)

ghost
fn forevery_ext_3
  (#a #b #c : Type0)
  (f g : a -> b -> c -> slprop)
  requires
    pure (forall x y z. f x y z == g x y z)
  requires
    forall+ (x:a) (y:b) (z:c). f x y z
  ensures
    forall+ (x:a) (y:b) (z:c). g x y z
{
  forevery_map_2
    (fun x y -> forall+ z. f x y z)
    (fun x y -> forall+ z. g x y z)
    fn x y {
      forevery_ext (fun z -> f x y z) (fun z -> g x y z)
    };
}

ghost
fn forevery_assoc_2
  (#a #b : Type0)
  (p1 p2 p3 : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). (p1 x y ** p2 x y) ** p3 x y
  ensures
    forall+ (x:a) (y:b). p1 x y ** p2 x y ** p3 x y
{
  forevery_map_2
    (fun x y -> (p1 x y ** p2 x y) ** p3 x y)
    (fun x y -> p1 x y ** p2 x y ** p3 x y)
    fn x y {};
}

let lem_div1 (n : nat) (d : pos) (r : natlt d)
: Lemma (requires true) (ensures (n * d + r) / d == n)
= Math.Lemmas.lemma_div_plus r n d

let lem_div2 (n : nat) (d : pos) (r : natlt d)
: Lemma (requires true) (ensures (n * d + r) % d == r)
= Math.Lemmas.lemma_mod_plus r n d

let barrier_step_lemma (i : nat)
  : Lemma (i * 2 + 1 + 1 == (i + 1) * 2)
  = ()

let block_count_commuted
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  : Lemma (SZ.v (nblocks p) == divup (SZ.v p.cols) (SZ.v p.blockItemsX) * SZ.v p.rows)
  = FStar.Math.Lemmas.swap_mul (SZ.v p.rows) (divup (SZ.v p.cols) (SZ.v p.blockItemsX))

let output_column_aligned
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et) (bid : natlt (nblocks_ p))
  : Lemma (SZ.v (chunk et) /? bcol p bid)
  = let _ = tcol p bid 0 in ()

let output_tile_bound
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (b : natlt (divup (SZ.v p.cols) (SZ.v p.blockItemsX)))
  : Lemma (b * SZ.v p.blockItemsX < SZ.v p.cols)
  = ()

#push-options "--z3rlimit 30"
ghost
fn setup
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (#rA : chest2 real p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fri #fB : perm)
  ()
  norewrite
  requires
    smatrix_approx' gA #fA elems col_ind row_off relems rA **
    row_indices |-> Frac fri (ordering row_perm) **
    gB |-> Frac fB eB **
    live gC
  ensures
    (forall+ (bid : natlt (nblocks p)) (tid : natlt p.blockWidth).
      block_pre
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        #relems rA eB rB
        fA fri fB
        bid tid
    ) **
      emp
{
  with eC. assert gC |-> eC;
  tensor_explode2 gC;
  forevery_unflatten' _;

  // Keep each cell's initial value while grouping rows into column tiles.
  Math.prod_divides p.blockWidth (chunk et) p.blockItemsX;
  Math.prod_cancel_divides p.blockItemsX (chunk et) p.blockWidth;
  forevery_map
    (fun r -> forall+ (c : natlt p.cols).
      tensor_pts_to_cell gC (idx2 r c) (acc2 eC r c))
    (fun r -> forall+ (b : natlt (divup p.cols p.blockItemsX)).
      pts_to_tile_in_matrix gC r (b * p.blockItemsX) eC p.blockItemsX)
    fn r {
      forevery_factor_ p.cols p.blockItemsX _;
      forevery_map #(natlt (divup p.cols p.blockItemsX))
        (fun b -> forall+ (ix : natlt p.blockItemsX {b * p.blockItemsX + ix < p.cols}).
          tensor_pts_to_cell gC (idx2 r (b * p.blockItemsX + ix))
            (acc2 eC r (b * p.blockItemsX + ix)))
        (fun b -> pts_to_tile_in_matrix gC r (b * p.blockItemsX) eC p.blockItemsX)
        fn b {
          forevery_unrefine_pred' #(natlt p.blockItemsX)
            (fun ix -> b * p.blockItemsX + ix < p.cols)
            (fun ix _ -> tensor_pts_to_cell gC (idx2 r (b * p.blockItemsX + ix))
              (acc2 eC r (b * p.blockItemsX + ix)));
          fold pts_to_tile_in_matrix gC r (b * p.blockItemsX) eC p.blockItemsX;
        };
    };
  forevery_iso (Kuiper.Bijection.bij_sym row_perm) _;

  // Column tiles are outermost, so adjacent blocks visit different rows.
  forevery_commute _;
  block_count_commuted p;
  forevery_unfactor'
    (nblocks p) (divup p.cols p.blockItemsX) p.rows _;
  forevery_ext #(natlt (nblocks p)) _
    (fun bid -> pts_to_tile_in_matrix gC
      (brow p bid |~> row_perm) (bcol p bid) eC p.blockItemsX);
  forevery_map #(natlt (nblocks p))
    (fun bid -> pts_to_tile_in_matrix gC
      (brow p bid |~> row_perm) (bcol p bid) eC p.blockItemsX)
    (fun bid -> forall+ (tid : natlt p.blockWidth).
      thread_live_tile_vec gC (brow p bid |~> row_perm) (tcol p bid tid)
        (p.blockItemsX / p.blockWidth) p.blockWidth)
    fn bid {
      output_column_aligned p bid;
      rewrite pts_to_tile_in_matrix gC
          (brow p bid |~> row_perm) (bcol p bid) eC p.blockItemsX
        as pts_to_tile_in_matrix gC
          (brow p bid |~> row_perm) (bcol p bid) eC
          ((p.blockItemsX / p.blockWidth) * p.blockWidth);
      thread_live_tile_vec_scatter gC (brow p bid |~> row_perm) (bcol p bid) eC
        p.blockWidth (p.blockItemsX / p.blockWidth);
      forevery_ext #(natlt p.blockWidth) _
        (fun tid -> thread_live_tile_vec gC
          (brow p bid |~> row_perm) (tcol p bid tid)
          (p.blockItemsX / p.blockWidth) p.blockWidth);
    };

  Kuiper.Array.Extra.array_share row_indices (allthreads p);
  forevery_factor (allthreads p) (nblocks p) p.blockWidth _;

  tensor_share_n gB (allthreads p) #fB;
  forevery_factor (allthreads p) (nblocks p) p.blockWidth _;

  forevery_zip3_2
    (fun _ _ ->
      row_indices |-> Frac (fri /. (allthreads p)) (ordering row_perm)
    )
    (fun _ _ -> gB |-> Frac (fB /. (allthreads p)) eB)
    _;

  smatrix_share_approx_n' gA #fA elems col_ind row_off relems rA (allthreads p);
  forevery_factor (allthreads p) (nblocks p) p.blockWidth _;

  forevery_zip_2
    (fun _ _ ->
      smatrix_approx' gA #(fA /. (allthreads p)) elems col_ind row_off relems rA
    )
    _;
}

#pop-options
ghost
fn block_setup
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (#rA : chest2 real p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt p.blockWidth).
      block_pre
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        #relems rA eB rB
        fA fri fB
        bid tid
    )
  ensures
    (forall+ (tid : natlt p.blockWidth).
      kpre
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        #relems rA eB rB
        fA fri fB
        sh
        bid tid
    ) **
      emp
{
  unfold_c_shmems sh (`%shmems_desc);

  let sh_e : larray et p.blockItemsK = fst sh;
  assert rewrites_to sh_e (fst sh);
  let sh_i : larray sz p.blockItemsK = fst (snd sh);
  assert rewrites_to sh_i (fst (snd sh));

  thread_share_chunks sh_e p.blockWidth;
  thread_share_chunks sh_i p.blockWidth;

  assume pure ( aligned 16 sh_e /\ aligned 16 sh_i);

  forevery_intro_pure #(natlt p.blockWidth)
    (fun _ ->
        aligned 16 sh_e /\
        aligned 16 sh_i
    );
  forevery_zip3 #(natlt p.blockWidth)
    (fun tid -> thread_live_chunks sh_e p.blockWidth tid)
    (fun tid -> thread_live_chunks sh_i p.blockWidth tid)
    (fun tid ->
      pure (
        aligned 16 sh_e /\
        aligned 16 sh_i
      )
    );

  forevery_zip #(natlt p.blockWidth)
    _
    (fun tid ->
      thread_live_chunks sh_e p.blockWidth tid **
      thread_live_chunks sh_i p.blockWidth tid **
      pure (
        aligned 16 sh_e /\
        aligned 16 sh_i
      )
    );

}

ghost
fn block_teardown
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (#rA : chest2 real p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p ))
  (bid : natlt (nblocks p))
  ()
  norewrite
  requires
    (forall+ (tid : natlt p.blockWidth).
      kpost
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        #relems rA eB rB
        fA fri fB
        sh bid tid
    ) **
    emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt p.blockWidth).
      block_post
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        #relems rA eB rB
        fA fri fB
        bid tid
    )
{
  let sh_e : larray et p.blockItemsK = fst sh;
  assert rewrites_to sh_e (fst sh);
  let sh_i : larray sz p.blockItemsK = fst (snd sh);
  assert rewrites_to sh_i (fst (snd sh));

  forevery_unzip3 _ _ _;
  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      pts_to sh_e #(1.0R /. p.blockWidth) x
    );
  with elems_tile.
    assert pts_to sh_e #(1.0R /. p.blockWidth) elems_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (pts_to sh_e #(1.0R /. p.blockWidth) elems_tile)
    (fun tid -> exists* x.
      pts_to sh_e #(1.0R /. p.blockWidth) x
    )
    (fun tid ->
      pts_to sh_e #(1.0R /. p.blockWidth) elems_tile
    )
    fn tid {
      Pulse.Lib.Array.pts_to_injective_eq sh_e;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      pts_to sh_e #(1.0R /. p.blockWidth) elems_tile
    );

  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      pts_to sh_i #(1.0R /. p.blockWidth) x
    );
  with col_ind_tile.
    assert pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile)
    (fun tid -> exists* x.
      pts_to sh_i #(1.0R /. p.blockWidth) x
    )
    (fun tid ->
      pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile
    )
    fn tid {
      Pulse.Lib.Array.pts_to_injective_eq sh_i;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile
    );

  Kuiper.Array.Extra.array_gather sh_e       p.blockWidth;
  Kuiper.Array.Extra.array_gather sh_i p.blockWidth;

  fold_c_shmems sh (`%shmems_desc);

  ();
}

(* Nonlinear bound relating a (column-block, row) pair to a flat block id.
   With one SMT query per proof obligation Z3 no longer finds this on its own,
   so we prove it once by an explicit multiplication lemma. *)
let block_index_bound #et {| sized et, has_vec_cpy et |} (p : parameters et)
  : Lemma (forall (r : natlt (SZ.v p.rows))
                  (b : natlt (divup (SZ.v p.cols) (SZ.v p.blockItemsX))).
             0 <= b * SZ.v p.rows + r /\
             b * SZ.v p.rows + r < nblocks_ p)
  = introduce forall (r : natlt (SZ.v p.rows))
                     (b : natlt (divup (SZ.v p.cols) (SZ.v p.blockItemsX))).
      0 <= b * SZ.v p.rows + r /\ b * SZ.v p.rows + r < nblocks_ p
    with FStar.Math.Lemmas.lemma_mult_le_right
           (SZ.v p.rows) (b + 1)
           (divup (SZ.v p.cols) (SZ.v p.blockItemsX))

#push-options "--z3rlimit 30 --using_facts_from '* -Kuiper.Sparse.Common.sorted_slice'"
ghost
fn teardown
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#relems : erased (lseq real gA.nnz))
  (#rA : chest2 real p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fri #fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (nblocks p)) (tid : natlt p.blockWidth).
      block_post
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        #relems rA eB rB
        fA fri fB
        bid tid
    ) **
    emp
  ensures
    smatrix_approx' gA #fA elems col_ind row_off relems rA **
    row_indices |-> Frac fri (ordering row_perm) **
    gB |-> Frac fB eB **
    matrix_approx gC (MS.matmul rA rB)
{
  forevery_unzip_2 _ _;
  forevery_unfactor' (allthreads p) _ _
    (fun _ _ ->
      smatrix_approx' gA #(fA /. allthreads p) elems col_ind row_off relems rA
    );
  smatrix_gather_approx_n' gA #fA elems col_ind row_off relems rA (allthreads p);

  forevery_unzip_2 _ _;
  forevery_unfactor' (allthreads p) _ _
    (fun _ _ ->
      row_indices |-> Frac (fri /. allthreads p) (ordering row_perm)
    );
  Kuiper.Array.Extra.array_gather row_indices (allthreads p);
  forevery_unzip_2 _ _;
  forevery_unfactor' (allthreads p) _ _
    (fun _ _ -> gB |-> Frac (fB /. allthreads p) eB);
  tensor_gather_n gB (allthreads p) #fB;

  // Gather each block's vectors into a scalar tile, preserving the value
  // of every cell. Then undo the block ordering and the row permutation.
  Math.prod_divides p.blockWidth (chunk et) p.blockItemsX;
  Math.prod_cancel_divides p.blockItemsX (chunk et) p.blockWidth;
  forevery_map #(natlt (nblocks p))
    (fun bid -> forall+ (tid : natlt p.blockWidth).
      thread_tile_approx gC
        (brow p bid |~> row_perm) (tcol p bid tid)
        (MS.matmul rA rB) (p.blockItemsX / p.blockWidth) p.blockWidth)
    (fun bid -> tile_approx gC
      (brow p bid |~> row_perm) (bcol p bid) (MS.matmul rA rB) p.blockItemsX)
    fn bid {
      output_column_aligned p bid;
      forevery_ext #(natlt p.blockWidth)
        _
        (fun tid -> thread_tile_approx gC
          (brow p bid |~> row_perm) (thread_offset et (bcol p bid) tid)
          (MS.matmul rA rB) (p.blockItemsX / p.blockWidth) p.blockWidth);
      thread_tile_approx_gather gC
        (brow p bid |~> row_perm) (bcol p bid) (MS.matmul rA rB)
        p.blockWidth (p.blockItemsX / p.blockWidth);
      rewrite tile_approx gC
          (brow p bid |~> row_perm) (bcol p bid) (MS.matmul rA rB)
          ((p.blockItemsX / p.blockWidth) * p.blockWidth)
        as tile_approx gC
          (brow p bid |~> row_perm) (bcol p bid) (MS.matmul rA rB) p.blockItemsX;
    };
  block_index_bound p;
  block_count_commuted p;
  forevery_factor
    (nblocks p) (divup p.cols p.blockItemsX) p.rows _;
  forevery_map_2 #(natlt (divup p.cols p.blockItemsX)) #(natlt p.rows)
    (fun b r -> tile_approx gC
      (brow p (b * p.rows + r) |~> row_perm) (bcol p (b * p.rows + r))
      (MS.matmul rA rB) p.blockItemsX)
    (fun b r -> tile_approx gC
      (r |~> row_perm) (b * p.blockItemsX) (MS.matmul rA rB) p.blockItemsX)
    fn b r {
      lem_div2 b p.rows r;
      rewrite each (brow p (b * p.rows + r)) as r;
      lem_div1 b p.rows r;
      output_tile_bound p b;
      rewrite each (bcol p (b * p.rows + r)) as (b * p.blockItemsX);
    };
  forevery_commute _;
  forevery_iso row_perm (fun r -> forall+ (b : natlt (divup p.cols p.blockItemsX)).
    tile_approx gC (r |~> row_perm) (b * p.blockItemsX)
      (MS.matmul rA rB) p.blockItemsX);
  forevery_ext_2 #(natlt p.rows) #(natlt (divup p.cols p.blockItemsX))
    _
    (fun r b -> tile_approx gC r (b * p.blockItemsX)
      (MS.matmul rA rB) p.blockItemsX);
  gather_approx_matrix gC (MS.matmul rA rB) p.blockItemsX;

  ();
}

#pop-options
