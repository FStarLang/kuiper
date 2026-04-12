module Kuiper.Poly.GEMM.Tiled

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Matrix.Reprs.Type
open Kuiper.Array2
open Kuiper.Tensor.Layout
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling

module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
module M = Kuiper.Array2
module MT = Kuiper.Tensor.Tiling

(* Move away somewhere, this is generic. *)
#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn matmul_tiled_dotprod_real
  (#et : Type0) {| scalar et, real_like et |}
  (#m #n #k : sz)
  (#tile : szp)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  {| ctlayout lA, ctlayout lB |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (#eA #eB : ematrix _ _ _)
  (rA rB : ematrix _ _ _)
  (bi : szlt m)
  (bj : szlt n)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB
  requires
    pure (eA %~ rA /\ eB %~ rB)
  returns
    res : et
  ensures
    pure (res %~ MS.matmul_single rA rB (bi * tile + i) (bj * tile + j))
{
  let grow : erased (natlt (m * tile)) = hide (bi * tile + i);
  let gcol : erased (natlt (n * tile)) = hide (bj * tile + j);

  let mut sum : et = zero;
  let mut bk  : szle k = 0sz;

  while (!bk <^ k)
    invariant live bk ** live sum
    invariant pure (!sum %~ MS.__matmul_single rA rB grow gcol (SZ.v !bk * tile))
    decreases (k - !bk)
  {
    let tA = MT.array2_extract_tile_ro' gA (SZ.v tile) (SZ.v tile) (SZ.v bi) (SZ.v !bk);
    let tB = MT.array2_extract_tile_ro' gB (SZ.v tile) (SZ.v tile) (SZ.v !bk) (SZ.v bj);

    let s' = Kuiper.DotProd.matmul_dotprod tA tB i j;

    ambig_trade_elim ();
    ambig_trade_elim ();

    let s = !sum;

    sum := s `add` s';

    (* Proof that s' %~ matmul_single (subtile rA) (subtile rB) i j:
       subtile eA %~ subtile rA (from eA %~ rA, subtiling preserves %~)
       Then __matmul_single_approx_real gives the approximation. *)
    let sub_rA = MT.ematrix_subtile rA tile tile bi (SZ.v !bk);
    let sub_rB = MT.ematrix_subtile rB tile tile (SZ.v !bk) bj;
    MU.__matmul_single_approx_real
      (ematrix_subtile eA tile tile bi (SZ.v !bk))
      (ematrix_subtile eB tile tile (SZ.v !bk) bj)
      sub_rA sub_rB
      i j tile;

    let r_partial = MS.__matmul_single rA rB grow gcol (SZ.v !bk * tile);
    let r_subtile = MS.__matmul_single sub_rA sub_rB i j tile;

    (* Step the partial sum: split property of __gmatmul_single over rA, rB *)
    MU.__gmatmul_single_split rA rB grow gcol (SZ.v !bk * tile) tile sub_rA sub_rB i j;

    a_add s s' r_partial r_subtile;

    bk := !bk +^ 1sz;

    ()
  };

  !sum
}
#pop-options

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : ematrix et _ _)
  (eC : ematrix et (m * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  : slprop
  =
  let mrow = bid / n in
  let mcol = bid % n in
  let brow = tid / tile in
  let bcol = tid % tile in
  let grow = mrow * tile + brow in
  let gcol = mcol * tile + bcol in
  pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
  gA |-> Frac (fA /. ((m * n) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((m * n) * (tile * tile))) eB **
  M.pts_to_cell
    (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
    (tid / tile, tid % tile) (macc eC grow gcol)

(* Functional postcondition: the cell contains a value approximating
   MS.gemm_single over external real matrices rA, rB, rC *)
unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : ematrix et _ _)
  (eC : ematrix et (m * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  : slprop
  =
  let mrow = bid / n in
  let mcol = bid % n in
  let brow = tid / tile in
  let bcol = tid % tile in
  let grow = mrow * tile + brow in
  let gcol = mcol * tile + bcol in
  gA |-> Frac (fA /. ((m * n) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((m * n) * (tile * tile))) eB **
  (exists* (v : et).
    M.pts_to_cell
      (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
      (tid / tile, tid % tile)
      v **
    pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol))

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB eC : ematrix et _ _)
  (rA rB rC : ematrix real _ _)
  (fA fB : perm)
  (bid : szlt (m * n))
  (tid : szlt (tile * tile))
  ()
  norewrite
  requires
    gpu **
    kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (m * n) bid
  ensures
    gpu **
    kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (m * n) bid
{
  let mrow, mcol = s_divmod n bid;
  let brow, bcol = s_divmod tile  tid;

  (* Global indices for this thread (ghost) *)
  let grow : erased nat = mrow * tile + brow;
  let gcol : erased nat = mcol * tile + bcol;

  (* Rewrite kpre's cell indices to use brow/bcol (which equal tid/tile, tid%tile) *)
  rewrite
    M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
      ((tid / tile <: natlt _), (tid % tile <: natlt _)) (macc eC grow gcol)
  as
    M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
      ((SZ.v brow <: natlt _), (SZ.v bcol <: natlt _)) (macc eC grow gcol);

  let gTile = MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n);
  assert rewrites_to gTile (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n));

  let s = matmul_tiled_dotprod_real gA gB rA rB mrow mcol brow bcol;

  let v0 = M.read_cell gTile (brow, bcol);
  let v1 = comb v0 s;
  M.write_cell gTile (brow, bcol) v1;

  rewrite
    M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
      ((SZ.v brow <: natlt _), (SZ.v bcol <: natlt _)) v1
  as
    M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
      ((tid / tile <: natlt _), (tid % tile <: natlt _)) v1;

  fold kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid;
  ()
}

#push-options "--z3rlimit 20"
ghost
fn setup
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA : ematrix et (m * tile) (k * tile))
  (eB : ematrix et (k * tile) (n * tile))
  (eC : ematrix et (m * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC)
  ensures
    (forall+ (bid : natlt (m *^ n))
             (tid : natlt (tile *^ tile)).
      kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp (* frame *)
{
  let n_threads = (m * n) * (tile * tile);

  (* Step 1: Share gA/gB, explode+tile gC *)
  M.share_n gA n_threads;
  M.share_n gB n_threads;
  MT.array2_explode_tiled gC (SZ.v tile) (SZ.v tile);

  (* Need to rewrite types: (rows/tile) == mrows, (cols/tile) == mcols *)
  forevery_rw_size4 ((m * tile) / tile) m ((n * tile) / tile) n (SZ.v tile) tile (SZ.v tile) tile;
  (* gC: forall+ mrow mcol brow bcol. subtile_cell *)

  (* Step 2: Factor gA/gB to 2D *)
  forevery_factor n_threads (m * n) (tile * tile) (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (m * n) (tile * tile) (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  (* Step 3: Convert 4D -> 2D using unfactor_2 *)
  assert pure (forall (mrow : natlt m) (mcol:natlt n). (mrow * n + mcol) / n == mrow /\ (mrow * n + mcol) % n == mcol);
  assert pure (forall (brow : natlt tile) (bcol:natlt tile). (brow * tile + bcol) / tile == brow /\ (brow * tile + bcol) % tile == bcol);
  forevery_ext_4
    (fun (mrow : natlt m) (mcol : natlt n) (brow : natlt tile) (bcol : natlt tile) ->
      M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) mrow mcol) (brow, bcol)
        (macc eC (mrow * tile + brow) (mcol * tile + bcol)))
    (fun (mrow : natlt m) (mcol : natlt n) (brow : natlt tile) (bcol : natlt tile) ->
      let bid = mrow * n + mcol in let tid = brow * tile + bcol in
      M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) ((tid / tile <: natlt _), (tid % tile <: natlt _))
        (macc eC ((bid / n) * tile + (tid / tile)) ((bid % n) * tile + (tid % tile))));

  forevery_unfactor_2 (m * n) m n (tile * tile) tile tile
    (fun bid tid ->
      M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) ((tid / tile <: natlt _), (tid % tile <: natlt _))
        (macc eC ((bid / n) * tile + (tid / tile)) ((bid % n) * tile + (tid % tile))));
  (* gC: forall+ bid tid. subtile_cell with div/mod indexing *)

  (* Duplicate pure facts into the bid/tid forall+ *)
  assert pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC);
  forevery_intro_pure_2 (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) -> eA %~ rA /\ eB %~ rB /\ eC %~ rC);

  (* Step 4: Zip gA, gB, gC together *)
  forevery_zip4_2
    #(natlt (m * n)) #(natlt (tile * tile))
    (fun bid tid -> pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC)) (* preserve precondition *)
    (fun bid tid -> gA |-> Frac (fA /. n_threads) eA)
    (fun bid tid -> gB |-> Frac (fB /. n_threads) eB)
    (fun bid tid ->
      M.pts_to_cell (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) ((tid / tile <: natlt _), (tid % tile <: natlt _))
        (macc eC ((bid / n) * tile + (tid / tile)) ((bid % n) * tile + (tid % tile))));

  (* Final ext match + making sure size is exactly equal. *)
  forevery_ext_2 _
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) -> kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  forevery_rw_size2
    (m * n) (SZ.v (m *^ n))
    (tile * tile) (SZ.v (tile *^ tile));
  ();
}
#pop-options

#push-options "--z3rlimit 40"
ghost
fn teardown
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA : ematrix et (m * tile) (k * tile))
  (eB : ematrix et (k * tile) (n * tile))
  (eC : ematrix et (m * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (m *^ n))
             (tid : natlt (tile *^ tile)).
      kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et (m * tile) (n * tile)).
      gC |-> eC' **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  let n_threads = (m * n) * (tile * tile);
  assert pure (SZ.fits (m * n));
  assert pure (SZ.fits (layout_size lC));

  forevery_rw_size2
    (SZ.v (m *^ n)) (m * n)
    (SZ.v (tile *^ tile)) (tile * tile);

  (* Step 1: Unfold kpost to expose the existentials *)
  forevery_ext_2 _
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
      let mrow = bid / n in
      let mcol = bid % n in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      (exists* (v : et).
        M.pts_to_cell
          (MT.array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
          ((tid / tile <: natlt _), (tid % tile <: natlt _)) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol)));

  (* Step 2: Unzip gA, gB *)
  forevery_unzip_2
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) -> gA |-> Frac (fA /. n_threads) eA) _;
  forevery_unzip_2
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) -> gB |-> Frac (fB /. n_threads) eB) _;

  (* Step 3: Gather gA and gB *)
  forevery_unfactor' n_threads (m * n) (tile * tile)
    (fun (_ : natlt (m * n)) (_ : natlt (tile * tile)) ->
      gA |-> Frac (fA /. n_threads) eA);
  forevery_unfactor' n_threads (m * n) (tile * tile)
    (fun (_ : natlt (m * n)) (_ : natlt (tile * tile)) ->
      gB |-> Frac (fB /. n_threads) eB);
  M.gather_n gA n_threads;
  M.gather_n gB n_threads;

  (* Step 4: Collect gC cells back into matrix, with approximation proof. We have a lemma for this. *)
  let _ = MT.array2_collect_approx_tiled gC (SZ.v tile) (SZ.v tile)
    (SZ.v m) (SZ.v n)
    (fun (row : natlt (m * tile)) (col : natlt (n * tile)) (v : et) ->
      v %~ MS.gemm_single comb_r rA rB rC row col);

  ();
}
#pop-options

(* No op *)
ghost
fn block_setup
  (nblk : nat)
  (nthr : nat)
  (#p : natlt nblk -> slprop)
  (bid : natlt nblk)
  norewrite
  requires
    p bid
  ensures
    p bid **
    emp
{
  ();
}

(* Sendability helpers — standalone definitions so the typeclass resolver
   runs in a minimal context with an explicit goal type.
   kpre/kpost are block_of-sendable (all components are global matrices, hence
   gpu_of-sendable, and gpu_of implies block_of).
   block_pre/post wrap kpre/kpost in forall+, needing gpu_of directly. *)
let kpre_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB eC : ematrix et _ _)
  (rA rB rC : ematrix real _ _)
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  : is_send_across block_of (kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = magic()

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB eC : ematrix et _ _)
  (rA rB rC : ematrix real _ _)
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  : is_send_across block_of (kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = magic()

let block_pre_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB eC : ematrix et _ _)
  (rA rB rC : ematrix real _ _)
  (fA fB : perm)
  (bid : natlt (m * n))
  : is_send_across gpu_of
      (forall+ (tid : natlt (tile * tile)).
        kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = magic()

let block_post_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB eC : ematrix et _ _)
  (rA rB rC : ematrix real _ _)
  (fA fB : perm)
  (bid : natlt (m * n))
  : is_send_across gpu_of
      (forall+ (tid : natlt (tile * tile)).
        kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = magic()

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#_ : squash (m * n <= max_blocks))
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB eC : ematrix et _ _)
  (rA rB rC : ematrix real _ _)
  (fA fB : perm)
  : kernel_desc_m_n
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC ** pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC))
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et (m * tile) (n * tile)).
          gC |-> eC' **
          pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
= {
  nblk = m *^ n;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  =
    (fun bid -> forall+ (tid : natlt2 tile tile). kpre  comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  block_post =
    (fun bid -> forall+ (tid : natlt2 tile tile). kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  setup     = setup    comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  teardown  = teardown comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  block_frame    = (fun _bid -> emp);
  block_setup    = block_setup (m * n) (tile * tile);
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  kpost     = kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  f = kf comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  kpre_sendable       = kpre_block_sendable     comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  kpost_sendable      = kpost_block_sendable    comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  block_pre_sendable  = block_pre_gpu_sendable  comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  block_post_sendable = block_post_gpu_sendable comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (#eC : ematrix et (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB);
  ()
}
