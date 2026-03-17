module Kuiper.Poly.GEMM.Tiled

#lang-pulse

open Kuiper
open Kuiper.Approximates
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Tiling

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  let mrow = bid / mcols in
  let mcol = bid % mcols in
  let brow = tid / tile in
  let bcol = tid % tile in
  let grow = mrow * tile + brow in
  let gcol = mcol * tile + bcol in
  gA |-> Frac (fA /. ((mrows * mcols) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((mrows * mcols) * (tile * tile))) eB **
  gpu_matrix_pts_to_cell
    (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
    (tid / tile)
    (tid % tile)
    (macc eC grow gcol)

(* Functional postcondition: the cell contains a value approximating
   MS.gemm_single over external real matrices rA, rB, rC *)
unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  let mrow = bid / mcols in
  let mcol = bid % mcols in
  let brow = tid / tile in
  let bcol = tid % tile in
  let grow = mrow * tile + brow in
  let gcol = mcol * tile + bcol in
  gA |-> Frac (fA /. ((mrows * mcols) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((mrows * mcols) * (tile * tile))) eB **
  (exists* (v : et).
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile)
      (tid % tile)
      v **
    pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol))



inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, cC : clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (_sq : squash (eA %~ rA /\ eB %~ rB /\ eC %~ rC))
  (fA fB : perm)
  (bid : szlt (mrows * mcols))
  (tid : szlt (tile * tile))
  ()
  norewrite
  requires
    gpu **
    kpre comb comb_r tile gA gB gC eA eB eC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
{
  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;

  (* Global indices for this thread (erased since they're ghost computations) *)
  let grow : erased nat = mrow * tile + brow;
  let gcol : erased nat = mcol * tile + bcol;

  assert (pure (mrow < mrows));
  assert (pure (mcol < mcols));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  (* Rewrite kpre's cell indices to use brow/bcol (which equal tid/tile, tid%tile) *)
  rewrite
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile) (tid % tile) (macc eC grow gcol)
  as
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      brow bcol (macc eC grow gcol);

  let gTile = gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  assert (rewrites_to gTile (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)));

  (* Introduce the pure approximation facts from the squash parameter *)
  assert pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC);

  (* Use matmul_tiled_dotprod_real which gives us the approximation over rA, rB *)
  let s = MU.matmul_tiled_dotprod_real gA gB rA rB mrow mcol brow bcol;
  (* s %~ MS.matmul_single rA rB grow gcol *)

  let v0 = gpu_matrix_read_cell gTile brow bcol;
  (* v0 == macc eC grow gcol *)
  let v1 = comb v0 s;
  gpu_matrix_write_cell gTile brow bcol v1;

  (* Prove v1 approximates MS.gemm_single comb_r rA rB rC grow gcol:
     - v0 = macc eC grow gcol, and eC %~ rC, so v0 %~ macc rC grow gcol
     - s %~ MS.matmul_single rA rB grow gcol (from matmul_tiled_dotprod_real)
     - v1 = comb v0 s
     - By approx2 comb comb_r: v1 %~ comb_r (macc rC grow gcol) (MS.matmul_single rA rB grow gcol)
     - Which is: v1 %~ MS.gemm_single comb_r rA rB rC grow gcol *)

  rewrite
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      brow bcol v1
  as
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile) (tid % tile) v1;

  fold (kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  ()
}

#push-options "--z3rlimit 20"
ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpre comb comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  let n_threads = (mrows * mcols) * (tile * tile);

  (* Step 1: Share gA/gB, explode+tile gC *)
  gpu_matrix_share_n gA n_threads;
  gpu_matrix_share_n gB n_threads;
  gpu_matrix_explode_tiled gC (SZ.v tile) (SZ.v tile);
  (* Need to rewrite types: (rows/tile) == mrows, (cols/tile) == mcols *)
  forevery_rw_size4 ((mrows * tile) / tile) mrows ((mcols * tile) / tile) mcols (SZ.v tile) tile (SZ.v tile) tile;
  (* gC: forall+ mrow mcol brow bcol. subtile_cell *)

  (* Step 2: Factor gA/gB to 2D *)
  forevery_factor n_threads (mrows * mcols) (tile * tile) (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (mrows * mcols) (tile * tile) (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  (* Step 3: Convert 4D -> 2D using unfactor_2 *)
  assert pure (forall (mrow:natlt mrows) (mcol:natlt mcols). (mrow * mcols + mcol) / mcols == mrow /\ (mrow * mcols + mcol) % mcols == mcol);
  assert pure (forall (brow:natlt tile) (bcol:natlt tile). (brow * tile + bcol) / tile == brow /\ (brow * tile + bcol) % tile == bcol);
  forevery_ext_4
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) mrow mcol) brow bcol (macc eC (mrow * tile + brow) (mcol * tile + bcol)))
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      let bid = mrow * mcols + mcol in let tid = brow * tile + bcol in
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));
  forevery_unfactor_2 (mrows * mcols) mrows mcols (tile * tile) tile tile
    (fun bid tid -> gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
      (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));
  (* gC: forall+ bid tid. subtile_cell with div/mod indexing *)

  (* Step 4: Zip gA, gB, gC together *)
  forevery_zip3_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) -> gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) -> gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));
  (* Combined: forall+ bid tid. gA ** gB ** cell *)

  (* Step 5: Convert types and match kpre *)
  forevery_rw_size2 (mrows * mcols) (SZ.v (mrows `SZ.mul` mcols)) (tile * tile) (SZ.v (tile `SZ.mul` tile));
  forevery_ext_2
    (fun (bid : natlt (SZ.v (mrows `SZ.mul` mcols))) (tid : natlt (SZ.v (tile `SZ.mul` tile))) ->
      gA |-> Frac (fA /. n_threads) eA ** gB |-> Frac (fB /. n_threads) eB **
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))))
    (fun (bid : natlt2 mrows mcols) (tid : natlt2 tile tile) -> kpre comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  ();
}
#pop-options

#push-options "--z3rlimit 40"
ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
      gC |-> eC' **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  let n_threads = (mrows * mcols) * (tile * tile);

  (* Step 1: Convert types from natlt2 to natlt *)
  assert pure (SZ.v (mrows `SZ.mul` mcols) == mrows * mcols);
  assert pure (SZ.v (tile `SZ.mul` tile) == tile * tile);
  forevery_rw_size2
    (SZ.v (mrows `SZ.mul` mcols)) (mrows * mcols)
    (SZ.v (tile `SZ.mul` tile)) (tile * tile);

  (* Step 2: Unfold kpost to expose the existentials *)
  forevery_ext_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      (exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol)));

  (* Step 3: Unzip gA, gB *)
  forevery_unzip_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gB |-> Frac (fB /. n_threads) eB **
      (exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol)));
  forevery_unzip_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC grow gcol));

  (* Step 4: Gather gA and gB *)
  forevery_unfactor' n_threads (mrows * mcols) (tile * tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (tile * tile)) ->
      gA |-> Frac (fA /. n_threads) eA);
  forevery_unfactor' n_threads (mrows * mcols) (tile * tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (tile * tile)) ->
      gB |-> Frac (fB /. n_threads) eB);
  gpu_matrix_gather_n gA n_threads;
  gpu_matrix_gather_n gB n_threads;

  (* Step 5: Collect gC cells back into matrix *)
  let _vf = gpu_matrix_collect_approx_tiled gC (SZ.v tile) (SZ.v tile)
    (SZ.v mrows) (SZ.v mcols)
    (fun (row : natlt (mrows * tile)) (col : natlt (mcols * tile)) (v : et) ->
      v %~ MS.gemm_single comb_r rA rB rC row col);

  (* Step 6: Prove eC' %~ MS.mmcomb comb_r rC rA rB *)
  with eC'. assert (gC |-> eC');
  assert pure (eC' %~ MS.mmcomb comb_r rC rA rB);
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

#push-options "--z3rlimit 40"
let kpre_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
: is_send_across block_of (kpre comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
= solve

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
: is_send_across block_of (kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
= solve

let block_pre_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      kpre comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
= solve

let block_post_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt2 tile tile).
      kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
= solve
#pop-options

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA #eB #eC : ematrix _ _ _)
  (rA : ematrix real (mrows   * tile) (mshared * tile))
  (rB : ematrix real (mshared * tile) (mcols   * tile))
  (rC : ematrix real (mrows   * tile) (mcols   * tile))
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads
               /\ eA %~ rA /\ eB %~ rB /\ eC %~ rC))
  : kernel_desc_m_n
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
          gC |-> eC' **
          pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre  comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  setup     = setup    tile comb comb_r gA gB gC #eA #eB #eC rA rB rC;
  teardown  = teardown tile comb comb_r gA gB gC #eA #eB #eC rA rB rC;

  block_frame    = (fun _bid -> emp);
  block_setup    = block_setup (mrows *^ mcols) (tile *^ tile);
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  comb comb_r tile gA gB gC eA eB eC fA fB;
  kpost     = kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  f = kf #et #_ #_ comb comb_r #mrows #mshared #mcols tile gA gB gC eA eB eC rA rB rC () fA fB;

  kpre_sendable = kpre_block_sendable comb comb_r tile gA gB gC eA eB eC fA fB;
  kpost_sendable = kpost_block_sendable comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  block_pre_sendable = block_pre_gpu_sendable comb comb_r tile gA gB gC eA eB eC fA fB;
  block_post_sendable = block_post_gpu_sendable comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#mrows #mshared #mcols : szp)
  (lA : mlayout (mrows   * tile) (mshared * tile))
  (lB : mlayout (mshared * tile) (mcols   * tile))
  (lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (rA  : ematrix real (mrows * tile) (mshared * tile))
  (rB  : ematrix real (mshared * tile) (mcols * tile))
  (rC  : ematrix real (mrows * tile) (mcols * tile))
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile comb comb_r gA gB gC rA rB rC ());
}
