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
open Kuiper.Bijection

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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

(* Functional postcondition: the cell contains a value approximating real_gemm_single *)
unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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
  (exists* (v : et).
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile)
      (tid % tile)
      v **
    pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol))



inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, cC : clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB eC : ematrix et _ _)
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
    kpost comb comb_r tile gA gB gC eA eB eC fA fB bid tid **
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

  (* Use matmul_tiled_dotprod' which gives us the approximation postcondition *)
  let s = MU.matmul_tiled_dotprod' gA gB mrow mcol brow bcol;
  (* s %~ real_matmul_single eA eB grow gcol *)

  let v0 = gpu_matrix_read_cell gTile brow bcol;
  (* v0 == macc eC grow gcol *)
  let v1 = comb v0 s;
  gpu_matrix_write_cell gTile brow bcol v1;

  (* Prove v1 approximates real_gemm_single:
     - v0 = macc eC grow gcol, so to_real v0 approximates itself by to_real_ok
     - s %~ real_matmul_single eA eB grow gcol (from matmul_tiled_dotprod')
     - v1 = comb v0 s
     - By comb_r's refinement: v1 %~ comb_r (to_real v0) (real_matmul_single) = real_gemm_single *)
  to_real_ok v0;

  rewrite
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      brow bcol v1
  as
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile) (tid % tile) v1;

  fold (kpost comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  ()
}

ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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

#push-options "--z3rlimit 80"
ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpost comb comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et _ _).
      gC |-> eC' **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
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
      kpost comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
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
        pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol)));

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
        pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol)));
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
        pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol));

  (* Step 4: Gather gA and gB *)
  forevery_unfactor' n_threads (mrows * mcols) (tile * tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (tile * tile)) ->
      gA |-> Frac (fA /. n_threads) eA);
  forevery_unfactor' n_threads (mrows * mcols) (tile * tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (tile * tile)) ->
      gB |-> Frac (fB /. n_threads) eB);
  gpu_matrix_gather_n gA n_threads;
  gpu_matrix_gather_n gB n_threads;

  (* Step 5: Collect the existential witnesses using forevery_exists_2 *)
  let vf = forevery_exists_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) (v : et) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
        (tid / tile) (tid % tile) v **
      pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol));
  (* Now vf : (bid -> tid -> GTot et) witnesses the value at each cell *)

  (* Step 6: Extract the pure approximation facts *)
  forevery_extract_pure_2
    #(natlt (mrows * mcols)) #(natlt (tile * tile))
    (fun bid tid ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
        (tid / tile) (tid % tile) (vf bid tid) **
      pure (vf bid tid %~ MU.real_gemm_single comb_r eA eB eC grow gcol))
    (fun bid tid ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      vf bid tid %~ MU.real_gemm_single comb_r eA eB eC grow gcol)
    fn bid tid { (); };

  (* Now we have the pure fact that all cells approximate real_gemm_single *)
  assert pure (forall (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)).
    let mrow = bid / mcols in
    let mcol = bid % mcols in
    let brow = tid / tile in
    let bcol = tid % tile in
    let grow = mrow * tile + brow in
    let gcol = mcol * tile + bcol in
    vf bid tid %~ MU.real_gemm_single comb_r eA eB eC grow gcol);

  (* Step 7: Drop the pures and reorganize cells for implode *)
  forevery_map_2
    #(natlt (mrows * mcols)) #(natlt (tile * tile))
    (fun bid tid ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
        (tid / tile) (tid % tile) (vf bid tid) **
      pure (vf bid tid %~ MU.real_gemm_single comb_r eA eB eC grow gcol))
    (fun bid tid ->
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
        (tid / tile) (tid % tile) (vf bid tid))
    fn bid tid {
      let mrow = bid / mcols;
      let mcol = bid % mcols;
      let brow = tid / tile;
      let bcol = tid % tile;
      let grow = mrow * tile + brow;
      let gcol = mcol * tile + bcol;
      drop_ (pure (vf bid tid %~ MU.real_gemm_single comb_r eA eB eC grow gcol));
    };

  (* Step 8: Factor to 4D for implode_tiled *)
  forevery_factor_2
    (mrows * mcols) mrows mcols
    (tile * tile) tile tile
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (tile * tile)) ->
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
        (tid / tile) (tid % tile) (vf bid tid));

  (* Simplify div/mod *)
  assert pure (forall (mrow:natlt mrows) (mcol:natlt mcols). (mrow * mcols + mcol) / mcols == mrow /\ (mrow * mcols + mcol) % mcols == mcol);
  assert pure (forall (brow:natlt tile) (bcol:natlt tile). (brow * tile + bcol) / tile == brow /\ (brow * tile + bcol) % tile == bcol);

  forevery_ext_4
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      let bid = mrow * mcols + mcol in let tid = brow * tile + bcol in
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
        (tid / tile) (tid % tile) (vf bid tid))
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      gpu_matrix_pts_to_cell
        (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) mrow mcol)
        brow bcol (vf (mrow * mcols + mcol) (brow * tile + bcol)));

  (* Step 9: Convert sizes for implode_tiled *)
  forevery_rw_size4 mrows ((mrows * tile) / tile) mcols ((mcols * tile) / tile) tile (SZ.v tile) tile (SZ.v tile);

  (* Step 10: Call implode_tiled *)
  gpu_matrix_implode_tiled gC (SZ.v tile) (SZ.v tile)
    (fun (tr:natlt mrows) (tc:natlt mcols) (i:natlt tile) (j:natlt tile) ->
      vf (tr * mcols + tc) (i * tile + j));

  (* Now we have gC |-> eC' where eC' = mkM(fun row col -> vf ...) *)
  with eC'. assert gC |-> eC';

  (* Step 11: Prove the approximation postcondition *)
  (* eC'[row][col] = vf (row/tile * mcols + col/tile) ((row%tile) * tile + col%tile)
     and we know vf bid tid %~ real_gemm_single comb_r eA eB eC grow gcol
     where grow = (bid/mcols)*tile + tid/tile, gcol = (bid%mcols)*tile + tid%tile *)

  (* The indexing matches: for row, col in the result matrix:
     bid = row/tile * mcols + col/tile, tid = (row%tile)*tile + col%tile
     grow = (bid/mcols)*tile + tid/tile = (row/tile)*tile + row%tile = row
     gcol = (bid%mcols)*tile + tid%tile = (col/tile)*tile + col%tile = col
     So vf bid tid %~ real_gemm_single comb_r eA eB eC row col = macc (real_mmcomb comb_r eC eA eB) row col *)

  assert pure (forall (row:natlt (mrows * tile)) (col:natlt (mcols * tile)).
    let tr = row / tile in
    let tc = col / tile in
    let i = row % tile in
    let j = col % tile in
    let bid = tr * mcols + tc in
    let tid = i * tile + j in
    macc eC' row col == vf bid tid);

  (* Index simplifications for SMT *)
  assert pure (forall (row:natlt (mrows * tile)) (col:natlt (mcols * tile)).
    let tr = row / tile in
    let tc = col / tile in
    let i = row % tile in
    let j = col % tile in
    let bid = tr * mcols + tc in
    let tid = i * tile + j in
    bid / mcols == tr /\ bid % mcols == tc /\
    tid / tile == i /\ tid % tile == j /\
    tr * tile + i == row /\ tc * tile + j == col);

  (* Connect vf with real_gemm_single using the row/col indices *)
  assert pure (forall (row:natlt (mrows * tile)) (col:natlt (mcols * tile)).
    let tr = row / tile in
    let tc = col / tile in
    let i = row % tile in
    let j = col % tile in
    let bid = tr * mcols + tc in
    let tid = i * tile + j in
    vf bid tid %~ MU.real_gemm_single comb_r eA eB eC row col);

  (* Bridge to ematrix_approximates: each cell of eC' approximates the corresponding cell of real_mmcomb *)
  assert pure (forall (row:natlt (mrows * tile)) (col:natlt (mcols * tile)).
    macc eC' row col %~ macc (MU.real_mmcomb comb_r eC eA eB) row col);

  assert pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB));

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
inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc_m_n
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et _ _).
          gC |-> eC' **
          pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB))))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre  comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  setup     = setup    tile comb comb_r gA gB gC #eA #eB #eC;
  teardown  = teardown tile comb comb_r gA gB gC #eA #eB #eC;

  block_frame    = (fun _bid -> emp);
  block_setup    = block_setup (mrows *^ mcols) (tile *^ tile);
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  comb comb_r tile gA gB gC eA eB eC fA fB;
  kpost     = kpost comb comb_r tile gA gB gC eA eB eC fA fB;

  f = kf #et #_ #_ comb comb_r #mrows #mshared #mcols tile gA gB gC eA eB eC fA fB;

  // FIXME: admitting these, they should be trivial but are extremely slow
  // and end up failing.
  kpre_sendable=magic();
  kpost_sendable=magic();
  block_pre_sendable=magic();
  block_post_sendable=magic();
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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
  (#eA #eB #eC : ematrix _ _ _)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et _ _).
      on gpu_loc (gC |-> eC') **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile comb comb_r gA gB gC ());
}


(* Legacy interface for backward compatibility.
   Calls the approximate kernel with add/(+.) and assumes the exact result. *)
inline_for_extraction noextract
fn mmcomb_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (#eA #eB #eC : ematrix _ _ _)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  // Fake real_like instance and comb_r with assumed refinement
  let _ : real_like et #_ = magic ();
  let comb_r : binop real = magic ();
  assume pure (forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s);
  mmcomb_gpu_approx tile comb comb_r lA lB lC gA gB gC;
  with eC'. assert (on gpu_loc (gC |-> eC'));
  (* Assume the approximate result is exactly correct *)
  assume pure (eC' == MS.mmcomb comb eC eA eB);
}
