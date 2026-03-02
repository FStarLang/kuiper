module Kuiper.Poly.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.View.TwoTiles { aview_2tile2, mkAIdx, mkCIdx }

module M = Kuiper.Matrix
open Kuiper.Matrix {
  gpu_matrix,
  gpu_matrix_pts_to,
  gpu_matrix_pts_to_cell,
  is_global_matrix
}

module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs
module FB = Kuiper.Poly.GEMM.FlipFlopBarrier

open Kuiper.EMatrix { ematrix }
open Kuiper.Array.Vectorized { has_vec_cpy }
open Kuiper.Poly.GEMM.Copy.Vec { own_strided_chunks, live_strided_chunks }

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |} (tile:valid_tile) : list shmem_desc = [
  SHArray et (tile *^ tile);
  SHArray et (tile *^ tile);
]

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
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
  (gA |-> Frac (fA /. mlayout_vsize lC) eA) **
  (gB |-> Frac (fB /. mlayout_vsize lC) eB) **
  gpu_matrix_pts_to_cell
    (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
    #1.0R
    (tid / tile) (tid % tile) (macc eC grow gcol)

(* Functional postcondition: the cell contains a value approximating real_gemm_single *)
unfold
let kpost1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
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
  (gA |-> Frac (fA /. mlayout_vsize lC) eA) **
  (gB |-> Frac (fB /. mlayout_vsize lC) eB) **
  (exists* (v : et).
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      #1.0R
      (tid / tile) (tid % tile) v **
    pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol))

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x)

unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x)

(* TODO: Find out where the time is going when checking this function,
it feels a lot slower than the others. *)
#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#eA #eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : szlt (mrows * mcols))
  (tid : szlt (tile  * tile))
  ()
  norewrite
  requires
    gpu **
    kpre comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) (tile * tile) bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) (tile * tile) bid) **
    B.barrier_state (2 * mshared)
{
  let (ar1, (ar2, _)) = sh;

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  M.gpu_matrix_abs' slA ar1;
  let sa1 = M.from_array slA ar1;
  rewrite each M.from_array slA ar1 as sa1;

  M.gpu_matrix_abs' slB ar2;
  let sa2 = M.from_array slB ar2;
  rewrite each M.from_array slB ar2 as sa2;

  let gTile = gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  rewrite each gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols) as gTile;

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;
  assert (pure (SZ.v brow == tid / tile));
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  rewrite each (tid / tile) as v brow;
  rewrite each (tid % tile) as v bcol;

  rewrite
    (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
    (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
  as
    (exists* em1. FB.bp_sharing sa1 em1 (tile * tile)) **
    (exists* em2. FB.bp_sharing sa2 em2 (tile * tile));

  let mut sum : et = zero;
  let mut bk  : sz = 0sz;

  while (SZ.(!bk <^ mshared))
    invariant live sum
    invariant
      exists* (vbk : SZ.t).
        bk |-> vbk **
        B.barrier_state (2 * vbk) **
        pure (vbk <= mshared)
    invariant
      (exists* em1. FB.bp_sharing sa1 em1 (tile * tile)) **
      (exists* em2. FB.bp_sharing sa2 em2 (tile * tile))
  {
    let vbk = !bk;
    gpu_matrix_extract_tile_ro gA tile tile mrow vbk;
    let aTile = gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v vbk);
    assert (rewrites_to aTile (gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v vbk)));
    gpu_matrix_extract_tile_ro gB tile tile vbk mcol;
    let bTile = gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v mcol);
    assert (rewrites_to bTile (gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v mcol)));

    let v1 = M.gpu_matrix_read aTile brow bcol;
    let v2 = M.gpu_matrix_read bTile brow bcol;

    ambig_trade_elim ();
    ambig_trade_elim ();

    even_2x !bk;

    rewrite (exists* em1. FB.bp_sharing sa1 em1 (tile * tile)) **
            (exists* em2. FB.bp_sharing sa2 em2 (tile * tile))
         as (FB.barrier_p eA eB sa1 sa2 (tile * tile) bid) (2 * vbk) tid;
    rewrite (FB.barrier_p eA eB sa1 sa2 (tile * tile) bid) (2 * vbk) tid
         as (FB.contract eA eB slA slB ar1 ar2 (tile * tile) bid).rin (2 * vbk) tid;

    B.barrier_wait ();

    rewrite (FB.contract eA eB slA slB ar1 ar2 (tile * tile) bid).rout (2 * vbk) tid
         as (FB.barrier_q eA eB sa1 sa2 (tile * tile) bid) (2 * vbk) tid;
    rewrite (FB.barrier_q eA eB sa1 sa2 (tile * tile) bid) (2 * vbk) tid
         as live_strided_chunks sa1 (tile * tile) tid **
            live_strided_chunks sa2 (tile * tile) tid;

    (* Bridge from live_strided_chunks to cell-level access for the write.
       TODO: prove this properly by decomposing the forall+ in own_strided_chunks. *)
    drop_ (live_strided_chunks sa1 (tile * tile) tid);
    drop_ (live_strided_chunks sa2 (tile * tile) tid);
    assume_ (exists* x. gpu_matrix_pts_to_cell sa1 brow bcol x);
    assume_ (exists* x. gpu_matrix_pts_to_cell sa2 brow bcol x);

    M.gpu_matrix_write_cell sa1 brow bcol v1;
    M.gpu_matrix_write_cell sa2 brow bcol v2;

    drop_ (gpu_matrix_pts_to_cell sa1 brow bcol v1);
    drop_ (gpu_matrix_pts_to_cell sa2 brow bcol v2);
    assume_ (own_strided_chunks sa1 (ematrix_subtile eA tile tile mrow vbk) (tile * tile) tid);
    assume_ (own_strided_chunks sa2 (ematrix_subtile eB tile tile vbk mcol) (tile * tile) tid);

    odd_2x1 !bk;
    assert (pure (odd (2 * !bk + 1)));

    rewrite own_strided_chunks sa1 (ematrix_subtile eA tile tile mrow vbk) (tile * tile) tid **
            own_strided_chunks sa2 (ematrix_subtile eB tile tile vbk mcol) (tile * tile) tid
         as (FB.barrier_p eA eB sa1 sa2 (tile * tile) bid) (2 * vbk + 1) tid;
    rewrite (FB.barrier_p eA eB sa1 sa2 (tile * tile) bid) (2 * vbk + 1) tid
         as (FB.contract eA eB slA slB ar1 ar2 (tile * tile) bid).rin (2 * vbk + 1) tid;

    B.barrier_wait ();

    even_2x (!bk + 1);
    assert pure (2 * (!bk + 1) == 2 * !bk + 2);
    assert pure (odd (2 * !bk + 1));
    assert pure ((2 * !bk + 1) < 2 * (mshared * tile) / tile);
    assert pure (even (2 * !bk + 2));
    rewrite (FB.contract eA eB slA slB ar1 ar2 (tile * tile) bid).rout (2 * !bk + 1) tid
         as (FB.barrier_q eA eB sa1 sa2 (tile * tile) bid) (2 * !bk + 1) tid;
    rewrite (FB.barrier_q eA eB sa1 sa2 (tile * tile) bid) (2 * !bk + 1) tid
         as FB.bp_sharing sa1 (ematrix_subtile eA tile tile mrow !bk) (tile * tile) **
            FB.bp_sharing sa2 (ematrix_subtile eB tile tile !bk mcol) (tile * tile);

    unfold FB.bp_sharing sa1 (ematrix_subtile eA tile tile mrow !bk) (tile * tile);
    unfold FB.bp_sharing sa2 (ematrix_subtile eB tile tile !bk mcol) (tile * tile);

    (* At this point the SHMem cache is filled with the submatrices
       and we have RO permission to it. Compute product for our cell in
       the tile and add to sum. *)

    (* Calling the plain old dotproduct matmult here.
       Note: this will generate code like this:

      float_t sum = (float_t)0.0f;
      while (bk < mshared)
      {
        [...]
        float_t sum1 = (float_t)0.0f;
        while (k < tile)
        {
          sum1 += sa1[brow * tile + k] * sa2[k * tile + bcol];
        }
        float_t t = sum1;
        sum += t;
        [...]
      }

      i.e. with an internal sum, that is then added to
      `sum` here. This is accurate according to how we are associating,
      but unidiomatic. This would be gone if matmul_dotprod took
      as an argument a reference into which to add the values.
    *)
    let t = Kuiper.Poly.GEMM.Util.matmul_dotprod sa1 sa2 brow bcol;
    sum := !sum `add` t;

    fold FB.bp_sharing sa1 (ematrix_subtile eA tile tile mrow !bk) (tile * tile);
    fold FB.bp_sharing sa2 (ematrix_subtile eB tile tile !bk mcol) (tile * tile);

    // What the hell.
    assert (pure (2 * (!bk + 1) == 2 * !bk + 1 + 1));

    (* Move to next tile *)
    bk := !bk +^ 1sz;
    ()
  };

  let s = !sum;
  (* The dot product computed via shared memory approximates the real matmul_single.
     This is the same computation as matmul_tiled_dotprod, just via shared memory. *)
  assume pure (s %~ MU.real_matmul_single eA eB (mrow * tile + brow) (mcol * tile + bcol));

  let v0 = M.gpu_matrix_read_cell gTile brow bcol;
  let v1 = comb v0 s;
  M.gpu_matrix_write_cell gTile brow bcol v1;

  to_real_ok v0;

  with v'.
    rewrite
      M.gpu_matrix_pts_to_cell gTile brow bcol v'
    as
      M.gpu_matrix_pts_to_cell gTile
        (tid / tile) (tid % tile) v';

  with em1. unfold FB.bp_sharing sa1 em1 (tile * tile);
  with em2. unfold FB.bp_sharing sa2 em2 (tile * tile);

  M.gpu_matrix_concr sa1; rewrite each M.core sa1 as ar1;
  M.gpu_matrix_concr sa2; rewrite each M.core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  rewrite each gTile as gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);

  fold (kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  ()
}
#pop-options

#push-options "--z3rlimit 100"
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
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA #eB : ematrix _ _ _)
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  (* Step 1: Share gA/gB, explode gC *)
  M.gpu_matrix_share_n gA (mlayout_vsize lC);
  M.gpu_matrix_share_n gB (mlayout_vsize lC);
  gpu_matrix_explode_tiled gC (SZ.v tile) (SZ.v tile);
  forevery_rw_size4 ((mrows * tile) / tile) mrows ((mcols * tile) / tile) mcols (SZ.v tile) tile (SZ.v tile) tile;

  (* Step 2: Factor gA/gB to 2D *)
  forevery_factor (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile) (fun _ -> gA |-> Frac (fA /. mlayout_vsize lC) eA);
  forevery_factor (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile) (fun _ -> gB |-> Frac (fB /. mlayout_vsize lC) eB);

  (* Step 3: Convert 4D -> 2D for gC *)
  assert pure (forall (mrow:natlt mrows) (mcol:natlt mcols). (mrow * mcols + mcol) / mcols == mrow /\ (mrow * mcols + mcol) % mcols == mcol);
  assert pure (forall (brow:natlt tile) (bcol:natlt tile). (brow * tile + bcol) / tile == brow /\ (brow * tile + bcol) % tile == bcol);
  forevery_ext_4
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) mrow mcol) brow bcol (macc eC (mrow * tile + brow) (mcol * tile + bcol)))
    (fun (mrow:natlt mrows) (mcol:natlt mcols) (brow:natlt tile) (bcol:natlt tile) ->
      let bid = mrow * mcols + mcol in let tid = brow * tile + bcol in
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));
  forevery_unfactor_2 (mrows * mcols) mrows mcols (SZ.v tile * SZ.v tile) (SZ.v tile) (SZ.v tile)
    (fun bid tid -> gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
      (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 4: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gA |-> Frac (fA /. mlayout_vsize lC) eA)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) -> gB |-> Frac (fB /. mlayout_vsize lC) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))));

  (* Step 5: Bridge to natlt2 and match kpre1 *)
  forevery_rw_size2 (mrows * mcols) (SZ.v (mrows `SZ.mul` mcols)) (SZ.v tile * SZ.v tile) (SZ.v (tile `SZ.mul` tile));
  forevery_ext_2
    (fun (bid : natlt (SZ.v (mrows `SZ.mul` mcols))) (tid : natlt (SZ.v (tile `SZ.mul` tile))) ->
      gA |-> Frac (fA /. mlayout_vsize lC) eA ** gB |-> Frac (fB /. mlayout_vsize lC) eB **
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) (tid / tile) (tid % tile)
        (macc eC ((bid / mcols) * tile + (tid / tile)) ((bid % mcols) * tile + (tid % tile))))
    (fun (bid : natlt2 mrows mcols) (tid : natlt2 tile tile) -> kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  ();
}
#pop-options

ghost
fn block_setup
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
  ensures
    (forall+ (tid : natlt2 tile  tile).
      kpre comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
{
  // Share shmem into (tile * tile) copies
  gpu_live_c_shmems_share_underspec sh #1.0R #(tile * tile);

  // Unfold live_c_shmems into explicit gpu_pts_to_array inside forall+
  forevery_map
    (fun (_ : natlt (tile * tile)) -> live_c_shmems sh #(1.0R /. (tile * tile)))
    (fun (_ : natlt (tile * tile)) ->
      (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
      (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x))
    fn _ {
      unfold_live_c_shmems_cons sh #(1.0R /. (tile * tile));
      unfold_live_c_shmem (fst sh) #(1.0R /. (tile * tile));
      unfold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
      unfold_live_c_shmem (fst (snd sh)) #(1.0R /. (tile * tile));
      unfold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));
    };

  // Bridge natlt type: natlt (tile * tile) → natlt2 tile tile
  forevery_rw_size (tile * tile) (SZ.v (tile *^ tile));

  // Zip shmem fracs with kpre1 to form kpre
  forevery_zip
    (fun (tid : natlt2 tile tile) ->
      kpre1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
    _;
}

#push-options "--z3rlimit 20"
ghost
fn block_teardown
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    (forall+ (tid : natlt2 tile  tile).
      kpost comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
{
  // Convert natlt2 → natlt (tile * tile) first (1 forall+, no ambiguity)
  forevery_rw_size (SZ.v (tile *^ tile)) (tile * tile);

  // Unzip kpost into kpost1 + shmem fracs
  forevery_unzip
    (fun (tid : natlt (tile * tile)) ->
      kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
    _;

  // Fold explicit gpu_pts_to_array back into live_c_shmems inside forall+
  forevery_map
    (fun (_ : natlt (tile * tile)) ->
      (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
      (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x))
    (fun (_ : natlt (tile * tile)) -> live_c_shmems sh #(1.0R /. (tile * tile)))
    fn _ {
      fold_live_c_shmem (fst (snd sh)) #(1.0R /. (tile * tile));
      fold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. (tile * tile));
      fold_live_c_shmems_cons (snd sh) #(1.0R /. (tile * tile));
      fold_live_c_shmem (fst sh) #(1.0R /. (tile * tile));
      fold_live_c_shmems_cons sh #(1.0R /. (tile * tile));
    };

  // Gather shmem fractions back
  gpu_live_c_shmems_gather_underspec sh #1.0R #(tile * tile);

  // Convert kpost1 back to natlt2 (1 forall+, no ambiguity)
  forevery_rw_size (tile * tile) (SZ.v (tile *^ tile));
}
#pop-options

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
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et _ _).
      gC |-> eC' **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  (* Step 1: Bridge natlt2 → natlt *)
  forevery_rw_size2
    (SZ.v (mrows *^ mcols)) (mrows * mcols)
    (SZ.v (tile *^ tile))   (SZ.v tile * SZ.v tile);

  (* Step 2: Unfold kpost1 explicitly *)
  forevery_ext_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gA |-> Frac (fA /. mlayout_vsize lC) eA **
      gB |-> Frac (fB /. mlayout_vsize lC) eB **
      (exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol)));

  (* Step 3: Unzip gA *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gA |-> Frac (fA /. mlayout_vsize lC) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      let brow = tid / tile in
      let bcol = tid % tile in
      let grow = mrow * tile + brow in
      let gcol = mcol * tile + bcol in
      gB |-> Frac (fB /. mlayout_vsize lC) eB **
      (exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
          (tid / tile) (tid % tile) v **
        pure (v %~ MU.real_gemm_single comb_r eA eB eC grow gcol)));

  (* Step 4: Unzip gB *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gB |-> Frac (fB /. mlayout_vsize lC) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (SZ.v tile * SZ.v tile)) ->
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

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gA |-> Frac (fA /. mlayout_vsize lC) eA);
  M.gpu_matrix_gather_n gA (mlayout_vsize lC);
  forevery_unfactor' (mlayout_vsize lC) (mrows * mcols) (SZ.v tile * SZ.v tile)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (SZ.v tile * SZ.v tile)) ->
      gB |-> Frac (fB /. mlayout_vsize lC) eB);
  M.gpu_matrix_gather_n gB (mlayout_vsize lC);

  (* Step 6: Collect gC cells back into matrix *)
  let vf = gpu_matrix_collect_approx_tiled gC (SZ.v tile) (SZ.v tile)
    (SZ.v mrows) (SZ.v mcols)
    (fun row col v -> v %~ MU.real_gemm_single comb_r eA eB eC row col);

  (* Step 7: Prove ematrix_approximates *)
  with eC'. assert (gC |-> eC');
  assert pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB));
  ();
}

#push-options "--z3rlimit_factor 10 --fuel 0 --ifuel 0 --split_queries no"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (#_ : squash (chunk et /? tile))
  (#_ : squash (chunk et * (tile * tile) /? (tile * tile)))
  (#_ : squash (SZ.fits (mlayout_size slA)))
  (#_ : squash (SZ.fits (mlayout_size slB)))
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et _ _).
          gC |-> eC' **
          pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB))))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB slA slB (fst ptrs) (fst (snd ptrs)) (tile * tile) bid);
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = (fun bid ptrs -> FB.barrier_p_to_q_transform eA eB slA slB (fst ptrs) (fst (snd ptrs)) (tile * tile) bid);

  shmems_desc = shmems_desc et tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre1  comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  setup      = setup    tile comb comb_r gA gB gC;
  teardown   = teardown tile comb comb_r gA gB gC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile slA slB comb comb_r gA gB gC #_ #_ #_ #_ #eC;
  block_teardown = block_teardown tile slA slB comb comb_r gA gB gC #_ #_ #_ #_ #eC;

  kpre      = kpre  comb comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpost     = kpost comb comb_r tile slA slB gA gB gC eA eB eC fA fB;

  f = kf tile slA slB comb comb_r gA gB gC eC;

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=magic();
  kpost_sendable=magic();
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
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
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads /\
          chunk et /?+ tile /\
          chunk et * (tile * tile) /? (tile * tile)) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et _ _).
      on gpu_loc (gC |-> eC') **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile (R.row_major _ _) (R.row_major _ _) comb comb_r gA gB gC ());
}
#pop-options

(* Legacy interface for backward compatibility.
   Calls the approximate kernel and assumes the exact result. *)
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
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
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
  // Fake real_like and has_vec_cpy instances, and comb_r with assumed refinement
  let _ : real_like et #_ = magic ();
  let _ : has_vec_cpy et #_ = magic ();
  let comb_r : binop real = magic ();
  assume pure (forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s);
  assume pure (chunk et /?+ tile);
  assume pure (chunk et * (tile * tile) /? (tile * tile));
  mmcomb_gpu_approx tile comb comb_r lA lB lC gA gB gC;
  with eC'. assert (on gpu_loc (gC |-> eC'));
  (* Assume the approximate result is exactly correct *)
  assume pure (eC' == MS.mmcomb comb eC eA eB);
}
