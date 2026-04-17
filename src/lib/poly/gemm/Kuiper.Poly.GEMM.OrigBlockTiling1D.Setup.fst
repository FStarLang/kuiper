module Kuiper.Poly.GEMM.OrigBlockTiling1D.Setup

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix { gpu_matrix, gpu_matrix_pts_to, gpu_matrix_pts_to_cell }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

module M = Kuiper.Matrix
module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.OrigBlockTiling1D.Defs

#push-options "--z3rlimit 80"
ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  // because of how the original code loads into shmem,
  //  the following is required
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt (mrows *^ mcols))
             (tid : natlt (bm /^ tm *^ bn)).
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  let n_threads = (mrows * mcols) * (bm/tm * bn);

  (* Step 1: Share gA/gB, explode gC *)
  M.gpu_matrix_share_n gA n_threads;
  M.gpu_matrix_share_n gB n_threads;
  gpu_matrix_explode_tiled gC (SZ.v bm) (SZ.v bn);
  forevery_rw_size4
    ((mrows * bm) / bm) mrows
    ((mcols * bn) / bn) mcols
    (SZ.v bm) bm
    (SZ.v bn) bn;

  (* Step 2: Rearrange inner (r,c) → (tid,i) per tile *)
  forevery_map_2
    (fun (tr : natlt mrows) (tc : natlt mcols) ->
      forall+ (r : natlt bm) (c : natlt bn).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc) r c
          (macc eC (tr * bm + r) (tc * bn + c)))
    (fun (tr : natlt mrows) (tc : natlt mcols) ->
      forall+ (tid : natlt (bm/tm * bn)) (i : natlt tm).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC (tr * bm + ((tid / bn * tm) + i)) (tc * bn + (tid % bn))))
    fn tr tc {
      forevery_factor bm (bm/tm) tm
        (fun (r : natlt bm) ->
          forall+ (c : natlt bn).
            gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc) r c
              (macc eC (tr * bm + r) (tc * bn + c)));
      forevery_mid_flip
        (fun (threadRow : natlt (bm/tm)) (i : natlt tm) (c : natlt bn) ->
          gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
            (threadRow * tm + i) c
            (macc eC (tr * bm + (threadRow * tm + i)) (tc * bn + c)));
      forevery_unfactor' (bm/tm * bn) (bm/tm) bn
        (fun (threadRow : natlt (bm/tm)) (c : natlt bn) ->
          forall+ (i : natlt tm).
            gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
              (threadRow * tm + i) c
              (macc eC (tr * bm + (threadRow * tm + i)) (tc * bn + c)));
    };

  (* Step 3: Collapse (tr,tc) → bid *)
  forevery_unfactor' (mrows * mcols) mrows mcols
    (fun (tr : natlt mrows) (tc : natlt mcols) ->
      forall+ (tid : natlt (bm/tm * bn)) (i : natlt tm).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) tr tc)
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC (tr * bm + ((tid / bn * tm) + i)) (tc * bn + (tid % bn))));

  (* Step 4: Factor gA/gB to 2D *)
  forevery_factor n_threads (mrows * mcols) (bm/tm * bn)
    (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (mrows * mcols) (bm/tm * bn)
    (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  (* Step 5: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gA |-> Frac (fA /. n_threads) eA)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      forall+ (i : natlt tm).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                   ((bid % mcols) * bn + (tid % bn))));

  (* Step 6: Bridge to SizeT and match kpre1 *)
  #set-options "--z3rlimit 100" {
    forevery_rw_size2
      (mrows * mcols) (mrows *^ mcols)
      (bm/tm * bn) (bm /^ tm *^ bn);
  };
  forevery_map_2
    (fun (bid : natlt (mrows *^ mcols)) (tid : natlt (bm /^ tm *^ bn)) ->
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (i : natlt tm).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
          ((tid / bn * tm) + i) (tid % bn)
          (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                   ((bid % mcols) * bn + (tid % bn))))
    (fun (bid : natlt (mrows *^ mcols)) (tid : natlt (bm /^ tm *^ bn)) ->
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
    fn bid tid {
      forevery_map
        (fun (i : natlt tm) ->
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn)
            (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                     ((bid % mcols) * bn + (tid % bn))))
        (fun (i : natlt tm) ->
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn)
            (macc (ematrix_subtile eC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  ((tid / bn * tm) + i) (tid % bn)))
        fn i {
          rewrite each
            (macc eC ((bid / mcols) * bm + ((tid / bn * tm) + i))
                     ((bid % mcols) * bn + (tid % bn)))
          as
            (macc (ematrix_subtile eC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  ((tid / bn * tm) + i) (tid % bn));
        };
    };
  ();
}
#pop-options

#push-options "--z3rlimit 80"
ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (mrows *^ mcols))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
  ensures
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpre comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
{
  // Bridge from SizeT-based to nat-based size
  forevery_rw_size (bm/^tm *^ bn) (bm/tm * bn);

  // Share all shmem buffers across all threads
  gpu_live_c_shmems_share_underspec sh #1.0R #(bm/tm * bn);

  // Zip with kpre1 to form kpre
  forevery_zip
    (fun (tid : natlt (bm/tm * bn)) -> kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
    (fun (_ : natlt (bm/tm * bn)) -> live_c_shmems sh #(1.0R /. (bm/tm * bn)));

  // Bridge back from nat-based to SizeT-based size
  forevery_rw_size (bm/tm * bn) (bm/^tm *^ bn);
}
#pop-options
