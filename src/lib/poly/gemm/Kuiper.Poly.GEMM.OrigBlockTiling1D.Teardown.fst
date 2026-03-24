module Kuiper.Poly.GEMM.OrigBlockTiling1D.Teardown

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.Approximates
open Kuiper.EMatrix
open Kuiper.Matrix { gpu_matrix, gpu_matrix_pts_to, gpu_matrix_pts_to_cell }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

module M = Kuiper.Matrix
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.OrigBlockTiling1D.Defs

#push-options "--z3rlimit 80 --fuel 1 --ifuel 1"
ghost
fn block_teardown
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
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
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ bn)).
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
{
  // Bridge from SizeT-based to nat-based size
  forevery_rw_size (bm/^tm *^ bn) (bm/tm * bn);

  // Split kpost (= kpost1 ** shmemA ** shmemB) into three components
  forevery_unzip3
    (fun (tid : natlt (bm/tm * bn)) -> kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x)
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x);

  // Fold each shmem buffer into live_c_shmem, then gather
  forevery_map
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * bn)) x)
    (fun (_ : natlt (bm/tm * bn)) -> live_c_shmem (fst sh) #(1.0R /. (bm/tm * bn)))
    fn _ { fold_live_c_shmem (fst sh) #(1.0R /. (bm/tm * bn)) };
  gpu_live_c_shmem_gather_underspec (fst sh) #1.0R #(bm/tm * bn);

  forevery_map
    (fun (_ : natlt (bm/tm * bn)) -> exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * bn)) x)
    (fun (_ : natlt (bm/tm * bn)) -> live_c_shmem (fst (snd sh)) #(1.0R /. (bm/tm * bn)))
    fn _ { fold_live_c_shmem (fst (snd sh)) #(1.0R /. (bm/tm * bn)) };
  gpu_live_c_shmem_gather_underspec (fst (snd sh)) #1.0R #(bm/tm * bn);

  // Combine into live_c_shmems
  fold_live_c_shmems_nil (snd (snd sh)) #1.0R;
  fold_live_c_shmems_cons (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;

  // Bridge back from nat-based to SizeT-based size
  forevery_rw_size (bm/tm * bn) (bm/^tm *^ bn);
}
#pop-options

#push-options "--z3rlimit 120 --fuel 1 --ifuel 1"
ghost
fn teardown
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
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
    (forall+ (bid : natlt (mrows *^ mcols))
             (tid : natlt (bm /^ tm *^ bn)).
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et _ _).
      gC |-> eC' **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  let n_threads = (mrows * mcols) * (bm/tm * bn);

  (* Step 1: Bridge from SizeT to nat *)
  forevery_rw_size2
    (mrows *^ mcols) (mrows * mcols)
    (bm /^ tm *^ bn) (bm/tm * bn);

  (* Step 2: Unfold kpost1 *)
  forevery_ext_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (mrow * bm + (tid / bn * tm) + i)
                  (mcol * bn + (tid % bn))));

  (* Step 3: Unzip gA *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (mrow * bm + (tid / bn * tm) + i)
                  (mcol * bn + (tid % bn))));

  (* Step 4: Unzip gB *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt (bm/tm * bn)) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      forall+ (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (mrow * bm + (tid / bn * tm) + i)
                  (mcol * bn + (tid % bn))));

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' n_threads (mrows * mcols) (bm/tm * bn)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gA |-> Frac (fA /. n_threads) eA);
  M.gpu_matrix_gather_n gA n_threads;

  forevery_unfactor' n_threads (mrows * mcols) (bm/tm * bn)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt (bm/tm * bn)) ->
      gB |-> Frac (fB /. n_threads) eB);
  M.gpu_matrix_gather_n gB n_threads;

  (* Step 6: Rearrange (bid, tid, i) → (bid, flatid) *)
  forevery_map
    (fun (bid : natlt (mrows * mcols)) ->
      forall+ (tid : natlt (bm/tm * bn)) (i : natlt tm).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            ((tid / bn * tm) + i) (tid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (bid / mcols * bm + (tid / bn * tm) + i)
                  (bid % mcols * bn + (tid % bn))))
    (fun (bid : natlt (mrows * mcols)) ->
      forall+ (flatid : natlt (bm * bn)).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
            (flatid / bn) (flatid % bn) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
                  (bid / mcols * bm + flatid / bn)
                  (bid % mcols * bn + flatid % bn)))
    fn bid {
      forevery_factor' (bm/tm * bn) (bm/tm) bn
        (fun (threadRow : natlt (bm/tm)) (c : natlt bn) ->
          forall+ (i : natlt tm).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                (threadRow * tm + i) c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + (threadRow * tm) + i)
                      (bid % mcols * bn + c)));
      forevery_mid_flip
        (fun (threadRow : natlt (bm/tm)) (c : natlt bn) (i : natlt tm) ->
          exists* (v : et).
            gpu_matrix_pts_to_cell
              (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
              (threadRow * tm + i) c v **
            pure (v %~ MU.real_gemm_single comb_r eA eB eC
                    (bid / mcols * bm + (threadRow * tm) + i)
                    (bid % mcols * bn + c)));
      // Re-associate addition: (a + b) + c → a + (b + c)
      forevery_map_2
        (fun (threadRow : natlt (bm/tm)) (i : natlt tm) ->
          forall+ (c : natlt bn).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                (threadRow * tm + i) c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + (threadRow * tm) + i)
                      (bid % mcols * bn + c)))
        (fun (threadRow : natlt (bm/tm)) (i : natlt tm) ->
          forall+ (c : natlt bn).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                (threadRow * tm + i) c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + (threadRow * tm + i))
                      (bid % mcols * bn + c)))
        fn threadRow i {
          assert pure (bid / mcols * bm + (threadRow * tm) + i == bid / mcols * bm + (threadRow * tm + i));
          forevery_map
            (fun (c : natlt bn) ->
              exists* (v : et).
                gpu_matrix_pts_to_cell
                  (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  (threadRow * tm + i) c v **
                pure (v %~ MU.real_gemm_single comb_r eA eB eC
                        (bid / mcols * bm + (threadRow * tm) + i)
                        (bid % mcols * bn + c)))
            (fun (c : natlt bn) ->
              exists* (v : et).
                gpu_matrix_pts_to_cell
                  (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                  (threadRow * tm + i) c v **
                pure (v %~ MU.real_gemm_single comb_r eA eB eC
                        (bid / mcols * bm + (threadRow * tm + i))
                        (bid % mcols * bn + c)))
            fn c {
              ();
            };
        };
      forevery_unfactor bm (bm/tm) tm
        (fun (r : natlt bm) ->
          forall+ (c : natlt bn).
            exists* (v : et).
              gpu_matrix_pts_to_cell
                (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
                r c v **
              pure (v %~ MU.real_gemm_single comb_r eA eB eC
                      (bid / mcols * bm + r)
                      (bid % mcols * bn + c)));
      forevery_unfactor' (bm * bn) bm bn
        (fun (r : natlt bm) (c : natlt bn) ->
          exists* (v : et).
            gpu_matrix_pts_to_cell
              (gpu_matrix_subtile gC (SZ.v bm) (SZ.v bn) (bid / mcols) (bid % mcols))
              r c v **
            pure (v %~ MU.real_gemm_single comb_r eA eB eC
                    (bid / mcols * bm + r)
                    (bid % mcols * bn + c)));
    };

  (* Step 7: Collect cells back into matrix *)
  let _ = gpu_matrix_collect_approx_tiled gC (SZ.v bm) (SZ.v bn)
    mrows mcols
    (fun (row : natlt (mrows * bm)) (col : natlt (mcols * bn)) (v : et) ->
      v %~ MU.real_gemm_single comb_r eA eB eC row col);

  (* Step 8: Prove ematrix_approximates *)
  with eC'. assert (gC |-> eC');

  assert pure (forall (row:natlt (mrows * bm)) (col:natlt (mcols * bn)).
    macc eC' row col %~ MU.real_gemm_single comb_r eA eB eC row col);

  assert pure (forall (row:natlt (mrows * bm)) (col:natlt (mcols * bn)).
    macc eC' row col %~ macc (MU.real_mmcomb comb_r eC eA eB) row col);

  assert pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB));
  ();
}
#pop-options
