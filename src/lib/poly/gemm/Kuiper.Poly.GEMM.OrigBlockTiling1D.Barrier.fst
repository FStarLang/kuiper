module Kuiper.Poly.GEMM.OrigBlockTiling1D.Barrier

#lang-pulse

#set-options "--z3rlimit 80 --fuel 0 --ifuel 0"

open Kuiper
open Kuiper.EMatrix
open Kuiper.Math { even, odd }
open Kuiper.Matrix { gpu_matrix, gpu_matrix_pts_to, gpu_matrix_pts_to_cell }
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

module M = Kuiper.Matrix
module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.Copy { live_cell }
open Kuiper.Poly.GEMM.OrigBlockTiling1D.Defs

ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| scalar et |}
  (#bm #bn #bk : szp)
  (tm : szp{tm `divides` bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#mrows #mshared #mcols : pos)
  (eA : ematrix et (mrows * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols * bn))
  (bid : natlt (mrows * mcols))
  (#l1 : full_mlayout bm bk)
  (#l2 : full_mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (#_ : squash (SZ.fits (mlayout_size l1)))
  (#_ : squash (SZ.fits (mlayout_size l2)))
  (it : nat)
  requires
    forall+ (tid : natlt (bm/tm * bn)).
      barrier_p tm eA eB bid m1 m2 it tid
  ensures
    forall+ (tid : natlt (bm/tm * bn)).
      barrier_q tm eA eB bid m1 m2 it tid
{
  if (it >= 2 * mshared) {
    forevery_map
      (fun (tid : natlt (bm/tm * bn)) -> barrier_p tm eA eB bid m1 m2 it tid)
      (fun (tid : natlt (bm/tm * bn)) -> barrier_q tm eA eB bid m1 m2 it tid)
      fn tid {
        rewrite barrier_p tm eA eB bid m1 m2 it tid as emp;
        rewrite emp as barrier_q tm eA eB bid m1 m2 it tid;
      };
  } else {
    let ev = even it;
    if ev {
      assert pure (it < 2 * mshared);
      assert pure (even it);
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> barrier_p tm eA eB bid m1 m2 it tid)
        (fun (tid : natlt (bm/tm * bn)) ->
          (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (bm/tm * bn)) x) **
          (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (bm/tm * bn)) x))
        fn tid {
          rewrite barrier_p tm eA eB bid m1 m2 it tid
               as (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (bm/tm * bn)) x) **
                  (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (bm/tm * bn)) x);
        };
      forevery_unzip _ _;
      M.gpu_matrix_gather_n_underspec m1 (bm/tm * bn);
      with em1. assert m1 |-> em1;
      M.gpu_matrix_explode m1;
      forevery_unfactor' (bm/tm * bn) bm bk
        (fun r c -> gpu_matrix_pts_to_cell m1 r c (macc em1 r c));
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> gpu_matrix_pts_to_cell m1 (tid/bk) (tid%bk) (macc em1 (tid/bk) (tid%bk)))
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m1 (tid/bk) (tid%bk))
        fn tid { fold (live_cell m1 (tid/bk) (tid%bk)) };
      M.gpu_matrix_gather_n_underspec m2 (bm/tm * bn);
      with em2. assert m2 |-> em2;
      M.gpu_matrix_explode m2;
      forevery_unfactor' (bm/tm * bn) bk bn
        (fun r c -> gpu_matrix_pts_to_cell m2 r c (macc em2 r c));
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> gpu_matrix_pts_to_cell m2 (tid/bn) (tid%bn) (macc em2 (tid/bn) (tid%bn)))
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m2 (tid/bn) (tid%bn))
        fn tid { fold (live_cell m2 (tid/bn) (tid%bn)) };
      forevery_zip
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m1 (tid/bk) (tid%bk)) _;
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> live_cell m1 (tid/bk) (tid%bk) ** live_cell m2 (tid/bn) (tid%bn))
        (fun (tid : natlt (bm/tm * bn)) -> barrier_q tm eA eB bid m1 m2 it tid)
        fn tid {
          rewrite live_cell m1 (tid/bk) (tid%bk) ** live_cell m2 (tid/bn) (tid%bn)
               as barrier_q tm eA eB bid m1 m2 it tid;
        };
    } else {
      assert pure (it < 2 * mshared);
      assert pure (odd it);
      let mrow = bid / mcols;
      let mcol = bid % mcols;
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) -> barrier_p tm eA eB bid m1 m2 it tid)
        (fun (tid : natlt (bm/tm * bn)) ->
          gpu_matrix_pts_to_cell m1 (tid/bk) (tid%bk) (macc (ematrix_subtile eA bm bk mrow (it/2)) (tid/bk) (tid%bk)) **
          gpu_matrix_pts_to_cell m2 (tid/bn) (tid%bn) (macc (ematrix_subtile eB bk bn (it/2) mcol) (tid/bn) (tid%bn)))
        fn tid {
          rewrite barrier_p tm eA eB bid m1 m2 it tid
               as gpu_matrix_pts_to_cell m1 (tid/bk) (tid%bk) (macc (ematrix_subtile eA bm bk mrow (it/2)) (tid/bk) (tid%bk)) **
                  gpu_matrix_pts_to_cell m2 (tid/bn) (tid%bn) (macc (ematrix_subtile eB bk bn (it/2) mcol) (tid/bn) (tid%bn));
        };
      forevery_unzip _ _;
      forevery_factor' (bm/tm * bn) bm bk
        (fun r c -> gpu_matrix_pts_to_cell m1 r c (macc (ematrix_subtile eA bm bk mrow (it/2)) r c));
      M.gpu_matrix_implode m1;
      M.gpu_matrix_share_n m1 (bm/tm * bn);
      forevery_factor' (bm/tm * bn) bk bn
        (fun r c -> gpu_matrix_pts_to_cell m2 r c (macc (ematrix_subtile eB bk bn (it/2) mcol) r c));
      M.gpu_matrix_implode m2;
      M.gpu_matrix_share_n m2 (bm/tm * bn);
      forevery_zip
        (fun (_ : natlt (bm/tm * bn)) -> m1 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow (it/2))) _;
      forevery_map
        (fun (tid : natlt (bm/tm * bn)) ->
          m1 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow (it/2)) **
          m2 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eB bk bn (it/2) mcol))
        (fun (tid : natlt (bm/tm * bn)) -> barrier_q tm eA eB bid m1 m2 it tid)
        fn tid {
          rewrite
            m1 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eA bm bk mrow (it/2)) **
            m2 |-> Frac (1.0R /. (bm/tm * bn)) (ematrix_subtile eB bk bn (it/2) mcol)
          as
            barrier_q tm eA eB bid m1 m2 it tid;
        };
    }
  }
}
