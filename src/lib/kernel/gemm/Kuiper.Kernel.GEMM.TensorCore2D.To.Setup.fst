module Kuiper.Kernel.GEMM.TensorCore2D.To.Setup

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy }
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

inline_for_extraction noextract
fn setup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#shared_fits : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#scratch_fits : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#warp_div : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (tid : szlt nthr)
  requires
    gpu **
    shared_thread_live bm bn bk tm tn nthr sh tid
  returns s :
    (array2 et_ab (rm bm bk) & array2 et_ab (rm bk bn))
  ensures
    pure (core s._1 == fst sh /\ core s._2 == fst (snd sh)) **
    gpu **
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    (exists* em1. FB.bp_sharing s._1 em1 nthr) **
    (exists* em2. FB.bp_sharing s._2 em2 nthr)
{
  unfold shared_thread_live bm bn bk tm tn nthr sh tid;
  let (sarA, (sarB, (sarAcc, sarTail))) = sh;
  unfold live_c_shmem sarA #(1.0R /. nthr);
  unfold live_c_shmem sarB #(1.0R /. nthr);
  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  tensor_abs' (rm bm bk) sarA;
  let sA = from_array (rm bm bk) sarA;
  rewrite each _ as sA;
  tensor_abs' (rm bk bn) sarB;
  let sB = from_array (rm bk bn) sarB;
  rewrite each _ as sB;

  rewrite
    (exists* (x : chest2 _ _ _). sA |-> Frac (1.0R /. nthr) x) **
    (exists* (x : chest2 _ _ _). sB |-> Frac (1.0R /. nthr) x)
  as
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr);
  rewrite
    scratch_tile_live bm bn bk tm tn nthr
      (sarA, (sarB, (sarAcc, sarTail))) tid
  as
    scratch_tile_live bm bn bk tm tn nthr sh tid;
  (sA, sB)
}
