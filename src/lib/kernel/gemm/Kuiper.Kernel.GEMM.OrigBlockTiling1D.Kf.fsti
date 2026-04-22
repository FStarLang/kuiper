module Kuiper.Kernel.GEMM.OrigBlockTiling1D.Kf

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix { gpu_matrix }
open Kuiper.Matrix.Reprs.Type

module B = Kuiper.Barrier
module M = Kuiper.Matrix
module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.OrigBlockTiling1D.Defs

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp{SZ.fits (bm * bk) /\ SZ.fits (bk * bn)})
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA : perm)
  (#fB : perm)
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (mrows * mcols))
  (tid : szlt (bm/tm * bn))
  ()
  norewrite
  requires
    gpu **
    kpre comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (bm/tm * bn) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (barrier_contract tm eA eB bid (M.from_array slA (fst sh)) (M.from_array slB (fst (snd sh)))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id (bm/tm * bn) tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (barrier_contract tm eA eB bid (M.from_array slA (fst sh)) (M.from_array slB (fst (snd sh)))) **
    B.barrier_state (2 * mshared)
