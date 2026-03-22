module Kuiper.Poly.GEMM.OrigBlockTiling1D

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.Approximates
open Kuiper.EMatrix
open Kuiper.Matrix { gpu_matrix }
open Kuiper.Matrix.Reprs.Type

module B = Kuiper.Barrier
module M = Kuiper.Matrix
module MU = Kuiper.Poly.GEMM.Util
module R = Kuiper.Matrix.Reprs
module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.OrigBlockTiling1D.Defs
open Kuiper.Poly.GEMM.OrigBlockTiling1D.Barrier
open Kuiper.Poly.GEMM.OrigBlockTiling1D.Kf
open Kuiper.Poly.GEMM.OrigBlockTiling1D.Setup
open Kuiper.Poly.GEMM.OrigBlockTiling1D.Teardown

#push-options "--z3rlimit 80"
let kpre_block_sendable
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
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (_: squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn))
: is_send_across block_of (kpre comb tm slA slB gA gB gC eA eB eC fA fB sh bid tid)
= magic()

let kpost_block_sendable
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
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (_: squash (c_shmems_inv sh))
  (bid : natlt (mrows * mcols))
  (tid : natlt (bm/tm * bn))
: is_send_across block_of (kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB sh bid tid)
= magic()
#pop-options

#push-options "--z3rlimit 80"

let block_pre_gpu_sendable
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt (bm/tm * bn)).
      kpre1 comb tm gA gB gC eA eB eC fA fB bid tid)
= magic()

let block_post_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA : ematrix et (mrows   * bm) (mshared * bk))
  (eB : ematrix et (mshared * bk) (mcols   * bn))
  (eC : ematrix et (mrows   * bm) (mcols   * bn))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
: is_send_across gpu_of
    (forall+ (tid : natlt (bm/tm * bn)).
      kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid)
= magic()
#pop-options

#push-options "--fuel 2 --ifuel 2 --z3rlimit 80"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#bm #bn #bk : szp)
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
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  (_ : squash (mrows * mcols <= max_blocks
               /\ (bm/tm * bn) <= max_threads))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et _ _).
          gC |-> eC' **
          pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB))))
= {
  nblk = mrows *^ mcols; //SZ.uint_to_t (SZ.v mrows * SZ.v mcols);
  nthr = (bm /^ tm *^ bn);

  shmems_desc = shmems_desc et bm bn bk;

  barrier_contract = (fun bid ptrs -> barrier_contract tm eA eB bid (M.from_array slA (fst ptrs)) (M.from_array slB (fst (snd ptrs))));
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = (fun bid ptrs -> barrier_p_to_q_transform tm eA eB bid (M.from_array slA (fst ptrs)) (M.from_array slB (fst (snd ptrs))));

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpre1  comb tm gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt (bm/^tm *^ bn)). kpost1 comb comb_r tm gA gB gC eA eB eC fA fB bid tid);
  setup      = setup    comb tm gA gB gC #eA #eB #eC;
  teardown   = teardown comb comb_r tm gA gB gC #eA #eB #eC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    =  block_setup    comb tm slA slB gA gB gC #eA #eB #eC;
  block_teardown = block_teardown comb comb_r tm slA slB gA gB gC #eA #eB #eC;

  kpre      = kpre  comb tm slA slB gA gB gC eA eB eC fA fB;
  kpost     = kpost comb comb_r tm slA slB gA gB gC eA eB eC fA fB;

  f = kf comb comb_r tm slA slB gA gB gC #fA #fB #eA #eB;

  block_pre_sendable = block_pre_gpu_sendable comb tm gA gB gC eA eB eC fA fB;
  block_post_sendable = block_post_gpu_sendable comb comb_r tm gA gB gC eA eB eC fA fB;
  kpre_sendable = kpre_block_sendable comb tm slA slB gA gB gC eA eB eC fA fB;
  kpost_sendable = kpost_block_sendable comb comb_r tm slA slB gA gB gC eA eB eC fA fB
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (bm bn bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et _ _).
      on gpu_loc (gC |-> eC') **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  launch_sync (mk_kernel comb comb_r tm (R.row_major _ _) (R.row_major _ _) gA #fA gB #fB gC #eA #eB #eC ());
}
