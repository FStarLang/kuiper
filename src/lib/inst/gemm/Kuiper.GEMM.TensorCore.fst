module Kuiper.GEMM.TensorCore

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU {
  specialize_as_gemm_to_type_and_reprs_gpu as spec_gemm_gpu,
}
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = FStar.SizeT

open Kuiper.Poly.GEMM.TensorCore

#push-options "--debug SMTFail --split_queries always"
inline_for_extraction noextract
fn specialize_gpu
  // specialize
  (et_ab et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (bm bn bk : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (rA rB rC : mrepr)
  {| ca : crepr rA, cB : crepr rB, cC : crepr rC |}
  
  // do not specialize
  // meaning that constraints must be checked dynamically (not enforced)
  (rows shared cols : szp)
  (gA : gpu_matrix et_ab (rA rows shared))
  (gB : gpu_matrix et_ab (rB shared cols))
  (gC : gpu_matrix et_c (row_major rows cols))
  (#eA : ematrix et_ab rows shared)
  (#eB : ematrix et_ab shared cols)
  (#eC : ematrix et_c rows cols)
  (#fA #fB : perm)
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    pure (valid_frag_et_dims et_ab FragA tm tn tk) **
    pure (valid_frag_et_dims et_ab FragB tm tn tk) **
    pure (valid_frag_et_dims et_c FragAcc tm tn tk) **
    pure (valid_frag_et_comb et_ab et_c) **
    pure (rows * cols <= max_blocks) **
    pure (bm/tm * bn/tn * warp_size <= max_threads) **
    pure (SZ.fits (rows * cols)) **
    pure (SZ.fits (bm * bk)) **
    pure (SZ.fits (bk * bn))
  requires
    gC |-> eC
  ensures
    (exists* eC'. gC |-> eC')
{
  // dassert (bm `SZ.gt` 0sz);
  // dassert (bn `SZ.gt` 0sz);
  // dassert (bk `SZ.gt` 0sz);
  // dassert (tm `SZ.gt` 0sz);

  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);

  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;

  // There is no way to prove this.
  assume (pure (SZ.fits (rows * shared)));
  assume (pure (SZ.fits (shared * cols)));

  // odd constraint, required for the implementation of copy
  // (we stride through a tile with all threads and in the last iteration the iteration variable may go up to (tile_size + nthr-1))
  let nthr: erased nat = bm/tm * (bn/tn) * warp_size;
  assume (pure (SZ.fits (bm*bk + nthr)));
  assume (pure (SZ.fits (bk*bn + nthr)));

  launch_sync (
    mk_kernel gA gB gC bm bn bk tm tn tk (rows/^bm *^ (cols/^bn)) (bm/^tm *^ (bn/^tn) *^ warp_sz) ()
  );

  ()
}
// Transposed A-tiles in shared memory
let g_gemm_f32_128x64x16_16x16 = specialize_gpu half half 128sz 64sz 16sz 16sz 16sz 16sz row_major row_major row_major
