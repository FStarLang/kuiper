module Kuiper.GEMM.TensorCore
#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = FStar.SizeT

open Kuiper.Poly.GEMM.TensorCore

inline_for_extraction noextract
fn specialize_gpu
  // specialize
  (et_ab et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (bm bn bk : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  // should be up here! if part of the precondition, then
  //  the value is not checked for correctness when
  //  the function is only partially applied!
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm * bk)))
  (#_ : squash (SZ.fits (bk * bn)))
  (#_ : squash (bm/tm * bn/tn * warp_size <= max_threads))
  (#_ : squash (SZ.fits (bm*bk + bm/tm * bn/tn * warp_size)))
  (#_ : squash (SZ.fits (bk*bn + bm/tm * bn/tn * warp_size)))
  (rA rB rC : mrepr)
  {| ca : crepr rA, cB : crepr rB, cC : crepr rC |}

  // do not specialize
  (rows shared cols : szp)
  (gA : gpu_matrix et_ab (rA rows shared))
  (gB : gpu_matrix et_ab (rB shared cols))
  (gC : gpu_matrix et_c (row_major rows cols))
  (#eA : ematrix et_ab rows shared)
  (#eB : ematrix et_ab shared cols)
  (#eC : ematrix et_c rows cols)
  (#fA #fB : perm)
  // non of these are are checked because the functions is only
  //  partially applied
  preserves
    cpu **
    // should be checked at runtime
    pure (rows * cols <= max_blocks) **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    gC |-> eC
  ensures
    (exists* eC'. gC |-> eC')
{
  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;


  // All assumes should be dynamically checked.
  // odd constraints, required for the implementation of copy
  // (we stride through a tile with all threads and in the last iteration the iteration variable may go up to (tile_size + nthr-1))
  assume (pure (SZ.fits (rows * shared)));
  assume (pure (SZ.fits (shared * cols)));
  assume (pure (SZ.fits (rows * cols)));

  // preconditions checcked at runtime
  // TODO should be checked at runtime but has ghost effect:
  //  dguard (SZ.lte (rows *^ cols) (SZ.uint_to_t max_blocks));
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);

  let nblk = rows/^bm *^ (cols/^bn);
  let nthr = bm/^tm *^ (bn/^tn) *^ warp_sz;
  launch_sync (
    mk_kernel gA gB gC bm bn bk tm tn tk nblk nthr ()
  );

  ()
}
let g_gemm_f16_f16_64x64x16_16x16x16_rrr = specialize_gpu half half 64sz 64sz 16sz 16sz 16sz 16sz row_major row_major row_major

let g_gemm_f16_f16_32x32x32_32x8x16_rrr = specialize_gpu half half 32sz 32sz 32sz 32sz 8sz 16sz row_major row_major row_major
let g_gemm_f16_f16_32x32x32_8x32x16_rrr = specialize_gpu half half 32sz 32sz 32sz 8sz 32sz 16sz row_major row_major row_major

let g_gemm_f16_f16_32x8x16_32x8x16_rrr = specialize_gpu half half 32sz 8sz 16sz 32sz 8sz 16sz row_major row_major row_major

let g_gemm_f16_f16_8x32x16_8x32x16_rrr = specialize_gpu half half 32sz 32sz 32sz 8sz 32sz 16sz row_major row_major row_major

// // These instances are tested.
let g_gemm_f16_f16_64x64x64_16x16x16_rrr = specialize_gpu half half 64sz 64sz 64sz 16sz 16sz 16sz row_major row_major row_major
let g_gemm_f16_f16_64x64x64_32x8x16_rrr = specialize_gpu half half 64sz 64sz 64sz 32sz 8sz 16sz row_major row_major row_major
let g_gemm_f16_f16_64x64x64_8x32x16_rrr = specialize_gpu half half 64sz 64sz 64sz 8sz 32sz 16sz row_major row_major row_major

let g_gemm_f16_f16_32x32x32_16x16x16_rrr = specialize_gpu half half 32sz 32sz 32sz 16sz 16sz 16sz row_major row_major row_major

let g_gemm_f16_f16_16x16x16_16x16x16_rrr = specialize_gpu half half 16sz 16sz 16sz 16sz 16sz 16sz row_major row_major row_major

// mixed precision
let g_gemm_f16_f32_32x32x32_16x16x16_rrr = specialize_gpu half float 32sz 32sz 32sz 16sz 16sz 16sz row_major row_major row_major
