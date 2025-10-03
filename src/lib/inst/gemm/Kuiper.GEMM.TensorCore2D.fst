module Kuiper.GEMM.TensorCore2D
#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.TensorCore
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

module SZ = FStar.SizeT

open Kuiper.Poly.GEMM.TensorCore2D

#push-options "--split_queries always" // very slow otherwise?
inline_for_extraction noextract
fn specialize_gpu
  // specialize
  (et_ab et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /? bk))
  (#_ : squash (chunk et_ab /? bn))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_sz) -1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_sz) -1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_sz)) <= max_threads))

  // do not specialize
  (rows shared cols : szp)
  (gA : gpu_matrix et_ab (row_major rows shared))
  (gB : gpu_matrix et_ab (row_major shared cols))
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
  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dassert (bk %^ tk = 0sz);

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
  let nthr = bm/^(wm*^tm) *^ (bn/^(wn*^tn)) *^ warp_sz;

  launch_sync (
    mk_kernel gA gB gC bm bn bk tm tn tk wm wn nblk nthr ()
  );

  ()
}
#pop-options

let g_gemm_f16_f16_64x64x16_16x16x16_4x4 = specialize_gpu half half 64sz 64sz 16sz 16sz 16sz 16sz 4sz 4sz

let g_gemm_f16_f16_32x32x32_32x8x16_1x2 = specialize_gpu half half 32sz 32sz 32sz 32sz 8sz 16sz 1sz 2sz
let g_gemm_f16_f16_32x32x32_8x32x16_2x1 = specialize_gpu half half 32sz 32sz 32sz 8sz 32sz 16sz 2sz 1sz

// Are these as fast as the non-tiled tensor core implementation?
// 1 tensor core operation per warp
let g_gemm_f16_f16_32x8x16_32x8x16 = specialize_gpu half half 32sz 8sz 16sz 32sz 8sz 16sz 1sz 1sz
let g_gemm_f16_f16_8x32x16_8x32x16 = specialize_gpu half half 32sz 32sz 32sz 8sz 32sz 16sz 1sz 1sz
let g_gemm_f16_f16_16x16x16_16x16x16 = specialize_gpu half half 16sz 16sz 16sz 16sz 16sz 16sz 1sz 1sz

// 16 tensor core operations per warp
let g_gemm_f16_f16_64x64x64_16x16x16_4x4 = specialize_gpu half half 64sz 64sz 64sz 16sz 16sz 16sz 4sz 4sz
let g_gemm_f16_f16_64x64x64_32x8x16_2x8 = specialize_gpu half half 64sz 64sz 64sz 32sz 8sz 16sz 2sz 8sz
let g_gemm_f16_f16_64x64x64_8x32x16_8x2 = specialize_gpu half half 64sz 64sz 64sz 8sz 32sz 16sz 8sz 2sz
let g_gemm_f16_f16_128x128x32_16x16x16_4x4 = specialize_gpu half half 128sz 128sz 32sz 16sz 16sz 16sz 4sz 4sz
let g_gemm_f16_f16_128x128x64_16x16x16_4x4 = specialize_gpu half half 128sz 128sz 64sz 16sz 16sz 16sz 4sz 4sz

let g_gemm_f16_f16_128x128x32_16x16x16_8x8 = specialize_gpu half half 128sz 128sz 32sz 16sz 16sz 16sz 8sz 8sz
let g_gemm_f16_f16_128x128x64_16x16x16_8x8 = specialize_gpu half half 128sz 128sz 64sz 16sz 16sz 16sz 8sz 8sz

// 4 tensor core operations per warp
let g_gemm_f16_f16_32x32x32_16x16x16_2x2   = specialize_gpu half half 32sz  32sz  32sz 16sz 16sz 16sz 2sz 2sz
let g_gemm_f16_f16_64x64x64_16x16x16_2x2   = specialize_gpu half half 64sz  64sz  64sz 16sz 16sz 16sz 2sz 2sz
let g_gemm_f16_f16_128x128x32_16x16x16_2x2 = specialize_gpu half half 128sz 128sz 32sz 16sz 16sz 16sz 2sz 2sz
let g_gemm_f16_f16_128x128x64_16x16x16_2x2 = specialize_gpu half half 128sz 128sz 64sz 16sz 16sz 16sz 2sz 2sz

let g_gemm_f16_f16_128x128x32_16x16x16_4x8 = specialize_gpu half half 128sz 128sz 32sz 16sz 16sz 16sz 4sz 8sz
let g_gemm_f16_f16_128x128x32_16x16x16_8x4 = specialize_gpu half half 128sz 128sz 32sz 16sz 16sz 16sz 8sz 4sz

// mixed precision
let g_gemm_f16_f32_32x32x32_16x16x16_2x2 = specialize_gpu half float 32sz 32sz 32sz 16sz 16sz 16sz 2sz 2sz


// Dynamic parameter version for shmem size
let g_gemm_f16_f16_16x16x16_2x2 bm bn bk =
  admit();
  specialize_gpu half half bm bn bk 16sz 16sz 16sz 2sz 2sz

let g_gemm_f16_f16_16x16x16_4x4 bm bn bk =
  admit();
  specialize_gpu half half bm bn bk 16sz 16sz 16sz 4sz 4sz

let g_gemm_f16_f16_16x16x16_8x8 bm bn bk =
  admit();
  specialize_gpu half half bm bn bk 16sz 16sz 16sz 8sz 8sz
