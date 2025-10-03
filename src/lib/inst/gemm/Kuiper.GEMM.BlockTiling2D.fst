module Kuiper.GEMM.BlockTiling2D

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm, crepr_row_major, crepr_col_major}

module M = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module P = Kuiper.Poly.GEMM.BlockTiling2D
module SZ = FStar.SizeT

inline_for_extraction noextract
fn spec_as_gemm
  (bm bn bk : szp)
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn /\ (bm/tm * bn/tn <= max_threads)})
  (#_ : squash (SZ.fits (bm*bk + (bm/tm * (bn/tn)))))
  (#_ : squash (SZ.fits (bk*bn + (bm/tm * (bn/tn)))))
  (et : Type0) {| scalar et, has_vec_cpy et |}
  (#_ : squash (chunk et /? bn))
  (#_ : squash (chunk et /? bk))
  (alpha beta : et)
  (#rows #shared #cols : szp)
  (gA : M.gpu_matrix et (rm rows shared))
  (#fA : perm)
  (gB : M.gpu_matrix et (rm shared cols))
  (#fB : perm)
  (gC : M.gpu_matrix et (rm rows cols))
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (rows * cols <= max_blocks) **
    gC |-> eC
  ensures
    gC |-> MS.gemm alpha beta eC eA eB
{
  // These didn't seem to be needed before!?
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;

  dassert (bm `SZ.gt` 0sz);
  dassert (bn `SZ.gt` 0sz);
  dassert (bk `SZ.gt` 0sz);
  dassert (tm `SZ.gt` 0sz);
  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);
  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;

  P.mmcomb_gpu
    (fun o n -> mul beta o `add` mul alpha n)
    #rows #shared #cols
    gA #eA gB #eB gC #eC
    bm bn bk
    tm tn
    slA slB #_ #_;

  ()
}

let g_gemm_f32_64x64x8_8x8_rr =
  spec_as_gemm 64sz 64sz 8sz (rm _ _) (rm _ _)
     8sz 8sz f32

let g_gemm_f32_128x128x8_8x8_rr =
  spec_as_gemm 128sz 128sz 8sz (rm _ _) (rm _ _)
    8sz 8sz f32

// Transposed A-tiles in shared memory
let g_gemm_f32_128x128x8_8x8_cr =
  spec_as_gemm 128sz 128sz 8sz (cm _ _) (rm _ _)
    8sz 8sz f32

let g_gemm_f32_128x128x16_8x8_cr =
  spec_as_gemm 128sz 128sz 16sz (cm _ _) (rm _ _)
    8sz 8sz f32

let g_gemm_f32_128x128x32_8x8_cr =
  spec_as_gemm 128sz 128sz 32sz (cm _ _) (rm _ _)
    8sz 8sz f32

// Dynamic parameter version. Only admitted since otherwise we
// need to repeat all requirements here. We only use this for some
// quick tuning, so it's not a big deal.
let g_gemm_f32_8x8_cr bm bn bk =
  admit();
  spec_as_gemm bm bn bk (cm _ _) (rm _ _)
    8sz 8sz f32
