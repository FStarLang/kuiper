module Klas.GEMM.TiledComb

#lang-pulse
open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
module K = Kuiper.Kernel.GEMM.TiledComb

module Reprs = Kuiper.Matrix.Reprs

noextract // broken
fn matmul_f32_tile32_rrr
  (#mrows #mshared #mcols : szp)
  (gA : gpu_matrix f32 (Reprs.row_major (mrows * 32sz) (mshared * 32sz)))
  (gB : gpu_matrix f32 (Reprs.row_major (mshared * 32sz) (mcols * 32sz)))
  (gC : gpu_matrix f32 (Reprs.row_major (mrows * 32sz) (mcols * 32sz)))
  (#eA #eB #eC : ematrix _ _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          32sz * 32sz <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (live gC)
{
  gpu_matrix_pts_to_ref_located gA;
  gpu_matrix_pts_to_ref_located gB;
  gpu_matrix_pts_to_ref_located gC;
  K.mmcomb_gpu 32sz #f32 (fun _ n -> n) gA gB gC;
}
