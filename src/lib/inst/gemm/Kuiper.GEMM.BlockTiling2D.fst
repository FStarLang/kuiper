module Kuiper.GEMM.BlockTiling2D

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU {
  specialize_as_matmul_to_type_and_reprs_cpu as spec_cpu,
  specialize_as_gemm_to_type_and_reprs_gpu as spec_gemm_gpu,
  mmcomb_gpu_shmem_block_tiled2d
}
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm, crepr_row_major, crepr_col_major}
module P = Kuiper.Poly.GEMM.BlockTiling2D

let matmul_f32_64x64x8_8x8_rrr_rr = spec_cpu (mmcomb_gpu_shmem_block_tiled2d P.mmcomb_gpu 64sz 64sz 8sz
  // cannot infer clayout :(
  (rm _ _) (rm _ _) #(crepr_row_major.map 64sz 8sz) #(crepr_row_major.map 8sz 64sz) 8sz 8sz) f32 rm rm rm
let matmul_f32_32x32x32_32x8_rrr_rr = spec_cpu (mmcomb_gpu_shmem_block_tiled2d P.mmcomb_gpu 32sz 32sz 32sz
  (rm _ _) (rm _ _) #(crepr_row_major.map 32sz 32sz) #(crepr_row_major.map 32sz 32sz) 32sz 8sz) f32 rm rm rm

let g_gemm_f32_64x64x8_8x8_rrr_rr = spec_gemm_gpu (mmcomb_gpu_shmem_block_tiled2d P.mmcomb_gpu 64sz 64sz 8sz (rm _ _) (rm _ _)
  #(crepr_row_major.map 64sz 8sz) #(crepr_row_major.map 8sz 64sz) 8sz 8sz) f32 rm rm rm
let g_gemm_f32_128x128x8_8x8_rrr_rr = spec_gemm_gpu (mmcomb_gpu_shmem_block_tiled2d P.mmcomb_gpu 128sz 128sz 8sz (rm _ _) (rm _ _)
  #(crepr_row_major.map 128sz 8sz) #(crepr_row_major.map 8sz 128sz) 8sz 8sz) f32 rm rm rm

// Transposed A-tiles in shared memory
let g_gemm_f32_128x128x8_8x8_rrr_cr = spec_gemm_gpu (mmcomb_gpu_shmem_block_tiled2d P.mmcomb_gpu 128sz 128sz 8sz (cm _ _) (rm _ _)
  #(crepr_col_major.map 128sz 8sz) #(crepr_row_major.map 8sz 128sz) 8sz 8sz) f32 rm rm rm
