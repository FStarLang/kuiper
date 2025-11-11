module Kuiper.Poly.GEMM.Tiled.Common.Vec

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.Matrix.Reprs
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy.Vec

open Kuiper.EMatrix

module SZ = Kuiper.SizeT

inline_for_extraction noextract
instance concrete_sz_32 : concrete_sz 32 = { x = 32sz }

(* Description of shared memory used for or tiled matmul kernels. *)
inline_for_extraction noextract
let shmems_desc
  (et:Type0) {| sized et |}
  (bm bn bk: szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  : list shmem_desc = [
  SHArray et (bm *^ bk);
  SHArray et (bk *^ bn);
]

// Vectorized version
inline_for_extraction noextract
fn copy_tiles_out_of_matrices_vec
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#rows #shared #cols : erased nat)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_ : squash (chunk et /? bk)) // extra req
  (#_ : squash (chunk et /? bn)) // extra req
  (#slA : mlayout bm bk) {| clayout slA |}
  (#slB : mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout rows shared) {| clayout lA, str_A : strided_row_major lA |}
  (#lB : mlayout shared cols) {| clayout lB, str_B : strided_row_major lB |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#fA #fB : perm)
  (#eB : ematrix et shared cols)
  (tile_row : szlt (rows/bm))
  (tile_shared : szlt (shared/bk))
  (tile_col : szlt (cols/bn))
  (nthr : szp)
  (#_ : squash (chunk et * nthr /?+ (bm * bk))) // extra req
  (#_ : squash (chunk et * nthr /?+ (bk * bn))) // extra req
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (tid : szlt nthr)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (aligned_strided_row_major (chunk et) str_A) **
    pure (aligned_strided_row_major (chunk et) str_B)
  requires
    live_strided_chunks sA nthr tid **
    live_strided_chunks sB nthr tid
  ensures
    own_strided_chunks sA (ematrix_subtile eA bm bk tile_row tile_shared) nthr tid **
    own_strided_chunks sB (ematrix_subtile eB bk bn tile_shared tile_col) nthr tid

unfold
let block_tile_idx_rows
  (rows cols : erased nat)
  (bm : erased nat {bm > 0 /\ bm /? rows})
  (bn : erased nat {bn > 0 /\ bn /? cols})
  (bid : enatlt (rows/bm * (cols/bn)))
  : enatlt (rows/bm)
  =
    bid / (cols/bn)

unfold
let block_tile_idx_cols
  (rows cols : erased nat)
  (bm : erased nat {bm > 0 /\ bm /? rows})
  (bn : erased nat {bn > 0 /\ bn /? cols})
  (bid : enatlt (rows/bm * (cols/bn)))
  : enatlt (cols/bn)
  = bid % (cols/bn)

unfold
let thread_tile_idx_rows
  (bm bn : erased nat)
  (tm : erased nat {tm > 0 /\ tm /? bm})
  (tn : erased nat {tn > 0 /\ tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : enatlt (bm/tm)
  = tid / (bn/tn)

unfold
let thread_tile_idx_cols
  (bm bn : erased nat)
  (tm : erased nat {tm > 0 /\ tm /? bm})
  (tn : erased nat {tn > 0 /\ tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : enatlt (bn/tn)
  = tid % (bn/tn)

inline_for_extraction noextract
let block_tile
  (#et : Type0) {| scalar et |}
  (#rows #cols : erased nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : erased nat{bm > 0 /\ bm /? rows})
  (bn : erased nat{bn > 0 /\ bn /? cols})
  (bid : enatlt (rows/bm * (cols/bn)))
  : Tot (gpu_matrix et
          (subtile_layout lC bm bn
            (block_tile_idx_rows rows cols bm bn bid)
            (block_tile_idx_cols rows cols bm bn bid)))
  =
    gpu_matrix_subtile gC bm bn
      (block_tile_idx_rows rows cols bm bn bid)
      (block_tile_idx_cols rows cols bm bn bid)

inline_for_extraction noextract
let thread_tile
  (#et : Type0) {| scalar et |}
  (#bm #bn : erased nat)
  (#lC_bt : mlayout bm bn)
  (gC_bt : gpu_matrix et lC_bt)
  (tm : erased nat{tm > 0 /\ tm /? bm})
  (tn : erased nat{tn > 0 /\ tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : Tot (gpu_matrix et
          (subtile_layout lC_bt tm tn
            (thread_tile_idx_rows bm bn tm tn tid) (thread_tile_idx_cols bm bn tm tn tid)))
  =
   gpu_matrix_subtile gC_bt tm tn
    (thread_tile_idx_rows bm bn tm tn tid) (thread_tile_idx_cols bm bn tm tn tid)

// The same as thread_tile* functions.
// Exists to clarify that instead of having number of tiles per block many threads per block,
// we have number of tiles per block many warps per block
unfold
let warp_tile_idx_rows
  (bm bn : erased nat)
  (tm : erased nat {tm > 0 /\ tm /? bm})
  (tn : erased nat {tn > 0 /\ tn /? bn})
  (wid : enatlt (bm/tm * (bn/tn)))
  : enatlt (bm/tm)
  = wid / (bn/tn)

unfold
let warp_tile_idx_cols
  (bm bn : erased nat)
  (tm : erased nat {tm > 0 /\ tm /? bm})
  (tn : erased nat {tn > 0 /\ tn /? bn})
  (wid : enatlt (bm/tm * (bn/tn)))
  : enatlt (bn/tn)
  = wid % (bn/tn)

inline_for_extraction noextract
let warp_tile
  (#et : Type0) {| scalar et |}
  (#bm #bn : erased nat)
  (#lC_bt : mlayout bm bn)
  (gC_bt : gpu_matrix et lC_bt)
  (tm : erased nat{tm > 0 /\ tm /? bm})
  (tn : erased nat{tn > 0 /\ tn /? bn})
  (wid : enatlt (bm/tm * (bn/tn)))
  : Tot (gpu_matrix et
          (subtile_layout lC_bt tm tn
            (warp_tile_idx_rows bm bn tm tn wid) (warp_tile_idx_cols bm bn tm tn wid)))
  =
   gpu_matrix_subtile gC_bt tm tn
    (warp_tile_idx_rows bm bn tm tn wid) (warp_tile_idx_cols bm bn tm tn wid)
