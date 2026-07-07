module Kuiper.Kernel.GEMM.Tiled.Common.Vec

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Kernel.GEMM.Copy.Vec2

open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

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
  (#m #n #k : erased nat)
  (bm : szp{bm /? m})
  (bn : szp{bn /? n})
  (bk : szp{bk /? k})
  (#_ : squash (chunk et /? bk)) // extra req
  (#_ : squash (chunk et /? bn)) // extra req
  (#slA : layout2 bm bk) {| T.ctlayout slA |}
  (#slB : layout2 bk bn) {| T.ctlayout slB |}
  (sA : array2 et slA)
  (sB : array2 et slB)
  (#lA : layout2 m k) {| T.ctlayout lA, str_A : strided_row_major lA |}
  (#lB : layout2 k n) {| T.ctlayout lB, str_B : strided_row_major lB |}
  (gA : array2 et lA)
  (#eA : chest2 et m k)
  (gB : array2 et lB)
  (#fA #fB : perm)
  (#eB : chest2 et k n)
  (tile_row : szlt (m/bm))
  (tile_shared : szlt (k/bk))
  (tile_col : szlt (n/bn))
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
  (m n : erased nat)
  (bm : erased nat {bm > 0 /\ bm /? m})
  (bn : erased nat {bn > 0 /\ bn /? n})
  (bid : enatlt (m/bm * (n/bn)))
  : enatlt (m/bm)
  =
    bid / (n/bn)

unfold
let block_tile_idx_cols
  (m n : erased nat)
  (bm : erased nat {bm > 0 /\ bm /? m})
  (bn : erased nat {bn > 0 /\ bn /? n})
  (bid : enatlt (m/bm * (n/bn)))
  : enatlt (n/bn)
  = bid % (n/bn)

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
  (#et : Type0)
  (#m #n : erased nat)
  (#lC : layout2 m n)
  (gC : array2 et lC)
  (bm : erased nat{bm > 0 /\ bm /? m})
  (bn : erased nat{bn > 0 /\ bn /? n})
  (bid : enatlt (m/bm * (n/bn)))
  : Tot (array2 et
          (subtile_layout lC bm bn
            (block_tile_idx_rows m n bm bn bid)
            (block_tile_idx_cols m n bm bn bid)))
  =
    array2_subtile gC bm bn
      (block_tile_idx_rows m n bm bn bid)
      (block_tile_idx_cols m n bm bn bid)

inline_for_extraction noextract
let thread_tile
  (#et : Type0)
  (#bm #bn : erased nat)
  (#lC_bt : layout2 bm bn)
  (gC_bt : array2 et lC_bt)
  (tm : erased nat{tm > 0 /\ tm /? bm})
  (tn : erased nat{tn > 0 /\ tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : Tot (array2 et
          (subtile_layout lC_bt tm tn
            (thread_tile_idx_rows bm bn tm tn tid) (thread_tile_idx_cols bm bn tm tn tid)))
  =
   array2_subtile gC_bt tm tn
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
  (#lC_bt : layout2 bm bn)
  (gC_bt : array2 et lC_bt)
  (tm : erased nat{tm > 0 /\ tm /? bm})
  (tn : erased nat{tn > 0 /\ tn /? bn})
  (wid : enatlt (bm/tm * (bn/tn)))
  : Tot (array2 et
          (subtile_layout lC_bt tm tn
            (warp_tile_idx_rows bm bn tm tn wid) (warp_tile_idx_cols bm bn tm tn wid)))
  =
   array2_subtile gC_bt tm tn
    (warp_tile_idx_rows bm bn tm tn wid) (warp_tile_idx_cols bm bn tm tn wid)
