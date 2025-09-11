module Kuiper.Poly.GEMM.Tiled.Common

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.Matrix.Reprs
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy

open Kuiper.EMatrix

module SZ = FStar.SizeT

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

inline_for_extraction noextract
fn populate_shmem
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : erased nat)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#slA : mlayout bm bk) {| clayout slA |}
  (#slB : mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout rows shared) {| clayout lA |}
  (#lB : mlayout shared cols) {| clayout lB |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#fA #fB : perm)
  (#eB : ematrix et shared cols)
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (tile_row : szlt (rows/bm))
  (tile_shared : szlt (shared/bk))
  (tile_col : szlt (cols/bn))
  (tid : szlt (bm/^tm *^ (bn/^tn)))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id (bm/^tm *^ (bn/^tn)) tid **
    live_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
    live_tile_stride_cells sB (bm/tm * (bn/tn)) tid
{
  let tileA = gpu_matrix_extract_tile_ro' gA
    (SZ.v bm) (SZ.v bk) (SZ.v tile_row) (SZ.v tile_shared);
  cp_matrix bm bk #_ #_ tileA sA (get_bdim()) tid;

  let tileB = gpu_matrix_extract_tile_ro' gB
    (SZ.v bk) (SZ.v bn) (SZ.v tile_shared) (SZ.v tile_col);
  cp_matrix bk bn tileB sB (get_bdim()) tid;

  ambig_trade_elim ();
  ambig_trade_elim ();
  ();
}

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
      (block_tile_idx_rows rows cols bm bn bid) (block_tile_idx_cols rows cols bm bn bid)

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
  (wid : natlt (bm/tm * (bn/tn)))
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
  (wid : natlt (bm/tm * (bn/tn)))
  : Tot (gpu_matrix et
          (subtile_layout lC_bt tm tn
            (warp_tile_idx_rows bm bn tm tn wid) (warp_tile_idx_cols bm bn tm tn wid)))
  =
   gpu_matrix_subtile gC_bt tm tn
    (warp_tile_idx_rows bm bn tm tn wid) (warp_tile_idx_cols bm bn tm tn wid)