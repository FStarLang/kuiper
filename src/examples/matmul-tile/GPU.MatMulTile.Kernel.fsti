module GPU.MatMulTile.Kernel

#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open GPU

module Impure = GPU.MatMul.Impure
module Pure = GPU.MatMul.Pure
module Barrier = GPU.MatMulTile.Barrier
module SZ = FStar.SizeT
module U64 = FStar.UInt64
module Layout4 = GPU.MatMulTile.Layout4

[@@pulse_unfold]
let kpre_pair (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (#s1: erased (seq u64))
  (#s2: erased (seq u64))
  (nthr: erased nat { nthr > 0 })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nthr s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nthr s2

[@@pulse_unfold]
let kpre (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (r: gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) )
  (#s2: erased (seq u64))
  (nthr: erased nat { nthr > 0 })
  (tid : nat{ tid < rows * columns})
  : slprop
  =
  kpre_pair rows shared columns ga1 ga2 #s1 #s2 nthr
  ** gpu_pts_to_array1 r tid

[@@pulse_unfold]
let kpost (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (r: gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (nthr: erased nat { nthr > 0 })
  (tid : nat {  tid < rows * columns })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nthr s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nthr s2
  ** gpu_pts_to_array1 r tid
  // ** (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)


// TODO: un-hardcode
[@@CPrologue "const"]
inline_for_extraction
let bdim : sz = 32sz // rows/columns of tiles

[@@CPrologue "const"]
inline_for_extraction
let rows_tile : sz = 32sz // rows of ga1/r (in tiles)

[@@CPrologue "const"]
inline_for_extraction
let shared_tile : sz = rows_tile // columns of ga1, rows of ga2 (in tiles)

[@@CPrologue "const"]
inline_for_extraction
let columns_tile : sz = rows_tile // columns of ga2/r (in tiles)

[@@CPrologue "const"]
inline_for_extraction
let rows : sz = SZ.(rows_tile *^ bdim) // rows of ga1/r

[@@CPrologue "const"]
inline_for_extraction
let shared : sz = SZ.(shared_tile *^ bdim) // columns of ga1, rows of ga2

[@@CPrologue "const"]
inline_for_extraction
let columns : sz = SZ.(columns_tile *^ bdim) // columns of ga2/r

let permute(): GTot (permutation (i: nat { 0 <= i /\ i < SZ.(rows_tile *^ columns_tile) * SZ.(bdim *^ bdim) })) = Layout4.titi_permutation bdim bdim rows_tile columns_tile

let tid_to_idx (tid: nat { 0 <= tid /\ tid < rows * columns })
  : GTot (tid: nat { 0 <= tid /\ tid < rows * columns })
  = assert (SZ.(rows_tile *^ columns_tile) * SZ.(bdim *^ bdim) == rows * columns);
    (permute()).f tid

[@@CPrologue "__global__"]
fn kernel
  // (rows: nat) (shared: nat { shared < pow2 16 }) (columns: nat)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (nblk : erased sz { SZ.v nblk == SZ.v SZ.(rows_tile *^ columns_tile) })
  (nthr : erased sz { SZ.v nthr == SZ.v SZ.(bdim *^ bdim) })
  (smem_sz : sz { SZ.v smem_sz == 2 * SZ.v nthr })
  (ear: erased (gpu_array u64 smem_sz))
  (etid : tid_t { gdim_x etid == nthr /\ bdim_x etid == nblk })
  requires gpu
    ** thread_id etid
    ** shmem_tok ear
    ** Barrier.shared_pre nthr 0 ear (SZ.v (bidx_x etid)) (SZ.v (tidx_x etid))
    ** kpre rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr) (tid_to_idx (thread_index etid))
  ensures  gpu
    ** thread_id etid
    ** Barrier.shared_pre nthr (2 * columns_tile) ear (SZ.v (bidx_x etid)) (SZ.v (tidx_x etid))
    ** kpost rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr) (tid_to_idx (thread_index etid))
