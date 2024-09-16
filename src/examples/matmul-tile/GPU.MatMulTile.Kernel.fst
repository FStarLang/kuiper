module GPU.MatMulTile.Kernel

#lang-pulse

open GPU
open GPU.Barrier.RPM
open FStar.SizeT
module SZ = FStar.SizeT

#set-options "--z3rlimit 20"
#push-options "--fuel 6 --ifuel 6"

// #push-options "--print_implicits --print_bound_var_types"
// #push-options "--debug SMTFail"

let lemma_pos_times_pos (a b: pos)
  : Lemma (a * b > 0) = ()

let lemma_nat_times_nat (a b: nat)
  : Lemma (a * b >= 0) = ()

inline_for_extraction noextract
fn calc_idxs
  (nblk : erased sz { SZ.v nblk == SZ.v SZ.(rows_tile *^ columns_tile) })
  (nthr : erased sz { SZ.v nthr == SZ.v SZ.(bdim *^ bdim) })
  (etid : tid_t { gdim_x etid == nthr /\ bdim_x etid == nblk })
  requires thread_id etid
  returns  idxs: (SZ.t & SZ.t & SZ.t)
  ensures  thread_id etid ** (pure (SZ.v idxs._1 == (tid_to_idx (thread_index etid)) /\ SZ.v idxs._2 < rows /\ SZ.v idxs._3 < columns))
{
  let tid = thread_idx_x () <: u32;
  let tid : sz = SZ.uint32_to_sizet tid;
  let trow = SZ.div tid bdim;
  let tcol = SZ.rem tid bdim;
  assert (pure (SZ.v trow <= SZ.v bdim - 1 /\ SZ.v tcol <= SZ.v bdim - 1));

  let bid = block_idx_x () <: u32;
  let bid : sz = SZ.uint32_to_sizet bid;
  let brow = SZ.div bid columns_tile;
  let bcol = SZ.rem bid columns_tile;
  assert (pure (SZ.v brow <= SZ.v rows_tile - 1 /\ SZ.v bcol <= SZ.v columns_tile - 1));

  let brow_idx = brow *^ bdim *^ columns_tile *^ bdim;
  let trow_idx = trow *^ bdim *^ columns_tile;
  let bcol_idx = bcol *^ bdim;
  assert (pure (SZ.v brow_idx <= (rows_tile - 1) * bdim * columns_tile * bdim));
  assert (pure (SZ.v trow_idx <= (bdim - 1) * columns_tile * bdim));
  assert (pure (SZ.v bcol_idx <= (columns_tile - 1) * bdim));

  assert (pure (SZ.v bcol_idx + SZ.v tcol < columns_tile * bdim));
  let bcol_tcol_idx = bcol_idx +^ tcol;

  assert (pure (SZ.v trow_idx + SZ.v bcol_tcol_idx < bdim * columns_tile * bdim));
  let trow_bcol_tcol_idx = trow_idx +^ bcol_tcol_idx;

  assert (pure (SZ.v brow_idx + SZ.v trow_bcol_tcol_idx < rows_tile * bdim * columns_tile * bdim));
  let idx : sz = brow_idx +^ trow_bcol_tcol_idx;

  lemma_nat_times_nat (SZ.v bid) (SZ.v nthr);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v nthr) (SZ.v bid) (SZ.v nblk - 1);
  assert (pure (SZ.v bid * SZ.v nthr <= (SZ.v nblk - 1) * SZ.v nthr));
  FStar.Math.Lemmas.distributivity_sub_left (SZ.v nblk) 1 (SZ.v nthr);
  assume (pure (SZ.v idx == tid_to_idx (SZ.v bid * SZ.v nthr + SZ.v tid))); // TODO

  assert (pure (SZ.v rows_tile == 32 /\ SZ.v columns_tile == 32 /\ SZ.v bdim == 32));

  let brow_bdim = brow *^ bdim;
  let bcol_bdim = bcol *^ bdim;
  let row : sz = brow_bdim +^ trow;
  let col : sz = bcol_bdim +^ tcol;

  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v brow) (SZ.v rows_tile - 1);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v bcol) (SZ.v columns_tile - 1);
  assert (pure (SZ.v row <= (SZ.v rows_tile - 1) * SZ.v bdim + (SZ.v bdim - 1)));
  FStar.Math.Lemmas.distributivity_sub_left (SZ.v rows_tile) 1 (SZ.v bdim);
  assert (pure ((SZ.v rows_tile - 1) * SZ.v bdim + (SZ.v bdim - 1) == SZ.v rows - SZ.v bdim + SZ.v bdim - 1));
  
  assert (pure (SZ.v col <= (SZ.v columns_tile - 1) * SZ.v bdim + (SZ.v bdim - 1)));
  FStar.Math.Lemmas.distributivity_sub_left (SZ.v columns_tile) 1 (SZ.v bdim);
  assert (pure ((SZ.v columns_tile - 1) * SZ.v bdim + (SZ.v bdim - 1) == SZ.v columns - 1));
  assert (pure (SZ.v row < SZ.v rows /\ SZ.v col < SZ.v columns));

  Mktuple3 idx row col
}

let sz_mult (x y: SZ.t) : Pure SZ.t
  (requires (SZ.fits (SZ.v x * SZ.v y)))
  (ensures (fun z -> SZ.v z == SZ.v x * SZ.v y)) = SZ.mul x y

let lemma_div_pos (a: real) (i: int)
  : Lemma (requires (a >. 0.0R /\ i > 0))
          (ensures (a /. Real.of_int i >. 0.0R)) = ()

let lemma_mod_lt (a : nat) (b : pos)
  : Lemma (a % b <= b - 1) = ()

let lemma_div_lt (a : nat) (b c: pos)
  : Lemma (requires (a < c * b))
          (ensures  (a / b < c)) = ()

[@@CPrologue "__device__"; "KrmlPrivate"]
inline_for_extraction
fn inner_loop
  (vv : sz{SZ.v vv <= SZ.v bdim - 1})
  (ga1_iidx : sz{SZ.v ga1_iidx <= (SZ.v bdim - 1) * SZ.v bdim})
  (ga2_iidx : sz{SZ.v ga2_iidx <= SZ.v bdim - 1})
  (nthr : erased nat{nthr == SZ.v SZ.(bdim *^ bdim)})
  (ar : gpu_array u64 (2 * nthr))
  (it: erased nat{it % 2 <> 0})
  (tid: erased nat{0 <= tid /\ tid < nthr})
  (sum : ref u64)
  requires gpu
        ** (exists* sumv. pts_to sum sumv)
        ** bigstar 0 nthr (Barrier.barrier_mm nthr ar it tid)
  ensures  gpu
        ** (exists* sumv. pts_to sum sumv)
        ** bigstar 0 nthr (Barrier.barrier_mm nthr ar it tid)
{
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v vv) (SZ.v bdim - 1);
  
  let ga1_idx = ga1_iidx +^ vv;
  assert (pure (SZ.v ga1_idx < nthr));

  let ga2_idx = ga2_iidx +^ (vv *^ bdim);
  assert (pure (SZ.v ga2_idx < nthr));

  bigstar_extract #0 0 nthr _ ga1_idx;
  unfold Barrier.barrier_mm nthr ar it tid ga1_idx;
  let ga1_val = gpu_array_read #_ #_ #(2 * ga1_idx) #(2 * ga1_idx + 2) ar #(1.0R /. Real.of_int nthr) (2sz *^ ga1_idx);
  fold Barrier.barrier_mm nthr ar it tid ga1_idx;
  bigstar_compose #0 0 nthr _ ga1_idx;

  bigstar_extract #0 0 nthr _ ga2_idx;
  unfold Barrier.barrier_mm nthr ar it tid ga2_idx;
  let ga2_val = gpu_array_read #_ #_ #(2 * ga2_idx) #(2 * ga2_idx + 2) ar #(1.0R /. Real.of_int nthr) (2sz *^ ga2_idx +^ 1sz);
  fold Barrier.barrier_mm nthr ar it tid ga2_idx;
  bigstar_compose #0 0 nthr _ ga2_idx;

  let s = !sum;
  sum := U64.add_mod (U64.mul_mod ga1_val ga2_val) s;
}

[@@CPrologue "__device__"; "KrmlPrivate"]
inline_for_extraction
fn outer_loop
  (iv: sz{SZ.v iv <= shared_tile - 1})
  (nthr: erased nat{nthr == SZ.v SZ.(bdim *^ bdim)})
  (smem_sz: erased nat{smem_sz == 2 * nthr})
  (ar: gpu_array u64 smem_sz)
  (tid: erased nat{0 <= tid /\ tid < nthr})
  (tcol: sz{SZ.v tcol < SZ.v bdim})
  (trow: sz{SZ.v trow < SZ.v bdim})
  (sum: ref u64)
  requires gpu
       ** (exists* sumv. pts_to sum sumv)
       ** (exists* s. gpu_pts_to_array_slice ar (2 * tid) (2 * tid + 2) s)
       ** mbarrier_tok nthr (Barrier.barrier_mm nthr ar) (2*iv) tid
  ensures gpu
       ** (exists* sumv. pts_to sum sumv)
       ** (exists* s. gpu_pts_to_array_slice ar (2 * tid) (2 * tid + 2) s)
       ** mbarrier_tok nthr (Barrier.barrier_mm nthr ar) (2*iv + 2) tid
{
  lemma_div_pos 1.0R nthr; // 1.0R /. Real.of_int nthr >. 0.0R

  FStar.Math.Lemmas.cancel_mul_mod (SZ.v iv) 2;
  FStar.Math.Lemmas.lemma_mod_plus 1 (SZ.v iv) 2;
  assert (pure ((SZ.v iv * 2) % 2 == 0 /\ (1 + SZ.v iv * 2) % 2 <> 0));

  // Share smem permission with all threads
  gpu_slice_share_underspec #0 #u64 #smem_sz ar #1.0R (2 * tid) (2 * tid + 2) nthr;
  bigstar_map #0 #0 #0 #nthr #_ #_ (Barrier.fold_barrier_mm_even nthr smem_sz ar (2*iv) tid);
  mbarrier_wait #nthr #(Barrier.barrier_mm nthr ar) #(2 * SZ.v iv) #tid;
  bigstar_map #0 #0 #0 #nthr #_ #_ (fun from -> Barrier.transfer_barrier_mm nthr smem_sz ar (2*iv) from tid);

  // Do stuff
  let ga1_iidx = trow *^ bdim;
  let ga2_iidx = tcol;

  let mut j = 0sz;
  while (let vv = !j; (vv <^ bdim))
    invariant b.
      exists* vv.
      pure (b == (SZ.v vv < bdim) /\ SZ.v vv <= bdim /\ SZ.v vv >= 0) **
      pts_to j vv **
      gpu
      ** (exists* sumv. pts_to sum sumv)
      ** bigstar 0 nthr (Barrier.barrier_mm nthr ar (2*iv + 1) tid)
  {
    let vv = !j;
    j := SZ.add vv 1sz;

    inner_loop vv ga1_iidx ga2_iidx nthr ar (2*iv + 1) tid sum;

    ()
  };

  // Send smem permission back to single thread
  mbarrier_wait #nthr #(Barrier.barrier_mm nthr ar) #(2 * SZ.v iv + 1) #tid;
  bigstar_map #0 #0 #0 #nthr #_ #_ (fun from -> Barrier.transfer_barrier_mm nthr smem_sz ar (2*iv + 1) from tid);
  bigstar_map #0 #0 #0 #nthr #_ #_ (Barrier.unfold_barrier_mm_even nthr smem_sz ar (2*iv + 2) tid);
  gpu_slice_gather_underspec #0 #u64 #smem_sz ar #1.0R (2 * tid) (2 * tid + 2) nthr;
}

#set-options "--z3rlimit 40"
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
  (smem_sz : erased nat { smem_sz == 2 * SZ.v nthr })
  (ear: erased (gpu_array u64 smem_sz))
  (etid : tid_t { gdim_x etid == nthr /\ bdim_x etid == nblk })
  requires gpu
    ** thread_id etid
    ** shmem_tok ear
    ** Barrier.shared_pre nthr 0 ear (SZ.v (bidx_x etid)) (SZ.v (tidx_x etid))
    ** kpre rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr) (tid_to_idx (thread_index etid))
  ensures  gpu
    ** thread_id etid
    ** Barrier.shared_pre nthr (2 * shared_tile) ear (SZ.v (bidx_x etid)) (SZ.v (tidx_x etid))
    ** kpost rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr) (tid_to_idx (thread_index etid))
{

  let idxs = calc_idxs nblk nthr etid;
  let idx = idxs._1;
  let row = idxs._2;
  let col = idxs._3;
  let ar = obtain_shmem ear;

  let tid = thread_idx_x () <: u32;
  let tid : sz = SZ.uint32_to_sizet tid;
  assert (pure (SZ.v tid < SZ.v bdim * SZ.v bdim));
  let trow : sz = SZ.div tid bdim;
  let tcol : sz = SZ.rem tid bdim;
  // lemmas
  FStar.Math.Lemmas.euclidean_division_definition (SZ.v tid) (SZ.v bdim);
  lemma_div_lt (SZ.v tid) (SZ.v bdim) (SZ.v bdim);
  lemma_mod_lt (SZ.v tid) (SZ.v bdim);
  assert (pure (SZ.v trow < SZ.v bdim /\ SZ.v tcol < SZ.v bdim));

  let smem_idx1 = 2sz *^ tid;
  let smem_idx2 = smem_idx1 +^ 1sz;

  let mut i = 0sz;
  let mut sum = 0UL;

  unfold Barrier.shared_pre;
  while (let iv = !i; (iv <^ shared_tile))
     invariant b.
       exists* iv.
       pure (b == (SZ.v iv < shared_tile) /\ SZ.v iv <= shared_tile /\ SZ.v iv >= 0) **
       pts_to i iv **
       gpu
       ** (exists* sumv. pts_to sum sumv)
       ** Impure.gpu_pts_to_matrix #u64 rows shared ga1 (SZ.v nblk * SZ.v nthr) s1
       ** Impure.gpu_pts_to_matrix #u64 shared columns ga2 (SZ.v nblk * SZ.v nthr) s2
       ** (exists* s. gpu_pts_to_array_slice ar (2 * tid) (2 * tid + 2) s)
       ** mbarrier_tok (SZ.v nthr) (Barrier.barrier_mm (SZ.v nthr) ar) (2*iv) tid
  {
    let iv = !i;
    i := SZ.add iv 1sz;
    FStar.Math.Lemmas.cancel_mul_mod (SZ.v iv) 2;

    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v iv) (SZ.v shared_tile - 1);
    assert (pure (SZ.v iv * SZ.v bdim <= (SZ.v shared_tile - 1) * SZ.v bdim));
    assert (pure (SZ.v shared_tile == 32 /\ SZ.v bdim == 32));
    SZ.fits_at_least_16 (SZ.v iv * SZ.v bdim);
    
    let v_bdim = SZ.mul iv bdim;

    assert (pure (SZ.v v_bdim + SZ.v tcol < shared /\ SZ.v v_bdim + SZ.v trow < shared));
    SZ.fits_at_least_16 (SZ.v v_bdim + SZ.v tcol);
    SZ.fits_at_least_16 (SZ.v v_bdim + SZ.v trow);
    assert (pure (SZ.v row < SZ.v rows /\ SZ.v col < SZ.v columns));

    let v1 = Impure.gpu_matrix_read #_ #rows #shared ga1 #(SZ.v nblk * SZ.v nthr) #s1 row (v_bdim +^ tcol);
    let v2 = Impure.gpu_matrix_read #_ #shared #columns ga2 #(SZ.v nblk * SZ.v nthr) #s2 (v_bdim +^ trow) col;
    
    gpu_array_write #u64 #smem_sz #(SZ.v smem_idx1) #(SZ.v smem_idx1 + 2) ar smem_idx1 v1;
    gpu_array_write #u64 #smem_sz #(SZ.v smem_idx1) #(SZ.v smem_idx1 + 2) ar smem_idx2 v2;

    outer_loop iv (SZ.v nthr) smem_sz ar (SZ.v tid) tcol trow sum;
    ()
  };
  fold Barrier.shared_pre nthr (2 * shared_tile) ar (SZ.v (bidx_x etid)) (SZ.v (tidx_x etid));

  let s = !sum;
  unfold gpu_pts_to_array1 r (tid_to_idx (thread_index etid));
  gpu_array_write #u64 #(rows * columns) #(SZ.v idx) #(SZ.v idx + 1) r idx s;
  fold gpu_pts_to_array1 r (tid_to_idx (thread_index etid));

  ()
}
