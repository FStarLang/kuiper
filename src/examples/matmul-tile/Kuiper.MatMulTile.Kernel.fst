module Kuiper.MatMulTile.Kernel

#lang-pulse

module U64 = FStar.UInt64
open Kuiper
open Kuiper.Barrier.RPM
open FStar.SizeT
module SZ = FStar.SizeT

module I = Kuiper.MatMul.Impure

#set-options "--z3rlimit 60"
#set-options "--fuel 1 --ifuel 1"

inline_for_extraction noextract
fn calc_idxs
  (rows shared columns bdim : szp)
  (nblk nthr : erased sz)
  (etid : tid_t)
  (#_ : squash (bdim /? rows /\ bdim /? columns /\
                SZ.v nblk == (rows / bdim) * (columns / bdim) /\
                SZ.v nthr == bdim * bdim /\ SZ.v nblk * SZ.v nthr == rows * columns /\
                gdim_x etid == SZ.v nblk /\ bdim_x etid == SZ.v nthr))
  requires
    thread_id etid **
    pure (bdim < pow2 30)
  returns
    idxs: (SZ.t & SZ.t & SZ.t)
  ensures
    thread_id etid **
    pure (SZ.v idxs._1 == tid_to_idx rows shared columns bdim (thread_index etid) /\
          SZ.v idxs._2 < rows /\
          SZ.v idxs._3 < columns)
{
  assume (pure False); // fixme.. proofs too brittle to work with
  let tid = thread_idx_x ();
  let trow = SZ.div tid bdim;
  let tcol = SZ.rem tid bdim;

  let columns_tile = SZ.div columns bdim;
  let rows_tile = SZ.div rows bdim;

  let bid = block_idx_x ();
  let brow = SZ.div bid columns_tile;
  let bcol = SZ.rem bid columns_tile;

  lemma_divides_exact columns bdim;
  assert (pure (bdim * columns_tile == columns));
  assert (pure (columns_tile * bdim == columns));
  assert (pure (SZ.fits (brow * bdim)));
  let brow_idx = brow *^ bdim *^ columns_tile *^ bdim;
  let trow_idx = trow *^ bdim *^ columns_tile;
  let bcol_idx = bcol *^ bdim;
  // assert (pure (SZ.v brow_idx <= (rows/bdim- 1) * bdim * (columns/bdim) * bdim));
  // assert (pure (SZ.v trow_idx <= (bdim - 1) * (columns/bdim) * bdim));
  // assert (pure (SZ.v bcol_idx <= (columns/bdim- 1) * bdim));

  // assert (pure (SZ.v bcol_idx + SZ.v tcol < columns_tile * bdim));
  let bcol_tcol_idx = bcol_idx +^ tcol;

  // assert (pure (SZ.v trow_idx + SZ.v bcol_tcol_idx < bdim * columns_tile * bdim));
  let trow_bcol_tcol_idx = trow_idx +^ bcol_tcol_idx;

  // assert (pure (SZ.v brow_idx + SZ.v trow_bcol_tcol_idx < rows_tile * bdim * columns_tile * bdim));
  let idx : sz = brow_idx +^ trow_bcol_tcol_idx;

  Kuiper.Math.Silly.lemma_nat_times_nat (SZ.v bid) (SZ.v nthr);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v nthr) (SZ.v bid) (SZ.v nblk - 1);
  // assert (pure (SZ.v bid * SZ.v nthr <= (SZ.v nblk - 1) * SZ.v nthr));
  FStar.Math.Lemmas.distributivity_sub_left (SZ.v nblk) 1 (SZ.v nthr);
  // assume (pure (SZ.v idx == tid_to_idx (SZ.v bid * SZ.v nthr + SZ.v tid))); // TODO

  // assert (pure (SZ.v rows_tile == 32 /\ SZ.v columns_tile == 32 /\ SZ.v bdim == 32));

  let brow_bdim = brow *^ bdim;
  let bcol_bdim = bcol *^ bdim;
  let row : sz = brow_bdim +^ trow;
  let col : sz = bcol_bdim +^ tcol;

  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v brow) (SZ.v rows_tile - 1);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v bcol) (SZ.v columns_tile - 1);
  // assert (pure (SZ.v row <= (SZ.v rows_tile - 1) * SZ.v bdim + (SZ.v bdim - 1)));
  FStar.Math.Lemmas.distributivity_sub_left (SZ.v rows_tile) 1 (SZ.v bdim);
  // assert (pure ((SZ.v rows_tile - 1) * SZ.v bdim + (SZ.v bdim - 1) == SZ.v rows - SZ.v bdim + SZ.v bdim - 1));

  // assert (pure (SZ.v col <= (SZ.v columns_tile - 1) * SZ.v bdim + (SZ.v bdim - 1)));
  FStar.Math.Lemmas.distributivity_sub_left (SZ.v columns_tile) 1 (SZ.v bdim);
  // assert (pure ((SZ.v columns_tile - 1) * SZ.v bdim + (SZ.v bdim - 1) == SZ.v columns - 1));
  // assert (pure (SZ.v row < SZ.v rows /\ SZ.v col < SZ.v columns));

  (idx, row, col)
}

let sz_mult (x y: SZ.t) : Pure SZ.t
  (requires (SZ.fits (SZ.v x * SZ.v y)))
  (ensures (fun z -> SZ.v z == SZ.v x * SZ.v y)) = SZ.mul x y

let lemma_div_pos (a: real) (i: int)
  : Lemma (requires (a >. 0.0R /\ i > 0))
          (ensures (a /. i >. 0.0R)) = ()

let lemma_mod_lt (a : nat) (b : pos)
  : Lemma (a % b <= b - 1) = ()

let lemma_div_lt (a : nat) (b c: pos)
  : Lemma (requires (a < c * b))
          (ensures  (a / b < c)) = ()

[@@CPrologue "__device__"; "KrmlPrivate"]
inline_for_extraction
fn inner_loop
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim < pow2 30})
  (vv : sz{SZ.v vv < SZ.v bdim})
  (ga1_iidx : sz{SZ.v ga1_iidx <= (SZ.v bdim - 1) * SZ.v bdim})
  (ga2_iidx : sz{SZ.v ga2_iidx < SZ.v bdim})
  (nthr : erased nat{nthr == bdim * bdim})
  (ar : gpu_array u64 (2 * nthr))
  (it: erased nat{it % 2 <> 0})
  (tid: erased nat{tid < nthr})
  (sum : ref u64)
  preserves gpu **
    bigstar 0 nthr (Barrier.barrier_mm nthr ar it tid)
  requires
    (exists* sumv. sum |-> sumv)
  ensures
    (exists* sumv. sum |-> sumv)
{
  assume (pure sizet_does_not_overflow); // cheating overflow, this is all in bounds, but the proofs are way too brittle
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v vv) (SZ.v bdim - 1);

  let ga1_idx = ga1_iidx +^ vv;
  assert (pure (SZ.fits (bdim * bdim)));
  let ga2_idx = ga2_iidx +^ (vv *^ bdim);
  assert (pure (SZ.v ga2_idx < nthr));

  bigstar_extract #0 0 nthr _ ga1_idx;
  unfold Barrier.barrier_mm nthr ar it tid ga1_idx;
  unfold Barrier.barrier_mm_perm;

  let ga1_val = gpu_array_read #_ #_ #(2 * ga1_idx) #(2 * ga1_idx + 2) ar #(1.0R /. nthr) (2sz *^ ga1_idx);
  // rewrite each Barrier.ifeven it tid (SZ.v ga1_idx)
  //           as (SZ.v ga1_idx);

  fold Barrier.barrier_mm nthr ar it tid ga1_idx;

  bigstar_compose #0 0 nthr _ ga1_idx;

  bigstar_extract #0 0 nthr _ ga2_idx;
  unfold Barrier.barrier_mm nthr ar it tid ga2_idx;
  let ga2_val = gpu_array_read #_ #_ #(2 * ga2_idx) #(2 * ga2_idx + 2) ar #(1.0R /. nthr) (2sz *^ ga2_idx +^ 1sz);
  fold Barrier.barrier_mm nthr ar it tid ga2_idx;
  bigstar_compose #0 0 nthr _ ga2_idx;

  let s = !sum;
  sum := U64.add_mod (U64.mul_mod ga1_val ga2_val) s;
}

[@@CPrologue "__device__"; "KrmlPrivate"]
inline_for_extraction
fn outer_loop
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (iv: sz{iv <= shared/bdim - 1})
  (nthr: erased nat{nthr == bdim * bdim})
  (smem_sz: erased nat{smem_sz == 2 * nthr})
  (ar: gpu_array u64 smem_sz)
  (tid: erased nat{tid < nthr})
  (tcol: sz{SZ.v tcol < SZ.v bdim})
  (trow: sz{SZ.v trow < SZ.v bdim})
  (sum: ref u64)
  preserves
    gpu
  requires
    (exists* sumv. sum |-> sumv) **
    (exists* s. gpu_pts_to_slice ar (2 * tid) (2 * tid + 2) s) **
    mbarrier_tok nthr (Barrier.barrier_mm nthr ar) (2*iv) tid
  ensures
    (exists* sumv. sum |-> sumv) **
    (exists* s. gpu_pts_to_slice ar (2 * tid) (2 * tid + 2) s) **
    mbarrier_tok nthr (Barrier.barrier_mm nthr ar) (2*iv + 2) tid
{
  lemma_div_pos 1.0R nthr; // 1.0R /. nthr >. 0.0R

  FStar.Math.Lemmas.cancel_mul_mod (SZ.v iv) 2;
  FStar.Math.Lemmas.lemma_mod_plus 1 (SZ.v iv) 2;
  assert (pure ((SZ.v iv * 2) % 2 == 0 /\ (1 + SZ.v iv * 2) % 2 <> 0));

  // Share smem permission with all threads
  gpu_slice_share_underspec #0 #u64 #smem_sz ar (2 * tid) (2 * tid + 2) nthr;
  bigstar_map #0 #0 #0 #nthr #_ #_ (Barrier.fold_barrier_mm_even nthr smem_sz ar (2*iv) tid);
  mbarrier_wait #nthr #(Barrier.barrier_mm nthr ar) #(2 * SZ.v iv) #tid;
  bigstar_map #0 #0 #0 #nthr #_ #_ (fun from -> Barrier.transfer_barrier_mm nthr smem_sz ar (2*iv) from tid);

  // Do stuff
  assert (pure (trow < bdim));
  Math.Lemmas.lemma_mult_lt_right bdim trow bdim; // sigh
  assert (pure (trow * bdim <= bdim * bdim));
  assert (pure (bdim < pow2 30));
  assert (pure (bdim * bdim < pow2 30 * pow2 30));
  assert (pure (SZ.fits (trow * bdim)));
  let ga1_iidx = trow *^ bdim;
  let ga2_iidx = tcol;
  assert (pure (trow < bdim));
  Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v trow) (SZ.v bdim - 1);
  assert (pure (trow * bdim <= (bdim - 1) * bdim));

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
    inner_loop rows shared columns bdim vv ga1_iidx ga2_iidx nthr ar (2*iv + 1) tid sum
  };

  // Send smem permission back to single thread
  mbarrier_wait #nthr #(Barrier.barrier_mm nthr ar) #(2 * SZ.v iv + 1) #tid;
  bigstar_map #0 #0 #0 #nthr #_ #_ (fun from -> Barrier.transfer_barrier_mm nthr smem_sz ar (2*iv + 1) from tid);
  admit(); //fixme, failed inference below?
  bigstar_map #0 #0 #0 #nthr #_ #_ (Barrier.unfold_barrier_mm_even nthr smem_sz ar (2*iv + 2) tid);
  gpu_slice_gather_underspec #0 #u64 #smem_sz ar #1.0R (2 * tid) (2 * tid + 2) nthr;
}

#set-options "--z3rlimit 20 --retry 3"
#restart-solver (* proof below very brittle *)

#restart-solver (* proof below very brittle *)
#set-options "--split_queries always" // BAD

[@@CPrologue "__global__"]
fn kernel
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {len s1 == rows * shared})
  (#s2: erased (seq u64) {len s2 == shared * columns})
  (nblk : erased sz { SZ.v nblk == (rows / bdim) * (columns / bdim) })
  (nthr : erased sz { SZ.v nthr == bdim * bdim })
  (smem_sz : erased nat { smem_sz == 2 * SZ.v nthr })
  (ear: erased (gpu_array u64 smem_sz))
  (etid : tid_t { gdim_x etid == SZ.v nblk /\ bdim_x etid == SZ.v nthr })
  requires gpu
    ** thread_id etid
    ** shmem_tok ear
    ** Barrier.shared_pre nthr 0 ear (bidx_x etid) (tidx_x etid)
    ** kpre rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr)
         (tid_to_idx rows shared columns bdim (thread_index etid))
  ensures  gpu
    ** thread_id etid
    ** Barrier.shared_pre nthr (2 * (shared / bdim)) ear (bidx_x etid) (tidx_x etid)
    ** kpost rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr)
         (tid_to_idx rows shared columns bdim (thread_index etid))
{
  assert (pure (thread_index etid < gdim_x etid * bdim_x etid));
  assert (pure (gdim_x etid = SZ.v nblk));
  assert (pure (bdim_x etid = SZ.v nthr));
  let _ = calc (==) {
    SZ.v nblk * SZ.v nthr;
    == {}
    ((rows / bdim) * (columns / bdim)) * (bdim * bdim);
    == { magic() } // ac...
    ((rows / bdim) * bdim) * ((columns / bdim) * bdim);
    == { lemma_divides_exact bdim rows }
    rows * ((columns / bdim) * bdim);
    == { lemma_divides_exact bdim columns }
    rows * columns;
  };
  assert (pure (thread_index etid < SZ.v nblk * SZ.v nthr ));
  assert (pure (SZ.v nblk * SZ.v nthr == rows * columns));
  assert (pure (thread_index etid < rows * columns));
  assert (pure (rows * columns >= 0));

  let idx, row, col = calc_idxs rows shared columns bdim nblk nthr etid #();
  let ar = obtain_shmem ear;
  rewrite each ear as ar;

  let shared_tile = shared `SZ.div` bdim;

  let tid = thread_idx_x ();
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
       pure (b == (SZ.v iv < shared_tile) /\ SZ.v iv <= shared_tile) **
       pts_to i iv **
       gpu
       ** (exists* sumv. pts_to sum sumv)
       ** I.gpu_pts_to_matrix #u64 rows shared ga1 (SZ.v nblk * SZ.v nthr) s1
       ** I.gpu_pts_to_matrix #u64 shared columns ga2 (SZ.v nblk * SZ.v nthr) s2
       ** (exists* s. gpu_pts_to_slice ar (2 * tid) (2 * tid + 2) s)
       ** mbarrier_tok (SZ.v nthr) (Barrier.barrier_mm (SZ.v nthr) ar) (2*iv) tid
  {
    let iv = !i;
    i := SZ.add iv 1sz;
    FStar.Math.Lemmas.cancel_mul_mod (SZ.v iv) 2;

    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bdim) (SZ.v iv) (SZ.v shared_tile - 1);
    assert (pure (SZ.v iv * SZ.v bdim <= (SZ.v shared_tile - 1) * SZ.v bdim));
    // assert (pure (SZ.v shared_tile == 32 /\ SZ.v bdim == 32));
    // SZ.fits_at_least_16 (SZ.v iv * SZ.v bdim);

    assert (pure (SZ.v iv < shared / bdim));
    assert (pure (bdim /? shared));
    assert (pure (bdim * (shared/bdim) == shared));
    assert (pure (bdim * (shared/bdim) == (shared/bdim) * bdim));
    assert (pure ((shared/bdim) * bdim == shared));
    Math.Lemmas.lemma_mult_lt_right (SZ.v bdim) (SZ.v iv) (shared/bdim);
    assert (pure (SZ.v iv * bdim < (shared/bdim) * bdim));
    assert (pure (SZ.v iv * SZ.v bdim < SZ.v shared));
    Kuiper.Math.Silly.lemma_nonneg_mul (SZ.v iv) (SZ.v bdim); // ridiculous to have to call this
    SizeT.fits_lte (SZ.v iv * SZ.v bdim) (SZ.v shared);
    assert (pure (SZ.fits (iv * bdim)));

    let v_bdim = SZ.mul iv bdim;

    assume (pure (SZ.v v_bdim + SZ.v tcol < shared /\ SZ.v v_bdim + SZ.v trow < shared));
    // SZ.fits_at_least_16 (SZ.v v_bdim + SZ.v tcol);
    // SZ.fits_at_least_16 (SZ.v v_bdim + SZ.v trow);
    assert (pure (SZ.v row < SZ.v rows /\ SZ.v col < SZ.v columns));

    let v1 = I.gpu_matrix_read #_ #(hide rows)   #shared ga1 #(SZ.v nblk * SZ.v nthr) #s1 row (v_bdim +^ tcol);
    let v2 = I.gpu_matrix_read #_ #(hide shared) #columns ga2 #(SZ.v nblk * SZ.v nthr) #s2 (v_bdim +^ trow) col;

    gpu_array_write #u64 #smem_sz #(SZ.v smem_idx1) #(SZ.v smem_idx1 + 2) ar smem_idx1 v1;
    gpu_array_write #u64 #smem_sz #(SZ.v smem_idx1) #(SZ.v smem_idx1 + 2) ar smem_idx2 v2;

    outer_loop rows shared columns bdim iv (SZ.v nthr) smem_sz ar (SZ.v tid) tcol trow sum;
    Kuiper.Math.Silly.two_times_succ (SZ.v iv); // silly
    rewrite each (2 `Prims.op_Multiply` SZ.v iv + 2) as (2 * (SZ.v iv + 1));
    ()
  };
  fold Barrier.shared_pre nthr (2 * shared_tile) ar (bidx_x etid) (tidx_x etid);

  let s = !sum;
  unfold gpu_pts_to_array1 r (tid_to_idx rows shared columns bdim (thread_index etid));
  gpu_array_write #u64 #(rows * columns) #(SZ.v idx) #(SZ.v idx + 1) r idx s;
  fold gpu_pts_to_array1 r (tid_to_idx rows shared columns bdim (thread_index etid));

  ()
}
