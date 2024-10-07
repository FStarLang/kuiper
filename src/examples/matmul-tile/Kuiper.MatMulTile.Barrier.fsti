module Kuiper.MatMulTile.Barrier

#lang-pulse

// #push-options "--log_queries"

open Kuiper
open Kuiper.Barrier.RPM
module SZ = FStar.SizeT

#push-options "--fuel 8 --ifuel 8"

// let lemma_pos (x: pos): Lemma (0 < x * x) [SMTPat (x * x)] = ()

[@@pulse_unfold]
let barrier_mm_perm
    (nthr: nat)
    (ar: gpu_array u64 (2 * nthr))
    (single: nat { single < nthr })
    : slprop
= exists* s. gpu_pts_to_array_slice ar #(1.0R /. Real.of_int nthr) (2 * single) (2 * single + 2) s

let barrier_mm
    (nthr: nat)
    (ar: gpu_array u64 (2 * nthr))
    (it: nat)
    (from: nat { 0 <= from /\ from < nthr })
    (to: nat { 0 <= to /\ to < nthr })
    : slprop
= barrier_mm_perm nthr ar (if (it % 2 = 0) then from else to)

ghost fn transfer_barrier_mm (nthr: nat) (smem_sz : nat{smem_sz == 2 * nthr}) (ar: gpu_array u64 smem_sz) (it: nat) (from to: (i: nat { 0 <= i /\ i < nthr }))
  requires Barrier.barrier_mm nthr ar it from to
  ensures  Barrier.barrier_mm nthr ar (it + 1) to from
{
  unfold Barrier.barrier_mm nthr ar it from to;
  fold Barrier.barrier_mm nthr ar (it + 1) to from;
}

ghost fn unfold_barrier_mm_odd (nthr: nat) (smem_sz : nat{smem_sz == 2 * nthr}) (ar: gpu_array u64 smem_sz) (it: nat{it % 2 <> 0}) (from to: (i: nat { 0 <= i /\ i < nthr }))
  requires Barrier.barrier_mm nthr ar it from to
  ensures exists* v. gpu_pts_to_array_slice ar #(1.0R /. Real.of_int nthr) (2 * to) (2 * to + 2) v
{
  unfold Barrier.barrier_mm nthr ar it from to;
}
ghost fn fold_barrier_mm_odd (nthr: nat) (smem_sz : nat{smem_sz == 2 * nthr}) (ar: gpu_array u64 smem_sz) (it: nat{it % 2 <> 0}) (from to: (i: nat { 0 <= i /\ i < nthr }))
  requires exists* v. gpu_pts_to_array_slice ar #(1.0R /. Real.of_int nthr) (2 * to) (2 * to + 2) v
  ensures Barrier.barrier_mm nthr ar it from to
{
  fold Barrier.barrier_mm nthr ar it from to;
}

ghost fn unfold_barrier_mm_even (nthr: nat) (smem_sz : nat{smem_sz == 2 * nthr}) (ar: gpu_array u64 smem_sz) (it: nat{it % 2 == 0}) (from to: (i: nat { 0 <= i /\ i < nthr }))
  requires Barrier.barrier_mm nthr ar it from to
  ensures exists* v. gpu_pts_to_array_slice ar #(1.0R /. Real.of_int nthr) (2 * from) (2 * from + 2) v
{
  unfold Barrier.barrier_mm nthr ar it from to;
}
ghost fn fold_barrier_mm_even (nthr: nat) (smem_sz : nat{smem_sz == 2 * nthr}) (ar: gpu_array u64 smem_sz) (it: nat{it % 2 == 0}) (from to: (i: nat { 0 <= i /\ i < nthr }))
  requires exists* v. gpu_pts_to_array_slice ar #(1.0R /. Real.of_int nthr) (2 * from) (2 * from + 2) v
  ensures Barrier.barrier_mm nthr ar it from to
{
  fold Barrier.barrier_mm nthr ar it from to;
}

let shared_pre
    (nthr: sz)
    (it: nat)
    (ar: gpu_array u64 (2 * nthr))
    (bid: nat)
    (tid: nat { 0 <= tid /\ tid < nthr })
: slprop
= (exists* s. gpu_pts_to_array_slice ar (2 * tid) (2 * tid + 2) s)
** mbarrier_tok nthr (barrier_mm nthr ar) it tid

ghost
fn block_setup_ghost
  (nthr: sz { 0 < nthr /\ nthr <= max_threads })
  (smem_sz : sz { SZ.v smem_sz == 2 * nthr })
  (ar: gpu_array u64 smem_sz)
  (bid: sz)
  requires block_setup nthr ** (exists* v. ar |-> v)
  ensures block_setup nthr ** bigstar 0 nthr (shared_pre nthr 0 ar bid)
{
  admit();
  // with v. assert (gpu_pts_to_array #u64 #smem_sz ar #1.0R v);

  // assert (pure ((multiply seq![2 <: pos; bdim; bdim] <: int) == (2 * bdim * bdim <: int)));
  // gpu_pts_to_ref ar #v;
  // assert (pure (FStar.len v = 2 * bdim * bdim /\ (get_dims (from_dims seq![2 <: pos; bdim; bdim])) == seq![2 <: pos; bdim; bdim]));
  // let vv: mseq (get_dims (from_dims seq![2 <: pos; bdim; bdim])) u64 = v;
  // gpu_matrix_from_array seq![2 <: pos; bdim; bdim] ar;

  // let ar_split = array_to_matrix ar;
  // // TODO: why does this need to be written out with the implicit like so?
  // rewrite each (array_to_matrix #u64 #(multiply (cons #pos 2 (cons #pos bdim (cons #pos bdim (empty #pos))))) ar) as ar_split;

  // let s1d: FStar.Seq.seq (erased pos & pos) = from_dims seq![2 <: pos; bdim; bdim];
  // let s2d: FStar.Seq.seq (erased pos & pos) = seq![(hide (2 <: pos), (bdim * bdim <: pos)); (hide (bdim <: pos), (bdim <: pos)); (hide (bdim <: pos), (1 <: pos))];
  // assert (pure (s1d.[0] == s2d.[0] /\ s1d.[1] == s2d.[1] /\ s1d.[2] == s2d.[2]));
  // assert (pure (FStar.len s1d == 3 /\ FStar.len s2d == 3));
  // assert (pure (forall (i:nat{i < FStar.len s1d}). i == 0 \/ i == 1 \/ i == 2));
  // FStar.Seq.lemma_eq_intro s1d s2d;

  // gpu_matrix_slice_permission #u64 #(from_dims seq![2 <: pos; bdim <: pos; bdim <: pos]) ar_split 0 #vv seq![(bdim <: pos); (bdim <: pos)];

  // assert (pure (((from_dims seq![2 <: pos; bdim; bdim]).[ 0 ])._1 == 2));

  // drop_ (bigstar 0 ((from_dims seq![2 <: pos; bdim; bdim]).[ 0 ])._1
  //     (fun i ->
  //         gpu_pts_to_matrix (slice_matrix ar_split
  //               i
  //               ((from_dims seq![2 <: pos; bdim; bdim]).[ 0 ])._2)
  //           (remove (from_dims seq![2 <: pos; bdim; bdim]) 0)
  //           (slice vv 0 i seq![(bdim <: pos); (bdim <: pos)])));
  // assume (bigstar 0 2
  //     (fun i ->
  //         gpu_pts_to_matrix (slice_matrix ar_split
  //               i
  //               (bdim * bdim))
  //           (from_dims seq![bdim; bdim])
  //           (slice vv 0 i seq![(bdim <: pos); (bdim <: pos)]))
  //     );

  // let ar1 = slice_matrix ar_split 0 (bdim * bdim);
  // let ar2 = slice_matrix ar_split 1 (bdim * bdim);
  // drop_ (bigstar 0 2 (fun i ->
  //         gpu_pts_to_matrix (slice_matrix ar_split i (bdim * bdim))
  //           (from_dims seq![bdim; bdim]) (slice vv 0 i seq![(bdim <: pos); (bdim <: pos)]))
  //     );

  // assume (gpu_pts_to_matrix ar1 (from_dims seq![bdim; bdim]) (slice vv 0 0 seq![(bdim <: pos); (bdim <: pos)]) **
  //          gpu_pts_to_matrix ar2 (from_dims seq![bdim; bdim]) (slice vv 0 1 seq![(bdim <: pos); (bdim <: pos)]));

  // gpu_matrix_slice_permission #u64 #(from_dims seq![bdim; bdim]) ar1 0 seq![(bdim <: pos)];
  // gpu_matrix_slice_permission #u64 #(from_dims seq![bdim; bdim]) ar2 0 seq![(bdim <: pos)];
  // bigstar_zip 0 (Seq.Base.index (from_dims seq![bdim; bdim]) 0)._1 _ _;

  // drop_ (bigstar 0
  //     (Seq.Base.index (from_dims seq![bdim; bdim]) 0)._1
  //     (fun i ->
  //         gpu_pts_to_matrix (slice_matrix ar1
  //               i
  //               (Seq.Base.index (from_dims seq![bdim; bdim]) 0)._2)
  //           (remove (from_dims seq![bdim; bdim]) 0)
  //           (slice #(get_dims (from_dims seq![bdim; bdim])) #u64
  //               (slice #(get_dims (from_dims seq![2 <: pos; bdim; bdim])) #u64 vv 0 0 seq![(bdim <: pos); (bdim <: pos)]) 0 i seq![(bdim <: pos)]) **
  //         gpu_pts_to_matrix (slice_matrix ar2
  //               i
  //               (Seq.Base.index (from_dims seq![bdim; bdim]) 0)._2)
  //           (remove (from_dims seq![bdim; bdim]) 0)
  //           (slice #(get_dims (from_dims seq![bdim; bdim])) #u64
  //               (slice #(get_dims (from_dims seq![2 <: pos; bdim; bdim])) #u64 vv 0 1 seq![(bdim <: pos); (bdim <: pos)]) 0 i seq![(bdim <: pos)])));
  // assume (bigstar 0 bdim (fun i ->
  //           (exists* v1. gpu_pts_to_matrix (slice_matrix ar1 i bdim) (from_dims seq![bdim]) v1) **
  //           (exists* v2. gpu_pts_to_matrix (slice_matrix ar2 i bdim) (from_dims seq![bdim]) v2)));
  
  // ghost fn slice_again (ari1 ari2: gpu_matrix u64)
  //   requires (exists* v1. gpu_pts_to_matrix ari1 (from_dims seq![bdim]) v1) **
  //            (exists* v2. gpu_pts_to_matrix ari2 (from_dims seq![bdim]) v2)
  //   ensures  bigstar 0 bdim (fun i ->
  //             (exists* v1. gpu_pts_to_matrix (slice_matrix ari1 i bdim) seq![] v1) **
  //             (exists* v2. gpu_pts_to_matrix (slice_matrix ari2 i bdim) seq![] v2))
  // {
  //   admit();
  //   ()
  // };
  // bigstar_map #_ #_ #0 #bdim (fun i -> slice_again (slice_matrix ar1 i bdim) (slice_matrix ar2 i bdim));
  // bigstar_flatten #_ #_ #bdim #bdim #_;

  // let bid_split = split_to_dims seq![bdim_cols; bdim_rows] bid;
  // mk_mbarrier nthr (barrier_mm s1 ar1 s2 ar2 bid_split.[0] bid_split.[1]);

  // rewrite (bigstar 0 nthr (mbarrier_tok nthr (barrier_mm s1 ar1 s2 ar2 bid_split.[0] bid_split.[1]) 0))
  //     as  (bigstar 0 (bdim * bdim) (mbarrier_tok (bdim * bdim) (barrier_mm s1 ar1 s2 ar2 bid_split.[0] bid_split.[1]) 0));
  // bigstar_zip 0 (bdim * bdim) _ _;

  // drop_ (bigstar 0 (bdim * bdim) (fun i ->
  //         ((exists* v1. gpu_pts_to_matrix (slice_matrix (slice_matrix ar1 (i / bdim) bdim) (i % bdim) bdim) seq![] v1) **
  //         (exists* v2. gpu_pts_to_matrix (slice_matrix (slice_matrix ar2 (i / bdim) bdim) (i % bdim) bdim) seq![] v2)) **
  //         mbarrier_tok (bdim * bdim) (barrier_mm s1 ar1 s2 ar2 bid_split.[0] bid_split.[1]) 0 i));
  // assume (bigstar 0 nthr (shared_pre ar_split s1 s2 0 bid));

  // // let dims_inner: seq pos = seq![bdim; bdim; 2 <: pos];
  // // let t: Type u#0 = (gpu_matrix u64 seq![bdim; bdim; 2] <: Type u#0);

  // // assert (pure (multiply seq![2 <: pos; bdim; bdim] == 2 * bdim * bdim));
  // // let ar_split = (to_gpu_matrix seq![2 <: pos; bdim; bdim] ar) <: gpu_matrix u64 seq![2 <: pos; bdim; bdim];
  // // FStar.Seq.lemma_eq_intro (remove seq![2 <: pos; bdim; bdim] 0) seq![bdim; bdim];
  // // assert (pure (remove seq![2 <: pos; bdim; bdim] 0 == seq![bdim; bdim] /\ gpu_matrix u64 (remove seq![2 <: pos; bdim; bdim] 0) == gpu_matrix u64 seq![bdim; bdim]));
  // // let ar1 = slice_matrix ar_split 0 0 <: gpu_matrix u64 (remove seq![2 <: pos; bdim; bdim] 0);
  
  // // let ar11 = (*coerce_eq ()*) ar1 <: gpu_matrix u64 seq![bdim; bdim];

  // ()
  // // with v. assert (gpu_pts_to_array #u64 #smem_sz ar #1.0R v);
  // // unfold gpu_pts_to_array ar v;
  // // gpu_slice_slice_1_underspec #1 ar #1.0R 0 smem_sz (bdim * bdim);
  // // drop_   (bigstar #1 0 (nthr - 0) (fun x -> gpu_pts_to_array1 ar (x + 0)));
  // // assume (bigstar #1 0 nthr            (fun x -> gpu_pts_to_array1 ar x));

  // // gpu_slice_slice_1_underspec #2 ar #1.0R nthr smem_sz smem_sz;
  // // drop_   (bigstar #2 0 (smem_sz - nthr) (fun x -> gpu_pts_to_array1 ar (x + nthr)));
  // // assume (bigstar #2 0 nthr             (fun x -> gpu_pts_to_array1 ar (x + nthr)));

  // // bigstar_zip #1 #2 #1 0 nthr _ _;

  // // mk_mbarrier nthr (barrier_mm nthr Seq.empty Seq.empty ar);
  // // bigstar_zip #1 #0 #0 0 nthr _ _;

  // // // FOLD:
  // // drop_   (bigstar #0 0 nthr (fun x -> gpu_pts_to_array1 ar x ** gpu_pts_to_array1 ar (x + nthr) ** mbarrier_tok nthr (barrier_mm nthr Seq.empty Seq.empty ar) 0 x));
  // // assume (bigstar #0 0 nthr (fun x -> shared_pre nthr Seq.empty Seq.empty 0 ar x));

  // // bigstar_uneta();
  // // gpu_slice_empty_elim ar smem_sz;
}

// bdim_shared == shared / tdim_y
// let split_input_a2 (shared: pos) (bidx_x: nat) (tdim_x bdim_x tdim_y bdim_shared: pos) (a2: seq u64 { FStar.len a2 == tdim_y * bdim_shared * tdim_x * bdim_x  }): seq (seq u64)
//   = FStar.Seq.init bdim_shared (fun bidx_y -> FStar.Seq.init (tdim_x * tdim_y) (fun tidx -> ))

// let shared_post (nthr : sz { 0 < nthr /\ nthr <= max_threads }) (ar: gpu_array u64 SZ.(2sz *^ nthr)) (i: nat { 0 <= i /\ i < nthr }): slprop =
//   exists* it. shared_pre nthr Seq.empty Seq.empty it ar i
