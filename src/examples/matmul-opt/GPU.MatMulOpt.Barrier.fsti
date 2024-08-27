module GPU.MatMulOpt.Barrier

#lang-pulse

// #push-options "--log_queries"

open GPU
open GPU.Barrier.RPM
open GPU.MatMulOpt.Array
open GPU.MatMulOpt.Layout
module Pure = GPU.MatMulOpt.Pure
module SZ = FStar.SizeT

#push-options "--fuel 8 --ifuel 8"

let lemma_pos (x: pos): Lemma (0 < x * x) [SMTPat (x * x)] = ()

let barrier_mm_share
    (#bdim #bdim_shared #bdim_cols #bdim_rows: pos)
    (s1: mseq seq![bdim; bdim_shared; bdim; bdim_rows] u64)
    (ar1: gpu_matrix u64 seq![bdim; bdim])
    (s2: mseq seq![bdim; bdim_cols; bdim; bdim_shared] u64)
    (ar2: gpu_matrix u64 seq![bdim; bdim])
    (#bid_x: nat { 0 <= bid_x /\ bid_x < bdim_cols })
    (#bid_y: nat { 0 <= bid_y /\ bid_y < bdim_rows })
    (it: nat { 0 <= it /\ it < bdim_shared })
    (from_x: nat { 0 <= from_x /\ from_x < bdim })
    (from_y: nat { 0 <= from_y /\ from_y < bdim })
    : slprop
=
  gpu_pts_to_matrix #u64 #seq![] (slice_matrix (slice_matrix ar1 1 from_y) 0 from_x) #(1.0R /. Real.of_int (bdim * bdim)) (Pure.singleton (index s1 seq![from_x; it; from_y; bid_y])) **
  gpu_pts_to_matrix #u64 #seq![] (slice_matrix (slice_matrix ar2 1 from_y) 0 from_x) #(1.0R /. Real.of_int (bdim * bdim)) (Pure.singleton (index s2 seq![from_x; bid_x; from_y; it]))

let barrier_mm_gather
    (#bdim #bdim_shared #bdim_cols #bdim_rows: pos)
    (s1: mseq seq![bdim; bdim_shared; bdim; bdim_rows] u64)
    (ar1: gpu_matrix u64 seq![bdim; bdim])
    (s2: mseq seq![bdim; bdim_cols; bdim; bdim_shared] u64)
    (ar2: gpu_matrix u64 seq![bdim; bdim])
    (#bid_x: nat { 0 <= bid_x /\ bid_x < bdim_cols })
    (#bid_y: nat { 0 <= bid_y /\ bid_y < bdim_rows })
    (it: nat { 0 <= it /\ it < bdim_shared })
    (to_x: nat { 0 <= to_x /\ to_x < bdim })
    (to_y: nat { 0 <= to_y /\ to_y < bdim })
    : slprop
=
  gpu_pts_to_matrix #u64 #seq![] (slice_matrix (slice_matrix ar1 1 to_y) 0 to_x) #(1.0R /. Real.of_int (bdim * bdim)) (Pure.singleton (index s1 seq![to_x; it; to_y; bid_y])) **
  gpu_pts_to_matrix #u64 #seq![] (slice_matrix (slice_matrix ar2 1 to_y) 0 to_x) #(1.0R /. Real.of_int (bdim * bdim)) (Pure.singleton (index s2 seq![to_x; bid_x; to_y; it]))

let multiply_2_lemma (a b: pos): Lemma (multiply seq![a; b] = a * b) [SMTPat (seq![a; b])] = ()
let multiply_3_lemma (a b c: pos): Lemma (multiply seq![a; b; c] = a * b * c) [SMTPat (seq![a; b; c])] = ()

let barrier_mm
    (#bdim #bdim_shared #bdim_cols #bdim_rows: pos)
    (s1: mseq seq![bdim; bdim_shared; bdim; bdim_rows] u64)
    (ar1: gpu_matrix u64 seq![bdim; bdim])
    (s2: mseq seq![bdim; bdim_cols; bdim; bdim_shared] u64)
    (ar2: gpu_matrix u64 seq![bdim; bdim])
    (bid_x: nat { 0 <= bid_x /\ bid_x < bdim_cols })
    (bid_y: nat { 0 <= bid_y /\ bid_y < bdim_rows })
    (it: nat)
    (from: nat { 0 <= from /\ from < bdim * bdim })
    (to: nat { 0 <= to /\ to < bdim * bdim })
    : slprop
= 
  let from_split = split_to_dims seq![bdim; bdim] from in
  let to_split = split_to_dims seq![bdim; bdim] to in
    if (it / 2 < bdim_shared)
    then (cond (it % 2 = 0) (barrier_mm_share s1 ar1 s2 ar2 #bid_x #bid_y (it / 2) from_split.[0] from_split.[1])
                            (barrier_mm_gather s1 ar1 s2 ar2 #bid_x #bid_y (it / 2) to_split.[0] to_split.[1]))
    else emp

let shared_pre
    (#bdim #bdim_shared #bdim_cols #bdim_rows: pos)
    (s1: mseq seq![bdim; bdim_shared; bdim; bdim_rows] u64)
    (ar1: gpu_matrix u64 seq![bdim; bdim])
    (s2: mseq seq![bdim; bdim_cols; bdim; bdim_shared] u64)
    (ar2: gpu_matrix u64 seq![bdim; bdim])
    (bid_x: nat { 0 <= bid_x /\ bid_x < bdim_cols })
    (bid_y: nat { 0 <= bid_y /\ bid_y < bdim_rows })
    (it: nat)
    (i: nat { 0 <= i /\ i < bdim * bdim })
: slprop
=
  let i_split = split_to_dims seq![bdim; bdim] i in
  gpu_pts_to_matrix #u64 #seq![] (slice_matrix (slice_matrix ar1 1 i_split.[1]) 0 i_split.[0]) #1.0R (Pure.singleton 0uL) **
  gpu_pts_to_matrix #u64 #seq![] (slice_matrix (slice_matrix ar2 1 i_split.[1]) 0 i_split.[0]) #1.0R (Pure.singleton 0uL) **
  mbarrier_tok (bdim * bdim) (barrier_mm s1 ar1 s2 ar2 bid_x bid_y) it i


// let gpu_matrix (a:Type u#0) (dims: FStar.Seq.seq pos) : Type u#0 = gpu_array a (multiply dims)
let to_gpu_matrix (#a:Type u#0) (dims: FStar.Seq.seq pos) (ar: gpu_array a (multiply dims)): gpu_matrix a dims = ar

val mk_gpu_array (a:Type u#0) (dims: FStar.Seq.seq pos): gpu_matrix a dims

let remove_lemma #a (x y z: a): Lemma (remove seq![x; y; z] 0 == seq![y; z])// [SMTPat (remove seq![x; y; z] 0)]
  = FStar.Seq.lemma_eq_intro (remove seq![x; y; z] 0) seq![y; z]

// #set-options "--debug SMTFail --split_queries always"

ghost
fn block_setup_ghost
  (#bdim #bdim_shared #bdim_cols #bdim_rows: pos)
  (s1: mseq seq![bdim; bdim_shared; bdim; bdim_rows] u64)
  // (ar1: gpu_matrix u64 seq![bdim; bdim])
  (s2: mseq seq![bdim; bdim_cols; bdim; bdim_shared] u64)
  // (ar2: gpu_matrix u64 seq![bdim; bdim])
  (nblk : sz { 0 < nblk /\ nblk <= max_blocks /\ SZ.v nblk == bdim_cols * bdim_rows })
  (nthr : sz { 0 < nthr /\ nthr <= max_threads /\ SZ.v nblk == bdim * bdim })
  (smem_sz : sz { SZ.v smem_sz == 2 * bdim * bdim })
  (ar: gpu_array u64 smem_sz)
  (bid: sz { 0 <= bid /\ SZ.v bid < bdim_cols * bdim_rows })
  requires block_setup nthr ** (exists* v. gpu_pts_to_array #u64 #smem_sz ar #1.0R v)
  ensures (let bid_split = split_to_dims seq![bdim_cols; bdim_rows] bid in
           let ar_split = to_gpu_matrix seq![2 <: pos; bdim; bdim] ar in
           (//remove_lemma (2 <: pos) bdim bdim; assert (remove seq![2 <: pos; bdim; bdim] 0 == seq![bdim; bdim]);
          //  let ar1: gpu_matrix u64 seq![bdim; bdim] = slice_matrix ar_split 0 0 in
          //  let ar2: gpu_matrix u64 seq![bdim; bdim] = slice_matrix ar_split 0 1 in
    block_setup nthr ** bigstar 0 nblk (shared_pre s1 (mk_gpu_array u64 seq![bdim; bdim]) s2 (mk_gpu_array u64 seq![bdim; bdim]) bid_split.[0] bid_split.[1] 0)))
{
  // let dims_inner: seq pos = seq![bdim; bdim; 2 <: pos];
  // let t: Type u#0 = (gpu_matrix u64 seq![bdim; bdim; 2] <: Type u#0);

  assert (pure (multiply seq![2 <: pos; bdim; bdim] == 2 * bdim * bdim));
  let ar_split = (to_gpu_matrix seq![2 <: pos; bdim; bdim] ar) <: gpu_matrix u64 seq![2 <: pos; bdim; bdim];
  FStar.Seq.lemma_eq_intro (remove seq![2 <: pos; bdim; bdim] 0) seq![bdim; bdim];
  assert (pure (remove seq![2 <: pos; bdim; bdim] 0 == seq![bdim; bdim] /\ gpu_matrix u64 (remove seq![2 <: pos; bdim; bdim] 0) == gpu_matrix u64 seq![bdim; bdim]));
  let ar1 = slice_matrix ar_split 0 0 <: gpu_matrix u64 (remove seq![2 <: pos; bdim; bdim] 0);
  
  let ar11 = (*coerce_eq ()*) ar1 <: gpu_matrix u64 seq![bdim; bdim];
  admit();
  ()
  // with v. assert (gpu_pts_to_array #u64 #smem_sz ar #1.0R v);
  // unfold gpu_pts_to_array ar v;
  // gpu_slice_slice_1_underspec #1 ar #1.0R 0 smem_sz (bdim * bdim);
  // drop_   (bigstar #1 0 (SZ.v nthr - 0) (fun x -> gpu_pts_to_array1 ar (x + 0)));
  // assume_ (bigstar #1 0 nthr            (fun x -> gpu_pts_to_array1 ar x));

  // gpu_slice_slice_1_underspec #2 ar #1.0R nthr smem_sz smem_sz;
  // drop_   (bigstar #2 0 (smem_sz - nthr) (fun x -> gpu_pts_to_array1 ar (x + nthr)));
  // assume_ (bigstar #2 0 nthr             (fun x -> gpu_pts_to_array1 ar (x + nthr)));

  // bigstar_zip #1 #2 #1 0 nthr _ _;

  // mk_mbarrier nthr (barrier_mm nthr Seq.empty Seq.empty ar);
  // bigstar_zip #1 #0 #0 0 nthr _ _;

  // // FOLD:
  // drop_   (bigstar #0 0 nthr (fun x -> gpu_pts_to_array1 ar x ** gpu_pts_to_array1 ar (x + nthr) ** mbarrier_tok nthr (barrier_mm nthr Seq.empty Seq.empty ar) 0 x));
  // assume_ (bigstar #0 0 nthr (fun x -> shared_pre nthr Seq.empty Seq.empty 0 ar x));

  // bigstar_uneta();
  // gpu_slice_empty_elim ar smem_sz;
}

// bdim_shared == shared / tdim_y
// let split_input_a2 (shared: pos) (bidx_x: nat) (tdim_x bdim_x tdim_y bdim_shared: pos) (a2: seq u64 { FStar.Seq.length a2 == tdim_y * bdim_shared * tdim_x * bdim_x  }): seq (seq u64)
//   = FStar.Seq.init bdim_shared (fun bidx_y -> FStar.Seq.init (tdim_x * tdim_y) (fun tidx -> ))

// let shared_post (nthr : sz { 0 < nthr /\ nthr <= max_threads }) (ar: gpu_array u64 SZ.(2sz *^ nthr)) (i: nat { 0 <= i /\ i < nthr }): slprop =
//   exists* it. shared_pre nthr Seq.empty Seq.empty it ar i
