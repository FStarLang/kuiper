module Kuiper.VectorizedScale
#lang-pulse
// open FStar.Tactics
// open Pulse.Lib
// open Pulse.Lib.Pervasives
open Kuiper
// open Pulse.Lib.BoundedIntegers
// open Pulse.Lib.PartitionRange
open FStar.Seq.Base
// open FStar.FiniteSet.Base
// open FStar.FiniteSet.Ambient
// module Set = FStar.FiniteSet.Base
module SZ = FStar.SizeT

open Kuiper.Array.Vectorized

// let lt_vectorized_block_elems (tid size block_elems: nat) : prop
//   = 4 /? block_elems /\ tid < block_elems / 4 

noextract
unfold
let global_id #nblk #nthr (bid : natlt nblk) (tid : natlt nthr) : natlt (nblk * nthr) = bid * nthr + tid

noextract
unfold
let kpre
  (size:sz)
  (a:gpu_array float size)
  (s:seq float{ len s ==  SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  gpu_pts_to_slice a (global_id bid tid * 4) (global_id bid tid * 4 + 4) (slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4))

noextract
let scale_seq (#len : nat) (s : seq float{length s == len}) (v : float)
  = Seq.seq_of_list (List.mapT (fun x -> v `mul` x) (Seq.seq_to_list s))

noextract
unfold
let kpost
  (v : float)
  (size:sz)
  (a:gpu_array float size)
  (s:seq float{ len s == SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  let s_slice = slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4) in
  gpu_pts_to_slice a (global_id bid tid * 4) (global_id bid tid * 4 + 4)
    (upd_seq_vec4 s_slice
      0
      (make_float4
        ((Seq.index s_slice 0) `mul` v)
        ((Seq.index s_slice 1) `mul` v)
        ((Seq.index s_slice 2) `mul` v)
        ((Seq.index s_slice 3) `mul` v)))

// #push-options "--debug SMTFail --split_queries always"
// inline_for_extraction noextract
[@@CPrologue "__device__"] // no KrmlPrivate, example
fn kf
  (size:sz)
  (#s:erased (seq float) { len s == SZ.v size })
  (nblk : erased (natle max_blocks))
  (a:gpu_array float size)
  (v : float)
  (nthr : sz{nthr*4 * nblk == SZ.v size})
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
requires
  gpu **
  kpre size a s nblk nthr bid tid **
  block_id nblk bid **
  thread_id nthr tid
ensures
  gpu **
  kpost v size a s nblk nthr bid tid **
  block_id nblk bid **
  thread_id nthr tid
{
  let global_idx = ((bid *^ nthr +^ tid) *^ 4sz); rewrite each ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4) as SZ.v global_idx;
  let fv = gpu_array_vec4_read a global_idx;
  let x = getx fv `mul` v;
  let y = gety fv `mul` v;
  let z = getz fv `mul` v;
  let w = getw fv `mul` v;

  let fv' = make_float4 x y z w;
  gpu_array_vec4_write a global_idx fv';

  rewrite each SZ.v global_idx as ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4);
  ()
}

// ghost
// fn setup
//   (size:sz)
//   (a:gpu_array float size)
//   (s:seq float{ len s == size })
//   (nblk : natle max_blocks)
//   (nthr : nat{(nthr*4 * nblk == SZ.v size)})
//   ()
//   requires
//     a |-> s
//   ensures
//     (forall+ (bid : natlt nblk) (tid : natlt nthr).
//       kpre size a s nblk nthr bid tid) **
//     emp (* frame *)
// {
//   // explode_cells a;
//   // partition_cells a;

//   // forevery_fromstar #(natlt (size `div` 2sz))
//   //   (fun tid ->
//   //     gpu_pts_to_cell a #1.0R tid (Seq.index s tid) **
//   //     gpu_pts_to_cell a #1.0R (SZ.v size - tid - 1) (index_flip s tid));
//   admit();
// }

// let scale_spec (s : seq float) (v : float) : GTot _ = Seq.init (len s) (fun i -> (index s i) `mul` v)

// ghost
// fn teardown
//   (v : float)
//   (size : sz)
//   (a:gpu_array float size)
//   (s:seq float{ len s == size })
//   (nblk : natle max_blocks)
//   (nthr : nat{(nthr*4 * nblk == size)})
//   ()
//   requires
//     (forall+ (bid : natlt nblk) (tid : natlt nthr).
//       kpost v size a s nblk nthr bid tid) **
//     emp (* frame *)
//   ensures
//     (a |-> (scale_spec s v))
// {
//   admit();
// }

// inline_for_extraction noextract
// let mk_kernel
//   (size : sz)
//   (v : float)
//   (a:gpu_array float size)
//   (s:seq float{ len s == SZ.v size })
//   (nblk : szp{nblk <= max_blocks})
//   (nthr : szp{(nthr*4 * nblk == SZ.v size) /\ nthr <= max_threads})
//   : kernel_desc_m_n
//       (a |-> s)
//       (a |-> scale_spec s v)
//   = {
//   nblk     = nblk;
//   nthr     = nthr;

//   frame    = emp;

//   block_pre = magic();
//   block_post = magic();

//   block_frame = (fun _ -> emp);

//   setup    = magic();//(setup size a s nblk nthr);
//   teardown = magic();//(teardown v size a s nblk nthr);

//   kpre     = kpre size a s nblk nthr;
//   kpost    = kpost v size a s nblk nthr;

//   block_setup = magic();
//   block_teardown = magic();

//   f        = magic();//kf size #s nblk a v nthr;
// }
