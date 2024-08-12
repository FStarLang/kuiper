module GPU.Kernel

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Seq
open Pulse.Lib.BigStar
open GPU.SizeT
open GPU.Array
open GPU.Base
open GPU.MatrixBarrier
open FStar.Mul
module U32 = FStar.UInt32
module SZ = FStar.SizeT

(* f<<<nblk, nthr>>>(...); *)
```pulse
val
fn launch_kernel_n_m_sync
  (#u1: erased int)
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < (nblk * nthr) } -> slprop))

  (a : Type u#0)
  (smem_sz : SZ.t)
  (#shared_pre : (ar: gpu_array a smem_sz) -> (tid: nat { 0 <= tid /\ tid < nthr } -> slprop))
  (#shared_post : (ar: gpu_array a smem_sz) -> (tid: nat { 0 <= tid /\ tid < nthr } -> slprop))
  (setup : (ar: gpu_array a smem_sz) -> (bid: SZ.t { 0 <= bid /\ bid < nblk }) ->
    stt_ghost unit emp_inames
      (block_setup nthr ** (exists* v. gpu_pts_to_array #a #smem_sz ar #1.0R v))
      (fun _ -> block_setup nthr ** bigstar 0 nthr (shared_pre ar)))

  (k :
    (ar: gpu_array a smem_sz) -> (etid: erased tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** shared_pre ar (SZ.v (tidx_x etid)) ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** shared_post ar (SZ.v (tidx_x etid)) ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (nblk * nthr) pre
  ensures  cpu ** bigstar #u1 0 (nblk * nthr) post
```

(* f<<<nblk, nthr>>>(...); *)
```pulse
val
fn launch_kernel_n_m_barrier
  (#u1: erased int)
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < (nblk * nthr) } -> slprop))

  (#p: (it:nat -> from: nat { 0 <= from /\ from < nthr } -> to: nat { 0 <= to /\ to < nthr } -> slprop))
  (k :
    (etid: erased tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** mbarrier_tok nthr p 0 (tidx_x etid) ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** (exists* it. mbarrier_tok nthr p it (tidx_x etid)) ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (nblk * nthr) pre
  ensures  cpu ** bigstar #u1 0 (nblk * nthr) post
```

(* f<<<nblk, nthr>>>(...); *)
```pulse
val
fn launch_kernel_n_m
  (#u1: erased int)
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < (nblk * nthr) } -> slprop))
  (k :
    (etid: erased tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (nblk * nthr) pre
  ensures  cpu ** bigstar #u1 0 (nblk * nthr) post
```

(* f<<<nblk, 1>>>(...); *)
```pulse
// Private
fn kernel_n_as_n_m
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < SZ.v nblk } -> slprop))
  (k :
    (etid:erased tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  (etid:erased tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz })
  requires gpu ** thread_id etid ** pre (thread_index etid)
  ensures  gpu ** thread_id etid ** post (thread_index etid)
{
  k etid;
}
```

```pulse
fn launch_kernel_n
  (#u1: erased int)
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < SZ.v nblk } -> slprop))
  (k :
    (etid:erased tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (SZ.v nblk) pre
  ensures  cpu ** bigstar #u1 0 (SZ.v nblk) post
{
  rewrite (bigstar #u1 0 (SZ.v nblk) pre) as (bigstar #u1 0 (SZ.v nblk * 1) pre);
  launch_kernel_n_m #u1 nblk 1sz #pre #post
    (fun etid -> kernel_n_as_n_m nblk #pre #post k etid);
  rewrite (bigstar #u1 0 (SZ.v nblk * 1) post) as (bigstar #u1 0 (SZ.v nblk) post);
}
```

(* f<<<1, 1>>>(...); *)
```pulse
// Private
fn kernel_1_as_n
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (etid:erased tid_t { gdim_x etid == 1sz /\ bdim_x etid == 1sz })
  requires gpu ** thread_id etid ** pre
  ensures  gpu ** thread_id etid ** post
{
  k ()
}
```

```pulse
fn launch_kernel_1
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  requires cpu ** pre
  ensures  cpu ** post
{
  bigstar_single_intro 0 (fun (i: nat { 0 <= i /\ i < 1 }) -> pre);
  launch_kernel_n 1sz (fun etid -> kernel_1_as_n #pre #post k etid);
  bigstar_single_elim #0;
}
```
