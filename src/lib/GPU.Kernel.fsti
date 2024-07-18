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

(* f<<<nblk, nthr>>>(...); *)
```pulse
val
fn launch_kernel_n_m_sync
  (#u1: int)
  (nblk : U32.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : U32.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < (nblk * nthr) } -> slprop))

  (barrier : (it:nat -> from: nat { 0 <= from /\ from < nthr } -> to: nat { 0 <= to /\ to < nthr } -> slprop))
  (smem_sz : U32.t)
  (smem : (to: nat { 0 <= to /\ to < nthr } -> slprop))
  (smem_split :
    (ar: gpu_array U32.t smem_sz) -> (v: FStar.Seq.seq U32.t) ->
    stt_ghost unit emp_inames
      (gpu_pts_to_array #U32.t #smem_sz ar #1.0R v) (fun _ -> bigstar 0 nthr smem))

  (k :
    (b : erased (GPU.Barrier2.barrier nthr)) -> (etid: erased tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit (         gpu ** thread_id etid ** smem (tidx_x etid) ** mbarrier_tok barrier b 0 (tidx_x etid) ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** smem (tidx_x etid) ** mbarrier_tok barrier b 0 (tidx_x etid) ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (nblk * nthr) pre
  ensures  cpu ** bigstar #u1 0 (nblk * nthr) post
```

(* f<<<nblk, nthr>>>(...); *)
```pulse
val
fn launch_kernel_n_m
  (#u1: int)
  (nblk : U32.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : U32.t { 0 < nthr /\ nthr <= max_threads })
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
  (nblk  : U32.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < U32.v nblk } -> slprop))
  (k :
    (etid:erased tid_t { gdim_x etid == nblk /\ bdim_x etid == 1ul }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  (etid:erased tid_t { gdim_x etid == nblk /\ bdim_x etid == 1ul })
  requires gpu ** thread_id etid ** pre (thread_index etid)
  ensures  gpu ** thread_id etid ** post (thread_index etid)
{
  k etid;
}
```

```pulse
fn launch_kernel_n
  (#u1: int)
  (nblk  : U32.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < U32.v nblk } -> slprop))
  (k :
    (etid:erased tid_t { gdim_x etid == nblk /\ bdim_x etid == 1ul }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  requires cpu ** bigstar #u1 0 (U32.v nblk) pre
  ensures  cpu ** bigstar #u1 0 (U32.v nblk) post
{
  rewrite (bigstar #u1 0 (U32.v nblk) pre) as (bigstar #u1 0 (U32.v nblk * 1) pre);
  launch_kernel_n_m #u1 nblk 1ul #pre #post (fun etid -> kernel_n_as_n_m nblk #pre #post k etid);
  rewrite (bigstar #u1 0 (U32.v nblk * 1) post) as (bigstar #u1 0 (U32.v nblk) post);
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
  (etid:erased tid_t { gdim_x etid == 1ul /\ bdim_x etid == 1ul })
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
  bigstar_single_intro 0 0 (fun (i: nat { 0 <= i /\ i < 1 }) -> pre);
  launch_kernel_n 1ul (fun etid -> kernel_1_as_n #pre #post k etid);
  bigstar_single_elim #0;
}
```
