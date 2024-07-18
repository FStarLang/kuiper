module GPU.DotProduct

open FStar.Mul
open Pulse.Lib.Array
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32
open Pulse.Lib.BigStar
open GPU

let m_size : SZ.t = 1024sz
let size : n:(erased nat){reveal n == SZ.v m_size} = SZ.v m_size

let elem_t = U32.t

(* This sucks, clean it up. Multiparam TC? *)
[@@coercion]
let uint32_to_int (x:U32.t) : GTot nat = U32.v x
[@@coercion]
let uint32_to_erased_nat (x:U32.t) : erased nat = U32.v x

[@@coercion]
let sizet_to_int (x:SZ.t) : GTot int = SZ.v x
[@@coercion]
let sizet_to_erased_nat (x:SZ.t) : erased nat = SZ.v x

[@@coercion]
inline_for_extraction
let uint32_to_sizet x = FStar.SizeT.uint32_to_sizet x

let kpre (ga1 ga2 r : gpu_array elem_t (SZ.v m_size)) (nthr : nat) (tid:nat{tid < nthr}) : slprop =
  gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid **
  gpu_pts_to_array1 r tid

let kpost (ga1 ga2 r : gpu_array elem_t (SZ.v m_size)) (nthr : nat) (tid:nat{tid < nthr}) : slprop =
  gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid **
  gpu_pts_to_array1 r tid

[@@CPrologue "__global__"]
```pulse
fn kernel
  (ga1 ga2 : gpu_array elem_t size)
  (r : gpu_array elem_t size)
  (nthr : erased nat)
  (etid : erased nat{etid < nthr})
  requires gpu ** thread_id etid ** kpre ga1 ga2 r nthr etid
  ensures  gpu ** thread_id etid ** kpost ga1 ga2 r nthr etid
{
  let tid = block_idx_x ();
  (* r[tid] = ga1[tid] * ga2[tid] *)

  (**)unfold (kpre ga1 ga2 r nthr tid);

  (**)unfold (gpu_pts_to_array1 ga1 tid);
  (**)gpu_pts_to_slice_ref ga1 tid (tid+1); // recall tid < size
  let v1 = gpu_array_read #_ #size #tid #(tid+1) ga1 tid;

  (**)unfold (gpu_pts_to_array1 ga2 tid);
  let v2 = gpu_array_read #_ #size #tid #(tid+1) ga2 tid;
  
  let v = v1 `U32.mul_underspec` v2;
  
  (**)unfold (gpu_pts_to_array1 r tid);
  gpu_array_write #_ #size #tid #(tid+1) r tid v;

  (**)fold (gpu_pts_to_array1 r tid);
  (**)fold (gpu_pts_to_array1 ga1 tid);
  (**)fold (gpu_pts_to_array1 ga2 tid);
  (**)fold (kpost ga1 ga2 r nthr tid);
}
```

```pulse
fn main (_:unit)
  requires cpu ** pure SizeT.fits_u32
  ensures  cpu
{
  let a1 = A.alloc #elem_t 0ul m_size;
  let a2 = A.alloc #elem_t 0ul m_size;
  let ar = A.alloc #elem_t 0ul m_size;

  let mut i = 0sz;

  while (let v = !i; (v `SZ.op_Less_Hat` m_size))
     invariant b.
       exists* v. pts_to i v **
       (exists* s. A.pts_to a1 s ** pure (Seq.length s == size)) **
       (exists* s. A.pts_to a2 s ** pure (Seq.length s == size)) **
       pure (b == (SZ.v v < size))
  {
    let v = !i;
    a1.(v) <- FStar.SizeT.sizet_to_uint32 v;
    a2.(v) <- FStar.SizeT.sizet_to_uint32 v;
    i := SZ.add v 1sz;
    ()
  };

  let ga1 = gpu_array_alloc #U32.t m_size;
  let ga2 = gpu_array_alloc #U32.t m_size;

  GPU.Array.gpu_memcpy_host_to_device a1 ga1 m_size;
  GPU.Array.gpu_memcpy_host_to_device a2 ga2 m_size;
  
  let gr = gpu_array_alloc #U32.t m_size;
  
  let nthr : U32.t = SZ.sizet_to_uint32 m_size <: U32.t;
  
  // Slicing the arrays
  (**)gpu_array_slice_1_underspec ga1;
  (**)gpu_array_slice_1_underspec ga2;
  (**)gpu_array_slice_1_underspec gr;

  // Boring combination of resources
  (**)bigstar_zip 0 (SZ.v m_size) (gpu_pts_to_array1 ga1) (gpu_pts_to_array1 ga2);
  (**)bigstar_zip 0 (SZ.v m_size) _ (gpu_pts_to_array1 gr);

  (**)rewrite
    (bigstar 0 (SZ.v m_size)
      (fun i -> (gpu_pts_to_array1 ga1 i **
                 gpu_pts_to_array1 ga2 i) **
                 gpu_pts_to_array1 gr i))
  as
    (bigstar 0 (SZ.v m_size) (fun i -> kpre ga1 ga2 gr nthr i));

  (**)bigstar_uneta ();

  assert (pure (SZ.v m_size == U32.v nthr));
  rewrite
    (bigstar 0 (SZ.v m_size) (kpre ga1 ga2 gr nthr))
  as
    (bigstar 0 (U32.v nthr)  (kpre ga1 ga2 gr nthr));

  launch_kernel_n #0 nthr (fun tid -> kernel ga1 ga2 gr (hide (U32.v nthr)) tid);

  rewrite
    (bigstar 0 (U32.v nthr)  (kpost ga1 ga2 gr nthr))
  as
    (bigstar 0 (SZ.v m_size) (kpost ga1 ga2 gr nthr));

  (**)bigstar_eta ();

  rewrite
    (bigstar 0 (SZ.v m_size) (fun i -> kpre ga1 ga2 gr nthr i))
  as
    (bigstar 0 (SZ.v m_size)
      (fun i -> gpu_pts_to_array1 ga1 i **
                gpu_pts_to_array1 ga2 i **
                gpu_pts_to_array1 gr i));
  (**)bigstar_unzip 0 (SZ.v m_size) _ _;
  (**)bigstar_unzip 0 (SZ.v m_size) _ _;
  
  // Unslicing
  (**)gpu_array_unslice_1_underspec ga1;
  (**)gpu_array_unslice_1_underspec ga2;
  (**)gpu_array_unslice_1_underspec gr;
  
  GPU.Array.gpu_memcpy_device_to_host ar gr m_size;
  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  i := 0sz;
  let mut psum = 0ul;
  while (let v = !i; (v `SZ.op_Less_Hat` m_size))
     invariant b. exists* vi vpsum.
       pts_to i vi **
       pts_to psum vpsum  **
       (exists* s. A.pts_to ar s ** pure (Seq.length s == size)) **
       pure (b == (SZ.v vi < size))
  {
    let vi = !i;
    let ri = ar.(vi);
    let vpsum = !psum;
    psum := vpsum `U32.add_underspec` ri;
    i := SZ.add vi 1sz;
  };
  
  A.free a1;
  A.free a2;
  A.free ar;

  ()
}
```
