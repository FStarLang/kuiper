module GPU.DotProduct2

open FStar.Mul
open Pulse.Lib.Array
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
open Pulse.Lib.BigStar
open GPU

let size : nat = 1024

let kpre (ga1 ga2 r : gpu_array int size) (nth : nat) (tid:nat{tid < nth}) : slprop =
  (gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid) **
  gpu_pts_to_array1 r tid

let kpost (ga1 ga2 r : gpu_array int size) (nth : nat) (tid:nat{tid < nth}) : slprop =
  (gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid) **
  gpu_pts_to_array1 r tid

```pulse
fn kernel
  (ga1 ga2 : gpu_array int size)
  (r : gpu_array int size)
  (nth : erased nat)
  (tid : nat{tid < nth})
  requires gpu ** kpre  ga1 ga2 r nth tid
  ensures  gpu ** kpost ga1 ga2 r nth tid
{
  (**)unfold (kpre ga1 ga2 r nth tid);

  (**)unfold (gpu_pts_to_array1 ga1 tid);
  (**)gpu_pts_to_slice_ref ga1 tid (tid+1); // recall tid < size
  let v1 = gpu_array_read #int #size #tid #(tid+1) ga1 tid;

  (**)unfold (gpu_pts_to_array1 ga2 tid);
  let v2 = gpu_array_read #int #size #tid #(tid+1) ga2 tid;
  
  let v = v1 * v2;
  
  (**)unfold (gpu_pts_to_array1 r tid);
  gpu_array_write #int #size #tid #(tid+1) r tid v;
  
  (* Reduction *)
  let mut n = size / 2 <: int;
  with vn. assert (pts_to n vn);


  assume_ (if tid < size / 2 then (exists* s. gpu_pts_to_array_slice r (tid+size/2) (tid+size/2+1) s) else emp);

  admit();

  rewrite (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)
       as (if tid < vn then (exists* s. gpu_pts_to_array_slice r tid (tid+1) s) else emp);

  let a = 123;
  while (let vn = !n; (vn > 1))
    invariant b. exists* vn.
      gpu **
      (if tid < vn
       then (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)
       else emp) **
      pts_to n vn **
      pure (vn <= size / 2 /\ b == (vn > 1))
  {
    let vn = !n;
    if (tid < vn) {
      (**)rewrite
             (if tid < vn then (exists* s. gpu_pts_to_array_slice r tid (tid+1) s) else emp)
          as (exists* s. gpu_pts_to_array_slice r tid (tid+1) s);
      let vl = gpu_array_read #int #size #tid #(tid+1) r tid;
      assume_ (exists* s. gpu_pts_to_array_slice r (tid+vn) (tid+vn+1) s);
      assume_ (pure (tid + vn < size));
      let vr = gpu_array_read #int #size #(tid+vn) #(tid+vn+1) r (tid + vn);
      gpu_array_write #int #size #tid #(tid+1) r tid (vl + vr);
      n := vn / 2;
      ()
    } else {
      n := vn / 2;
      assert (if tid < (vn / 2) then (exists* s. gpu_pts_to_array_slice r tid (tid+1) s) else emp);
      (* sync_threads() *)
      ()
    };
  };

  (**)fold (gpu_pts_to_array1 r tid);
  (**)fold (gpu_pts_to_array1 ga1 tid);
  (**)fold (gpu_pts_to_array1 ga2 tid);
  (**)fold (kpost ga1 ga2 r nth tid);
}
```

```pulse
fn main (_:unit)
  requires cpu
  ensures  cpu
{
  let a1 = A.alloc 0 (SZ.uint_to_t size);
  let a2 = A.alloc 0 (SZ.uint_to_t size);
  let ar = A.alloc 0 (SZ.uint_to_t size);

  let mut i = 0sz;

  while (let v = !i; (SZ.v v < size))
     invariant b.
       exists* v. pts_to i v **
       (exists* s. A.pts_to a1 s ** pure (Seq.length s == size)) **
       (exists* s. A.pts_to a2 s ** pure (Seq.length s == size)) **
       pure (b == (SZ.v v < size))
  {
    let v = !i;
    a1.(v) <- SZ.v v;
    a2.(v) <- SZ.v v;
    i := SZ.add v 1sz;
    ()
  };

  let ga1 = gpu_array_alloc #int size;
  let ga2 = gpu_array_alloc #int size;

  GPU.Array.gpu_memcpy_host_to_device a1 ga1;
  GPU.Array.gpu_memcpy_host_to_device a2 ga2;
  
  let gr = gpu_array_alloc #int size;

  // Slicing the arrays
  (**)gpu_array_slice_1_underspec ga1;
  (**)gpu_array_slice_1_underspec ga2;
  (**)gpu_array_slice_1_underspec gr;
  
  // Boring combination of resources
  (**)bigstar_zip 0 size (gpu_pts_to_array1 ga1) (gpu_pts_to_array1 ga2);
  (**)bigstar_zip 0 size _ (gpu_pts_to_array1 gr);
  (**)rewrite
    (bigstar 0 size
      (fun i -> (gpu_pts_to_array1 ga1 i **
                 gpu_pts_to_array1 ga2 i) **
                 gpu_pts_to_array1 gr i))
  as
    (bigstar 0 size (fun i -> kpre ga1 ga2 gr size i));
  (**)bigstar_uneta ();
  
  launch_kernel_n size (kernel ga1 ga2 gr size);
  
  (**)bigstar_eta ();
  (**)rewrite
    (bigstar 0 size (fun i -> kpost ga1 ga2 gr size i))
  as
    (bigstar 0 size
      (fun i -> gpu_pts_to_array1 ga1 i **
                gpu_pts_to_array1 ga2 i **
                gpu_pts_to_array1 gr i));
  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;
  
  // Unslicing
  (**)gpu_array_unslice_1_underspec ga1;
  (**)gpu_array_unslice_1_underspec ga2;
  (**)gpu_array_unslice_1_underspec gr;
  
  GPU.Array.gpu_memcpy_device_to_host ar gr;

  i := 0sz;
  let mut psum = 0;
  while (let v = !i; (SZ.v v < size))
     invariant b. exists* vi vpsum.
       pts_to i vi **
       pts_to psum vpsum  **
       (exists* s. A.pts_to ar s ** pure (Seq.length s == size)) **
       pure (b == (SZ.v vi < size))
  {
    let vi = !i;
    let ri = ar.(vi);
    let vpsum = !psum;
    psum := vpsum + ri;
    i := SZ.add vi 1sz;
  };
  
  A.free a1;
  A.free a2;
  A.free ar;
  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ()
}
```
