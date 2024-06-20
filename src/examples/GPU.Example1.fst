module GPU.Example1

open Pulse.Lib.Pervasives
open GPU

```pulse
fn kernel (r : gpu_ref int)
  requires gpu ** (exists* v. gpu_pts_to r v)
  ensures  gpu ** gpu_pts_to r 2
{
   gpu_write r 2; // *r = 2
}
```

```pulse
fn main (_:unit)
  requires cpu
  ensures  cpu
{
   let r = alloc 1;
   let gr = gpu_alloc0 #int ();
   
  //  GPU.Ref.gpu_memcpy_host_to_device r gr;
   
   (* kernel<<1,1>>(gr); *)
   launch_kernel_1 (fun () -> kernel gr);

   GPU.Ref.gpu_memcpy_device_to_host r gr;
   let v = !r;
   
   assert (pure (v == 2));
   
   gpu_free gr;
   free r;
}
```
