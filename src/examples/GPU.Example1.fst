module GPU.Example1

open Pulse.Lib
open Pulse.Lib.Pervasives
open GPU

module U64 = FStar.UInt64

```pulse
fn kernel (r : gpu_ref U64.t)
  requires gpu ** (exists* v. gpu_pts_to r v)
  ensures  gpu ** gpu_pts_to r 2uL
{
   gpu_write r 2uL; // *r = 2
}
```

```pulse
fn main (_:unit)
  requires cpu
  ensures  cpu
{
  let r  = Box.alloc #U64.t 1uL;
  let gr = gpu_alloc0 #U64.t ();
   
  Box.to_ref_pts_to r;
  GPU.Ref.gpu_memcpy_host_to_device #U64.t (Box.box_to_ref r) gr;

  (* kernel<<1,1>>(gr); *)
  launch_kernel_1 (fun () -> kernel gr);

  GPU.Ref.gpu_memcpy_device_to_host (Box.box_to_ref r) gr;
  Box.to_box_pts_to r;

  let v = Pulse.Lib.Box.(!r);
   
  assert (pure (v == 2uL));
   
  gpu_free gr;
  Box.free r;
}
```
