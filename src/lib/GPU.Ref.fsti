module GPU.Ref

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open GPU.Base

val gpu_ref (a:Type u#0) : Type u#0

val gpu_pts_to
  (#a:Type u#0)
  (x:gpu_ref a)
  (#[exact (`1.0R)] f : perm)
  (v : a)
: vprop

val gpu_alloc0
  (#a:Type u#0)
  ()
: stt (gpu_ref a)
      cpu
      (fun x -> cpu ** (exists* (v:a). gpu_pts_to x v))

val gpu_alloc
  (#a:Type u#0)
  (v:a)
: stt (gpu_ref a)
      cpu
      (fun x -> cpu ** gpu_pts_to x v)

val gpu_free
  (#a:Type u#0)
  (r : gpu_ref a)
  (#v : erased a)
: stt unit
      (cpu ** gpu_pts_to r v)
      (fun _ -> cpu)

val gpu_read (#a:Type u#0) (x:gpu_ref a) (#f:perm) (#v0:erased a)
  : stt a (gpu ** gpu_pts_to x #f v0)
          (fun v -> gpu ** gpu_pts_to x #f v ** pure (v == reveal v0))

val gpu_write (#a:Type u#0) (x:gpu_ref a) (v:a)
  : stt unit (gpu ** (exists* v0. gpu_pts_to x v0))
             (fun _ -> gpu ** gpu_pts_to x v)

(* cudaMemcpy (_, _, _, cudaMemcpyHostToDevice) *)
val gpu_memcpy_host_to_device
  (#a:Type u#0)
  (r  : ref a)
  (#f : perm)
  (#v : erased a)
  (gr : gpu_ref a)
  (#gv : erased a)
  : stt unit (cpu ** pts_to r #f v ** gpu_pts_to gr #1.0R gv)
             (fun _ -> cpu ** pts_to r #f v ** gpu_pts_to gr #1.0R v)

(* cudaMemcpy (_, _, _, cudaMemcpyDeviceToHost) *)
val gpu_memcpy_device_to_host
  (#a:Type u#0)
  (r  : ref a)
  (#v : erased a)
  (gr : gpu_ref a)
  (#f:perm)
  (#gv : erased a)
  : stt unit (cpu ** pts_to r #1.0R v ** gpu_pts_to gr #f gv)
             (fun _ -> cpu ** pts_to r #1.0R gv ** gpu_pts_to gr #f gv)
