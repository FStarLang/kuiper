module GPU.Base

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Seq
open Pulse.Lib.BigStar
module U32 = FStar.UInt32

(* Token for being in CPU code *)
val cpu : slprop

(* Token for being in GPU code *)
val gpu : slprop

let tid_t = nat

(* Token for being a particular thread *)
val thread_id : tid_t -> slprop

```pulse
val
fn block_idx_x () (#n:erased tid_t)
  requires thread_id n
  returns  id : U32.t
  ensures  thread_id n ** pure (n == U32.v id)
``` 

(* f<<<1, 1>>>(...); *)
```pulse
val
fn launch_kernel_1
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  requires cpu ** pre
  ensures  cpu ** post
```

(* f<<<n, 1>>>(...); *)
```pulse
val
fn launch_kernel_n
  (#u1: int)
  (nthr  : U32.t)
  (#pre  : (tid:nat{tid < U32.v nthr} -> slprop))
  (#post : (tid:nat{tid < U32.v nthr} -> slprop))
  (k :
    (etid:erased nat{etid < U32.v nthr}) ->
    stt unit (gpu ** thread_id etid ** pre etid)
             (fun _ -> gpu ** thread_id etid ** post etid)
  )
  requires cpu ** bigstar #u1 0 (U32.v nthr) pre
  ensures  cpu ** bigstar #u1 0 (U32.v nthr) post
```
