module GPU.Base

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Seq
open Pulse.Lib.BigStar

(* Token for being in CPU code *)
val cpu : slprop

(* Token for being in GPU code *)
val gpu : slprop

(*
  __device__
  void f() {
    ...
  }

  f<<<1, 1>>>();
*)
val launch_kernel_1
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  : stt unit (cpu ** pre) (fun _ -> cpu ** post)

(*
  f<<<n, 1>>>();
*)
val launch_kernel_n
  (#u1: int)
  (nthr  : pos)
  (#pre  : (tid:nat{tid < nthr} -> slprop))
  (#post : (tid:nat{tid < nthr} -> slprop))
  (f :
    (tid:nat{tid < nthr}) ->
    stt unit (gpu ** pre tid) (fun _ -> gpu ** post tid)
  )
  : stt unit (cpu ** bigstar #u1 0 nthr pre)
             (fun _ -> cpu ** bigstar #u1 0 nthr post)
