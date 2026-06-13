module Klas.Rot

(* See Klas.Rot.fsti for the specification. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Map = Kuiper.Kernel.Map

inline_for_extraction noextract
fn rot_gen (#et:Type0) {| floating et |}
  (c s : et)
  (lena : szp { lena <= max_blocks * max_threads })
  (x : array1 et (l1_forward lena) { is_global x })
  (y : array1 et (l1_forward lena) { is_global y })
  (#vx : erased (lseq et lena))
  (#vy : erased (lseq et lena))
  preserves cpu
  requires
    on gpu_loc (x |-> vx) ** on gpu_loc (y |-> vy)
  ensures
    on gpu_loc (x |-> s_rot_x c s vx vy) ** on gpu_loc (y |-> s_rot_y c s vx vy)
{
  let tmp = alloc0 #et lena (l1_forward lena);
  memcpy_device_to_device tmp x lena;
  Map.map_gpu2 (fun (xi yi : et) -> add (mul c xi) (mul s yi)) lena x y;
  Map.map_gpu2 (fun (yi ti : et) -> sub (mul c yi) (mul s ti)) lena y tmp;
  free tmp;
}

let rot_f16 = rot_gen #f16
let rot_f32 = rot_gen #f32
let rot_f64 = rot_gen #f64
