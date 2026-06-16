module Klas.Rot

(* See Klas.Rot.fsti for the specification. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Complex32
open Kuiper.Complex64
open Kuiper.Complex.Class { complex, of_real }
module CC = Kuiper.Complex.Class
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

(* Csrot/Zdrot: real cosine/sine, complex vectors. See Klas.Rot.fsti. *)
inline_for_extraction noextract
fn csrot_gen (#c #r:Type0) {| scalar c |} {| floating r |} {| complex c r |}
  (cc ss : r)
  (lena : szp { lena <= max_blocks * max_threads })
  (x : array1 c (l1_forward lena) { is_global x })
  (y : array1 c (l1_forward lena) { is_global y })
  (#vx : erased (lseq c lena))
  (#vy : erased (lseq c lena))
  preserves cpu
  requires
    on gpu_loc (x |-> vx) ** on gpu_loc (y |-> vy)
  ensures
    on gpu_loc (x |-> s_csrot_x cc ss vx vy) ** on gpu_loc (y |-> s_csrot_y cc ss vx vy)
{
  let tmp = alloc0 #c lena (l1_forward lena);
  memcpy_device_to_device tmp x lena;
  Map.map_gpu2 (fun (xi yi : c) -> add (mul (of_real cc) xi) (mul (of_real ss) yi)) lena x y;
  Map.map_gpu2 (fun (yi ti : c) -> add (mul (of_real cc) yi) (mul (of_real (sub zero ss)) ti)) lena y tmp;
  free tmp;
}

let csrot_cf32 = csrot_gen #cf32 #f32
let csrot_cf64 = csrot_gen #cf64 #f64

(* Crot/Zrot: real cosine, COMPLEX sine; needs conj + complex subtraction. *)
inline_for_extraction noextract
fn crot_gen (#c #r:Type0) {| scalar c |} {| floating r |} {| complex c r |}
  (cc : r) (ss : c)
  (lena : szp { lena <= max_blocks * max_threads })
  (x : array1 c (l1_forward lena) { is_global x })
  (y : array1 c (l1_forward lena) { is_global y })
  (#vx : erased (lseq c lena))
  (#vy : erased (lseq c lena))
  preserves cpu
  requires
    on gpu_loc (x |-> vx) ** on gpu_loc (y |-> vy)
  ensures
    on gpu_loc (x |-> s_crot_x cc ss vx vy) ** on gpu_loc (y |-> s_crot_y cc ss vx vy)
{
  let tmp = alloc0 #c lena (l1_forward lena);
  memcpy_device_to_device tmp x lena;
  Map.map_gpu2 (fun (xi yi : c) -> add (mul (of_real cc) xi) (mul ss yi)) lena x y;
  Map.map_gpu2 (fun (yi ti : c) -> CC.csub #c #r (mul (of_real cc) yi) (mul (CC.cconj #c #r ss) ti)) lena y tmp;
  free tmp;
}

let crot_cf32 = crot_gen #cf32 #f32
let crot_cf64 = crot_gen #cf64 #f64
