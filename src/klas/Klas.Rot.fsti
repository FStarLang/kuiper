module Klas.Rot

(* BLAS level-1 rot: apply a Givens rotation to two vectors,
     x := c*x + s*y
     y := c*y - s*x        (using the old x)
   Corresponds to cublasSrot/Drot (real cosine/sine).

   Implemented out-of-place via a scratch copy of x and two pointwise maps, so
   it reuses the verified copy + map kernels. Specs are exact at the element
   type (pure pointwise floating-point arithmetic, no approximation). *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Map = Kuiper.Kernel.Map

let s_rot_x (#et:Type0) {| floating et |} (#n:nat) (c s : et) (vx vy : lseq et n)
  : GTot (lseq et n)
  = Map.lseq_map2 (fun (xi yi : et) -> add (mul c xi) (mul s yi)) vx vy

let s_rot_y (#et:Type0) {| floating et |} (#n:nat) (c s : et) (vx vy : lseq et n)
  : GTot (lseq et n)
  = Map.lseq_map2 (fun (yi xi : et) -> sub (mul c yi) (mul s xi)) vy vx

inline_for_extraction noextract
type rot_ty (et:Type0) {| floating et |} =
  fn (c s : et)
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

val rot_f16 : rot_ty f16
val rot_f32 : rot_ty f32
val rot_f64 : rot_ty f64
