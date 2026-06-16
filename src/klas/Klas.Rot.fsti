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
open Kuiper.Complex32           (* cf32 -> cuFloatComplex *)
open Kuiper.Complex64           (* cf64 -> cuDoubleComplex *)
open Kuiper.Complex.Class { complex, of_real }
module CC = Kuiper.Complex.Class
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

(* ----- Csrot / Zdrot: a Givens rotation with REAL cosine/sine applied to two
   COMPLEX vectors (cublasCsrot / cublasZdrot). Same formulas as rot, but the
   coefficients c, s are real and x, y are complex:
       x := c*x + s*y
       y := c*y - s*x      (using the old x)
   We lift c, s to the complex type with the [complex] class's [of_real], and
   realise the subtraction by negating s in the reals (-s, then of_real),
   so the complex side uses only `scalar` add/mul (no complex subtraction).
   Specs are exact at the element type. ----- *)

let s_csrot_x (#c #r:Type0) {| scalar c |} {| floating r |} {| complex c r |} (#n:nat)
  (cc ss : r) (vx vy : lseq c n) : GTot (lseq c n)
  = Map.lseq_map2 (fun (xi yi : c) -> add (mul (of_real cc) xi) (mul (of_real ss) yi)) vx vy

let s_csrot_y (#c #r:Type0) {| scalar c |} {| floating r |} {| complex c r |} (#n:nat)
  (cc ss : r) (vx vy : lseq c n) : GTot (lseq c n)
  = Map.lseq_map2 (fun (yi xi : c) -> add (mul (of_real cc) yi) (mul (of_real (sub zero ss)) xi)) vy vx

inline_for_extraction noextract
type csrot_ty (c r:Type0) {| scalar c |} {| floating r |} {| complex c r |} =
  fn (cc ss : r)
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

val csrot_cf32 : csrot_ty cf32 f32   (* cuBLAS Csrot *)
val csrot_cf64 : csrot_ty cf64 f64   (* cuBLAS Zdrot *)

(* ----- Crot / Zrot: a Givens rotation with REAL cosine c but COMPLEX sine s,
   applied to two COMPLEX vectors (cublasCrot / cublasZrot):
       x := c*x + s*y
       y := c*y - conj(s)*x       (using the old x)
   Here s is complex, so the y-update genuinely needs the complex conjugate and
   complex subtraction (csub) from the [complex] class. Specs exact. ----- *)

let s_crot_x (#c #r:Type0) {| scalar c |} {| floating r |} {| complex c r |} (#n:nat)
  (cc : r) (ss : c) (vx vy : lseq c n) : GTot (lseq c n)
  = Map.lseq_map2 (fun (xi yi : c) -> add (mul (of_real cc) xi) (mul ss yi)) vx vy

let s_crot_y (#c #r:Type0) {| scalar c |} {| floating r |} {| complex c r |} (#n:nat)
  (cc : r) (ss : c) (vx vy : lseq c n) : GTot (lseq c n)
  = Map.lseq_map2 (fun (yi xi : c) -> CC.csub #c #r (mul (of_real cc) yi) (mul (CC.cconj #c #r ss) xi)) vy vx

inline_for_extraction noextract
type crot_ty (c r:Type0) {| scalar c |} {| floating r |} {| complex c r |} =
  fn (cc : r) (ss : c)
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

val crot_cf32 : crot_ty cf32 f32   (* cuBLAS Crot *)
val crot_cf64 : crot_ty cf64 f64   (* cuBLAS Zrot *)
