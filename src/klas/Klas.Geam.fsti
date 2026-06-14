module Klas.Geam

(* BLAS-like extension geam (no-transpose, contiguous case):
     C := alpha * A + beta * B,  elementwise.
   Corresponds to cublasSgeam/Dgeam with op(A)=A, op(B)=B and contiguous
   storage (an m x n matrix is passed as its length m*n flattened array).

   Reuses the verified device copy and pointwise-map kernels; the spec is exact
   at the element type. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Map = Kuiper.Kernel.Map

let s_geam (#et:Type0) {| scalar et |} (#n:nat) (alpha beta : et) (sa sb : lseq et n)
  : GTot (lseq et n)
  = Map.lseq_map2 (fun (ai bi : et) -> add (mul alpha ai) (mul beta bi)) sa sb

inline_for_extraction noextract
type geam_ty (et:Type0) {| scalar et |} =
  fn (alpha beta : et)
     (len : szp { len <= max_blocks * max_threads })
     (c : array1 et (l1_forward len) { is_global c })
     (a : array1 et (l1_forward len) { is_global a })
     (b : array1 et (l1_forward len) { is_global b })
     (#sc : erased (lseq et len))
     (#sa : erased (lseq et len))
     (#sb : erased (lseq et len))
     (#fa : perm)
     (#fb : perm)
  preserves
    cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  requires
    on gpu_loc (c |-> sc)
  ensures
    on gpu_loc (c |-> s_geam alpha beta sa sb)

val geam_f16 : geam_ty f16
val geam_f32 : geam_ty f32
val geam_f64 : geam_ty f64
val geam_u32 : geam_ty u32
val geam_u64 : geam_ty u64
