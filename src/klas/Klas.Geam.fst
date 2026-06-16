module Klas.Geam

(* See Klas.Geam.fsti for the specification. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Complex32           (* cf32 -> cuFloatComplex *)
open Kuiper.Complex64           (* cf64 -> cuDoubleComplex *)
module Map = Kuiper.Kernel.Map

inline_for_extraction noextract
fn geam_gen (#et:Type0) {| scalar et |}
  (alpha beta : et)
  (len : szp { len <= max_blocks * max_threads })
  (c : array1 et (l1_forward len) { is_global c })
  (a : array1 et (l1_forward len) { is_global a })
  (b : array1 et (l1_forward len) { is_global b })
  (#sc : erased (lseq et len))
  (#sa : erased (lseq et len))
  (#sb : erased (lseq et len))
  (#fa : perm)
  (#fb : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  requires
    on gpu_loc (c |-> sc)
  ensures
    on gpu_loc (c |-> s_geam alpha beta sa sb)
{
  memcpy_device_to_device c a len;
  Map.map_gpu2 (fun (ci bi : et) -> add (mul alpha ci) (mul beta bi)) len c b;
}

let geam_f16 = geam_gen #f16
let geam_f32 = geam_gen #f32
let geam_f64 = geam_gen #f64
let geam_u32 = geam_gen #u32
let geam_u64 = geam_gen #u64
let geam_cf32 = geam_gen #cf32   (* cuBLAS Cgeam (op(A)=A, op(B)=B) *)
let geam_cf64 = geam_gen #cf64   (* cuBLAS Zgeam *)
