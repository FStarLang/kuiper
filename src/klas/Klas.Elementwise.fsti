module Klas.Elementwise

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Array1 = Kuiper.Array1
module Map = Kuiper.Kernel.Map

inline_for_extraction
let silu_step (#t:Type0) {| floating t |} (x : t) : t =
  mul x (div one (add one (fexp (sub zero x))))

inline_for_extraction
let neg_step (#t:Type0) {| floating t |} (x : t) : t = sub zero x

inline_for_extraction
let rsqrt_step (#t:Type0) {| floating t |} (x : t) : t = rsqrt x

inline_for_extraction
let square_step (#t:Type0) {| floating t |} (x : t) : t = mul x x

inline_for_extraction
let cos_step (#t:Type0) {| floating t |} (x : t) : t = cos x

inline_for_extraction
let sin_step (#t:Type0) {| floating t |} (x : t) : t = sin x

inline_for_extraction
let add_step (#t:Type0) {| floating t |} (x y : t) : t = add x y

inline_for_extraction
let mul_step (#t:Type0) {| floating t |} (x y : t) : t = mul x y

inline_for_extraction
let add_const_step (#t:Type0) {| floating t |} (c : t) (x : t) : t = add x c

inline_for_extraction
let mul_const_step (#t:Type0) {| floating t |} (c : t) (x : t) : t = mul x c

inline_for_extraction noextract
type unary_fw_ty (t:Type0) {| floating t |} (f : t -> t) =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : Array1.t t (l1_forward lena) { Array1.is_global a })
     (#s : erased (lseq t lena))
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> lseq_map f s)

val silu_fw_bf16  : unary_fw_ty bf16 silu_step
val neg_fw_bf16   : unary_fw_ty bf16 neg_step
val rsqrt_fw_f32  : unary_fw_ty f32 rsqrt_step
val square_fw_f32 : unary_fw_ty f32 square_step
val cos_fw_f32    : unary_fw_ty f32 cos_step
val sin_fw_f32    : unary_fw_ty f32 sin_step

inline_for_extraction noextract
type binary_fw_ty (t:Type0) {| floating t |} (f : t -> t -> t) =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : Array1.t t (l1_forward lena) { Array1.is_global a })
     (b : Array1.t t (l1_forward lena) { Array1.is_global b })
     (#sa : erased (lseq t lena))
     (#sb : erased (lseq t lena))
     (#fb : perm)
     norewrite
     preserves cpu ** on gpu_loc (b |-> Frac fb sb)
     requires  on gpu_loc (a |-> sa)
     ensures   on gpu_loc (a |-> (Map.lseq_map2 f sa sb <: lseq t lena))

val add_fw_bf16 : binary_fw_ty bf16 add_step
val mul_fw_bf16 : binary_fw_ty bf16 mul_step
val mul_fw_f32  : binary_fw_ty f32 mul_step

inline_for_extraction noextract
type unary_const_fw_ty (t:Type0) {| floating t |} (f : t -> t -> t) =
  fn (c : t)
     (lena : szp { lena <= max_blocks * max_threads })
     (a : Array1.t t (l1_forward lena) { Array1.is_global a })
     (#s : erased (lseq t lena))
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> lseq_map (f c) s)

val add_const_fw_f32 : unary_const_fw_ty f32 add_const_step
val mul_const_fw_f32 : unary_const_fw_ty f32 mul_const_step
