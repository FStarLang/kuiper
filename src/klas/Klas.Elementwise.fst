module Klas.Elementwise

#lang-pulse
open Kuiper
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
let add_const_step (#t:Type0) {| floating t |} (c x : t) : t = add x c

inline_for_extraction
let mul_const_step (#t:Type0) {| floating t |} (c x : t) : t = mul x c

fn silu_fw_bf16
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq bf16 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map silu_step s)
{ Map.map_gpu silu_step lena a }

fn neg_fw_bf16
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq bf16 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map neg_step s)
{ Map.map_gpu neg_step lena a }

fn rsqrt_fw_f32
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq f32 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map rsqrt_step s)
{ Map.map_gpu rsqrt_step lena a }

fn square_fw_f32
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq f32 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map square_step s)
{ Map.map_gpu square_step lena a }

fn cos_fw_f32
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq f32 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map cos_step s)
{ Map.map_gpu cos_step lena a }

fn sin_fw_f32
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq f32 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map sin_step s)
{ Map.map_gpu sin_step lena a }

fn add_fw_bf16
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward lena) { Array1.is_global a })
  (b : Array1.t bf16 (l1_forward lena) { Array1.is_global b })
  (#sa : erased (lseq bf16 lena))
  (#sb : erased (lseq bf16 lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (a |-> sa)
  ensures  on gpu_loc (a |-> (Map.lseq_map2 add_step sa sb <: lseq bf16 lena))
{ Map.map_gpu2 add_step lena a b }

fn mul_fw_bf16
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward lena) { Array1.is_global a })
  (b : Array1.t bf16 (l1_forward lena) { Array1.is_global b })
  (#sa : erased (lseq bf16 lena))
  (#sb : erased (lseq bf16 lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (a |-> sa)
  ensures  on gpu_loc (a |-> (Map.lseq_map2 mul_step sa sb <: lseq bf16 lena))
{ Map.map_gpu2 mul_step lena a b }

fn mul_fw_f32
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward lena) { Array1.is_global a })
  (b : Array1.t f32 (l1_forward lena) { Array1.is_global b })
  (#sa : erased (lseq f32 lena))
  (#sb : erased (lseq f32 lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (a |-> sa)
  ensures  on gpu_loc (a |-> (Map.lseq_map2 mul_step sa sb <: lseq f32 lena))
{ Map.map_gpu2 mul_step lena a b }

fn add_const_fw_f32
  (c : f32)
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq f32 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map (add_const_step c) s)
{ Map.map_gpu (add_const_step c) lena a }

fn mul_const_fw_f32
  (c : f32)
  (lena : szp { lena <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward lena) { Array1.is_global a })
  (#s : erased (lseq f32 lena))
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> lseq_map (mul_const_step c) s)
{ Map.map_gpu (mul_const_step c) lena a }
