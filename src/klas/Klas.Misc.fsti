module Klas.Misc

#lang-pulse
open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }

inline_for_extraction noextract
type arange_i64_ty =
  fn (len : szp { len <= max_blocks * max_threads })
     (out : array1 u64 (l1_forward len) { is_global out })
     (#s : erased (lseq u64 len))
  preserves cpu
  requires  on gpu_loc (out |-> s)
  ensures   exists* s'. on gpu_loc (out |-> s')

val arange_i64 : arange_i64_ty

inline_for_extraction noextract
type gather_bf16_u32_ty =
  fn (lens : szp)
     (leni : szp { leni <= max_blocks * max_threads })
     (src : array1 bf16 (l1_forward lens) { is_global src })
     (idx : array1 u32 (l1_forward leni) { is_global idx })
     (out : array1 bf16 (l1_forward leni) { is_global out })
     (#ss : erased (lseq bf16 lens))
     (#si : erased (lseq u32 leni))
     (#so : erased (lseq bf16 leni))
     (#fs #fi : perm)
  norewrite
  preserves cpu ** on gpu_loc (src |-> Frac fs ss) ** on gpu_loc (idx |-> Frac fi si)
  requires  on gpu_loc (out |-> so)
  ensures   exists* so'. on gpu_loc (out |-> so')

val gather_bf16_u32 : gather_bf16_u32_ty
