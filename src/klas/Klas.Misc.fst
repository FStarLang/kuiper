module Klas.Misc

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.Map
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }

inline_for_extraction noextract
fn kf_arange_i64
  (#len : erased (n:nat { SZ.fits n }))
  (out : array1 u64 (l1_forward len))
  (#s : erased (lseq u64 len))
  (i : szlt len)
  ()
  requires
    gpu **
    Cell out (i <: natlt len) |-> (s @! i)
  ensures
    gpu **
    Cell out (i <: natlt len) |-> FStar.UInt64.(uint_to_t (SZ.v i))
{
  Array1.write_cell out i FStar.UInt64.(uint_to_t (SZ.v i));
}

ghost
fn arange_setup
  (len : nat)
  (out : array1 u64 (l1_forward len))
  (#s : erased (lseq u64 len))
  ()
  norewrite
  requires
    out |-> s
  ensures
    (forall+ (i : natlt len). Cell out i |-> (s @! i)) **
    pure (SZ.fits len)
{
  Array1.pts_to_ref out;
  Array1.explode out;
}

ghost
fn arange_teardown
  (len : nat { SZ.fits len })
  (out : array1 u64 (l1_forward len))
  ()
  norewrite
  requires
    (forall+ (i : natlt len).
      Cell out i |-> FStar.UInt64.(uint_to_t i)) **
    pure (SZ.fits len)
  ensures
    exists* s'. out |-> s'
{
  forevery_map
    (fun (i : natlt len) -> Cell out i |-> FStar.UInt64.(uint_to_t i))
    (fun (i : natlt len) -> Cell out i |-> (Seq.init_ghost len (fun j -> FStar.UInt64.(uint_to_t j)) @! i))
    fn x { () };
  Array1.implode out;
}

inline_for_extraction noextract
let karange_i64
  (len : szp { len <= max_blocks * max_threads })
  (out : array1 u64 (l1_forward len))
  (#_ : squash (is_global out))
  (#s : erased (lseq u64 len))
  : kernel_desc
      (requires out |-> s)
      (ensures  exists* s'. out |-> s')
= {
    nthr = len;
    f = kf_arange_i64 out;
    frame = pure (SZ.fits len);
    teardown = arange_teardown len out;
    setup = arange_setup len out;
    kpre = (fun (i:natlt len) -> Cell out i |-> (s @! i));
    kpost = (fun (i:natlt len) -> Cell out i |-> FStar.UInt64.(uint_to_t i));
    kpost_sendable = solve;
    kpre_sendable = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn inst_arange_i64
  (len : szp { len <= max_blocks * max_threads })
  (out : array1 u64 (l1_forward len) { is_global out })
  (#s : erased (lseq u64 len))
  preserves cpu
  requires  on gpu_loc (out |-> s)
  ensures   exists* s'. on gpu_loc (out |-> s')
{
  launch_sync (karange_i64 len out);
}

inline_for_extraction noextract
fn inst_gather_bf16_u32
  (lens : szp)
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
{
  K.gather_gpu lens leni src idx out;
}

let arange_i64 = inst_arange_i64
let gather_bf16_u32 = inst_gather_bf16_u32
