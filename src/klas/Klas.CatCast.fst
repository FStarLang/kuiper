module Klas.CatCast

#lang-pulse
open Kuiper

module Array1 = Kuiper.Array1
module Map = Kuiper.Kernel.Map
module Cat = Kuiper.Kernel.Cat
module Casts = Kuiper.Float.Casts.Base
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Layout.Alg

fn cat2_bf16
  (lena : szp)
  (lenb : szp { SZ.fits (lena + lenb) /\ lena + lenb <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward lena) { Array1.is_global a })
  (b : Array1.t bf16 (l1_forward lenb) { Array1.is_global b })
  (#sa : erased (lseq bf16 lena))
  (#sb : erased (lseq bf16 lenb))
  (#fa #fb : perm)
  norewrite
  preserves cpu
  preserves on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  returns c : Array1.t bf16 (l1_forward (lena + lenb))
  ensures
    on gpu_loc (c |-> (Cat.cat2_seq sa sb <: lseq bf16 (lena + lenb))) **
    pure (Array1.is_global c)
{
  Cat.cat2_gpu lena lenb a b
}

inline_for_extraction noextract
fn cast
  (#et #ot : Type0) {| sized ot |}
  (f : et -> ot)
  (len : szp { len <= max_blocks * max_threads })
  (a : Array1.t et (l1_forward len) { Array1.is_global a })
  (#sa : erased (lseq et len))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  returns c : Array1.t ot (l1_forward len)
  ensures
    on gpu_loc (c |-> (Kuiper.Seq.Common.lseq_map f sa <: lseq ot len)) **
    pure (Array1.is_global c)
{
  let c = Array1.alloc0 #ot len (l1_forward len);
  Map.map_gpu_notinplace f len a c;
  c
}

fn cast_bf16_to_f32
  (len : szp { len <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward len) { Array1.is_global a })
  (#sa : erased (lseq bf16 len))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  returns c : Array1.t f32 (l1_forward len)
  ensures
    on gpu_loc (c |-> (Kuiper.Seq.Common.lseq_map Casts.cast_bf16_to_f32 sa <: lseq f32 len)) **
    pure (Array1.is_global c)
{
  cast #bf16 #f32 Casts.cast_bf16_to_f32 len a
}

fn cast_f32_to_bf16
  (len : szp { len <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward len) { Array1.is_global a })
  (#sa : erased (lseq f32 len))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  returns c : Array1.t bf16 (l1_forward len)
  ensures
    on gpu_loc (c |-> (Kuiper.Seq.Common.lseq_map Casts.cast_f32_to_bf16 sa <: lseq bf16 len)) **
    pure (Array1.is_global c)
{
  cast #f32 #bf16 Casts.cast_f32_to_bf16 len a
}

fn cast_bf16_to_bf16
  (len : szp { len <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward len) { Array1.is_global a })
  (#sa : erased (lseq bf16 len))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  returns c : Array1.t bf16 (l1_forward len)
  ensures
    on gpu_loc (c |-> (Kuiper.Seq.Common.lseq_map id sa <: lseq bf16 len)) **
    pure (Array1.is_global c)
{
  cast #bf16 #bf16 id len a
}
