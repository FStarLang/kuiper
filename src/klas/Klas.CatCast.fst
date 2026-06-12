module Klas.CatCast

#lang-pulse
open Kuiper

module Array1 = Kuiper.Array1
module Map = Kuiper.Kernel.Map
module Casts = Kuiper.Float.Casts.Base
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Layout.Alg

let cat2_seq
  #et
  #len
  (sa : lseq et len)
  (sb : lseq et len)
  : GTot (lseq et (len + len))
  = Seq.init_ghost (len + len) (fun i -> if i < len then sa @! i else sb @! (i - len))

fn cat2_bf16
  (len : szp { SZ.fits (len + len) /\ len + len <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward len) { Array1.is_global a })
  (b : Array1.t bf16 (l1_forward len) { Array1.is_global b })
  (#sa #sb : erased (lseq bf16 len))
  (#fa #fb : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> Frac fa sa) **
    on gpu_loc (b |-> Frac fb sb)
  returns c : Array1.t bf16 (l1_forward (len + len))
  ensures exists* sc. on gpu_loc (c |-> sc) ** pure (Array1.is_global c)
{
  Array1.alloc0 #bf16 (len +^ len) (l1_forward (len +^ len))
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
  ensures exists* sc. on gpu_loc (c |-> sc) ** pure (Array1.is_global c)
{
  let c = Array1.alloc0 #ot len (l1_forward len);
  with sc. assert on gpu_loc (c |-> sc);
  Map.map_gpu_cast f len a c #_ #_ #sa #sc #fa;
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
  ensures exists* sc. on gpu_loc (c |-> sc) ** pure (Array1.is_global c)
{
  cast #bf16 #f32 Casts.cast_bf16_to_f32 len a #sa #fa
}

fn cast_f32_to_bf16
  (len : szp { len <= max_blocks * max_threads })
  (a : Array1.t f32 (l1_forward len) { Array1.is_global a })
  (#sa : erased (lseq f32 len))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  returns c : Array1.t bf16 (l1_forward len)
  ensures exists* sc. on gpu_loc (c |-> sc) ** pure (Array1.is_global c)
{
  cast #f32 #bf16 Casts.cast_f32_to_bf16 len a #sa #fa
}

fn cast_bf16_to_bf16
  (len : szp { len <= max_blocks * max_threads })
  (a : Array1.t bf16 (l1_forward len) { Array1.is_global a })
  (#sa : erased (lseq bf16 len))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  returns c : Array1.t bf16 (l1_forward len)
  ensures exists* sc. on gpu_loc (c |-> sc) ** pure (Array1.is_global c)
{
  cast #bf16 #bf16 id len a #sa #fa
}
