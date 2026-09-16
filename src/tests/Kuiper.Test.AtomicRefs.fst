module Kuiper.Test.AtomicRefs

#lang-pulse

open Kuiper
open Kuiper.Atomics
open Kuiper.Shape { up }

module R = Kuiper.Ref
module A = Kuiper.Array.Core
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

// A GPU execution token and local ownership do not make a stack address a
// valid CUDA atomic target. Check each primitive and the typeclass entry point.
[@@expect_failure [19]]
inline_for_extraction noextract
fn local_u32 ()
  preserves gpu
{
  let mut storage = [| 0ul; 1sz |];
  let r = R.of_array storage;
  rewrite Pulse.Lib.Array.pts_to storage (Seq.create 1 0ul)
       as R.pts_to r 0ul;
  let _ = gpu_faa_u32 r 1ul;
  ()
}

[@@expect_failure [19]]
inline_for_extraction noextract
fn local_u64 ()
  preserves gpu
{
  let mut storage = [| 0uL; 1sz |];
  let r = R.of_array storage;
  rewrite Pulse.Lib.Array.pts_to storage (Seq.create 1 0uL)
       as R.pts_to r 0uL;
  let _ = gpu_faa_u64 r 1uL;
  ()
}

[@@expect_failure [19]]
inline_for_extraction noextract
fn local_f32 ()
  preserves gpu
{
  let mut storage = [| zero #f32 #_; 1sz |];
  let r = R.of_array storage;
  rewrite Pulse.Lib.Array.pts_to storage (Seq.create 1 (zero #f32 #_))
       as R.pts_to r (zero #f32 #_);
  let _ = gpu_faa_f32 r (zero #f32 #_);
  ()
}

[@@expect_failure [19]]
inline_for_extraction noextract
fn local_f64 ()
  preserves gpu
{
  let mut storage = [| zero #f64 #_; 1sz |];
  let r = R.of_array storage;
  rewrite Pulse.Lib.Array.pts_to storage (Seq.create 1 (zero #f64 #_))
       as R.pts_to r (zero #f64 #_);
  let _ = gpu_faa_f64 r (zero #f64 #_);
  ()
}

[@@expect_failure [19]]
inline_for_extraction noextract
fn local_typeclass ()
  preserves gpu
{
  let mut storage = [| 0ul; 1sz |];
  let r = R.of_array storage;
  rewrite Pulse.Lib.Array.pts_to storage (Seq.create 1 0ul)
       as R.pts_to r 0ul;
  let _ = atomic_add r 1ul;
  ()
}

// Ownership of a parameter is insufficient without its storage provenance.
[@@expect_failure [19]]
inline_for_extraction noextract
fn unknown_storage (r : ref u32)
  preserves gpu
  requires r |-> 'v
  ensures r |-> add 1ul 'v
{
  let _ = gpu_faa_u32 r 1ul;
  ()
}

inline_for_extraction noextract
fn global_ref (r : ref u32)
  preserves gpu
  requires pure (is_global_ref r)
  requires r |-> 'v
  returns old : u32
  ensures r |-> add 1ul 'v ** pure (old == reveal 'v)
{
  gpu_faa_u32 r 1ul
}

inline_for_extraction noextract
fn shared_ref (r : ref f32) (increment : f32)
  preserves gpu
  requires pure (is_shared_ref r)
  requires r |-> 'v
  returns old : f32
  ensures r |-> add increment 'v ** pure (old == reveal 'v)
{
  gpu_faa_f32 r increment
}

inline_for_extraction noextract
fn shared_typeclass (r : ref u64)
  preserves gpu
  requires pure (R.visibility_of_ref r == block_of)
  requires r |-> 'v
  returns old : u64
  ensures r |-> add 1uL 'v ** pure (old == reveal 'v)
{
  atomic_add r 1uL
}

inline_for_extraction noextract
fn global_allocation ()
  preserves cpu
{
  let r = Kuiper.Ref.alloc0 #u32 ();
  assert pure (is_atomic_ref r);
  Kuiper.Ref.free r;
}

// References into shared arrays inherit the array's storage provenance.
inline_for_extraction noextract
fn shared_array_cell (arr : array u32) (i : sz)
  preserves gpu
  requires pure (Kuiper.SHMem.is_block arr)
  requires pts_to_cell arr (SZ.v i) 'v
  returns old : u32
  ensures pts_to_cell arr (SZ.v i) (add 1ul 'v)
  ensures pure (old == reveal 'v)
{
  A.cell_to_ref arr (SZ.v i);
  let r = A.get_ref_of_cell arr i;
  assert rewrites_to r (A.ref_of_cell arr (SZ.v i));
  let old = gpu_faa_u32 r 1ul;
  A.cell_from_ref arr (SZ.v i);
  old
}

// Converting a local array cell to a reference must not enable atomics.
[@@expect_failure [19]]
inline_for_extraction noextract
fn local_array_cell (arr : array u32) (i : sz)
  preserves gpu
  requires pure (visibility_of arr == process_of)
  requires pts_to_cell arr (SZ.v i) 'v
  returns old : u32
  ensures pts_to_cell arr (SZ.v i) (add 1ul 'v)
  ensures pure (old == reveal 'v)
{
  A.cell_to_ref arr (SZ.v i);
  let r = A.get_ref_of_cell arr i;
  assert rewrites_to r (A.ref_of_cell arr (SZ.v i));
  let old = gpu_faa_u32 r 1ul;
  A.cell_from_ref arr (SZ.v i);
  old
}

// The tensor wrapper must preserve the same fact through its layout mapping.
inline_for_extraction noextract
fn tensor_cell
  (#rank : nat) (#s : T.shape rank) (#l : T.tlayout s)
  (arr : T.tensor u32 l) {| T.ctlayout l |} (i : T.conc s)
  preserves gpu
  requires pure (visibility_of (T.core arr) == gpu_of \/
                 visibility_of (T.core arr) == block_of)
  requires T.pts_to_cell arr (up i) 'v
  returns old : u32
  ensures T.pts_to_cell arr (up i) (add 1ul 'v)
  ensures pure (old == reveal 'v)
{
  T.cell_to_ref arr (up i);
  let r = T.get_ref_of_cell arr i;
  assert rewrites_to r (T.ref_of_cell arr (up i));
  let old = gpu_faa_u32 r 1ul;
  T.cell_from_ref arr (up i);
  old
}
