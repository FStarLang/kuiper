module Kuiper.Example1

#lang-pulse

open Kuiper

module U64 = FStar.UInt64

inline_for_extraction noextract
fn kf (r : ref u64) (#v0 : erased u64)
  preserves gpu
  requires r |-> v0
  ensures  r |-> U64.add_mod v0 1uL
{
  open U64;
  r := !r +%^ 1uL;
}

// An ordinary reference into an array can use GPU reference atomics directly.
inline_for_extraction noextract
fn atomic_add_cell (arr : array u32) (i : sz) (increment : u32)
  preserves gpu
  requires pts_to_cell arr (Kuiper.SizeT.v i) 'v
  returns old : u32
  ensures pts_to_cell arr (Kuiper.SizeT.v i) (add increment 'v)
  ensures pure (old == reveal 'v)
{
  array_cell_to_ref arr (Kuiper.SizeT.v i);
  let r = get_ref_of_array_cell arr i;
  assert rewrites_to r (ref_of_array_cell arr (Kuiper.SizeT.v i));
  let old = gpu_faa_u32 r increment;
  array_cell_from_ref arr (Kuiper.SizeT.v i);
  old
}

fn main (_:unit)
  preserves cpu
  requires emp
  returns  _ : u64
  ensures emp
{
  let mut r = 1uL;
  let gr = alloc0 #u64 ();

  Kuiper.Ref.memcpy_host_to_device gr r;
  with v. assert on gpu_loc (gr |-> v);
  launch_kernel_1 (fun _ -> kf gr #v);

  Kuiper.Ref.memcpy_device_to_host r gr;

  let v = !r;

  assert (pure (v == 2uL));

  free gr;
  v
}

// Device-to-device reference copy round trip.
fn test_device_to_device (_:unit)
  preserves cpu
  requires emp
  returns v : u64
  ensures pure (v == 2uL)
{
  let mut r = 2uL;
  let src = alloc0 #u64 ();
  let dst = alloc0 #u64 ();

  Kuiper.Ref.memcpy_host_to_device src r;
  Kuiper.Ref.memcpy_device_to_device dst src;
  r := 0uL;
  Kuiper.Ref.memcpy_device_to_host r dst;

  let v = !r;
  free src;
  free dst;
  v
}

// The pointer type is shared by host and device. Located ownership must still
// be brought to the current location before either a read or a write.
[@@expect_failure [228]]
fn host_cannot_read_device (r : ref u64)
  preserves cpu
  preserves on gpu_loc (r |-> 'v)
  returns x : u64
{
  !r
}

[@@expect_failure [228]]
fn host_cannot_write_device (r : ref u64)
  preserves cpu
  requires on gpu_loc (r |-> 'v)
  ensures on gpu_loc (r |-> 0uL)
{
  r := 0uL;
}

[@@expect_failure [228]]
fn device_cannot_read_host (r : ref u64) (#l : loc_id)
  preserves gpu
  requires pure (is_cpu_loc l)
  preserves on l (r |-> 'v)
  returns x : u64
{
  !r
}

[@@expect_failure [228]]
fn device_cannot_write_host (r : ref u64) (#l : loc_id)
  preserves gpu
  requires pure (is_cpu_loc l)
  requires on l (r |-> 'v)
  ensures on l (r |-> 0uL)
{
  r := 0uL;
}

// Ordinary references can point into larger allocations; ownership alone
// does not authorize freeing an interior pointer.
[@@expect_failure [19]]
fn cannot_free_without_full_allocation (r : ref u64)
  preserves cpu
  requires on gpu_loc (r |-> 'v)
  ensures emp
{
  Kuiper.Ref.free r;
}

// A reference's type alone does not let its ownership cross GPU blocks.
[@@expect_failure [19]]
let cannot_send_without_visibility (r : ref u64) (v : u64)
  : is_send_across gpu_of (r |-> v)
  = solve
