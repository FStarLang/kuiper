module Kuiper.Reduction

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open FStar.SizeT { op_Less_Hat }

module GA = Kuiper.Array
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32

let size : sz = 1024sz

(* Reduces a simple scalar array with the addition operation, leaving
the result in the 0th index of the original array (overwriting that element). *)
[@@CPrologue "__device__"]
fn k_reduce
  (#tt : Type0)
  {| d : simple_scalar tt |}
  (size : sz)
  (a : gpu_array tt size)
  (#v : erased (seq tt))
  requires gpu ** GA.gpu_pts_to_array a v ** pure (size > 0)
  returns  r : tt
  ensures  gpu ** GA.gpu_pts_to_array a v
{
  let mut i = 0sz;
  let mut r : tt = zero #tt #d;
  unfold (gpu_pts_to_array a v);

  while (let vi = !i; (FStar.SizeT.op_Less_Hat vi size))
    invariant b.
      exists* vi vr.
        gpu ** // infer automatically
        GA.gpu_pts_to_array_slice a 0 (SZ.v size) v ** // infer automatically
        Pulse.Lib.Reference.pts_to i vi **
        Pulse.Lib.Reference.pts_to r vr **
        pure (b == (vi <^ size))
  {
    let vi = !i;
    // FIXME: using #t instead of #tt (t = SizeT.t) gives a terrible error
    let v = gpu_array_read #tt #(SZ.v size) #0 #(SZ.v size) a #1.0R vi;
    let vr = !r;
    let vr' = add vr v;
    r := vr';
  };
  fold (gpu_pts_to_array a v);
  !r
}

(* Reduces a simple scalar array with the addition operation, leaving
the result in the 0th index of the original array (overwriting that element). *)
[@@CPrologue "__global__"]
fn k_reduce_and_set
  (#tt : Type0)
  {| d : simple_scalar tt |}
  (size : sz)
  (a : gpu_array tt size)
  (#v : erased (seq tt))
  ()
  requires gpu ** (GA.gpu_pts_to_array a v ** pure (size > 0))
  ensures  gpu ** (exists* v'. GA.gpu_pts_to_array a v')
{
  let r = k_reduce size a;
  unfold (gpu_pts_to_array a v);
  gpu_array_write #tt #(SZ.v size) #0 #(SZ.v size) a 0sz r;
  with v'. assert (gpu_pts_to_array_slice a 0 (SZ.v size) v');
  fold (gpu_pts_to_array a v');
}

inline_for_extraction noextract
[@@noextract_to "krml"]	
fn copy_to_gpu
  (#t:Type0)
  {| d : sized t |}
  (sz : sz)
  (a : A.array t)
  (#v : erased (seq t))
  requires cpu ** A.pts_to a v
  returns  ga : GA.gpu_array t sz
  ensures  cpu ** A.pts_to a v ** GA.gpu_pts_to_array ga v
{
  let ga = gpu_array_alloc #t #d sz;
  gpu_memcpy_host_to_device ga a sz;
  ga
}

inline_for_extraction noextract
[@@noextract_to "krml"]	
fn reduce
  (#t : Type0)
  {| d : simple_scalar t |}
  (a : array t)
  (size : sz)
  (#v : erased (seq t))
  requires cpu
        ** A.pts_to a v
        ** pure (size > 0 /\
                 len v == size)
  returns  r : t
  ensures  cpu ** (exists* v'. A.pts_to a v')
{
  let ga = copy_to_gpu size a;
  launch_kernel_1
    (k_reduce_and_set #t #d size ga #v);
  gpu_memcpy_device_to_host a ga size;
  gpu_array_free ga;
  with v'. assert (A.pts_to a v');
  assert (pure (len v' > 0));
  let r = Pulse.Lib.Array.op_Array_Access #t a 0sz #1.0R #v';
  r
}

(*

fn reduce_F32
  (a : array f32)
  (size : sz)
  (#v : erased (seq f32))
  requires cpu
        ** A.pts_to a v
        ** pure (size > 0 /\
                 len v == size)
  returns  r : f32
  ensures  cpu ** (exists* v'. A.pts_to a v')
{
  reduce #f32 a size
}

