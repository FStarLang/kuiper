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
  preserves gpu ** (a |-> v)
  requires pure (size > 0)
  returns  r : tt
  ensures  emp
{
  let mut i = 0sz;
  let mut r : tt = zero #tt #d;

  while (let vi = !i; (FStar.SizeT.op_Less_Hat vi size))
    invariant b.
      exists* vi vr.
        gpu ** // infer automatically
        GA.gpu_pts_to_slice a 0 (SZ.v size) v ** // infer automatically
        (i |-> vi) **
        (r |-> vr) **
        pure (b == (vi <^ size))
  {
    let vi = !i;
    // FIXME: using #t instead of #tt (t = SizeT.t) gives a terrible error
    let v = gpu_array_read #tt #(SZ.v size) #0 #(SZ.v size) a #1.0R vi;
    let vr = !r;
    let vr' = add vr v;
    r := vr';
  };
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
  preserves gpu
  requires ((a |-> v) ** pure (size > 0))
  ensures  (exists* v'. (a |-> v') ** pure (Seq.length v' == size))
{
  let r = k_reduce size a;
  gpu_array_write #tt #(SZ.v size) #0 #(SZ.v size) a 0sz r;
  with v'. assert (gpu_pts_to_slice a 0 (SZ.v size) v');
}

inline_for_extraction noextract
[@@noextract_to "krml"]	
fn copy_to_gpu
  (#t:Type0)
  {| d : sized t |}
  (sz : sz)
  (a : vec t)
  (#v : erased (seq t){len v == reveal sz})
  preserves cpu ** (a |-> v)
  requires emp
  returns  ga : GA.gpu_array t sz
  ensures  ga |-> v
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
  (a : vec t)
  (size : sz)
  (#v : erased (seq t))
  preserves cpu
  requires
    (a |-> v) **
    pure (size > 0 /\
          len v == size)
  returns  r : t
  ensures 
    exists* v'.
      a |-> v'
{
  let ga = copy_to_gpu size a;
  launch_kernel_1
    (k_reduce_and_set #t #d size ga #v);
  gpu_memcpy_device_to_host a ga size;
  gpu_array_free ga;
  with v'. assert (a |-> v');
  assert (pure (len v' > 0));
  let r = a.(0sz);
  r
}

(*

fn reduce_F32
  (a : vec f32)
  (size : sz)
  (#v : erased (seq f32))
  requires cpu
        ** (a |-> v)
        ** pure (size > 0 /\
                 len v == size)
  returns  r : f32
  ensures  cpu ** (exists* v'. a |-> v')
{
  reduce #f32 a size
}

