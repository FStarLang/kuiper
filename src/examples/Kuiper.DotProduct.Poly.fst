module Kuiper.DotProduct.Poly

#lang-pulse

open Kuiper
module V = Pulse.Lib.Vec
module SZ = FStar.SizeT

inline_for_extraction
let m_size : sz = 1024sz
let size = m_size


[@@coercion]
inline_for_extraction
let uint32_to_sizet x = FStar.SizeT.uint32_to_sizet x

let kpre #et (size:nat) (ga1 ga2 r : gpu_array et size) (tid:nat) : slprop =
  gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid **
  gpu_pts_to_array1 r tid

let kpost #et (size:nat) (ga1 ga2 r : gpu_array et size) (tid:nat) : slprop =
  gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid **
  gpu_pts_to_array1 r tid

fn kf
  (#et:Type0) {| scalar et |}
  (#size : erased nat)
  (ga1 ga2 r : gpu_array et size)
  (tid : szlt size)
  ()
  requires
    gpu **
    kpre size ga1 ga2 r tid **
    thread_id size tid
  ensures
    gpu **
    kpost size ga1 ga2 r tid **
    thread_id size tid
{
  (* r[id] = ga1[id] * ga2[id] *)

  (**)unfold (kpre size ga1 ga2 r tid);

  (**)unfold (gpu_pts_to_array1 ga1 tid);
  let v1 = gpu_array_read #_ #(reveal size) #tid #(tid+1) ga1 tid;

  (**)unfold (gpu_pts_to_array1 ga2 tid);
  let v2 = gpu_array_read #_ #(reveal size) #tid #(tid+1) ga2 tid;

  let v = v1 `mul` v2;

  (**)unfold (gpu_pts_to_array1 r tid);
  gpu_array_write #_ #(reveal size) #tid #(tid+1) r tid v;

  (**)fold (gpu_pts_to_array1 r tid);
  (**)fold (gpu_pts_to_array1 ga1 tid);
  (**)fold (gpu_pts_to_array1 ga2 tid);
  (**)fold (kpost size ga1 ga2 r tid);
  ()
}

ghost
fn block_setup
    (#et:Type) (ga1 ga2 gr : gpu_array et size)
    ()
    requires
      block_setup_tok m_size **
      (exists* s1 s2 sr.
        (ga1 |-> s1) **
        (ga2 |-> s2) **
        (gr |-> sr))
    ensures
      block_setup_tok m_size **
      (forall+ (tid : natlt m_size). kpre m_size ga1 ga2 gr tid) **
      emp (* frame *)
{
  // Slicing the arrays
  (**)gpu_array_slice_1_underspec #1 ga1;
  (**)gpu_array_slice_1_underspec #2 ga2;
  (**)gpu_array_slice_1_underspec #3 gr;

  // Boring combination of resources
  (**)bigstar_zip #1 #2 #1 0 size _ _;
  (**)bigstar_zip #1 #3 #0 0 size _ _;

  (**)bigstar_uneta ();

  forevery_fromnat size
    (fun i -> kpre m_size ga1 ga2 gr i);

  forevery_rw_size size (SZ.v m_size);
}

ghost
fn block_teardown
    (#et:Type) (ga1 ga2 gr : gpu_array et size)
    ()
    requires
      (forall+ (tid : natlt m_size). kpre m_size ga1 ga2 gr tid) **
      emp (* frame *)
    ensures
      (exists* s1 s2 sr.
        (ga1 |-> s1) **
        (ga2 |-> s2) **
        (gr |-> sr))
{
  forevery_rw_size (SZ.v m_size) size;
  forevery_tonat size
    (fun i -> kpost m_size ga1 ga2 gr i);

  (**)bigstar_unzip #1 #2 #0 0 size _ _;
  (**)bigstar_unzip #3 #4 #1 0 size _ _;

  rewrite each SZ.v m_size as size;

  (* Why is this needed? *)
  bigstar_uneta () #3 #0 #size #(gpu_pts_to_array1 ga1 #1.0R);
  bigstar_uneta () #4 #0 #size #(gpu_pts_to_array1 ga2 #1.0R);
  bigstar_uneta () #_ #0 #size #(gpu_pts_to_array1 gr  #1.0R);

  // Unslicing
  (**)gpu_array_unslice_1_underspec #3 ga1;
  (**)gpu_array_unslice_1_underspec ga2;
  (**)gpu_array_unslice_1_underspec gr;
}


inline_for_extraction noextract
let kdesc (#et:Type) {| scalar et |} (ga1 ga2 r : gpu_array et size)
  : kernel_desc
      (exists* s1 s2 sr.
        (ga1 |-> s1) **
        (ga2 |-> s2) **
        (r |-> sr))
      (exists* s1 s2 sr.
        (ga1 |-> s1) **
        (ga2 |-> s2) **
        (r |-> sr))
= {
  f = kf #et ga1 ga2 r;
  nthr = m_size;
  frame = emp;
  kpre  = kpre #et size ga1 ga2 r;
  kpost = kpost #et size ga1 ga2 r;
  block_setup = block_setup #et ga1 ga2 r;
  block_teardown = block_teardown #et ga1 ga2 r;
} <: kernel_desc_1_n _ _

inline_for_extraction noextract
fn main (#et:Type0) {| scalar et |} (_:unit)
  requires cpu ** pure SZ.fits_u32
  ensures  cpu
{
  let a1 = V.alloc #et zero m_size;
  let a2 = V.alloc #et zero m_size;
  let ar = V.alloc #et zero m_size;

  let mut i = 0sz;
  let mut a = zero #et #_;

  nuwhile (below i m_size)
    invariant live i ** live a
    invariant (exists* (s:seq et). (a1 |-> s) ** pure (len s == size))
    invariant (exists* (s:seq et). (a2 |-> s) ** pure (len s == size))
  {
    let v = !i;
    let va = !a;
    a1.(v) <- va;
    a2.(v) <- va;

    i := SZ.add v 1sz;
    a := va `add` one #et;
    ()
  };

  let ga1 = gpu_array_alloc #et m_size;
  let ga2 = gpu_array_alloc #et m_size;



  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 m_size;
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 m_size;

  let gr = gpu_array_alloc #et m_size;

  launch_sync (kdesc ga1 ga2 gr);

  Kuiper.Array.gpu_memcpy_device_to_host ar gr m_size;
  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  i := 0sz;
  let mut psum = zero #et #_;
  nuwhile (below i m_size)
    invariant live i ** live psum
  {
    let vi = !i;
    let ri = ar.(vi);
    let vpsum = !psum;
    psum := vpsum `add` ri;
    i := SZ.add vi 1sz;
  };

  V.free a1;
  V.free a2;
  V.free ar;

  ()
}
