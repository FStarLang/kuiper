module Kuiper.DotProduct.Poly

#lang-pulse

open Kuiper
module V = Pulse.Lib.Vec
module SZ = Kuiper.SizeT

inline_for_extraction
let m_size : sz = 1024sz
unfold let size = m_size

unfold
let kpre #et (size:nat) (ga1 ga2 r : gpu_array et size) (tid:nat) : slprop =
  gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid **
  gpu_pts_to_array1 r   tid

unfold
let kpost #et (size:nat) (ga1 ga2 r : gpu_array et size) (tid:nat) : slprop =
  gpu_pts_to_array1 ga1 tid **
  gpu_pts_to_array1 ga2 tid **
  gpu_pts_to_array1 r   tid

fn kf
  (#et:Type0) {| scalar et |}
  (#size : erased nat)
  (ga1 ga2 r : gpu_array et size)
  (tid : szlt size)
  ()
  norewrite
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

  (**)unfold gpu_pts_to_array1 ga1 tid;
  (**)unfold gpu_pts_to_array1 ga2 tid;
  (**)unfold gpu_pts_to_array1 r   tid;

  // Needed after API change in gpu_array_read
  gpu_pts_to_slice_ref ga1 _ _;
  gpu_pts_to_slice_ref ga2 _ _;

  gpu_array_write r tid (gpu_array_read ga1 tid `mul` gpu_array_read ga2 tid);

  (**)fold gpu_pts_to_array1 r   tid;
  (**)fold gpu_pts_to_array1 ga1 tid;
  (**)fold gpu_pts_to_array1 ga2 tid;
  ()
}

ghost
fn gpu_array_slice_1_underspec
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
  requires arr |-> Frac f v
  ensures  forall+ (i: natlt sz). gpu_pts_to_array1 arr #f i
{
  gpu_pts_to_ref arr;
  gpu_array_slice_1 arr;
  forevery_map
    (fun (i: natlt sz) ->
      gpu_pts_to_slice arr #f i (i + 1) seq![Seq.Base.index v i])
    (fun (i: natlt sz) -> gpu_pts_to_array1 arr #f i)
    fn i { fold gpu_pts_to_array1 arr #f i };
}

ghost
fn gpu_array_unslice_1_underspec
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  requires forall+ (i: natlt sz). gpu_pts_to_array1 arr #f i
  ensures exists* (v : seq a). arr |-> Frac f v
{
  forevery_map
    (fun (i: natlt sz) -> gpu_pts_to_array1 arr #f i)
    (fun (i: natlt sz) -> exists* v. gpu_pts_to_slice arr #f i (i + 1) seq![v])
    fn i {
      unfold gpu_pts_to_array1 arr #f i; with v. _;
      gpu_pts_to_slice_ref arr i (i+1);
      assert pure (v `Seq.equal` seq![Seq.index v 0]);
      rewrite each v as seq![Seq.index v 0];
    };
  let v = forevery_exists (fun (i: natlt sz) v ->
    gpu_pts_to_slice arr #f i (i + 1) seq![v]);
  let v' = Seq.init_ghost sz v;
  forevery_ext
    (fun (i: natlt sz) -> gpu_pts_to_slice arr #f i (i + 1) seq![v i])
    (fun (i: natlt sz) -> gpu_pts_to_slice arr #f i (i + 1) seq![Seq.index v' i]);
  gpu_array_unslice_1 arr;
}

ghost
fn block_setup
  (#et:Type) (ga1 ga2 gr : gpu_array et size)
  ()
  norewrite
  requires
    (live ga1 ** live ga2 ** live gr)
  ensures
    (forall+ (tid : natlt m_size). kpre m_size ga1 ga2 gr tid) **
    emp (* frame *)
{
  // Slicing the arrays
  (**)gpu_array_slice_1_underspec ga1;
  (**)gpu_array_slice_1_underspec ga2;
  (**)gpu_array_slice_1_underspec gr;

  // Boring combination of resources
  forevery_zip3 #(natlt (v m_size))
    (gpu_pts_to_array1 ga1)
    (gpu_pts_to_array1 ga2)
    (gpu_pts_to_array1 gr);
}

ghost
fn block_teardown
  (#et:Type) (ga1 ga2 gr : gpu_array et size)
  ()
  norewrite
  requires
    (forall+ (tid : natlt m_size). kpre m_size ga1 ga2 gr tid) **
    emp (* frame *)
  ensures
    (live ga1 ** live ga2 ** live gr)
{
  forevery_unzip3 _ _ _;

  // Unslicing
  (**)gpu_array_unslice_1_underspec ga1;
  (**)gpu_array_unslice_1_underspec ga2;
  (**)gpu_array_unslice_1_underspec gr;
}


inline_for_extraction noextract
let kdesc (#et:Type) {| scalar et |}
    (ga1:gpu_array et size{ is_global_array ga1 })
    (ga2:gpu_array et size{ is_global_array ga2 })
    (r : gpu_array et size{ is_global_array r })

  : kernel_desc
    (live ga1 ** live ga2 ** live r)
    (live ga1 ** live ga2 ** live r)
= {
  f = kf #et ga1 ga2 r;
  nthr = m_size;
  frame = emp;
  kpre  = kpre #et size ga1 ga2 r;
  kpost = kpost #et size ga1 ga2 r;
  block_setup = block_setup #et ga1 ga2 r;
  block_teardown = block_teardown #et ga1 ga2 r;
  kpost_sendable=solve;
  kpre_sendable=solve;
  full_post_sendable=solve;
  full_pre_sendable=solve;
} <: kernel_desc_1_n _ _

inline_for_extraction noextract
fn main (#et:Type0) {| scalar et |} (_:unit)
  requires cpu
  ensures  cpu
{
  let a1 = V.alloc #et zero m_size;
  let a2 = V.alloc #et zero m_size;
  let ar = V.alloc #et zero m_size;

  let mut i = 0sz;
  let mut a : et = zero;

  while (SZ.(!i <^ m_size))
     invariant live i ** live a
     invariant exists* (s:seq et). a1 |-> s ** pure (len s == size)
     invariant exists* (s:seq et). a2 |-> s ** pure (len s == size)
     decreases (m_size - !i)
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
  let mut psum : et = zero;
  while (SZ.(!i <^ m_size))
   invariant live i ** live psum
   decreases (m_size - !i)
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
