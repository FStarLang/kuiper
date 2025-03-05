module Kuiper.DotProduct.Poly

#lang-pulse

open Kuiper
module V = Pulse.Lib.Vec
module SZ = FStar.SizeT

let m_size : sz = 1024sz
let size : n:(erased nat){reveal n == SZ.v m_size} = SZ.v m_size

[@@coercion]
inline_for_extraction
let uint32_to_sizet x = FStar.SizeT.uint32_to_sizet x

let kpre #et (size: sz) (ga1 ga2 r : gpu_array et (SZ.v size)) (tid:nat) : slprop =
  gpu_pts_to_array1 #et #(SZ.v size) ga1 tid **
  gpu_pts_to_array1 #et #(SZ.v size) ga2 tid **
  gpu_pts_to_array1 #et #(SZ.v size) r tid

let kpost #et (size: sz) (ga1 ga2 r : gpu_array et (SZ.v size)) (tid:nat) : slprop =
  gpu_pts_to_array1 #et #(SZ.v size) ga1 tid **
  gpu_pts_to_array1 #et #(SZ.v size) ga2 tid **
  gpu_pts_to_array1 #et #(SZ.v size) r tid

[@@CPrologue "__global__"]
fn kernel
  (#et:Type0)
  {| simple_scalar et |}
  (#nblk : erased sz { 0 < SZ.v nblk /\ SZ.v nblk <= 1024 * 1024 })
  (size : erased sz { SZ.v size == SZ.v nblk })
  (ga1 ga2 : gpu_array et (reveal size))
  (r : gpu_array et (reveal size))
  (etid : erased tid_t { gdim_x etid == SZ.v nblk /\ bdim_x etid == 1 })
  requires gpu ** thread_id etid ** kpre size ga1 ga2 r (thread_index etid)
  ensures  gpu ** thread_id etid ** kpost size ga1 ga2 r (thread_index etid)
{
  let id = thread_idx_all ();
  rewrite each thread_index etid as id;
  (* r[id] = ga1[id] * ga2[id] *)

  (**)unfold (kpre size ga1 ga2 r id);

  (**)unfold (gpu_pts_to_array1 ga1 id);
  (**)gpu_pts_to_slice_ref ga1 id (id+1); // recall tid < size
  let v1 = gpu_array_read #_ #(reveal size) #id #(id+1) ga1 id;

  (**)unfold (gpu_pts_to_array1 ga2 id);
  let v2 = gpu_array_read #_ #(reveal size) #id #(id+1) ga2 id;

  let v = v1 `mul` v2;

  (**)unfold (gpu_pts_to_array1 r id);
  gpu_array_write #_ #(reveal size) #id #(id+1) r id v;

  (**)fold (gpu_pts_to_array1 r id);
  (**)fold (gpu_pts_to_array1 ga1 id);
  (**)fold (gpu_pts_to_array1 ga2 id);
  (**)fold (kpost size ga1 ga2 r id);
  ()
}

inline_for_extraction noextract
fn main (#et:Type0) {| simple_scalar et |} (_:unit)
  requires cpu ** pure SZ.fits_u32
  ensures  cpu
{
  let a1 = V.alloc #et zero m_size;
  let a2 = V.alloc #et zero m_size;
  let ar = V.alloc #et zero m_size;

  let mut i = 0sz;
  let mut a = zero #et #_;

  while (let v = !i; (v `SZ.op_Less_Hat` m_size))
     invariant b.
       exists* v va. pts_to i v ** pts_to a va **
       (exists* (s:seq et). (a1 |-> s) ** pure (len s == size)) **
       (exists* (s:seq et). (a2 |-> s) ** pure (len s == size)) **
       pure (b == (SZ.v v < size))
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

  let nthr : sz = m_size;

  // Slicing the arrays
  (**)gpu_array_slice_1_underspec #1 ga1;
  (**)gpu_array_slice_1_underspec #2 ga2;
  (**)gpu_array_slice_1_underspec #3 gr;

  // Boring combination of resources
  (**)bigstar_zip #1 #2 #1 0 (SZ.v m_size) _ _;
  (**)bigstar_zip #1 #3 #0 0 (SZ.v m_size) _ _;

  (**)bigstar_uneta ();

  assert (pure (SZ.v m_size == SZ.v nthr));
  rewrite
    (bigstar 0 (SZ.v m_size) (kpre m_size ga1 ga2 gr))
  as
    (bigstar 0 (SZ.v nthr)  (kpre m_size ga1 ga2 gr));

  launch_kernel_n nthr
    #(kpre m_size ga1 ga2 gr) #(kpost m_size ga1 ga2 gr)
    (fun etid -> kernel #et #_ #(hide nthr) (hide m_size) ga1 ga2 gr etid);

  rewrite
    (bigstar 0 (SZ.v nthr)  (kpost m_size ga1 ga2 gr))
  as
    (bigstar 0 (SZ.v m_size) (kpost m_size ga1 ga2 gr));

  (**)bigstar_eta ();

  (**)bigstar_unzip 0 (SZ.v m_size) _ _;
  (**)bigstar_unzip 0 (SZ.v m_size) _ _;

  // Unslicing
  (**)gpu_array_unslice_1_underspec ga1;
  (**)gpu_array_unslice_1_underspec ga2;
  (**)gpu_array_unslice_1_underspec gr;

  Kuiper.Array.gpu_memcpy_device_to_host ar gr m_size;
  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  i := 0sz;
  let mut psum = zero #et #_;
  while (let v = !i; (v `SZ.op_Less_Hat` m_size))
     invariant b. exists* vi vpsum.
       pts_to i vi **
       pts_to psum vpsum  **
       (exists* (s : seq et). (ar |-> s) ** pure (len s == size)) **
       pure (b == (SZ.v vi < size))
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

(* These do not extract due to the typeclass dictionaries
   being passed explicitly in the krml. *)

// fn main_u64 (_:unit)
//   requires cpu ** pure SZ.fits_u32
//   ensures  cpu
// {
//   main #u64 ();
// }

// fn main_f32 (_:unit)
//   requires cpu ** pure SZ.fits_u32
//   ensures  cpu
// {
//   main #f32 ();
// }
