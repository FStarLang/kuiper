module Kuiper.DotProduct

#lang-pulse

open Kuiper
module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module U32 = FStar.UInt32

let m_size : sz = 1024sz
let size : n:(erased nat){reveal n == SZ.v m_size} = SZ.v m_size

let elem_t = u32

[@@coercion]
inline_for_extraction
let uint32_to_sizet x = FStar.SizeT.uint32_to_sizet x

let kpre (size: sz) (ga1 ga2 r : gpu_array elem_t (SZ.v size)) (tid:nat) : slprop =
  gpu_pts_to_array1 #elem_t #(SZ.v size) ga1 tid **
  gpu_pts_to_array1 #elem_t #(SZ.v size) ga2 tid **
  gpu_pts_to_array1 #elem_t #(SZ.v size) r tid

let kpost (size: sz) (ga1 ga2 r : gpu_array elem_t (SZ.v size)) (tid:nat) : slprop =
  gpu_pts_to_array1 #elem_t #(SZ.v size) ga1 tid **
  gpu_pts_to_array1 #elem_t #(SZ.v size) ga2 tid **
  gpu_pts_to_array1 #elem_t #(SZ.v size) r tid

[@@CPrologue "__global__"]
fn kernel
  (#nblk : erased sz { 0 < SZ.v nblk /\ SZ.v nblk <= 1024 * 1024 })
  (size : sz { SZ.v size == SZ.v nblk })
  (ga1 ga2 : gpu_array elem_t size)
  (r : gpu_array elem_t size)
  (etid : erased tid_t { gdim_x etid == SZ.v nblk /\ bdim_x etid == 1 })
  requires gpu ** thread_id etid ** kpre size ga1 ga2 r (thread_index etid)
  ensures  gpu ** thread_id etid ** kpost size ga1 ga2 r (thread_index etid)
{
  let id = thread_idx_all ();
  (* r[id] = ga1[id] * ga2[id] *)

  (**)unfold (kpre size ga1 ga2 r (thread_index etid));

  (**)unfold (gpu_pts_to_array1 ga1 id);
  (**)gpu_pts_to_slice_ref ga1 id (id+1); // recall tid < size
  let v1 = gpu_array_read #_ #size #id #(id+1) ga1 id;

  (**)unfold (gpu_pts_to_array1 ga2 id);
  let v2 = gpu_array_read #_ #size #id #(id+1) ga2 id;

  let v = v1 `U32.mul_underspec` v2;

  (**)unfold (gpu_pts_to_array1 r id);
  gpu_array_write #_ #size #id #(id+1) r id v;

  (**)fold (gpu_pts_to_array1 r id);
  (**)fold (gpu_pts_to_array1 ga1 id);
  (**)fold (gpu_pts_to_array1 ga2 id);
  (**)fold (kpost size ga1 ga2 r (thread_index etid));
}


fn main (_:unit)
  requires cpu ** pure SZ.fits_u32
  ensures  cpu
{
  let a1 = A.alloc #elem_t 0ul m_size;
  let a2 = A.alloc #elem_t 0ul m_size;
  let ar = A.alloc #elem_t 0ul m_size;

  let mut i = 0sz;

  while (let v = !i; (v `SZ.op_Less_Hat` m_size))
     invariant b.
       exists* v. pts_to i v **
       (exists* s. A.pts_to a1 s ** pure (Seq.length s == size)) **
       (exists* s. A.pts_to a2 s ** pure (Seq.length s == size)) **
       pure (b == (SZ.v v < size))
  {
    let v = !i;
    a1.(v) <- FStar.SizeT.sizet_to_uint32 v;
    a2.(v) <- FStar.SizeT.sizet_to_uint32 v;
    i := SZ.add v 1sz;
    ()
  };

  let ga1 = gpu_array_alloc #u32 m_size;
  let ga2 = gpu_array_alloc #u32 m_size;

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 m_size;
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 m_size;
  
  let gr = gpu_array_alloc #u32 m_size;
  
  let nthr : sz = m_size;

  // Slicing the arrays
  (**)gpu_array_slice_1_underspec #1 ga1;
  (**)gpu_array_slice_1_underspec #2 ga2;
  (**)gpu_array_slice_1_underspec #3 gr;

  // Boring combination of resources
  (**)bigstar_zip #1 #2 #1 0 (SZ.v m_size) _ _;
  (**)bigstar_zip #1 #3 #0 0 (SZ.v m_size) _ _;

  (**)rewrite
    (bigstar 0 (SZ.v m_size)
      (fun i -> (gpu_pts_to_array1 ga1 i **
                 gpu_pts_to_array1 ga2 i) **
                 gpu_pts_to_array1 gr i))
  as
    (bigstar 0 (SZ.v m_size) (fun i -> kpre m_size ga1 ga2 gr i));

  (**)bigstar_uneta ();

  assert (pure (SZ.v m_size == SZ.v nthr));
  rewrite
    (bigstar 0 (SZ.v m_size) (kpre m_size ga1 ga2 gr))
  as
    (bigstar 0 (SZ.v nthr)  (kpre m_size ga1 ga2 gr));

  launch_kernel_n nthr
    #(kpre m_size ga1 ga2 gr) #(kpost m_size ga1 ga2 gr)
    (fun etid -> kernel #(hide nthr) m_size ga1 ga2 gr etid);

  rewrite
    (bigstar 0 (SZ.v nthr)  (kpost m_size ga1 ga2 gr))
  as
    (bigstar 0 (SZ.v m_size) (kpost m_size ga1 ga2 gr));

  (**)bigstar_eta ();

  rewrite
    (bigstar 0 (SZ.v m_size) (fun i -> kpre m_size ga1 ga2 gr i))
  as
    (bigstar 0 (SZ.v m_size)
      (fun i -> gpu_pts_to_array1 ga1 i **
                gpu_pts_to_array1 ga2 i **
                gpu_pts_to_array1 gr i));
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
  let mut psum = 0ul;
  while (let v = !i; (v `SZ.op_Less_Hat` m_size))
     invariant b. exists* vi vpsum.
       pts_to i vi **
       pts_to psum vpsum  **
       (exists* s. A.pts_to ar s ** pure (Seq.length s == size)) **
       pure (b == (SZ.v vi < size))
  {
    let vi = !i;
    let ri = ar.(vi);
    let vpsum = !psum;
    psum := vpsum `U32.add_underspec` ri;
    i := SZ.add vi 1sz;
  };
  
  A.free a1;
  A.free a2;
  A.free ar;

  ()
}
