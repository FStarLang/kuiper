module Kuiper.MatMul
#lang-pulse

open Kuiper

module SZ   = FStar.SizeT

module K = Kuiper.MatMul.Kernel
module I = Kuiper.MatMul.Impure
module P = Kuiper.MatMul.Pure

ghost
fn setup
  (rows: szp) (shared: szp) (columns: szp{rows * columns < pow2 64})
  (size: sz { SZ.v size == rows * columns })
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 size)
  requires
    (gr |-> 's) **
    (ga |-> 'va) **
    (gb |-> 'vb)
  ensures
    bigstar 0 size (fun i ->
      K.kpre rows shared columns ga gb gr 'va 'vb size i)
{
  (* recall *)
  gpu_pts_to_ref ga; gpu_pts_to_ref gb;

  // Sharing the input matrices (splitting permissions)
  fold I.gpu_pts_to_matrix rows   shared  ga 1 'va;
  fold I.gpu_pts_to_matrix shared columns gb 1 'vb;
  I.gpu_matrix_share_underspec #_ #1 rows   shared  ga size 'va;
  I.gpu_matrix_share_underspec #_ #2 shared columns gb size 'vb;

  // Sharing the output matrix (splitting each cell)
  gpu_pts_to_ref gr; (* obtain length v == size *)
  gpu_array_slice_1 #4 gr;

  // Join resources into a single bigstar
  bigstar_zip #1 #2 #3 0 size _ _;
  bigstar_zip #3 #4 #0 0 size _ _;

  // Rewrite inside the bigstar
  ghost
  fn aux (i:nat{0 <= i /\ i < size})
    requires
      I.gpu_pts_to_matrix rows   shared  ga size 'va **
      I.gpu_pts_to_matrix shared columns gb size 'vb **
      gpu_pts_to_slice gr i (i + 1) seq!['s `Seq.index` i]
    ensures
      K.kpre rows shared columns ga gb gr 'va 'vb size i
  {
    ()
  };
  bigstar_map #_ #_ #0 #size aux;
  bigstar_eta();
}

fn main
  (rows shared columns : szp)
  (a b : vec u64)
  (#va #vb : erased (seq u64))
  preserves
    cpu **
    (a |-> va) **
    (b |-> vb)
  requires
    pure (
      len va == rows * shared /\
      len vb == shared * columns /\
      rows * columns < pow2 64 /\
      rows * columns < max_blocks
    )
    // ^ Some of these could be ommited if we had some "core" pure inference from slprops.
    // Since we have a1 |-> v1, the length of v1 must fit, etc.
  returns
    ar: (_ : vec u64
        { len va == rows * shared /\ len vb == shared * columns })
        (* ^ This refinement just a hack to check the post. *)
  ensures
    ar |-> P.matmul rows shared columns va vb
{
  open FStar.SizeT;
  Pulse.Lib.Vec.pts_to_len a;
  assert (pure (SZ.fits (rows * shared)));
  Pulse.Lib.Vec.pts_to_len b;
  assert (pure (SZ.fits (shared * columns)));
  let size = rows *^ columns;
  let ar = Pulse.Lib.Vec.alloc 0UL size;

  let rs = rows *^ shared;
  let sc = shared *^ columns;

  let ga = gpu_array_alloc #u64 rs;
  let gb = gpu_array_alloc #u64 sc;

  Kuiper.Array.gpu_memcpy_host_to_device ga a rs;
  Kuiper.Array.gpu_memcpy_host_to_device gb b sc;

  let gr = gpu_array_alloc #u64 size;

  (**)setup rows shared columns size ga gb gr;

  launch_kernel_n #0
    size
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> K.kpre  rows shared columns ga gb gr va vb (SZ.v size) tid)
    #(fun (tid: nat {0 <= tid /\ tid < size} ) -> K.kpost rows shared columns ga gb gr va vb (SZ.v size) tid)
    (fun etid -> K.kernel rows shared columns ga gb gr (hide size) etid);

  (**)bigstar_unzip 0 size _ _;
  (**)bigstar_unzip 0 size _ _;

  (**)I.gpu_matrix_unshare_underspec rows shared ga size va;
  (**)I.gpu_matrix_unshare_underspec shared columns gb size vb;
  (**)unfold I.gpu_pts_to_matrix rows shared ga 1 va;
  (**)unfold I.gpu_pts_to_matrix shared columns gb 1 vb;

  ghost
  fn aux1 (i:nat{0 <= i /\ i < size})
    requires
      gpu_pts_to_slice gr i (i + 1)
            seq![P.matmul_single rows shared columns
                   va vb (i / columns) (i % columns) shared]
    ensures
      gpu_pts_to_slice gr i (i + 1)
            seq![P.matmul rows shared columns va vb @! i]
  {
    P.lemma_matmul_index rows shared columns va vb i;
    () (* cf. issue #181 in Pulse *)
  };
  bigstar_map #0 #0 #0 #size aux1;

  (**)gpu_array_unslice_1 gr;

  Kuiper.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga;
  gpu_array_free gb;
  gpu_array_free gr;

  ar
}
