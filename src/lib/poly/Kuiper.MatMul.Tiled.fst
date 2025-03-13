module Kuiper.MatMul.Tiled

#lang-pulse

open Kuiper
module M   = Kuiper.Matrix
module M4  = Kuiper.Matrix4
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}


unfold
let kpre
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols #bdim : pos)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bdim bdim)
  (eB : ematrix4 et mshared mcols bdim bdim)
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bdim * bdim))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lA) eA **
  m4_pts_to gB #(f /. mlayout_size lB) eB **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / bdim) (tid % bdim) v)

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols #bdim : pos)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared bdim bdim)
  (eB : ematrix4 et mshared mcols bdim bdim)
  (f : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (bdim * bdim))
  : slprop
  =
  m4_pts_to gA #(f /. mlayout_size lA) eA **
  m4_pts_to gB #(f /. mlayout_size lB) eB **
  (exists* v.
    m4_pts_to_cell gC #1.0R
      (bid / mcols) (bid % mcols)
      (tid / bdim) (tid % bdim) v)

inline_for_extraction
type kernel_fixed_ty
  (bdim : szp) (* block dim *)
  (et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (lA : mlayout4 mrows   mshared bdim bdim)
  (lB : mlayout4 mshared mcols   bdim bdim)
  (lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
: Type0
=
  (gA : gpu_matrix4 et lA) ->
  (gB : gpu_matrix4 et lB) ->
  (gC : gpu_matrix4 et lC) ->
  (#eA : ematrix4 et mrows   mshared bdim bdim) ->
  (#eB : ematrix4 et mshared mcols   bdim bdim) ->
  (#f : perm) ->
  (ebid : enatlt (mrows * mcols)) ->
  (etid : enatlt (bdim * bdim)) ->
  stt unit
  (requires
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpre gA gB gC eA eB f ebid etid)
  (ensures fun _ ->
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpost gA gB gC eA eB f ebid etid)

inline_for_extraction noextract
fn kernel_fixed_f
  (bdim : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (lA : mlayout4 mrows   mshared bdim bdim)
  (lB : mlayout4 mshared mcols   bdim bdim)
  (lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#f : perm)
  (ebid : enatlt (mrows * mcols))
  (etid : enatlt (bdim * bdim))
  requires
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpre gA gB gC eA eB f ebid etid
  ensures
    gpu **
    block_id (mrows * mcols) ebid **
    thread_id (bdim * bdim) etid **
    kpost gA gB gC eA eB f ebid etid
{
  admit();
  // let tid = get_tid (); rewrite each etid as SZ.v tid;

  // let trow = SZ.div tid cols;
  // let tcol = SZ.rem tid cols;
  // with v0.
  //   rewrite
  //     M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols) v0
  //   as
  //     M.gpu_matrix_pts_to_cell gC #1.0R trow tcol v0;

  // assert (pure (trow < rows));
  // assert (pure (tcol < cols));

  // let mut i : sz = 0sz;
  // let mut sum : et = zero #et #_;

  // while (let vi = !i; SZ.(vi <^ shared))
  //   invariant b.
  //     exists* (vi : SZ.t{ vi <= shared}).
  //       pure (0 <= shared /\ b == (SZ.v vi < shared) /\ vi <= shared /\ vi >= 0) **
  //       pts_to i vi **
  //       pts_to #_ #et sum (MS.matmul_single eA eB trow tcol vi) **
  //       M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  //       M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  //       gpu
  // {
  //   let vi = !i;
  //   let s = !sum;
  //   let v1 = M.gpu_matrix_read gA trow vi;
  //   let v2 = M.gpu_matrix_read gB vi tcol;

  //   sum := s `add` mul v1 v2;
  //   i := SZ.add vi 1sz;

  //   (**)MS.matmul_single_lemma eA eB trow tcol (vi + 1);
  //   ();
  // };

  // let s = !sum;
  // M.gpu_matrix_write_cell gC trow tcol s;

  // assert (pure (SZ.v trow == thread_index etid / cols));
  // assert (pure (SZ.v tcol == thread_index etid % cols));
  // rewrite
  //   M.gpu_matrix_pts_to_cell gC trow tcol
  //     (MS.matmul_single eA eB trow tcol shared)
  // as
  //   M.gpu_matrix_pts_to_cell gC (thread_index etid / cols) (thread_index etid % cols)
  //     (MS.matmul_single eA eB (thread_index etid / cols) (thread_index etid % cols) shared);

  // ()
}

let kernel_fixed = kernel_fixed_f

// let mksz = SZ.uint_to_t

ghost
fn setup
  (bdim : pos) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : nat)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#eC : ematrix4 et mrows   mcols   bdim bdim)
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    forall+ (bid : natlt (mrows * mcols))
            (tid : natlt (bdim  * bdim)).
      kpost gA gB gC eA eB 1.0R bid tid
{
  admit();
  // // Sharing the input matrices (splitting permissions)
  // M.gpu_matrix_share_n #_ #0 gA (rows * cols);
  // forevery_fromstar #(natlt (rows * cols)) _;
  // M.gpu_matrix_share_n #_ #0 gB (rows * cols);
  // forevery_fromstar #(natlt (rows * cols)) _;

  // // Sharing the output matrix (splitting each cell)
  // M.gpu_matrix_explode #_ gC;

  // forevery_unfactor' (rows * cols) rows cols _;

  // // Join resources into a single bigstar
  // forevery_zip #(natlt (rows * cols))
  //   (fun _ -> M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA)
  //   (fun _ -> M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB);
  // forevery_zip #(natlt (rows * cols))
  //   _
  //   (fun i -> M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols)));

  // // Rewrite inside the bigstar
  // ghost
  // fn aux1 (i : natlt (rows * cols))
  //   requires
  //     (M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
  //     M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB) **
  //     M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols))
  //   ensures
  //     kpre gA gB gC eA eB 1.0R (Kuiper.Enumerable.of_nat #(natlt (rows * cols)) i)
  // {
  //   ()
  // };
  // forevery_map #(natlt (rows * cols))
  //   (fun i ->
  //     (M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
  //     M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB) **
  //     M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols)))
  //   _
  //   aux1;
}

ghost
fn teardown
  (bdim : pos) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : nat)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#eC : ematrix4 et mrows   mcols   bdim bdim)
  requires
    forall+ (bid : natlt (mrows * mcols))
            (tid : natlt (bdim  * bdim)).
      kpre gA gB gC eA eB 1.0R bid tid
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.matmul eA eB)
{
  admit();
  // forevery_unzip #(natlt (rows * cols)) _ _;
  // forevery_unzip #(natlt (rows * cols)) _ _;

  // forevery_tostar #(natlt (rows * cols))
  //   (fun i -> M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA);
  // M.gpu_matrix_gather_n gA _;
  // forevery_tostar #(natlt (rows * cols))
  //   (fun i -> M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB);
  // M.gpu_matrix_gather_n gB _;

  // forevery_factor (rows * cols) rows cols _;

  // (* we get things back with some arithmetic in it *)
  // assert (forall+ (r:natlt rows) (c:natlt cols).
  //     M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
  //        (MS.matmul_single eA eB ((r * cols + c) / cols) ((r * cols + c) % cols) shared));

  // (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  // assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  // assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));
  // forevery_ext_2
  //   (fun (r:natlt rows) (c:natlt cols) ->
  //     M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
  //        (MS.matmul_single eA eB ((r * cols + c) / cols) ((r * cols + c) % cols) shared))
  //   (fun (r:natlt rows) (c:natlt cols) ->
  //     M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared));

  // assert (forall+ r c.
  //     M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared));

  // ghost
  // fn aux (r:natlt rows) (c:natlt cols)
  //   requires
  //     M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared)
  //   ensures
  //     M.gpu_matrix_pts_to_cell gC r c (macc (MS.matmul eA eB) r c)
  // {
  //   MS.lemma_matmul_index eA eB r c;
  //   () (* BUG! Should not be needed. *)
  // };
  // forevery_map_2 #(natlt rows) #_ #(natlt cols)
  //   (fun r c -> M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared))
  //   _
  //   aux;

  // M.gpu_matrix_implode gC;
  // ()
}

ghost
fn forevery_rw_size2
  (n1 : nat)
  (n2 : nat{n1 == n2})
  (n3 : nat)
  (n4 : nat{n3 == n4})
  (#p : natlt n1 -> natlt n3 -> slprop)
  requires
    forall+ (i : natlt n1) (j : natlt n3). p i j
  ensures
    forall+ (i : natlt n2) (j : natlt n4). p i j
{
  admit();
}

inline_for_extraction noextract
fn matmul_gpu_fixed
  (bdim : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : SZ.t)
  (lA : mlayout4 mrows   mshared bdim bdim)
  (lB : mlayout4 mshared mcols   bdim bdim)
  (lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
  (kk : kernel_fixed_ty bdim et lA lB lC #_ #_ #_)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared bdim bdim)
  (#eB : ematrix4 et mshared mcols   bdim bdim)
  (#eC : ematrix4 et mrows   mcols   bdim bdim)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bdim * bdim <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.matmul eA eB
{
  open FStar.SizeT;
  setup bdim gA gB gC;

  let nblk = mrows *^ mcols;
  let nthr = bdim *^ bdim;

  forevery_rw_size2 (mrows * mcols) nblk
                    (bdim * bdim) nthr;

  (* FIXME: F* inference failure means we need to annotate pre/post (somewhat) *)
  (* We also need eta due to the extraction rules looking for it. *)
  launch_kernel_n_m
    nblk
    nthr
    #(kpre  _ _ _ _ _ _)
    #(kpost _ _ _ _ _ _)
    (fun ebid etid -> kk gA gB gC ebid etid);

  forevery_rw_size2 nblk (mrows * mcols)
                    nthr (bdim * bdim);

  teardown bdim gA gB gC #eA #eB #eC;
  ()
}
