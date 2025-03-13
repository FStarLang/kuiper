module Kuiper.MatMul.Naive

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
module MU = Kuiper.MatMul.Util
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (f : perm)
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  (exists* v.
    M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols) v)

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (f : perm)
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols)
    (MS.matmul_single eA eB (tid / cols) (tid % cols) shared)

inline_for_extraction
type kernel_fixed_ty
  (et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (lA : mlayout rows shared)
  (lB : mlayout shared cols)
  (lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
: Type0
=
  (gA : M.gpu_matrix et lA) ->
  (gB : M.gpu_matrix et lB) ->
  (gC : M.gpu_matrix et lC) ->
  (#eA : ematrix et rows shared) ->
  (#eB : ematrix et shared cols) ->
  (#f : perm) ->
  (ebid : enatlt (rows * cols)) ->
  stt unit
  (requires
    gpu **
    block_id (rows * cols) ebid **
    kpre gA gB gC eA eB f ebid)
  (ensures fun _ ->
    gpu **
    block_id (rows * cols) ebid **
    kpost gA gB gC eA eB f ebid)

inline_for_extraction noextract
fn kernel_fixed
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (lA : mlayout rows shared)
  (lB : mlayout shared cols)
  (lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#f : perm)
  (ebid : enatlt (rows * cols))
  requires
    gpu **
    block_id (rows * cols) ebid **
    kpre gA gB gC eA eB f ebid
  ensures
    gpu **
    block_id (rows * cols) ebid **
    kpost gA gB gC eA eB f ebid
{
  let id = get_bid (); rewrite each ebid as SZ.v id;

  let trow = SZ.div id cols;
  let tcol = SZ.rem id cols;
  with v0.
    rewrite
      M.gpu_matrix_pts_to_cell gC #1.0R (id / cols) (id % cols) v0
    as
      M.gpu_matrix_pts_to_cell gC #1.0R trow tcol v0;

  assert (pure (trow < rows));
  assert (pure (tcol < cols));

  let s = MU.matmul_dotprod gA gB trow tcol;
  M.gpu_matrix_write_cell gC trow tcol s;

  assert (pure (SZ.v trow == ebid / cols));
  assert (pure (SZ.v tcol == ebid % cols));
  rewrite
    M.gpu_matrix_pts_to_cell gC trow tcol
      (MS.matmul_single eA eB trow tcol shared)
  as
    M.gpu_matrix_pts_to_cell gC (ebid / cols) (ebid % cols)
      (MS.matmul_single eA eB (ebid / cols) (ebid % cols) shared);

  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : pos)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    forall+ (rc : natlt (rows * cols)).
      kpre gA gB gC eA eB 1.0R rc
{
  // Sharing the input matrices (splitting permissions)
  M.gpu_matrix_share_n #_ #0 gA (rows * cols);
  forevery_fromstar #(natlt (rows * cols)) _;
  M.gpu_matrix_share_n #_ #0 gB (rows * cols);
  forevery_fromstar #(natlt (rows * cols)) _;

  // Sharing the output matrix (splitting each cell)
  M.gpu_matrix_explode #_ gC;

  forevery_unfactor' (rows * cols) rows cols _;

  // Join resources into a single bigstar
  forevery_zip #(natlt (rows * cols))
    (fun _ -> M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA)
    (fun _ -> M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB);
  forevery_zip #(natlt (rows * cols))
    _
    (fun i -> M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols)));

  // Rewrite inside the bigstar
  ghost
  fn aux1 (i : natlt (rows * cols))
    requires
      (M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
      M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB) **
      M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols))
    ensures
      kpre gA gB gC eA eB 1.0R (Kuiper.Enumerable.of_nat #(natlt (rows * cols)) i)
  {
    ()
  };
  forevery_map #(natlt (rows * cols))
    (fun i ->
      (M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
      M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB) **
      M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols)))
    _
    aux1;
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : pos)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  requires
    forall+ (rc : natlt (rows * cols)).
      kpost gA gB gC eA eB 1.0R rc
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.matmul eA eB)
{
  forevery_unzip #(natlt (rows * cols)) _ _;
  forevery_unzip #(natlt (rows * cols)) _ _;

  forevery_tostar #(natlt (rows * cols))
    (fun i -> M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA);
  M.gpu_matrix_gather_n gA _;
  forevery_tostar #(natlt (rows * cols))
    (fun i -> M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB);
  M.gpu_matrix_gather_n gB _;

  forevery_factor (rows * cols) rows cols _;

  (* we get things back with some arithmetic in it *)
  assert (forall+ (r:natlt rows) (c:natlt cols).
      M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
         (MS.matmul_single eA eB ((r * cols + c) / cols) ((r * cols + c) % cols) shared));

  (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));
  forevery_ext_2
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
         (MS.matmul_single eA eB ((r * cols + c) / cols) ((r * cols + c) % cols) shared))
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared));

  assert (forall+ r c.
      M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared));

  ghost
  fn aux (r:natlt rows) (c:natlt cols)
    requires
      M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared)
    ensures
      M.gpu_matrix_pts_to_cell gC r c (macc (MS.matmul eA eB) r c)
  {
    MS.lemma_matmul_index eA eB r c;
    () (* BUG! Should not be needed. *)
  };
  forevery_map_2 #(natlt rows) #_ #(natlt cols)
    (fun r c -> M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c shared))
    _
    aux;

  M.gpu_matrix_implode gC;
  ()
}

inline_for_extraction noextract
fn matmul_gpu_fixed
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
  (kk : kernel_fixed_ty et lA lB lC #_ #_ #_)
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (rows * cols <= max_blocks) **
    (gC |-> eC)
  ensures
    gC |-> MS.matmul eA eB
{
  open FStar.SizeT;
  setup gA gB gC;

  let size = rows *^ cols;
  forevery_rw_size (rows * cols) size;

  (* FIXME: F* inference failure means we need to annotate pre/post (somewhat) *)
  (* We also need eta due to the extraction rules looking for it. *)

  launch_kernel_n_blocks
    size
    #(kpre  #et gA gB gC eA eB 1.0R)
    #(kpost #et gA gB gC eA eB 1.0R)
    (fun ebid -> kk gA gB gC ebid);

  forevery_rw_size size (rows * cols);

  teardown gA gB gC;
}
