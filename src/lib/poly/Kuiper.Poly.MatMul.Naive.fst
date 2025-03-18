module Kuiper.Poly.MatMul.Naive

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
module MU = Kuiper.Poly.MatMul.Util
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : (et -> et -> et))
  (#rows #shared #cols : nat)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (f : perm)
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  (* Note: as far as this algorithm is concerned, we could have
  an existential for the gC cell and not state anything interesting.
  However it is actually more comfortable to not have an existential here,
  and we will need it anyway for the  GEMM. *)
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols)
    (macc eC (tid / cols) (tid % cols))

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : (et -> et -> et))
  (#rows #shared #cols : nat)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (f : perm)
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols)
    (MS.gemm_single comb eA eB eC (tid / cols) (tid % cols) shared)

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : (et -> et -> et))
  (#rows #shared #cols : SZ.t)
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
  (#f : perm)
  (bid : szlt (rows *^ cols))
  ()
  requires
    gpu **
    kpre comb gA gB gC eA eB eC f bid **
    block_id (rows *^ cols) bid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC f bid **
    block_id (rows *^ cols) bid
{
  let trow = SZ.div bid cols;
  let tcol = SZ.rem bid cols;
  rewrite each (SZ.v bid / SZ.v cols) as trow;
  rewrite each (SZ.v bid % SZ.v cols) as tcol;

  let s = MU.matmul_dotprod gA gB trow tcol;
  let v0 = M.gpu_matrix_read_cell gC trow tcol;
  let v1 = comb v0 s;
  M.gpu_matrix_write_cell gC trow tcol v1;

  rewrite each SZ.v trow as (bid / cols);
  rewrite each SZ.v tcol as (bid % cols);
  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : (et -> et -> et))
  (#rows #shared #cols : szp)
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
  ()
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    (forall+ (rc : natlt (rows *^ cols)).
      kpre comb gA gB gC eA eB eC 1.0R rc) **
    emp (* frame *)
{
  // Sharing the input matrices (splitting permissions)
  M.gpu_matrix_share_n #_ #0 gA (rows *^ cols);
  forevery_fromstar #(natlt2 rows cols) _;
  M.gpu_matrix_share_n #_ #0 gB (rows *^ cols);
  forevery_fromstar #(natlt2 rows cols) _;

  // Sharing the output matrix (splitting each cell)
  M.gpu_matrix_explode #_ gC;

  forevery_unfactor' (rows *^ cols) rows cols (fun r c ->
    M.gpu_matrix_pts_to_cell gC r c (macc eC r c));

  // Join resources into a single bigstar
  forevery_zip #(natlt2 rows cols)
    (fun _ -> M.gpu_matrix_pts_to gA #(1.0R /. (rows *^ cols)) eA)
    (fun _ -> M.gpu_matrix_pts_to gB #(1.0R /. (rows *^ cols)) eB);
  forevery_zip #(natlt2 rows cols)
    _
    (fun i -> M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols)));

  (* We're done actually, but the encoding will not match the lambdas. *)
  forevery_ext #(natlt2 rows cols)
    (fun i ->
      M.gpu_matrix_pts_to gA #(1.0R /. (rows *^ cols)) eA **
      M.gpu_matrix_pts_to gB #(1.0R /. (rows *^ cols)) eB **
      M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i / cols) (i % cols)))
    (fun i ->
      kpre comb gA gB gC eA eB eC 1.0R i);
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : (et -> et -> et))
  (#rows #shared #cols : szp)
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
  ()
  requires
    (forall+ (rc : natlt (rows *^ cols)).
      kpost comb gA gB gC eA eB eC 1.0R rc) **
    emp (* frame *)
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> matrix_comb comb eC (MS.matmul eA eB))
{
  forevery_unzip #(natlt2 rows cols) _ _;
  forevery_unzip #(natlt2 rows cols) _ _;

  forevery_tostar #(natlt2 rows cols)
    (fun i -> M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA);
  M.gpu_matrix_gather_n gA _;
  forevery_tostar #(natlt2 rows cols)
    (fun i -> M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB);
  M.gpu_matrix_gather_n gB _;

  forevery_factor (rows *^ cols) rows cols _;

  (* we get things back with some arithmetic in it *)
  assert (forall+ (r:natlt rows) (c:natlt cols).
      M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
         (MS.gemm_single comb eA eB eC ((r * cols + c) / cols) ((r * cols + c) % cols) shared));

  (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));
  forevery_ext_2
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
         (MS.gemm_single comb eA eB eC ((r * cols + c) / cols) ((r * cols + c) % cols) shared))
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gC r c (MS.gemm_single comb eA eB eC r c shared));

  assert (forall+ r c.
      M.gpu_matrix_pts_to_cell gC r c (MS.gemm_single comb eA eB eC r c shared));

  ghost
  fn aux (r:natlt rows) (c:natlt cols)
    requires
      M.gpu_matrix_pts_to_cell gC r c (MS.gemm_single comb eA eB eC r c shared)
    ensures
      M.gpu_matrix_pts_to_cell gC r c (macc (matrix_comb comb eC (MS.matmul eA eB)) r c)
  {
    MS.lemma_matmul_index eA eB r c;
    () (* BUG! Should not be needed. *)
  };
  forevery_map_2 #(natlt rows) #_ #(natlt cols)
    (fun r c -> M.gpu_matrix_pts_to_cell gC r c (MS.gemm_single comb eA eB eC r c shared))
    _
    aux;

  M.gpu_matrix_implode gC;
  ()
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : (et -> et -> et))
  (#rows #shared #cols : szp)
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
  (_ : squash (rows * cols <= max_blocks))
  : kernel_desc_m_1
    ((gA |-> eA) ** (gB |-> eB) ** (gC |-> eC))
    ((gA |-> eA) ** (gB |-> eB) ** (gC |-> MS.gemm comb eC eA eB))
= {
  nblk = rows *^ cols;

  frame = emp;

  setup    = setup    comb gA gB gC #eA #eB #eC;
  teardown = teardown comb gA gB gC #eA #eB;

  kpre  = kpre  comb gA gB gC eA eB eC 1.0R;
  kpost = kpost comb gA gB gC eA eB eC 1.0R;

  f = kf comb gA gB gC #eA #eB #eC #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu
  (#et : Type0) {| scalar et |}
  (comb : (et -> et -> et))
  (#rows #shared #cols : szp)
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
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (rows * cols <= max_blocks) **
    (gC |-> eC)
  ensures
    gC |-> MS.gemm comb eC eA eB
{
  launch_sync (kdesc comb gA gB gC ());
}
