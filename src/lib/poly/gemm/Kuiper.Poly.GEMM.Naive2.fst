module Kuiper.Poly.GEMM.Naive2

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = FStar.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
friend Kuiper.Poly.GEMM.Naive (* We reuse some lemmas from Naive *)

inline_for_extraction noextract
let blocksz = 1024sz

let in_bounds (rows cols bid tid : nat) : GTot bool =
  bid * blocksz + tid < rows * cols

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (bid : natlt (divup (rows * cols) blocksz))
  (tid : natlt blocksz)
  : slprop
  =
  if in_bounds rows cols bid tid then (
    (gA |-> Frac (fA /. (rows * cols)) eA) **
    (gB |-> Frac (fB /. (rows * cols)) eB) **
    M.gpu_matrix_pts_to_cell gC #1.0R ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols)
      (macc eC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols))
   ) else emp

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (bid : natlt (divup (rows * cols) blocksz))
  (tid : natlt blocksz)
  : slprop
  =
  if in_bounds rows cols bid tid then (
    (gA |-> Frac (fA /. (rows * cols)) eA) **
    (gB |-> Frac (fB /. (rows * cols)) eB) **
    M.gpu_matrix_pts_to_cell gC #1.0R ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols)
      (MS.gemm_single comb eA eB eC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols))
  ) else emp

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA)
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (bid : szlt (divup (rows *^ cols) blocksz))
  (tid : szlt blocksz)
  ()
  norewrite
  requires
    gpu **
    kpre comb gA gB gC eA eB eC fA fB bid tid **
    thread_id blocksz tid **
    block_id (divup (rows * cols) blocksz) bid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC fA fB bid tid **
    thread_id blocksz tid **
    block_id (divup (rows * cols) blocksz) bid
{
  (* Should remove this admit by constraining the sizes, but it's
     pretty benign. We know bid*blocksz does not overflow a size_t,
     and size_t is almost certainly a multiple of blocksz, so this
     cannot fail. *)
  assert (pure (SZ.fits (bid * blocksz)));
  assume (pure (SZ.fits (bid * blocksz + tid)));
  let id = bid *^ blocksz +^ tid;

  if SZ.lt id (rows *^ cols) {
    rewrite each in_bounds rows cols bid tid as true;

    let trow, tcol = s_divmod cols id;
    with i0 j0 v0.
      rewrite
        M.gpu_matrix_pts_to_cell gC #1.0R i0 j0 v0
      as
        M.gpu_matrix_pts_to_cell gC #1.0R trow tcol v0;

    assert (pure (trow < rows));
    assert (pure (tcol < cols));

    let s = MU.matmul_dotprod gA gB trow tcol;
    let v0 = M.gpu_matrix_read_cell gC trow tcol;
    let v1 = comb v0 s;
    M.gpu_matrix_write_cell gC trow tcol v1;

    assert (pure (SZ.v trow == id / cols));
    assert (pure (SZ.v tcol == id % cols));
    rewrite
      (gA |-> Frac (fA /. (rows * cols)) eA) **
      (gB |-> Frac (fB /. (rows * cols)) eB) **
      M.gpu_matrix_pts_to_cell gC trow tcol
        (MS.gemm_single comb eA eB eC trow tcol)
    as kpost comb gA gB gC eA eB eC fA fB bid tid;

    ()
  } else {
    (* Out of bounds, do nothing *)
    assert (pure (in_bounds rows cols bid tid == false));
    (* Funny, we need to go via emp to convince Pulse. *)
    rewrite
      (if in_bounds rows cols bid tid
       then
         (gA |-> Frac (fA /. (rows * cols)) eA) **
         (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols)
            (macc eC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols))
       else
         emp)
    as emp;
    rewrite emp
    as
      (if in_bounds rows cols bid tid
       then
         (gA |-> Frac (fA /. (rows * cols)) eA) **
         (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols)
           (MS.gemm_single comb eA eB eC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols))
       else emp);
  }
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA)
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (_ : squash (rows * cols <= max_blocks))
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt (sdivup (rows *^ cols) blocksz))
            (tid : natlt blocksz).
      kpre comb gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  Kuiper.Poly.GEMM.Naive.setup comb gA gB gC #eA #eB #eC ();

  (* At this point we split the matrices in cells. We now factor
  that forall+ into chunks of blocksz. But first, we gotta pad it with empties. *)

  assert (pure ((sdivup (rows *^ cols) blocksz) * blocksz >= rows *^ cols));
  forevery_pad (rows *^ cols) (SZ.v (sdivup (rows *^ cols) blocksz) * blocksz)
    (fun (rc : natlt (rows *^ cols)) -> Naive.kpre comb gA gB gC eA eB eC fA fB rc);
  forevery_factor
    ((sdivup (rows *^ cols) blocksz) * blocksz)
    (sdivup (rows *^ cols) blocksz)
    blocksz
    _;

  (* Convince Z3 *)
  forevery_ext_2
    (fun (bid : natlt (sdivup (rows *^ cols) blocksz)) (tid : natlt blocksz) ->
       if bid * blocksz + tid < rows *^ cols
       then
         (gA |-> Frac (fA /. (rows * cols)) eA) **
         (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols)
           (macc eC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols))
       else emp)
    (fun bid tid -> kpre comb gA gB gC eA eB eC fA fB bid tid);
  ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA)
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (_ : squash (rows * cols <= max_blocks))
  ()
  norewrite
  requires
    (forall+ (bid : natlt (sdivup (rows *^ cols) blocksz))
            (tid : natlt blocksz).
      kpost comb gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  (* Idem. *)
  forevery_ext_2 #(natlt (sdivup (rows *^ cols) blocksz)) #_ #(natlt blocksz)
    (fun bid tid -> kpost comb gA gB gC eA eB eC fA fB bid tid)
    (fun (bid : natlt (sdivup (rows *^ cols) blocksz)) (tid : natlt blocksz) ->
       if bid * blocksz + tid < rows *^ cols
       then
        (gA |-> Frac (fA /. (rows * cols)) eA) **
        (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols)
           (MS.gemm_single comb eA eB eC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols))
       else emp);
  forevery_unfactor'
    ((sdivup (rows *^ cols) blocksz) * blocksz)
    (sdivup (rows *^ cols) blocksz)
    blocksz
    (fun (bid : natlt (sdivup (rows *^ cols) blocksz)) (tid : natlt blocksz) ->
       if bid * blocksz + tid < rows *^ cols
       then
         (gA |-> Frac (fA /. (rows * cols)) eA) **
         (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols)
           (MS.gemm_single comb eA eB eC ((bid * blocksz + tid) / cols) ((bid * blocksz + tid) % cols))
       else emp);
  forevery_ext #(natlt (sdivup (rows *^ cols) blocksz * blocksz))
    (fun i ->
       if i / blocksz * blocksz + i % blocksz < rows *^ cols
       then
         (gA |-> Frac (fA /. (rows * cols)) eA) **
         (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC #1.0R ((i/blocksz * blocksz + i%blocksz) / cols) ((i/blocksz * blocksz + i%blocksz) % cols)
           (MS.gemm_single comb eA eB eC ((i/blocksz * blocksz + i%blocksz) / cols) ((i/blocksz * blocksz + i%blocksz) % cols))
       else emp)
    (fun i ->
       if i < rows *^ cols
       then
         (gA |-> Frac (fA /. (rows * cols)) eA) **
         (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC #1.0R (i / cols) (i % cols)
           (MS.gemm_single comb eA eB eC (i / cols) (i % cols))
       else emp);
  forevery_unpad (rows *^ cols) (SZ.v (sdivup (rows *^ cols) blocksz) * blocksz)
    (fun (i : natlt (rows *^ cols)) ->
         (gA |-> Frac (fA /. (rows * cols)) eA) **
         (gB |-> Frac (fB /. (rows * cols)) eB) **
         M.gpu_matrix_pts_to_cell gC #1.0R (i / cols) (i % cols)
           (MS.gemm_single comb eA eB eC (i / cols) (i % cols))
       );
  Naive.teardown comb gA gB gC #eA #eB #eC ();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA)
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (_ : squash (rows * cols <= max_blocks))
  : kernel_desc_m_n
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
=
{
  nblk = sdivup (rows *^ cols) blocksz;
  nthr = blocksz;

  frame = emp;

  block_pre  = (fun bid -> forall+ (tid : natlt blocksz). kpre  comb gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt blocksz). kpost comb gA gB gC eA eB eC fA fB bid tid);
  setup    = setup    comb gA #fA gB #fB gC #eA #eB #eC ();
  teardown = teardown comb gA #fA gB #fB gC #eA #eB ();

  block_setup = (fun bid -> Kuiper.Frame.emp_intro_r2 ());
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());
  block_frame = (fun _bid -> emp);

  kpre  = kpre  comb gA gB gC eA eB eC fA fB;
  kpost = kpost comb gA gB gC eA eB eC fA fB;

  f = kf comb gA #fA gB #fB gC #eA #eB #eC;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : M.gpu_matrix et lA)
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (rows * cols <= max_blocks) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  launch_sync (kdesc comb gA gB gC #eA #eB #eC ());
}
