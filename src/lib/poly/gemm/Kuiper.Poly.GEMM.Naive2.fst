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

let in_bounds (rows cols bid tid : nat) : GTot bool =
  bid * 1024 + tid < rows * cols

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  (bid : natlt (divup (rows * cols) 1024sz))
  (tid : natlt 1024sz)
  : slprop
  =
  (if in_bounds rows cols bid tid
   then
    M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
    M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
    M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
       (macc eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols))
   else
     emp)

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  (bid : natlt (divup (rows * cols) 1024sz))
  (tid : natlt 1024sz)
  : slprop
  =
  (if in_bounds rows cols bid tid
   then
    M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
    M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
    M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
      (MS.gemm_single comb eA eB eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols) shared)
   else
     emp)

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  (bid : szlt (divup (rows *^ cols) 1024sz))
  (tid : szlt 1024sz)
  ()
  requires
    gpu **
    kpre comb gA gB gC eA eB eC f bid tid **
    thread_id 1024 tid **
    block_id (divup (rows * cols) 1024) bid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC f bid tid **
    thread_id 1024 tid **
    block_id (divup (rows * cols) 1024) bid
{
  (* Should remove this admit by constraining the sizes, but it's
     pretty benign. We know bid*1024 does not overflow a size_t,
     and size_t is almost certainly a multiple of 1024, so this
     cannot fail. *)
  assert (pure (SZ.fits (bid * 1024)));
  assume (pure (SZ.fits (bid * 1024 + tid)));
  let id = bid *^ 1024sz +^ tid;

  if SZ.lt id (rows *^ cols) {
    rewrite each in_bounds (SZ.v rows) (SZ.v cols) (SZ.v bid) (SZ.v tid) as true;

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
      M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
      M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
      M.gpu_matrix_pts_to_cell gC trow tcol
        (MS.gemm_single comb eA eB eC trow tcol shared)
    as
      (if (in_bounds rows cols bid tid)
       then
        M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
        M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
        M.gpu_matrix_pts_to_cell gC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
         (MS.gemm_single comb eA eB eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols) shared)
       else emp);

    ()
  } else {
    (* Out of bounds, do nothing *)
    (* Funny, we need to go via emp to convince Pulse. *)
    rewrite
      (if in_bounds rows cols bid tid
       then
         M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
            (macc eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols))
       else
         emp)
    as emp;
    rewrite emp as
      (if in_bounds rows cols bid tid
       then
         M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
           (MS.gemm_single comb eA eB eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols) shared)
       else emp);
  }
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  ()
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    (forall+ (bid : natlt (sdivup (rows *^ cols) 1024sz))
            (tid : natlt 1024).
      kpre comb gA gB gC eA eB eC 1.0R bid tid) **
    emp (* frame *)
{
  Kuiper.Poly.GEMM.Naive.setup comb gA gB gC #eA #eB #eC ();

  (* At this point we split the matrices in cells. We now factor
  that forall+ into chunks of 1024. But first, we gotta pad it with empties. *)

  assert (pure ((sdivup (rows *^ cols) 1024sz) * 1024 >= rows *^ cols));
  forevery_pad (rows *^ cols) (SZ.v (sdivup (rows *^ cols) 1024sz) * 1024)
    (fun (rc : natlt (rows *^ cols)) -> Naive.kpre comb gA gB gC eA eB eC 1.0R rc);
  forevery_factor
    ((sdivup (rows *^ cols) 1024sz) * 1024)
    (sdivup (rows *^ cols) 1024sz)
    1024
    _;

  (* Convince Z3 *)
  forevery_ext_2
    (fun (bid : natlt (sdivup (rows *^ cols) 1024sz)) (tid : natlt 1024) ->
       if bid * 1024 + tid < rows *^ cols
       then
         M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
           (macc eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols))
       else emp)
    (fun bid tid -> kpre comb gA gB gC eA eB eC 1.0R bid tid);
  ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  ()
  requires
    (forall+ (bid : natlt (sdivup (rows *^ cols) 1024sz))
            (tid : natlt 1024).
      kpost comb gA gB gC eA eB eC 1.0R bid tid) **
    emp (* frame *)
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.gemm comb eC eA eB)
{
  (* Idem. *)
  forevery_ext_2 #(natlt (sdivup (rows *^ cols) 1024sz)) #_ #(natlt 1024)
    (fun bid tid -> kpost comb gA gB gC eA eB eC 1.0R bid tid)
    (fun (bid : natlt (sdivup (rows *^ cols) 1024sz)) (tid : natlt 1024) ->
       if bid * 1024 + tid < rows *^ cols
       then
         M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
           (MS.gemm_single comb eA eB eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols) shared)
       else emp);
  forevery_unfactor'
    ((sdivup (rows *^ cols) 1024sz) * 1024)
    (sdivup (rows *^ cols) 1024sz)
    1024
    (fun (bid : natlt (sdivup (rows *^ cols) 1024sz)) (tid : natlt 1024) ->
       if bid * 1024 + tid < rows *^ cols
       then
         M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
           (MS.gemm_single comb eA eB eC ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols) shared)
       else emp);
  forevery_ext #(natlt (sdivup (rows *^ cols) 1024sz * 1024))
    (fun i ->
       if i / 1024 * 1024 + i % 1024 < rows *^ cols
       then
         M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC #1.0R ((i/1024 * 1024 + i%1024) / cols) ((i/1024 * 1024 + i%1024) % cols)
           (MS.gemm_single comb eA eB eC ((i/1024 * 1024 + i%1024) / cols) ((i/1024 * 1024 + i%1024) % cols) shared)
       else emp)
    (fun i ->
       if i < rows *^ cols
       then
         M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC #1.0R (i / cols) (i % cols)
           (MS.gemm_single comb eA eB eC (i / cols) (i % cols) shared)
       else emp);
  forevery_unpad (rows *^ cols) (SZ.v (sdivup (rows *^ cols) 1024sz) * 1024)
    (fun (i : natlt (rows *^ cols)) ->
         M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
         M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB **
         M.gpu_matrix_pts_to_cell gC #1.0R (i / cols) (i % cols)
           (MS.gemm_single comb eA eB eC (i / cols) (i % cols) shared)
       );
  Naive.teardown comb gA gB gC #eA #eB #eC ();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  : kernel_desc_m_n
    (((gA |-> eA) ** (gB |-> eB)) **
     ((gC |-> eC)))
    (((gA |-> eA) ** (gB |-> eB)) **
      (gC |-> MS.gemm comb eC eA eB))
=
{
  nblk = sdivup (rows *^ cols) 1024sz;
  nthr = 1024sz;

  frame = emp;

  block_pre  = (fun bid -> forall+ (tid : natlt 1024). kpre  comb gA gB gC eA eB eC 1.0R bid tid);
  block_post = (fun bid -> forall+ (tid : natlt 1024). kpost comb gA gB gC eA eB eC 1.0R bid tid);
  setup    = setup    comb gA gB gC #eA #eB #eC ();
  teardown = teardown comb gA gB gC #eA #eB ();

  block_setup = (fun bid -> Kuiper.Frame.emp_intro_r ());
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());
  block_frame = (fun _bid -> emp);

  kpre  = kpre  comb gA gB gC eA eB eC 1.0R;
  kpost = kpost comb gA gB gC eA eB eC 1.0R;

  f = kf comb gA gB gC #eA #eB #eC #1.0R;
}

inline_for_extraction noextract
fn matmul_gpu
  (#et : Type0) {| scalar et |}
  (comb : et -> et -> et)
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
  launch_sync (kdesc comb gA gB gC #eA #eB #eC ());
}
