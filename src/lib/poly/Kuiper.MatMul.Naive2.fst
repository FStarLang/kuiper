module Kuiper.MatMul.Naive2

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module MU = Kuiper.MatMul.Util
module SZ = FStar.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

let in_bounds (rows cols bid tid : nat) : GTot bool =
  bid * 1024 + tid < rows * cols

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
  (bid : natlt (divup (rows * cols) 1024))
  (tid : natlt 1024)
  : slprop
  =
  pure (SZ.fits (bid * 1024 + tid)) **
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  (if in_bounds rows cols bid tid
   then
    exists* v.
      M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols) v
   else
     emp)

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
  (bid : natlt (divup (rows * cols) 1024))
  (tid : natlt 1024)
  : slprop
  =
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  (if in_bounds rows cols bid tid
   then
    M.gpu_matrix_pts_to_cell gC #1.0R ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols)
      (MS.matmul_single eA eB ((bid * 1024 + tid) / cols) ((bid * 1024 + tid) % cols) shared)
   else
     emp)

inline_for_extraction noextract
fn kernel
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
  (ebid : enatlt (divup (rows * cols) 1024))
  (etid : enatlt 1024)
  requires
    gpu **
    block_id (divup (rows * cols) 1024) ebid **
    thread_id (1024) etid **
    kpre gA gB gC eA eB f ebid etid
  ensures
    gpu **
    block_id (divup (rows * cols) 1024) ebid **
    thread_id (1024) etid **
    kpost gA gB gC eA eB f ebid etid
{
  let bid = get_bid ();
  let tid = get_tid ();
  let id = bid *^ 1024sz +^ tid;

  if SZ.lt id (rows *^ cols) {
    rewrite each in_bounds (SZ.v rows) (SZ.v cols) ebid etid as true;
    rewrite each ebid as SZ.v bid;
    rewrite each etid as SZ.v tid;

    let trow, tcol = s_divmod cols id;
    with i0 j0 v0.
      rewrite
        M.gpu_matrix_pts_to_cell gC #1.0R i0 j0 v0
      as
        M.gpu_matrix_pts_to_cell gC #1.0R trow tcol v0;

    assert (pure (trow < rows));
    assert (pure (tcol < cols));

    let s = MU.matmul_dotprod gA gB trow tcol;
    M.gpu_matrix_write_cell gC trow tcol s;

    assert (pure (SZ.v trow == id / cols));
    assert (pure (SZ.v tcol == id % cols));
    rewrite
      M.gpu_matrix_pts_to_cell gC trow tcol
        (MS.matmul_single eA eB trow tcol shared)
    as
      (if (in_bounds rows cols ebid etid)
       then
         M.gpu_matrix_pts_to_cell gC ((ebid * 1024 + etid) / cols) ((ebid * 1024 + etid) % cols)
          (MS.matmul_single eA eB ((ebid * 1024 + etid) / cols) ((ebid * 1024 + etid) % cols) shared)
       else emp);

    ()
  } else {
    (* Out of bounds, do nothing *)
    (* Funny, we need to go via emp to convince Pulse. *)
    rewrite
      (if in_bounds rows cols ebid etid
       then
         exists* v.
           M.gpu_matrix_pts_to_cell gC #1.0R ((ebid * 1024 + etid) / cols) ((ebid * 1024 + etid) % cols) v
       else
         emp)
    as emp;
    rewrite emp as
      (if in_bounds rows cols ebid etid
       then M.gpu_matrix_pts_to_cell gC ((ebid * 1024 + etid) / cols) ((ebid * 1024 + etid) % cols)
             (MS.matmul_single eA eB ((ebid * 1024 + etid) / cols) ((ebid * 1024 + etid) % cols) shared)
       else emp);
  }
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
    forall+ (bid : natlt (divup (rows * cols) 1024))
            (tid : natlt 1024).
      kpre gA gB gC eA eB 1.0R bid tid
{
  (* Reuse Naive.setup + factor the forevery from N into (divup N 1024) * 1024.
     Should be really easy. *)
  admit();
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
    forall+ (bid : natlt (divup (rows * cols) 1024))
            (tid : natlt 1024).
      kpost gA gB gC eA eB 1.0R bid tid
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.matmul eA eB)
{
  (* Idem. *)
  admit();
}

inline_for_extraction noextract
fn matmul_gpu
  (#et : Type0) {| scalar et |}
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
    gC |-> MS.matmul eA eB
{
  open FStar.SizeT;
  setup gA gB gC;

  let nblk = sdivup (rows *^ cols) 1024sz;
  forevery_rw_size (divup (rows * cols) 1024) nblk;

  (* FIXME: F* inference failure means we need to annotate pre/post (somewhat) *)
  (* We also need eta due to the extraction rules looking for it. *)

  launch_kernel_n_m
    nblk
    1024sz
    #(kpre  #et gA gB gC eA eB 1.0R)
    #(kpost #et gA gB gC eA eB 1.0R)
    (fun ebid etid -> kernel _ _ _ gA gB gC ebid etid);

  forevery_rw_size nblk (divup (rows * cols) 1024);

  teardown gA gB gC;

  ();
}
