module Kuiper.Kernel.GEMM.Naive2

friend Kuiper.Kernel.GEMM.Naive (* We reuse setup/teardown from Naive *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor { tensor_pts_to_cell as pts_to_cell }
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT
module N = Kuiper.Kernel.GEMM.Naive
open Kuiper.Index
open Kuiper.Chest

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lA : layout2 rows shared)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (eC : chest2 et rows cols)
  (fA fB : perm)
  (gid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  pts_to_cell gC
    (gid / cols, (gid % cols, ()))
    (acc eC (gid / cols, (gid % cols, ())))

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lA : layout2 rows shared)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (eC : chest2 et rows cols)
  (fA fB : perm)
  (gid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  pts_to_cell gC (gid / cols, (gid % cols, ()))
    (MS.gemm_single comb eA eB eC (gid / cols) (gid % cols))

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : SZ.t)
  (#lA : layout2 rows shared)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (#eA : chest2 et rows shared)
  (#eB : chest2 et shared cols)
  (#eC : chest2 et rows cols)
  (#fA #fB : perm)
  (gid : szlt (rows * cols))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA gB gC eA eB eC fA fB gid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC fA fB gid
{
  let trow : szlt rows = gid /^ cols; assert (rewrites_to trow (gid /^ cols));
  let tcol : szlt cols = gid %^ cols; assert (rewrites_to tcol (gid %^ cols));

  let s = Kuiper.DotProd.matmul_dotprod_t gA gB trow tcol;

  let v0 = tensor_read_cell gC (trow, (tcol, ()));
  let v1 = comb v0 s;
  tensor_write_cell gC (trow, (tcol, ())) v1;

  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : layout2 rows shared)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (#fA : perm)
  (gB : tensor et lB)
  (#fB : perm)
  (gC : tensor et lC)
  (#eA : chest2 et rows shared)
  (#eB : chest2 et shared cols)
  (#eC : chest2 et rows cols)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (gid : natlt (rows *^ cols)).
      kpre comb gA gB gC eA eB eC fA fB gid) **
    emp (* frame *)
{
  N.setup comb gA gB gC #eA #eB #eC ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : layout2 rows shared)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (#fA : perm)
  (gB : tensor et lB)
  (#fB : perm)
  (gC : tensor et lC)
  (#eA : chest2 et rows shared)
  (#eB : chest2 et shared cols)
  (#eC : chest2 et rows cols)
  ()
  norewrite
  requires
    (forall+ (gid : natlt (rows *^ cols)).
      kpost comb gA gB gC eA eB eC fA fB gid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  N.teardown comb gA gB gC #eA #eB #eC ();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA #eB #eC : chest2 et _ _)
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
=
{
  nthr = m *^ n;

  frame = emp;

  setup    = setup    comb gA gB gC;
  teardown = teardown comb gA gB gC;

  kpre  = kpre  comb gA gB gC eA eB eC fA fB;
  kpost = kpost comb gA gB gC eA eB eC fA fB;

  f = kf comb gA gB gC;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA #eB #eC : chest2 et _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  launch_sync (kdesc comb gA gB gC #eA #eB #eC);
}

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA : chest2 et m k)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
{
  mmcomb_gpu_exact comb gA gB gC;
  MU.mmcomb_approx_real comb comb_r eC eA eB rA rB rC;
  ()
}
