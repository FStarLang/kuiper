module Kuiper.Poly.GEMM.Naive2

friend Kuiper.Poly.GEMM.Naive (* We reuse some lemmas from Naive *)

#lang-pulse
open Kuiper
open Kuiper.Approximates
module T = Kuiper.Tensor
module M = Kuiper.Array2
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
open Kuiper.EMatrix

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lA : M.layout rows shared)
  (#lB : M.layout shared cols)
  (#lC : M.layout rows cols)
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (gid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.pts_to_cell gC (gid / cols, gid % cols)
    (macc eC (gid / cols) (gid % cols))

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lA : M.layout rows shared)
  (#lB : M.layout shared cols)
  (#lC : M.layout rows cols)
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (gid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.pts_to_cell gC (gid / cols, gid % cols)
    (MS.gemm_single comb eA eB eC (gid / cols) (gid % cols))

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : SZ.t)
  (#lA : M.layout rows shared)
  (#lB : M.layout shared cols)
  (#lC : M.layout rows cols)
  {| cA : T.ctlayout lA, cB : T.ctlayout lB, cC : T.ctlayout lC |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
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
  let trow = gid /^ cols; assert (rewrites_to trow (gid /^ cols));
  let tcol = gid %^ cols; assert (rewrites_to tcol (gid %^ cols));

  let s = Kuiper.DotProd.matmul_dotprod gA gB trow tcol;

  let v0 = M.read_cell' gC trow tcol;
  let v1 = comb v0 s;
  M.write_cell' gC trow tcol v1;

  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : M.layout rows shared)
  (#lB : M.layout shared cols)
  (#lC : M.layout rows cols)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA)
  (#fA : perm)
  (gB : M.array2 et lB)
  (#fB : perm)
  (gC : M.array2 et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
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
  Kuiper.Poly.GEMM.Naive.setup comb gA gB gC #eA #eB #eC ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : M.layout rows shared)
  (#lB : M.layout shared cols)
  (#lC : M.layout rows cols)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA)
  (#fA : perm)
  (gB : M.array2 et lB)
  (#fB : perm)
  (gC : M.array2 et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
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
  Naive.teardown comb gA gB gC #eA #eB #eC ();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (#eA #eB #eC : ematrix et _ _)
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
=
{
  nthr = m *^ n;

  frame = emp;

  setup    = setup    comb gA #fA gB #fB gC #eA #eB #eC;
  teardown = teardown comb gA #fA gB #fB gC #eA #eB;

  kpre  = kpre  comb gA gB gC eA eB eC fA fB;
  kpost = kpost comb gA gB gC eA eB eC fA fB;

  f = kf comb gA gB gC #eA #eB #eC #fA #fB;
  kpre_sendable=magic();
  kpost_sendable=magic();
} <: kernel_desc_n _ _

// FIXME: extraction of this function (in the inst module) is very slow, around
// 1.5s for each one. This is *after* a lot of tweaking in the definition of the
// kn_as_kmn cast. We seem to spend a lot of time normalizing, probably with an
// exponential explosion somewhere.
inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (#eA #eB #eC : ematrix et _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  launch_sync (kdesc comb gA gB gC);
}

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (#eA : ematrix et m k)
  (#eB : ematrix et k n)
  (#eC : ematrix et m n)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : ematrix et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
{
  mmcomb_gpu_exact comb gA gB gC;
  MU.mmcomb_approx_real comb comb_r eC eA eB rA rB rC;
  ()
}
