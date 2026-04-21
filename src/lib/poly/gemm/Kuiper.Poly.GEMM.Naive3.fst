module Kuiper.Poly.GEMM.Naive3

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
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : nat)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et m n)
  (rA : ematrix real m k{eA %~ rA})
  (rB : ematrix real k n{eB %~ rB})
  (rC : ematrix real m n{eC %~ rC})
  (fA fB : perm)
  (gid : natlt (m * n))
  : slprop
  =
  gA |-> Frac (fA /. (m * n)) eA **
  gB |-> Frac (fB /. (m * n)) eB **
  M.pts_to_cell gC (gid / n, gid % n)
    (macc eC (gid / n) (gid % n))

unfold
let kpost
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : nat)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et m n)
  (rA : ematrix real m k{eA %~ rA})
  (rB : ematrix real k n{eB %~ rB})
  (rC : ematrix real m n{eC %~ rC})
  (fA fB : perm)
  (gid : natlt (m * n))
  : slprop
  =
  gA |-> Frac (fA /. (m * n)) eA **
  gB |-> Frac (fB /. (m * n)) eB **
  exists* v.
    M.pts_to_cell gC (gid / n, gid % n) v **
      pure (v %~ MS.gemm_single comb_r rA rB rC (gid / n) (gid % n))

inline_for_extraction noextract
fn kf
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| cA : T.ctlayout lA, cB : T.ctlayout lB, cC : T.ctlayout lC |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et m n)
  (rA : ematrix real m k{eA %~ rA})
  (rB : ematrix real k n{eB %~ rB})
  (rC : ematrix real m n{eC %~ rC})
  (fA fB : perm)
  (gid : szlt (m *^ n))
  ()
  norewrite
  requires
    gpu **
    kpre comb comb_r gA gB gC eA eB eC rA rB rC fA fB gid
  ensures
    gpu **
    kpost comb comb_r gA gB gC eA eB eC rA rB rC fA fB gid
{
  let trow = gid /^ n; assert (rewrites_to trow (gid /^ n));
  let tcol = gid %^ n; assert (rewrites_to tcol (gid %^ n));

  let s = Kuiper.DotProd.matmul_kahan_dotprod gA gB trow tcol rA rB;

  let v0 = M.read_cell' gC trow tcol;
  let v1 = comb v0 s;
  M.write_cell' gC trow tcol v1;

  ()
}

ghost
fn setup
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et m n)
  (rA : ematrix real m k{eA %~ rA})
  (rB : ematrix real k n{eB %~ rB})
  (rC : ematrix real m n{eC %~ rC})
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (gid : natlt (m *^ n)).
      kpre comb comb_r gA gB gC eA eB eC rA rB rC fA fB gid) **
    emp (* frame *)
{
  M.share_n gA (m *^ n);
  M.share_n gB (m *^ n);

  M.explode gC;
  forevery_rw_type (M.ait m n) (natlt m & natlt n) _;
  forevery_unflatten' _;
  forevery_unfactor' (m *^ n) m n (fun r c ->
    M.pts_to_cell gC (r, c) (macc eC r c));

  forevery_zip #(natlt2 m n)
    (fun _ -> gB |-> Frac (fB /. (m *^ n)) eB)
    (fun i -> M.pts_to_cell gC ((i/n <: natlt m), (i%n <: natlt n)) (macc eC (i/n) (i%n)));
  forevery_zip #(natlt2 m n)
    (fun _ -> gA |-> Frac (fA /. (m *^ n)) eA)
    _;

  forevery_ext #(natlt2 m n) _ (kpre comb comb_r gA gB gC eA eB eC rA rB rC fA fB);
  ();
}

ghost
fn teardown
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et m n)
  (rA : ematrix real m k{eA %~ rA})
  (rB : ematrix real k n{eB %~ rB})
  (rC : ematrix real m n{eC %~ rC})
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (gid : natlt (m *^ n)).
      kpost comb comb_r gA gB gC eA eB eC rA rB rC fA fB gid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC : ematrix et m n).
      gC |-> eC ** pure (eC %~ MS.mmcomb comb_r rC rA rB))
{
  forevery_unzip3
    (fun (gid : natlt (m *^ n)) -> gA |-> Frac (fA /. (m * n)) eA)
    (fun (gid : natlt (m *^ n)) -> gB |-> Frac (fB /. (m * n)) eB)
    _;

  forevery_rw_type (natlt (m *^ n)) (natlt (m * n))
    (fun _ -> M.pts_to #et gA #(fA /. (v m * v n)) eA);
  forevery_rw_type (natlt (m *^ n)) (natlt (m * n))
    (fun _ -> M.pts_to #et gB #(fB /. (v m * v n)) eB);

  M.gather_n gA _;
  M.gather_n gB _;

  forevery_factor (m *^ n) m n _;
  let vf = forevery_exists_2 #(natlt m) #_ #(natlt n) _;

  forevery_ext_2 _
    (fun (r : natlt m) (c : natlt n) ->
      M.pts_to_cell gC (r, c) (vf r c) **
        pure (vf r c %~ MS.gemm_single comb_r rA rB rC r c));

  forevery_extract_pure_2
    (fun (r : natlt m) (c : natlt n) ->
      M.pts_to_cell gC (r, c) (vf r c) **
        pure (vf r c %~ MS.gemm_single comb_r rA rB rC r c))
    (fun (r : natlt m) (c : natlt n) ->
      vf r c %~ MS.gemm_single comb_r rA rB rC r c)
    fn r c { (); };

  let eC' : ematrix et m n = mkM (fun (r : natlt m) (c : natlt n) -> vf r c);
  forevery_map_2
    (fun (r : natlt m) (c : natlt n) ->
      M.pts_to_cell gC (r, c) (vf r c) **
        pure (vf r c %~ MS.gemm_single comb_r rA rB rC r c))
    (fun (r : natlt m) (c : natlt n) ->
      M.pts_to_cell gC (r, c) (macc eC' r c))
    fn r c { () };

  forevery_flatten' (fun (rc : natlt m & natlt n) ->
    M.pts_to_cell gC rc (macc eC' (fst rc) (snd rc)));
  M.implode gC;
  assert (pure (eC' %~ MS.mmcomb comb_r rC rA rB));
  ();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (#eA #eB #eC : ematrix et _ _)
  (rA : ematrix real m k{eA %~ rA})
  (rB : ematrix real k n{eB %~ rB})
  (rC : ematrix real m n{eC %~ rC})
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** (exists* (eC : ematrix et m n). gC |-> eC ** pure (eC %~ MS.mmcomb comb_r rC rA rB)))
=
{
  nthr     = m *^ n;
  frame    = emp;
  setup    = setup    comb comb_r gA gB gC eA eB eC rA rB rC fA fB;
  teardown = teardown comb comb_r gA gB gC eA eB eC rA rB rC fA fB;
  kpre     = kpre     comb comb_r gA gB gC eA eB eC rA rB rC fA fB;
  kpost    = kpost    comb comb_r gA gB gC eA eB eC rA rB rC fA fB;
  f        = kf       comb comb_r gA gB gC eA eB eC rA rB rC fA fB;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
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
  launch_sync (kdesc comb comb_r gA gB gC rA rB rC);
  ()
}
