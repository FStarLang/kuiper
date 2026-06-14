module Kuiper.Kernel.GEMM.Naive3

#set-options "--z3rlimit 20"

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor { tensor_pts_to_cell as pts_to_cell }
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.Index
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.EMatrix { mkM, macc }

let abs_bij (#m #n : nat)
  : (abs (m @| n @| INil) =~ (natlt m & natlt n)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
    ff_gg = ez;
    gg_ff = ez;
  }

unfold
let kpre
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : nat)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (rA : chest2 real m k{eA %~ rA})
  (rB : chest2 real k n{eB %~ rB})
  (rC : chest2 real m n{eC %~ rC})
  (fA fB : perm)
  (gid : natlt (m * n))
  : slprop
  =
  gA |-> Frac (fA /. (m * n)) eA **
  gB |-> Frac (fB /. (m * n)) eB **
  pts_to_cell gC
    (gid / n, (gid % n, ()))
    (acc eC (gid / n, (gid % n, ())))

unfold
let kpost
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : nat)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (rA : chest2 real m k{eA %~ rA})
  (rB : chest2 real k n{eB %~ rB})
  (rC : chest2 real m n{eC %~ rC})
  (fA fB : perm)
  (gid : natlt (m * n))
  : slprop
  =
  gA |-> Frac (fA /. (m * n)) eA **
  gB |-> Frac (fB /. (m * n)) eB **
  exists* v.
    pts_to_cell gC (gid / n, (gid % n, ())) v **
      pure (v %~ MS.gemm_single comb_r rA rB rC (gid / n) (gid % n))

#set-options "--split_queries always"

inline_for_extraction noextract
fn kf
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : SZ.t)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (rA : chest2 real m k{eA %~ rA})
  (rB : chest2 real k n{eB %~ rB})
  (rC : chest2 real m n{eC %~ rC})
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
  let trow : szlt m = gid /^ n; assert (rewrites_to trow (gid /^ n));
  let tcol : szlt n = gid %^ n; assert (rewrites_to tcol (gid %^ n));

  let s = Kuiper.DotProd.matmul_kahan_dotprod_t gA gB trow tcol rA rB;

  let v0 = tensor_read_cell gC (trow, (tcol, ()));
  let v1 = comb v0 s;
  tensor_write_cell gC (trow, (tcol, ())) v1;

  ()
}

ghost
fn setup
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (rA : chest2 real m k{eA %~ rA})
  (rB : chest2 real k n{eB %~ rB})
  (rC : chest2 real m n{eC %~ rC})
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
  // Sharing the input matrices (splitting permissions)
  tensor_share_n gA (m *^ n);
  tensor_share_n gB (m *^ n);

  // Sharing the output matrix (splitting each cell)
  tensor_explode gC;
  forevery_iso (abs_bij #m #n) _;
  forevery_ext _ (fun (ij : natlt m & natlt n) ->
    pts_to_cell gC (fst ij, (snd ij, ())) (acc eC (fst ij, (snd ij, ()))));
  forevery_unflatten' _;

  forevery_unfactor' (m *^ n) m n (fun r c ->
    pts_to_cell gC (r, (c, ())) (acc eC (r, (c, ()))));

  // Join resources into a single bigstar
  forevery_zip #(natlt2 m n)
    (fun _ -> gB |-> Frac (fB /. (m *^ n)) eB)
    (fun i -> pts_to_cell gC ((i/n <: natlt m), ((i%n <: natlt n), ())) (acc eC ((i/n <: natlt m), ((i%n <: natlt n), ()))));
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
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (rA : chest2 real m k{eA %~ rA})
  (rB : chest2 real k n{eB %~ rB})
  (rC : chest2 real m n{eC %~ rC})
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
    (exists* (eC : chest2 et m n).
      gC |-> eC ** pure (eC %~ MS.mmcomb comb_r rC rA rB))
{
  forevery_unzip3
    (fun (gid : natlt (m *^ n)) -> gA |-> Frac (fA /. (m * n)) eA)
    (fun (gid : natlt (m *^ n)) -> gB |-> Frac (fB /. (m * n)) eB)
    _;

  forevery_rw_type (natlt (m *^ n)) (natlt (m * n))
    (fun _ -> gA |-> Frac (fA /. (v m * v n)) eA);
  forevery_rw_type (natlt (m *^ n)) (natlt (m * n))
    (fun _ -> gB |-> Frac (fB /. (v m * v n)) eB);

  tensor_gather_n gA _;
  tensor_gather_n gB _;

  forevery_factor (m *^ n) m n _;

  let vf = forevery_exists_2 #(natlt m) #_ #(natlt n) _;

  (* need to use ext to get rid of the [(r*n+c)/n] arithmetic *)
  assert (pure (forall (r c : nat). c < n ==> (r * n + c) / n == r));
  assert (pure (forall (r c : nat). c < n ==> (r * n + c) % n == c));
  forevery_ext_2 _
    (fun (r : natlt m) (c : natlt n) ->
      pts_to_cell gC (r, (c, ())) (vf r c) **
        pure (vf r c %~ MS.gemm_single comb_r rA rB rC r c));

  forevery_extract_pure_2
    (fun (r : natlt m) (c : natlt n) ->
      pts_to_cell gC (r, (c, ())) (vf r c) **
        pure (vf r c %~ MS.gemm_single comb_r rA rB rC r c))
    (fun (r : natlt m) (c : natlt n) ->
      vf r c %~ MS.gemm_single comb_r rA rB rC r c)
    fn r c { (); };

  let eC' : chest2 et m n = mkM (fun (r : natlt m) (c : natlt n) -> vf r c);

  ghost
  fn aux (r:natlt m) (c:natlt n)
    requires
      pts_to_cell gC (r, (c, ())) (vf r c) **
        pure (vf r c %~ MS.gemm_single comb_r rA rB rC r c)
    ensures
      pts_to_cell gC (r, (c, ())) (acc eC' (r, (c, ())))
  {
    assert (pure (macc eC' r c == vf r c));
    ()
  };
  forevery_map_2 #(natlt m) #(natlt n)
    (fun (r : natlt m) (c : natlt n) ->
      pts_to_cell gC (r, (c, ())) (vf r c) **
        pure (vf r c %~ MS.gemm_single comb_r rA rB rC r c))
    (fun (r : natlt m) (c : natlt n) ->
      pts_to_cell gC (r, (c, ())) (acc eC' (r, (c, ()))))
    aux;

  forevery_flatten' (fun (rc : natlt m & natlt n) ->
    pts_to_cell gC (fst rc, (snd rc, ())) (acc eC' (fst rc, (snd rc, ()))));

  forevery_iso (bij_sym (abs_bij #m #n)) _;
  forevery_ext _ (fun (i : abs (m @| n @| INil)) ->
    pts_to_cell gC i (acc eC' i));
  tensor_implode gC;

  assert (pure (eC' %~ MS.mmcomb comb_r rC rA rB));
  ();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
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
  (#eA #eB #eC : chest2 et _ _)
  (rA : chest2 real m k{eA %~ rA})
  (rB : chest2 real k n{eB %~ rB})
  (rC : chest2 real m n{eC %~ rC})
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** (exists* (eC : chest2 et m n). gC |-> eC ** pure (eC %~ MS.mmcomb comb_r rC rA rB)))
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
  launch_sync (kdesc comb comb_r gA gB gC rA rB rC);
  ()
}
