module Kuiper.Poly.GEMM.Naive

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
  (bid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.pts_to_cell gC (bid / cols, bid % cols)
    (macc eC (bid / cols) (bid % cols))

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
  (bid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.pts_to_cell gC (bid / cols, bid % cols)
    (MS.gemm_single comb eA eB eC (bid / cols) (bid % cols))

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
  (bid : szlt (rows * cols))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA gB gC eA eB eC fA fB bid **
    block_id (rows *^ cols) bid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC fA fB bid **
    block_id (rows *^ cols) bid
{
  let trow = bid /^ cols; assert (rewrites_to trow (bid /^ cols));
  let tcol = bid %^ cols; assert (rewrites_to tcol (bid %^ cols));

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
    (forall+ (rc : natlt (rows *^ cols)).
      kpre comb gA gB gC eA eB eC fA fB rc) **
    emp (* frame *)
{
  // Sharing the input matrices (splitting permissions)
  M.share_n gA (rows *^ cols);
  M.share_n gB (rows *^ cols);

  // Sharing the output matrix (splitting each cell)
  M.explode gC;
  forevery_rw_type (M.ait rows cols) (natlt rows & natlt cols) _;
  forevery_unflatten' _;

  forevery_unfactor' (rows *^ cols) rows cols (fun r c ->
    M.pts_to_cell gC (r, c) (macc eC r c));

  // Join resources into a single bigstar
  forevery_zip #(natlt2 rows cols)
    (fun _ -> gB |-> Frac (fB /. (rows *^ cols)) eB)
    (fun i -> M.pts_to_cell gC ((i/cols <: natlt rows), (i%cols <: natlt cols)) (macc eC (i/cols) (i%cols)));
  forevery_zip #(natlt2 rows cols)
    (fun _ -> gA |-> Frac (fA /. (rows *^ cols)) eA)
    _;

  (* We're done actually. Just need extensionality. *)
  forevery_ext #(natlt2 rows cols) _ (kpre comb gA gB gC eA eB eC fA fB);

  ();
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
    (forall+ (rc : natlt (rows *^ cols)).
      kpost comb gA gB gC eA eB eC fA fB rc) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  forevery_unzip #(natlt2 rows cols) _ _;
  forevery_unzip #(natlt2 rows cols) _ _;

  forevery_rw_type
    (natlt (v (SizeT.mul rows cols)))
    (natlt (v rows * v cols))
    (fun _ ->
      M.pts_to #et gA #(fA /. (v rows * v cols)) eA);

  forevery_rw_type
    (natlt (v (SizeT.mul rows cols)))
    (natlt (v rows * v cols))
    (fun _ ->
      M.pts_to #et gB #(fB /. (v rows * v cols)) eB);

  M.gather_n gA _;
  M.gather_n gB _;

  forevery_factor (rows *^ cols) rows cols _;

  (* we get things back with some arithmetic in it *)
  assert (forall+ (r:natlt rows) (c:natlt cols).
      M.pts_to_cell gC (((r * cols + c) / cols <: natlt rows), ((r * cols + c) % cols <: natlt cols))
         (MS.gemm_single comb eA eB eC ((r * cols + c) / cols) ((r * cols + c) % cols)));

  (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));
  forevery_ext_2 _ (fun (r : natlt rows) (c : natlt cols) ->
      M.pts_to_cell gC (r, c) (MS.gemm_single comb eA eB eC r c));

  ghost
  fn aux (r:natlt rows) (c:natlt cols)
    requires
      M.pts_to_cell gC (r, c) (MS.gemm_single comb eA eB eC r c)
    ensures
      M.pts_to_cell gC (r, c) (macc (MS.mmcomb comb eC eA eB) r c)
  {
    ()
  };
  forevery_map_2 #(natlt rows) #(natlt cols)
    (fun r c -> M.pts_to_cell gC (r, c) (MS.gemm_single comb eA eB eC r c))
    _
    aux;

  forevery_flatten' (fun (rc : natlt rows & natlt cols) ->
    M.pts_to_cell gC rc (macc (MS.mmcomb comb eC eA eB) (fst rc) (snd rc)));
  M.implode gC;
  ()
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
  (#_ : squash (m * n <= max_blocks))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
= {
  nblk = m *^ n;

  frame = emp;

  setup    = setup    comb gA gB gC;
  teardown = teardown comb gA gB gC;

  kpre  = kpre  comb gA gB gC eA eB eC fA fB;
  kpost = kpost comb gA gB gC eA eB eC fA fB;

  f = kf comb gA gB gC;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_m_1 _ _

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
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) ** (* size_req *)
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
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
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
