module Kuiper.Kernel.GEMM.Naive

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor { tensor_pts_to_cell as pts_to_cell }
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Bijection

(* The reshaping bridge between the nested tensor index
   [abs (rows @| cols @| INil) = natlt rows & (natlt cols & unit)]
   and the flat pair [natlt rows & natlt cols], mirroring
   [Kuiper.Array2.abs_bij]. *)
let abs_bij (#rows #cols : nat)
  : (abs (rows @| cols @| INil) =~ (natlt rows & natlt cols)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
    ff_gg = ez;
    gg_ff = ez;
  }

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
  (bid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  pts_to_cell gC
    (bid / cols, (bid % cols, ()))
    (acc eC (bid / cols, (bid % cols, ())))

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
  (bid : natlt (rows * cols))
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  pts_to_cell gC (bid / cols, (bid % cols, ()))
    (MS.gemm_single comb eA eB eC (bid / cols) (bid % cols))

#set-options "--split_queries always"

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
  let trow : szlt rows = bid /^ cols; assert (rewrites_to trow (bid /^ cols));
  let tcol : szlt cols = bid %^ cols; assert (rewrites_to tcol (bid %^ cols));

  let s = Kuiper.DotProd.matmul_dotprod gA gB trow tcol;

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
    (forall+ (rc : natlt (rows *^ cols)).
      kpre comb gA gB gC eA eB eC fA fB rc) **
    emp (* frame *)
{
  // Sharing the input matrices (splitting permissions)
  tensor_share_n gA (rows *^ cols);
  tensor_share_n gB (rows *^ cols);

  // Sharing the output matrix (splitting each cell)
  tensor_explode gC;
  forevery_iso (abs_bij #rows #cols) _;
  forevery_ext _ (fun (ij : natlt rows & natlt cols) ->
    pts_to_cell gC (fst ij, (snd ij, ())) (acc eC (fst ij, (snd ij, ()))));
  forevery_unflatten' _;

  forevery_unfactor' (rows *^ cols) rows cols (fun r c ->
    pts_to_cell gC (r, (c, ())) (acc eC (r, (c, ()))));

  // Join resources into a single bigstar
  forevery_zip #(natlt2 rows cols)
    (fun _ -> gB |-> Frac (fB /. (rows *^ cols)) eB)
    (fun i -> pts_to_cell gC ((i/cols <: natlt rows), ((i%cols <: natlt cols), ())) (acc eC ((i/cols <: natlt rows), ((i%cols <: natlt cols), ()))));
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
    (fun _ -> gA |-> Frac (fA /. (v rows * v cols)) eA);

  forevery_rw_type
    (natlt (v (SizeT.mul rows cols)))
    (natlt (v rows * v cols))
    (fun _ -> gB |-> Frac (fB /. (v rows * v cols)) eB);

  tensor_gather_n gA _;
  tensor_gather_n gB _;

  forevery_factor (rows *^ cols) rows cols _;

  (* we get things back with some arithmetic in it *)
  assert (forall+ (r:natlt rows) (c:natlt cols).
      pts_to_cell gC (((r * cols + c) / cols <: natlt rows), (((r * cols + c) % cols <: natlt cols), ()))
         (MS.gemm_single comb eA eB eC ((r * cols + c) / cols) ((r * cols + c) % cols)));

  (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));
  forevery_ext_2 _ (fun (r : natlt rows) (c : natlt cols) ->
      pts_to_cell gC (r, (c, ())) (MS.gemm_single comb eA eB eC r c));

  ghost
  fn aux (r:natlt rows) (c:natlt cols)
    requires
      pts_to_cell gC (r, (c, ())) (MS.gemm_single comb eA eB eC r c)
    ensures
      pts_to_cell gC (r, (c, ())) (acc (MS.mmcomb comb eC eA eB) (r, (c, ())))
  {
    ()
  };
  forevery_map_2 #(natlt rows) #(natlt cols)
    (fun r c -> pts_to_cell gC (r, (c, ())) (MS.gemm_single comb eA eB eC r c))
    _
    aux;

  forevery_flatten' (fun (rc : natlt rows & natlt cols) ->
    pts_to_cell gC (fst rc, (snd rc, ())) (acc (MS.mmcomb comb eC eA eB) (fst rc, (snd rc, ()))));

  forevery_iso (bij_sym (abs_bij #rows #cols)) _;
  forevery_ext _ (fun (i : abs (rows @| cols @| INil)) ->
    pts_to_cell gC i (acc (MS.mmcomb comb eC eA eB) i));
  tensor_implode gC;
  ()
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
    pure (m * n <= max_blocks) **
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
