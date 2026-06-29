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
  }

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (bid : natlt (m * n))
  : slprop
  =
  gA |-> Frac (fA /. (m * n)) eA **
  gB |-> Frac (fB /. (m * n)) eB **
  pts_to_cell gC
    (bid / n, (bid % n, ()))
    (acc eC (bid / n, (bid % n, ())))

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
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
  (fA fB : perm)
  (bid : natlt (m * n))
  =
  gA |-> Frac (fA /. (m * n)) eA **
  gB |-> Frac (fB /. (m * n)) eB **
  pts_to_cell gC (bid / n, (bid % n, ()))
    (MS.gemm_single comb eA eB eC (bid / n) (bid % n))

#set-options "--split_queries always"

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : SZ.t)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (#eA : chest2 et m k)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  (#fA #fB : perm)
  (bid : szlt (m * n))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA gB gC eA eB eC fA fB bid **
    block_id (m *^ n) bid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC fA fB bid **
    block_id (m *^ n) bid
{
  let trow : szlt m = bid /^ n; assert (rewrites_to trow (bid /^ n));
  let tcol : szlt n = bid %^ n; assert (rewrites_to tcol (bid %^ n));

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
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (#fA : perm)
  (gB : tensor et lB)
  (#fB : perm)
  (gC : tensor et lC)
  (#eA : chest2 et m k)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (rc : natlt (m *^ n)).
      kpre comb gA gB gC eA eB eC fA fB rc) **
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

  (* We're done actually. Just need extensionality. *)
  forevery_ext #(natlt2 m n) _ (kpre comb gA gB gC eA eB eC fA fB);

  ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (#fA : perm)
  (gB : tensor et lB)
  (#fB : perm)
  (gC : tensor et lC)
  (#eA : chest2 et m k)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  ()
  norewrite
  requires
    (forall+ (rc : natlt (m *^ n)).
      kpost comb gA gB gC eA eB eC fA fB rc) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  forevery_unzip #(natlt2 m n) _ _;
  forevery_unzip #(natlt2 m n) _ _;

  forevery_rw_type
    (natlt (v (SizeT.mul m n)))
    (natlt (v m * v n))
    (fun _ -> gA |-> Frac (fA /. (v m * v n)) eA);

  forevery_rw_type
    (natlt (v (SizeT.mul m n)))
    (natlt (v m * v n))
    (fun _ -> gB |-> Frac (fB /. (v m * v n)) eB);

  tensor_gather_n gA _;
  tensor_gather_n gB _;

  forevery_factor (m *^ n) m n _;

  (* we get things back with some arithmetic in it *)
  assert (forall+ (r:natlt m) (c:natlt n).
      pts_to_cell gC (((r * n + c) / n <: natlt m), (((r * n + c) % n <: natlt n), ()))
         (MS.gemm_single comb eA eB eC ((r * n + c) / n) ((r * n + c) % n)));

  (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  assert (pure (forall (r c : nat). c < n ==> (r * n + c) / n == r));
  assert (pure (forall (r c : nat). c < n ==> (r * n + c) % n == c));
  forevery_ext_2 _ (fun (r : natlt m) (c : natlt n) ->
      pts_to_cell gC (r, (c, ())) (MS.gemm_single comb eA eB eC r c));

  ghost
  fn aux (r:natlt m) (c:natlt n)
    requires
      pts_to_cell gC (r, (c, ())) (MS.gemm_single comb eA eB eC r c)
    ensures
      pts_to_cell gC (r, (c, ())) (acc (MS.mmcomb comb eC eA eB) (r, (c, ())))
  {
    ()
  };
  forevery_map_2 #(natlt m) #(natlt n)
    (fun r c -> pts_to_cell gC (r, (c, ())) (MS.gemm_single comb eA eB eC r c))
    _
    aux;

  forevery_flatten' (fun (rc : natlt m & natlt n) ->
    pts_to_cell gC (fst rc, (snd rc, ())) (acc (MS.mmcomb comb eC eA eB) (fst rc, (snd rc, ()))));

  forevery_iso (bij_sym (abs_bij #m #n)) _;
  forevery_ext _ (fun (i : abs (m @| n @| INil)) ->
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
