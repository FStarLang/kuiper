module Kuiper.Example.Sparse.GEMM

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.EMatrix
open Kuiper.Sparse
open Kuiper.Sparse.Approx
open Kuiper.Sparse.Approx.Share
open Kuiper.Tensor
open Kuiper.Seq.Common { op_At_Bang }
open Kuiper.Tensor.Layout.Alg { l2_row_major }

let _sparse_matmul_single
  (#nnz #shared #cols : nat)
  (elems : lseq real nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (eB : chest2 real shared cols)
  (ri re : nat{ri <= re /\ re <= nnz /\ sorted_slice col_ind ri re})
  (j : natlt cols)
  (to : nat {ri <= to /\ to <= re})
  : GTot real
=
  let open Kuiper.Sparse.DotProduct in
  _sparse_dprod
    (Seq.slice elems ri re <: lseq real (re - ri))
    (Seq.slice col_ind ri re)
    (ematrix_col eB j)
    (to - ri)

let sparse_matmul_single
  (#nnz #shared #cols : nat)
  (elems : lseq real nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (eB : chest2 real shared cols)
  (ri re : nat{ri <= re /\ re <= nnz /\ sorted_slice col_ind ri re})
  (j : natlt cols)
  : GTot real
=
  _sparse_matmul_single elems col_ind eB ri re j re

(* Expose one real accumulation step before proving its approximation. *)
let sparse_matmul_single_step
  (#nnz #shared #cols : nat)
  (elems : lseq real nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (eB : chest2 real shared cols)
  (ri re : nat{ri <= re /\ re <= nnz /\ sorted_slice col_ind ri re})
  (j : natlt cols)
  (k : nat{ri <= k /\ k < re})
  : Lemma
    (ensures
      _sparse_matmul_single elems col_ind eB ri re j (k + 1) ==
      add (_sparse_matmul_single elems col_ind eB ri re j k)
        (mul (elems @! k) (acc2 eB (col_ind @! k) j)))
  = ()


let rec matmul_all_zeros_lemma
  (#rows #shared #columns : nat)
  (m1 : chest2 real rows shared)
  (m2 : chest2 real shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (from to : nat{from <= to /\ to <= shared})
  : Lemma
    (requires forall i. from <= i /\ i < to ==> acc2 m1 row i == zero)
    (ensures MS.__matmul_single m1 m2 row col from == MS.__matmul_single m1 m2 row col to)
=
  if from = to
    then ()
    else (
      MS.matmul_single_lemma m1 m2 row col to;
      matmul_all_zeros_lemma m1 m2 row col from (to - 1)
    )

let matmul_dotprod_lemma
  (#nnz #rows #shared #cols : nat)
  (elems : lseq real nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (row_off : lseq nat (rows + 1))
  (eB : chest2 real shared cols)
  (i : natlt rows)
  (j : natlt cols)
  : Lemma
    (requires valid_smatrix rows shared col_ind row_off)
    (ensures
      sparse_matmul_single elems col_ind eB (row_off @! i) (row_off @! (i + 1)) j ==
      MS.matmul_single (matrix_unsparse rows shared elems col_ind row_off) eB i j
    )
=
  let open Kuiper.Sparse.DotProduct in

  let eA = matrix_unsparse rows shared elems col_ind row_off in

  let ri = row_off @! i in
  let re = row_off @! (i + 1) in

  let relems   : lseq real  (re - ri) = Seq.slice elems   ri re in
  let rcol_ind : lseq nat (re - ri) = Seq.slice col_ind ri re in

  unsparse_row_lemma rows shared elems col_ind row_off i;
  assert seq_unsparse (re - ri) shared relems rcol_ind == ematrix_row eA i;

  sparse_dprod_lemma relems rcol_ind (ematrix_col eB j);

  dprod_is_matmul_single eA eB i j

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #shared #cols : SZ.t)
  (#lB : layout2 shared cols)
  {| ctlayout lB |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (gB : array2 et lB)
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols { eB %~ rB })
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    smatrix_approx gA #fA eA rA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures pure (res %~ MS.matmul_single rA rB i j)
{
  unfold smatrix_approx gA #fA eA rA;
  with v_elems v_ind v_row_off relems.
    unfold smatrix_model gA #fA v_elems v_ind v_row_off relems eA rA;

  assert pure (forall k. v_ind @! k < shared);

  let ri = Pulse.Lib.Array.(gA.row_off.(i));
  let re = Pulse.Lib.Array.(gA.row_off.(i +^ 1sz));

  let mut dp : et = zero;

  let mut k = ri;

  while (!k <^ re)
    invariant
      live dp **
      live k **
      pure (
        ri <= !k /\ !k <= re /\
        !dp %~ _sparse_matmul_single relems (cast_pos v_ind) rB ri re j !k
      )

    decreases (re - !k)
  {
    let x = Pulse.Lib.Array.(gA.elems.(!k));
    let c = Pulse.Lib.Array.(gA.col_ind.(!k));

    let y = tensor_read gB ((c <: szlt _), ((j <: szlt _), ()));

    with v_k. assert k |-> v_k;
    with v_dp. assert dp |-> v_dp;
    sparse_matmul_single_step relems (cast_pos v_ind) rB ri re j v_k;
    Kuiper.Approximates.Scalar.a_mul x y
      (relems @! v_k) (acc2 rB c j);
    Kuiper.Approximates.Scalar.a_add v_dp (x `mul` y)
      (_sparse_matmul_single relems (cast_pos v_ind) rB ri re j v_k)
      (mul (relems @! v_k) (acc2 rB c j));

    dp := !dp `add` (x `mul` y);

    k := !k +^ 1sz;
  };

  fold smatrix_model gA #fA v_elems v_ind v_row_off relems eA rA;
  fold smatrix_approx gA #fA eA rA;

  matmul_dotprod_lemma relems (cast_pos v_ind) (cast_pos v_row_off) rB i j;

  !dp;
}

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : nat)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  (gA : smatrix et rows shared)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA : chest2 et rows shared)
  (rA : chest2 real rows shared)
  (eB : chest2 et shared cols)
  (rB : chest2 real shared cols { eB %~ rB })
  (eC : chest2 et rows cols)
  (rC : chest2 real rows cols { eC %~ rC })
  (fA fB : perm)
  (gid : natlt (rows * cols))
  : slprop
  =
  smatrix_approx gA #(fA /. (rows * cols)) eA rA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  tensor_pts_to_cell gC (idx2 (gid / cols) (gid % cols))
    (acc2 eC (gid / cols) (gid % cols))

unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : nat)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  (gA : smatrix et rows shared)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA : chest2 et rows shared)
  (rA : chest2 real rows shared)
  (eB : chest2 et shared cols)
  (rB : chest2 real shared cols { eB %~ rB })
  (eC : chest2 et rows cols)
  (rC : chest2 real rows cols { eC %~ rC })
  (fA fB : perm)
  (gid : natlt (rows * cols))
  : slprop
  =
  smatrix_approx gA #(fA /. (rows * cols)) eA rA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  cell_approx gC (gid / cols) (gid % cols)
    (MS.gemm_single comb_r rA rB rC (gid / cols) (gid % cols))

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : SZ.t)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols { eB %~ rB })
  (#eC : chest2 et rows cols)
  (#rC : chest2 real rows cols { eC %~ rC })
  (#fA #fB : perm)
  (gid : szlt (rows * cols))
  ()
  norewrite
  preserves
    gpu
  requires
    kpre comb comb_r gA gB gC eA rA eB rB eC rC fA fB gid
  ensures
    kpost comb comb_r gA gB gC eA rA eB rB eC rC fA fB gid
{
  let trow : sz = gid /^ cols; assert (rewrites_to trow (gid /^ cols));
  let tcol : sz = gid %^ cols; assert (rewrites_to tcol (gid %^ cols));

  let s = matmul_dotprod gA gB #eA #rA #eB #rB trow tcol;
  let v0 = tensor_read_cell gC ((trow <: szlt _), ((tcol <: szlt _), ()));
  let v1 = comb v0 s;
  tensor_write_cell gC ((trow <: szlt _), ((tcol <: szlt _), ())) v1;

  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (nthr : szp { SZ.v nthr == rows * cols })
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (#fA : perm)
  (gB : array2 et lB)
  (#fB : perm)
  (gC : array2 et lC)
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols { eB %~ rB })
  (#eC : chest2 et rows cols)
  (#rC : chest2 real rows cols { eC %~ rC })
  ()
  norewrite
  requires
    smatrix_approx gA #fA eA rA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (rc : natlt nthr).
      kpre comb comb_r gA gB gC eA rA eB rB eC rC fA fB rc) **
    emp (* frame *)
{
  // Sharing the input matrices (splitting permissions)
  smatrix_share_approx_n gA (rows * cols);
  tensor_share_n gB (rows * cols);

  // Sharing the output matrix (splitting each cell)
  tensor_ilower2 gC;

  forevery_unfactor' (rows * cols) rows cols (fun r c ->
    tensor_pts_to_cell gC (idx2 (r) (c)) (acc2 eC r c));

  // Join resources into a single bigstar
  forevery_zip #(natlt (rows * cols))
    (fun _ -> gB |-> Frac (fB /. (rows * cols)) eB)
    (fun i -> tensor_pts_to_cell gC (idx2 ((i/cols <: natlt rows)) ((i%cols <: natlt cols))) (acc2 eC (i/cols) (i%cols)));
  forevery_zip #(natlt (rows * cols))
    (fun _ -> smatrix_approx gA #(fA /. (rows * cols)) eA rA)
    _;

  (* We're done actually, but the encoding will not match the lambdas. *)
  forevery_ext #(natlt (rows * cols))
    (fun i ->
      (smatrix_approx gA #(fA /. (rows * cols)) eA rA) **
      (gB |-> Frac (fB /. (rows * cols)) eB) **
      tensor_pts_to_cell gC (idx2 ((i/cols <: natlt rows)) ((i%cols <: natlt cols))) (acc2 eC (i / cols) (i % cols)))
    (fun i ->
      kpre comb comb_r gA gB gC eA rA eB rB eC rC fA fB i);
  forevery_rw_size (rows * cols) nthr;
}

ghost
fn teardown
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (nthr : szp { SZ.v nthr == rows * cols })
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (#fA : perm)
  (gB : array2 et lB)
  (#fB : perm)
  (gC : array2 et lC)
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols { eB %~ rB })
  (#eC : chest2 et rows cols)
  (#rC : chest2 real rows cols { eC %~ rC })
  ()
  norewrite
  requires
    (forall+ (rc : natlt nthr).
      kpost comb comb_r gA gB gC eA rA eB rB eC rC fA fB rc) **
    emp (* frame *)
  ensures
    smatrix_approx gA #fA eA rA **
    gB |-> Frac fB eB **
    matrix_approx gC (chest_comb comb_r rC (MS.matmul rA rB))
{
  forevery_rw_size nthr (rows * cols);

  forevery_unzip #(natlt (rows * cols)) _ _;
  forevery_unzip #(natlt (rows * cols)) _ _;

  smatrix_gather_approx_n gA (rows * cols) #fA #eA #rA;
  tensor_gather_n gB _;

  forevery_factor (rows * cols) rows cols _;

  forevery_map_2
    (fun (r:natlt rows) (c:natlt cols) ->
      cell_approx gC ((r * cols + c) / cols) ((r * cols + c) % cols)
        (MS.gemm_single comb_r rA rB rC ((r * cols + c) / cols) ((r * cols + c) % cols)))
    (fun (r:natlt rows) (c:natlt cols) ->
      cell_approx gC r c (acc2 (MS.mmcomb comb_r rC rA rB) r c))
    fn r c {
      Kuiper.Math.div_mod_of_mul_add cols r c;
      rewrite each ((r * cols + c) / cols) as r;
      rewrite each ((r * cols + c) % cols) as c;
    };

  let values = forevery_exists_2
    (fun r c (v : et) -> Cell gC (idx2 r c) |-> v **
      pure (v %~ acc2 (MS.mmcomb comb_r rC rA rB) r c));
  forevery_extract_pure_2
    (fun r c -> Cell gC (idx2 r c) |-> values r c **
      pure (values r c %~ acc2 (MS.mmcomb comb_r rC rA rB) r c))
    (fun r c -> values r c %~ acc2 (MS.mmcomb comb_r rC rA rB) r c)
    fn r c {};
  forevery_map_2
    (fun r c -> Cell gC (idx2 r c) |-> values r c **
      pure (values r c %~ acc2 (MS.mmcomb comb_r rC rA rB) r c))
    (fun r c -> Cell gC (idx2 r c) |-> acc2 (mk2 values) r c)
    fn r c {};
  tensor_iraise2 gC #_ #(mk2 values);
  lemma_approximates_intro (mk2 values) (MS.mmcomb comb_r rC rA rB);
  fold matrix_approx gC (MS.mmcomb comb_r rC rA rB);
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared) { is_global_smatrix gA })
  (#fA : perm)
  (gB : array2 et lB { is_global gB })
  (#fB : perm)
  (gC : array2 et lC { is_global gC })
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols { eB %~ rB })
  (#eC : chest2 et rows cols)
  (#rC : chest2 real rows cols { eC %~ rC })
  (#_ : squash (rows * cols <= max_blocks * max_threads))
  : kernel_desc_n
    (smatrix_approx gA #fA eA rA ** gB |-> Frac fB eB ** gC |-> eC)
    (
      smatrix_approx gA #fA eA rA ** gB |-> Frac fB eB **
      matrix_approx gC (MS.mmcomb comb_r rC rA rB)
    )
= [@@inline_let] let nthr : (x : szp { SZ.v x == rows * cols }) = rows *^ cols in {
  nthr = nthr;

  frame = emp;

  kpre  = kpre  comb comb_r gA gB gC eA rA eB rB eC rC fA fB;
  kpost = kpost comb comb_r gA gB gC eA rA eB rB eC rC fA fB;

  f = kf comb comb_r gA gB gC #eA #rA #eB #rB #eC #rC #fA #fB;

  setup    = setup    comb comb_r nthr gA gB gC;
  teardown = teardown comb comb_r nthr gA gB gC;

  kpre_sendable = solve;
  kpost_sendable = solve;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared) { is_global_smatrix gA })
  (#fA : perm)
  (gB : array2 et lB { is_global gB })
  (#fB : perm)
  (gC : array2 et lC { is_global gC })
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols { eB %~ rB })
  (#eC : chest2 et rows cols)
  (#rC : chest2 real rows cols { eC %~ rC })
  norewrite
  preserves
    cpu **
    on gpu_loc (smatrix_approx gA #fA eA rA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (rows * cols <= max_blocks * max_threads) ** (* size_req *)
    on gpu_loc (gC |-> eC)
  ensures on gpu_loc (matrix_approx gC (MS.mmcomb comb_r rC rA rB))
{
  launch_sync (kdesc comb comb_r gA gB gC #eA #rA #eB #rB #eC #rC);
}

let _mm_u32_rr (rows shared cols : szp { SZ.fits (rows * cols) /\ SZ.fits (shared * cols) }) =
  mmcomb_gpu #u32 #_ #_ (fun _ x -> x) (fun _ x -> x)
  #rows #shared #cols
  #(l2_row_major _ _) #(l2_row_major _ _)
