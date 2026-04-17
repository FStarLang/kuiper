module Kuiper.Sparse.GEMM

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.Sparse
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs

noextract
let rec __dprod
  (#et : Type0) {| scalar et |}
  (#nnz #shared #cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (eB : ematrix et shared cols)
  (ri re : nat{ri <= re /\ re <= nnz /\ sorted_slice col_ind ri re})
  (j : natlt cols)
  (to : nat{ri <= to /\ to <= re})
  : GTot et
=
  if to = ri
    then zero
    else (
      add
        (__dprod elems col_ind eB ri re j (to - 1))
        (mul
          (elems @! (to - 1))
          (macc eB (col_ind @! (to - 1)) j))
    )

noextract
let dprod
  (#et : Type0) {| scalar et |}
  (#nnz #shared #cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (eB : ematrix et shared cols)
  (ri re : nat{ri <= re /\ re <= nnz /\ sorted_slice col_ind ri re})
  (j : natlt cols)
  : GTot et
=
  __dprod elems col_ind eB ri re j re




let rec matmul_all_zeros_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (from to : nat{from <= to /\ to <= shared})
  : Lemma
    (requires forall i. from <= i /\ i < to ==> macc m1 row i == zero)
    (ensures MS.__matmul_single m1 m2 row col from == MS.__matmul_single m1 m2 row col to)
=
  if from = to
    then ()
    else (
      MS.matmul_single_lemma m1 m2 row col to;
      matmul_all_zeros_lemma m1 m2 row col from (to - 1)
    )

#push-options "--z3rlimit 20"
let rec __matmul_dotprod_lemma
  (#et : Type0) {| scalar et |}
  (#nnz #rows #shared #cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (row_off : lseq nat (rows + 1))
  (eB : ematrix et shared cols)
  (i : natlt rows)
  (j : natlt cols)
  (to : nat{(row_off @! i) <= to /\ to < (row_off @! (i + 1))})
  : Lemma
    (requires valid_smatrix rows shared col_ind row_off)
    (ensures
      __dprod elems col_ind eB (row_off @! i) (row_off @! (i + 1)) j (to + 1) ==
      MS.__matmul_single (smatrix_unsparse _ _ elems col_ind row_off) eB i j ((col_ind @! to) + 1)
    )
=
  let eA = smatrix_unsparse rows shared elems col_ind row_off in

  let ri = row_off @! i in
  let re = row_off @! (i + 1) in

  if to = ri then (
    MS.matmul_single_lemma eA eB i j ((col_ind @! to) + 1);
    matmul_all_zeros_lemma eA eB i j 0 (col_ind @! to)
  ) else (
    MS.matmul_single_lemma eA eB i j ((col_ind @! to) + 1);
    smatrix_all_zeros rows shared elems col_ind row_off i to;
    matmul_all_zeros_lemma eA eB i j ((col_ind @! to - 1) + 1) (col_ind @! to);
    __matmul_dotprod_lemma elems col_ind row_off eB i j (to - 1);
    assert macc eA i (col_ind @! to) == elems @! to;
    ()
  )
#pop-options

let matmul_dotprod_lemma
  (#et : Type0) {| scalar et |}
  (#nnz #rows #shared #cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz{in_bounds 0 shared col_ind})
  (row_off : lseq nat (rows + 1))
  (eB : ematrix et shared cols)
  (i : natlt rows)
  (j : natlt cols)
  : Lemma
    (requires valid_smatrix rows shared col_ind row_off)
    (ensures
      dprod elems col_ind eB (row_off @! i) (row_off @! (i + 1)) j ==
      MS.matmul_single (smatrix_unsparse rows shared elems col_ind row_off) eB i j
    )
=
  let eA = smatrix_unsparse rows shared elems col_ind row_off in

  let ri = row_off @! i in
  let re = row_off @! (i + 1) in

  if ri = re
    then matmul_all_zeros_lemma eA eB i j 0 shared
    else (
      __matmul_dotprod_lemma elems col_ind row_off eB i j (re - 1);
      matmul_all_zeros_lemma eA eB i j ((col_ind @! (re - 1)) + 1) shared
    )

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lB : mlayout shared cols)
  {| clayout lB |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures pure (res == MS.matmul_single eA eB i j)
{
  unfold smatrix_pts_to gA #fA eA;
  with v_elems.
    assert gpu_pts_to_array gA.elems #fA v_elems;
  with v_row_off.
    assert gpu_pts_to_array gA.row_off #fA v_row_off;
  with v_ind.
    assert gpu_pts_to_array gA.col_ind # fA v_ind;

  assert pure (forall k. v_ind @! k < shared);

  let ri = gpu_array_read gA.row_off i;
  let re = gpu_array_read gA.row_off (i +^ 1sz);

  let mut dp : et = zero;

  let mut k = ri;

  while (!k <^ re)
    invariant
      live dp **
      live k **
      pure (
        ri <= !k /\ !k <= re /\
        !dp == __dprod v_elems (cast_pos v_ind) eB ri re j !k
      )

  {
    let x = gpu_array_read gA.elems !k;
    let c = gpu_array_read gA.col_ind !k;

    let y = M.gpu_matrix_read gB c j;

    dp := !dp `add` (x `mul` y);

    k := !k +^ 1sz;
  };

  fold smatrix_pts_to gA #fA eA;

  matmul_dotprod_lemma v_elems (cast_pos v_ind) (cast_pos v_row_off) eB i j;

  !dp;
}

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (bid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC (bid / cols) (bid % cols)
    (macc eC (bid / cols) (bid % cols))

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (bid : natlt (rows * cols))
  : slprop
  =
  gA |-> Frac (fA /. (rows * cols)) eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC (bid / cols) (bid % cols)
    (MS.gemm_single comb eA eB eC (bid / cols) (bid % cols))

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : SZ.t)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (#fA #fB : perm)
  (bid : szlt (rows * cols))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA gB gC eA eB eC fA fB bid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC fA fB bid
{
  let trow = bid /^ cols; assert (rewrites_to trow (bid /^ cols));
  let tcol = bid %^ cols; assert (rewrites_to tcol (bid %^ cols));

  let s = matmul_dotprod gA gB trow tcol;
  let v0 = M.gpu_matrix_read_cell gC trow tcol;
  let v1 = comb v0 s;
  M.gpu_matrix_write_cell gC trow tcol v1;

  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
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
  smatrix_share_n gA (rows *^ cols);
  M.gpu_matrix_share_n gB (rows *^ cols);

  // Sharing the output matrix (splitting each cell)
  M.gpu_matrix_explode gC;

  forevery_unfactor' (rows *^ cols) rows cols (fun r c ->
    M.gpu_matrix_pts_to_cell gC r c (macc eC r c));

  // Join resources into a single bigstar
  forevery_zip #(natlt2 rows cols)
    (fun _ -> gB |-> Frac (fB /. (rows *^ cols)) eB)
    (fun i -> M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i/cols) (i%cols)));
  forevery_zip #(natlt2 rows cols)
    (fun _ -> gA |-> Frac (fA /. (rows *^ cols)) eA)
    _;

  (* We're done actually, but the encoding will not match the lambdas. *)
  forevery_ext #(natlt2 rows cols)
    (fun i ->
      (gA |-> Frac (fA /. (rows *^ cols)) eA) **
      (gB |-> Frac (fB /. (rows *^ cols)) eB) **
      M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (macc eC (i / cols) (i % cols)))
    (fun i ->
      kpre comb gA gB gC eA eB eC fA fB i);
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
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
    gC |-> matrix_comb comb eC (MS.matmul eA eB)
{
  forevery_rw_size (rows *^ cols) (rows * cols);

  forevery_unzip #(natlt (rows * cols)) _ _;
  forevery_unzip #(natlt (rows * cols)) _ _;

  smatrix_gather_n gA (rows * cols) #fA #eA;
  M.gpu_matrix_gather_n gB _;

  forevery_factor (rows * cols) rows cols _;

  (* we get things back with some arithmetic in it *)
  assert forall+ (r:natlt rows) (c:natlt cols).
      M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
        (MS.gemm_single comb eA eB eC ((r * cols + c) / cols) ((r * cols + c) % cols)
  );

  (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));
  forevery_ext_2
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
         (MS.gemm_single comb eA eB eC ((r * cols + c) / cols) ((r * cols + c) % cols)))
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gC r c (MS.gemm_single comb eA eB eC r c));

  ghost
  fn aux (r:natlt rows) (c:natlt cols)
    requires
      M.gpu_matrix_pts_to_cell gC r c (MS.gemm_single comb eA eB eC r c)
    ensures
      M.gpu_matrix_pts_to_cell gC r c (macc (matrix_comb comb eC (MS.matmul eA eB)) r c)
  {
    ()
  };
  forevery_map_2
    (fun r c -> M.gpu_matrix_pts_to_cell gC r c (MS.gemm_single comb eA eB eC r c))
    _
    aux;

  M.gpu_matrix_implode gC;
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared) { is_global_smatrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (#_ : squash (rows * cols <= max_blocks * max_threads))
  : kernel_desc_n
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (
      gA |-> Frac fA eA ** gB |-> Frac fB eB **
      gC |-> MS.mmcomb comb eC eA eB
    )
= {
  nthr = rows *^ cols;

  frame = emp;

  setup    = setup    comb gA gB gC;
  teardown = teardown comb gA gB gC;

  kpre  = kpre  comb gA gB gC eA eB eC fA fB;
  kpost = kpost comb gA gB gC eA eB eC fA fB;

  f = kf comb gA gB gC;

  kpre_sendable = solve;
  kpost_sendable = solve;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared) { is_global_smatrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et lB { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et lC { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (rows * cols <= max_blocks * max_threads) ** (* size_req *)
    on gpu_loc (gC |-> eC)
  ensures on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  launch_sync (kdesc comb gA gB gC);
}

let _gemm_u32_rr (rows shared cols : szp { SZ.fits (rows * cols) /\ SZ.fits (shared * cols) }) =
  mmcomb_gpu #u32 #_ (fun _ x -> x)
  #rows #shared #cols
  #(row_major _ _) #(row_major _ _)
