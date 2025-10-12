module Kuiper.Sparse.GEMM

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module MU = Kuiper.Poly.GEMM.Util
open Kuiper.Sparse
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs

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
  //ensures pure (res == MS.matmul_single eA eB i j)
{
  unfold smatrix_pts_to gA #fA eA;
  with v_row_off.
    assert gpu_pts_to_array gA.row_off #fA v_row_off;
  with v_ind.
    assert gpu_pts_to_array gA.col_ind # fA v_ind;
  
  let ri = gpu_array_read gA.row_off i;
  let re = gpu_array_read gA.row_off (i +^ 1sz);

  let row_cols = hide (slice_row (cast_pos v_row_off) (cast_pos v_ind) i);

  let mut dp : et = zero;

  let mut k = ri;
  
  while ((!k <^ re))
    invariant
      exists* v_k.
        k |-> v_k **
        live dp **
        pure (
          ri <= v_k /\
          (v_k < re ==> SZ.v (v_ind @! v_k) == row_cols @! (v_k - ri)) 
        )
      
  {
    let x = gpu_array_read gA.elems !k;
    let c = gpu_array_read gA.col_ind !k;

    let y = M.gpu_matrix_read gB c j;

    dp := !dp `add` (x `mul` y);

    k := !k +^ 1sz;
  };

  fold smatrix_pts_to gA;

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
  (exists* v.
    M.gpu_matrix_pts_to_cell gC #1.0R (bid / cols) (bid % cols)
    //(MS.gemm_single comb eA eB eC (bid / cols) (bid % cols))
    v)

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
    kpre comb gA gB gC eA eB eC fA fB bid **
    block_id (rows *^ cols) bid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC fA fB bid **
    block_id (rows *^ cols) bid
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
  forevery_fromstar #(natlt2 rows cols) _;
  M.gpu_matrix_share_n gB (rows *^ cols);
  forevery_fromstar #(natlt2 rows cols) _;

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
    //gC |-> matrix_comb comb eC (MS.matmul eA eB)
    live gC
{
  forevery_unzip #(natlt2 rows cols) _ _;
  forevery_unzip #(natlt2 rows cols) _ _;

  forevery_tostar #(natlt2 rows cols)
    (fun i -> gA |-> Frac (fA /. (rows * cols)) eA);
  smatrix_gather_n gA _;
  forevery_tostar #(natlt2 rows cols)
    (fun i -> gB |-> Frac (fB /. (rows * cols)) eB);
  M.gpu_matrix_gather_n gB _;

  forevery_factor (rows *^ cols) rows cols _;

  (* we get things back with some arithmetic in it *)
  assert (forall+ (r:natlt rows) (c:natlt cols).
    exists* v.
      M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols)
      //(MS.gemm_single comb eA eB eC ((r * cols + c) / cols) ((r * cols + c) % cols)
      v
  );

  (* need to use ext to get rid of it-- automatically applying ext would be really useful. *)
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));

  ghost
  fn aux (r : natlt rows) (c : natlt cols)
    norewrite
    requires (
      exists* v. M.gpu_matrix_pts_to_cell gC
        ((r * cols + c) / cols) ((r * cols + c) % cols) v
    )
    ensures (exists* v. M.gpu_matrix_pts_to_cell gC r c v)
  {
    with v. assert M.gpu_matrix_pts_to_cell gC
      ((r * cols + c) / cols) ((r * cols + c) % cols) v;
    
    assert pure ((r * cols + c) / cols == r); 
    assert pure ((r * cols + c) % cols == c); 
    
    admit();
    assert M.gpu_matrix_pts_to_cell gC r c v;
  };
  
  forevery_map_2
    #(natlt (SZ.v rows)) #_
    #(natlt (SZ.v cols)) #_ 
    (fun (r:natlt rows) (c:natlt cols) ->
      exists* v.
        M.gpu_matrix_pts_to_cell gC ((r * cols + c) / cols) ((r * cols + c) % cols) v)
    (fun (r:natlt rows) (c:natlt cols) ->
      exists* v.
        M.gpu_matrix_pts_to_cell gC r c v)
    aux;
  assert (forall+ r c.
      exists* v.
      M.gpu_matrix_pts_to_cell gC r c v);

  admit();

  M.gpu_matrix_implode gC;
  ()
}

inline_for_extraction noextract
let kdesc
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
  (#_ : squash (rows * cols <= max_blocks))
  : kernel_desc_m_1
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (
      gA |-> Frac fA eA ** gB |-> Frac fB eB **
      live gC//gC |-> MS.mmcomb comb eC eA eB
    )
= {
  nblk = rows *^ cols;

  frame = emp;

  setup    = setup    comb gA gB gC;
  teardown = teardown comb gA gB gC;

  kpre  = kpre  comb gA gB gC eA eB eC fA fB;
  kpost = kpost comb gA gB gC eA eB eC fA fB;

  f = kf comb gA gB gC;
}

inline_for_extraction noextract
fn mmcomb_gpu
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
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (rows * cols <= max_blocks) ** (* size_req *)
    gC |-> eC
  //ensures gC |-> MS.mmcomb comb eC eA eB
  ensures live gC
{
  launch_sync (kdesc comb gA gB gC);
}

let _gemm_u32_rr (rows shared cols : szp { SZ.fits (rows * cols) /\ SZ.fits (shared * cols) }) =
  mmcomb_gpu #u32 #_ (fun x _ -> x)
  #rows #shared #cols
  #(row_major _ _) #(row_major _ _)