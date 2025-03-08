module Kuiper.MatMul

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : pos)
  (gA : M.gpu_matrix et rows shared)
  (gB : M.gpu_matrix et shared cols)
  (gC : M.gpu_matrix et rows cols)
  (eA : M.ematrix et rows shared)
  (eB : M.ematrix et shared cols)
  (f : perm)
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  (exists* v.
    M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols) v)

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : pos)
  (gA : M.gpu_matrix et rows shared)
  (gB : M.gpu_matrix et shared cols)
  (gC : M.gpu_matrix et rows cols)
  (eA : M.ematrix et rows shared)
  (eB : M.ematrix et shared cols)
  (f : perm)
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  M.gpu_matrix_pts_to gA #(f /. (rows * cols)) eA **
  M.gpu_matrix_pts_to gB #(f /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols)
    (MS.matmul_single eA eB (tid / cols) (tid % cols) shared)

// unfold
inline_for_extraction
type kernel_ty (et : Type0) {| scalar et |} =
  (#rows : erased szp) ->
  (#shared : szp) ->
  (#cols : szp{reveal rows * cols < pow2 64}) ->
  (gA : M.gpu_matrix et (reveal rows) shared) ->
  (gB : M.gpu_matrix et shared cols) ->
  (gC : M.gpu_matrix et (reveal rows) cols) ->
  (#eA : M.ematrix et (reveal rows) shared) ->
  (#eB : M.ematrix et shared cols) ->
  (#f : perm) ->
  (etid : tid_t { gdim_x etid == SZ.v rows * cols /\ bdim_x etid == 1 }) ->
  stt unit
  (requires
    gpu **
    thread_id etid **
    kpre gA gB gC eA eB f (thread_index etid))
  (ensures fun _ ->
    gpu **
    thread_id etid **
    kpost gA gB gC eA eB f (thread_index etid))

inline_for_extraction noextract
fn kernel
  (#et : Type0) {| scalar et |}
  (#rows : erased szp) (#shared : szp) (#cols : szp{reveal rows * cols < pow2 64})
  (gA : M.gpu_matrix et (reveal rows) shared)
  (gB : M.gpu_matrix et shared cols)
  (gC : M.gpu_matrix et (reveal rows) cols)
  (#eA : M.ematrix et (reveal rows) shared)
  (#eB : M.ematrix et shared cols)
  (#f : perm)
  (etid : tid_t { gdim_x etid == SZ.v rows * cols /\ bdim_x etid == 1 })
  requires gpu
    ** thread_id etid
    ** kpre gA gB gC eA eB f (thread_index etid)
  ensures  gpu
    ** thread_id etid
    ** kpost gA gB gC eA eB f (thread_index etid)
{
  let tid = block_idx_x ();
  rewrite each thread_index etid as tid;

  let trow = SZ.div tid cols;
  let tcol = SZ.rem tid cols;
  with v0.
    rewrite
      M.gpu_matrix_pts_to_cell gC #1.0R (tid / cols) (tid % cols) v0
    as
      M.gpu_matrix_pts_to_cell gC #1.0R trow tcol v0;

  assert (pure (trow < (reveal rows)));
  assert (pure (tcol < cols));

  let mut i : sz = 0sz;
  let mut sum : et = zero #et #_;

  while (let vi = !i; SZ.(vi <^ shared))
    invariant b.
      exists* (vi : SZ.t{ vi <= shared}).
        pure (0 <= shared /\ b == (SZ.v vi < shared) /\ vi <= shared /\ vi >= 0) **
        pts_to i vi **
        pts_to #_ #et sum (MS.matmul_single eA eB trow tcol vi) **
        M.gpu_matrix_pts_to gA #(f /. (reveal rows * cols)) eA **
        M.gpu_matrix_pts_to gB #(f /. (reveal rows * cols)) eB **
        gpu
  {
    let vi = !i;
    let s = !sum;
    let v1 = M.gpu_matrix_read gA trow vi;
    let v2 = M.gpu_matrix_read gB vi tcol;

    sum := s `add` mul v1 v2;
    i := SZ.add vi 1sz;

    (**)MS.matmul_single_lemma eA eB trow tcol (vi + 1);
    ();
  };

  let s = !sum;
  M.gpu_matrix_write_cell gC trow tcol s; // r[tid] = s

  (* ugh *)
  rewrite
    M.gpu_matrix_pts_to_cell gC trow tcol
      (MS.matmul_single eA eB trow tcol shared)
  as
    M.gpu_matrix_pts_to_cell gC (thread_index etid / cols) (thread_index etid % cols)
      (MS.matmul_single eA eB (thread_index etid / cols) (thread_index etid % cols) shared);

  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : pos)
  (gA : M.gpu_matrix et rows shared)
  (gB : M.gpu_matrix et shared cols)
  (gC : M.gpu_matrix et rows cols)
  (#eA : M.ematrix et rows shared)
  (#eB : M.ematrix et shared cols)
  (#eC : M.ematrix et rows cols)
  requires
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> eC)
  ensures
    forevery (natlt (rows * cols)) (kpre gA gB gC eA eB 1.0R)
{
  // Sharing the input matrices (splitting permissions)
  M.gpu_matrix_share_n #_ #0 gA (rows * cols);
  M.gpu_matrix_share_n #_ #1 gB (rows * cols);

  // Sharing the output matrix (splitting each cell)
  M.gpu_matrix_explode #_ #2 gC;

  // Join resources into a single bigstar
  bigstar_zip #0 #1 #3 0 (rows * cols) _ _;
  bigstar_zip #3 #2 #4 0 (rows * cols) _ _;

  // Rewrite inside the bigstar
  ghost
  fn aux1 (i:nat{0 <= i /\ i < rows * cols})
    requires
      M.gpu_matrix_pts_to gA #(1.0R /. (rows * cols)) eA **
      M.gpu_matrix_pts_to gB #(1.0R /. (rows * cols)) eB **
      M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (M.macc eC (i/cols) (i%cols))
    ensures
      kpre gA gB gC eA eB 1.0R (Enumerable.of_nat #(natlt (rows * cols)) i)
  {
    ()
  };
  bigstar_map #_ #_ #0 #(rows * cols) aux1;

  forevery_fromstar #(natlt (rows * cols)) (kpre gA gB gC eA eB 1.0R);
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : pos)
  (gA : M.gpu_matrix et rows shared)
  (gB : M.gpu_matrix et shared cols)
  (gC : M.gpu_matrix et rows cols)
  (#eA : M.ematrix et rows shared)
  (#eB : M.ematrix et shared cols)
  requires
    forevery (natlt (rows * cols))
      (kpost gA gB gC eA eB 1.0R)
  ensures
    (gA |-> eA) **
    (gB |-> eB) **
    (gC |-> MS.matmul eA eB)
{
  forevery_tostar #(natlt (rows * cols)) (kpost gA gB gC eA eB 1.0R);
  (* FIXME: the #_ is important. *)
  rewrite each Enumerable.cardinal (natlt (op_Multiply rows cols)) #_ as (op_Multiply rows cols);

  bigstar_unzip 0 (rows * cols) _ _;
  bigstar_unzip 0 (rows * cols) _ _;

  M.gpu_matrix_gather_n gA _;
  M.gpu_matrix_gather_n gB _;

  Classical.forall_intro_2 (MS.lemma_matmul_index eA eB);

  ghost
  fn aux (i:nat{0 <= i /\ i < rows * cols})
    requires
      M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (MS.matmul_single eA eB (i/cols) (i%cols) shared)
    ensures
      M.gpu_matrix_pts_to_cell gC (i/cols) (i%cols) (M.macc (MS.matmul eA eB) (i/cols) (i%cols))
  {
    ()
  };
  bigstar_map #0 #0 #0 #(rows * cols) aux;

  M.gpu_matrix_implode gC;
}

inline_for_extraction noextract
fn matmul_gpu
  (#et : Type0) {| scalar et |}
  (kk : kernel_ty et #_)
  (#rows #shared #cols : szp) (* concrete args *)
  (gA : M.gpu_matrix et rows shared)
  (gB : M.gpu_matrix et shared cols)
  (gC : M.gpu_matrix et rows cols)
  (#eA : M.ematrix et rows shared)
  (#eB : M.ematrix et shared cols)
  (#eC : M.ematrix et rows cols)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (rows * cols <= max_blocks) **
    (gC |-> eC)
  ensures
    gC |-> MS.matmul eA eB
{
  open FStar.SizeT;
  setup gA gB gC;

  (* FIXME: F* inference failure means we need to annotate pre/post (somewhat) *)
  (* We also need eta due to the extraction rules looking for it. *)
  launch_kernel_n
    (rows *^ cols)
    #(kpre  _ _ _ _ _ _)
    #(kpost _ _ _ _ _ _)
    (fun etid -> kk gA gB gC #eA #eB #1.0R etid);

  teardown gA gB gC;
}

inline_for_extraction noextract
fn matmul
  (#et : Type0) {| scalar et |}
  (kk : kernel_ty et #_)
  (#rows #shared #cols : szp) (* concrete args *)
  (a : vec et)
  (b : vec et)
  (#sa : erased (seq et){ len sa == rows * shared })
  (#sb : erased (seq et){ len sb == shared * cols })
  preserves
    cpu **
    (a |-> sa) **
    (b |-> sb)
  requires
    pure (SZ.fits (rows * shared) /\ SZ.fits (shared * cols) /\ SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks)
  returns
    c : vec et
  ensures
    (c |-> matrix_as_seq <| MS.matmul (seq_as_matrix rows shared sa) (seq_as_matrix shared cols sb))
{
  let gA = M.gpu_matrix_alloc #et rows shared;
  let gB = M.gpu_matrix_alloc #et shared cols;
  let gC = M.gpu_matrix_alloc #et rows cols;

  M.gpu_matrix_from_array a gA;
  M.gpu_matrix_from_array b gB;

  matmul_gpu kk gA gB gC;

  let c = Pulse.Lib.Vec.alloc #et zero (SZ.mul rows cols);
  M.gpu_matrix_to_array c gC;

  M.gpu_matrix_free gA;
  M.gpu_matrix_free gB;
  M.gpu_matrix_free gC;

  c
}
