module Kuiper.Poly.GEMM.BlockTiling1D

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.EMatrix4
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

module M = Kuiper.Matrix
open Kuiper.Matrix {
  gpu_matrix,
  gpu_matrix_pts_to,
  gpu_matrix_pts_to_cell,
}

open Kuiper.Matrix4 {
  gpu_matrix as gpu_matrix4,
  gpu_matrix_pts_to as m4_pts_to,
  gpu_matrix_pts_to_cell as m4_pts_to_cell,
  mlayout4,
  clayout4
}

module M4 = Kuiper.Matrix4
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs

open Kuiper.EMatrix { ematrix }

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |} (tile:valid_tile) : list shmem_desc = [
  SHArray et (tile *^ tile);
  SHArray et (tile *^ tile);
]

(* The barrier flip-flops between an initial state
where every threads shares all of the array, and
a second state where each thread owns two cells
of the array, related to their tid.

So in even steps, they give their shared ownership,
and receive their cells. (p)
*)

(* To verify functional correctness: the existentials here should be made
precise, and parametrize this over the starting input matrices. *)
let own_1_col
  (#et : Type0)
  (#tile : valid_tile)
  (#l : mlayout tile tile) (m : gpu_matrix et l)
  (tid : natlt tile)
  : slprop =
  forall+ (ii : natlt tile).
    (exists* x. gpu_matrix_pts_to_cell m ii tid x)

let barrier_p
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : mlayout tile tile) (m1 : gpu_matrix et l1)
  (#l2 : mlayout tile tile) (m2 : gpu_matrix et l2)
  : B.barrier_side tile =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. tile) x)
    else
      own_1_col m1 tid ** own_1_col m2 tid

let barrier_q
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : mlayout tile tile) (m1 : gpu_matrix et l1)
  (#l2 : mlayout tile tile) (m2 : gpu_matrix et l2)
  : B.barrier_side tile =
  fun it tid -> barrier_p m1 m2 (it+1) tid (* flip flop *)

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  (* mlayout_size lC: wrong, should be (mrows*mcols)*tile *)
  gA |-> Frac (fA /. mlayout_size lC) eA **
  gB |-> Frac (fB /. mlayout_size lC) eB **
  forall+ (ii : natlt tile).
   (exists* v.
     m4_pts_to_cell gC #1.0R
       (bid / mcols) (bid % mcols)
       ii tid v)

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  gA |-> Frac (fA /. mlayout_size lC) eA **
  gB |-> Frac (fB /. mlayout_size lC) eB **
  forall+ (ii : natlt tile).
   (exists* v.
     m4_pts_to_cell gC #1.0R
       (bid / mcols) (bid % mcols)
       ii tid v)

let barrier_tok
  (#et : Type0)
  (tile : valid_tile)
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 l2 : full_mlayout tile tile)
  (ar1 ar2 : gpu_array et (tile * tile))
  (it : nat)
  (tid : natlt tile)
  : slprop
  =
  B.barrier_tok
    (barrier_p (M.from_array l1 ar1) (M.from_array l2 ar2))
    (barrier_q (M.from_array l1 ar1) (M.from_array l2 ar2))
    it tid

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  kpre1 comb tile gA gB gC eA eB fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. tile) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. tile) x) **
  barrier_tok tile slA slB (fst sh) (fst (snd sh)) 0 tid

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (gC : gpu_matrix4 et lC)
  (eA : ematrix4 et mrows mshared tile tile)
  (eB : ematrix4 et mshared mcols tile tile)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  kpost1 comb tile gA gB gC eA eB fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. tile) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. tile) x) **
  barrier_tok tile slA slB (fst sh) (fst (snd sh)) (2 * mshared) tid

inline_for_extraction noextract
fn bring_2cols
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : erased nat)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  {| clayout4 lA, clayout4 lB |}
  (gA : gpu_matrix4 et lA)
  (gB : gpu_matrix4 et lB)
  (#l1 : mlayout tile tile) {| clayout l1 |} (sa1 : gpu_matrix et l1)
  (#l2 : mlayout tile tile) {| clayout l2 |} (sa2 : gpu_matrix et l2)
  (mrow : szlt mrows)
  (mk : szlt mshared)
  (mcol : szlt mcols)
  (tid : szlt tile)
  (#fA #fB : perm)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  preserves
    gpu
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    own_1_col sa1 tid **
    own_1_col sa2 tid
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    own_1_col sa1 tid **
    own_1_col sa2 tid
{
  let mut i = 0sz;
  while (SZ.(!i <^ tile))
    invariant live i
  {
    let vi = !i;

    unfold own_1_col sa1 tid;
    forevery_extract #(natlt tile) vi _;
    let v1 = M4.gpu_matrix_read gA mrow mk vi tid;
    M.gpu_matrix_write_cell sa1 vi tid v1;
    Pulse.Lib.Trade.elim_trade _ _;
    fold own_1_col sa1 tid;

    unfold own_1_col sa2 tid;
    forevery_extract #(natlt tile) vi _;
    let v2 = M4.gpu_matrix_read gB mk mcol vi tid;
    M.gpu_matrix_write_cell sa2 vi tid v2;
    Pulse.Lib.Trade.elim_trade _ _;
    fold own_1_col sa2 tid;

    i := !i +^ 1sz;
  }
}

inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (sh : c_shmems (shmems_desc et tile))
  (bid : szlt (mrows * mcols))
  (tid : szlt tile)
  ()
  norewrite
  requires
    gpu **
    kpre comb tile slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id tile tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tile slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id tile tid **
    block_id (mrows * mcols) bid
{
  let ar1 : gpu_array et (tile * tile) = fst sh;
  let ar2 : gpu_array et (tile * tile) = fst (snd sh);
  rewrite each fst sh as ar1;
  rewrite each fst (snd sh) as ar2;

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  unfold barrier_tok tile slA slB ar1 ar2 0 tid;

  M.gpu_matrix_abs' slA ar1;
  let sa1 = M.from_array slA ar1;
  rewrite each M.from_array slA ar1 as sa1;

  M.gpu_matrix_abs' slB ar2;
  let sa2 = M.from_array slB ar2;
  rewrite each M.from_array slB ar2 as sa2;


  let mrow, mcol = s_divmod mcols bid;
  let bcol = tid;
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (bcol < tile));


  (* thread-local result cache *)
  let mut sums : Pulse.Lib.Array.array et = [| zero ; tile |];
  let mut bk  : sz = 0sz;

  while (let vbk = !bk; SZ.(vbk <^ mshared))
    invariant live sums
    invariant
      exists* (vbk : SZ.t).
        bk |-> vbk **
        B.barrier_tok (barrier_p sa1 sa2) (barrier_q sa1 sa2) (2 * vbk) tid **
        pure (vbk <= mshared)
    invariant
        (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
        (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. tile) x)
  {
    pts_to_len sums;
    let vbk = !bk;

    rewrite
        (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
        (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. tile) x)
      as barrier_p sa1 sa2 (2 * vbk) tid;

    B.barrier_wait ();
    rewrite (barrier_q sa1 sa2 (2 * vbk) tid)
         as own_1_col sa1 tid ** own_1_col sa2 tid;

    (* At this point we exclusively own a full column of the SHMEM
       cache. Populate it. *)
    let vbk = !bk;
    bring_2cols tile gA gB sa1 sa2 mrow vbk mcol tid;

    rewrite own_1_col sa1 tid ** own_1_col sa2 tid
         as (barrier_p sa1 sa2 (2 * vbk + 1) tid);
    B.barrier_wait ();
    rewrite (barrier_q sa1 sa2 (2 * vbk + 1) tid)
    as
      (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. tile) x);

    (* At this point the SHMem cache is filled with the submatrices
       and we have RO permission to it. Compute product for our cell in
       the tile and add to sum. *)
    Kuiper.Poly.GEMM.Util.subproduct_cols tile sums sa1 sa2 bcol;

    // What the hell.
    assert (pure (2 * (vbk + 1) == 2 * vbk + 1 + 1));

    (* Move to next tile *)
    bk := !bk +^ 1sz;
  };

  (* Write all the accumulated sums. *)

  let mut row : sz = 0sz;
  Pulse.Lib.Array.pts_to_len sums;
  while (SZ.(!row <^ tile))
    invariant live row ** live sums
  {
    Pulse.Lib.Array.pts_to_len sums;
    let vrow = !row;
    forevery_extract #(natlt tile) (SZ.v vrow) _;

    (* tedious *)
    with bi0 bj0 i0 j0 v0.
      rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
          as m4_pts_to_cell gC mrow mcol vrow bcol v0;

    let v0 = M4.gpu_matrix_read_cell gC mrow mcol !row bcol;
    open Pulse.Lib.Array;
    let v1 = sums.(!row);
    let v' = comb v0 v1;
    M4.gpu_matrix_write_cell gC mrow mcol !row bcol v';

    with bi0 bj0 i0 j0 v0.
      rewrite m4_pts_to_cell gC bi0  bj0  i0   j0   v0
          as m4_pts_to_cell gC (bid / mcols) (bid % mcols) vrow tid v0;

    row := !row +^ 1sz;
    Pulse.Lib.Trade.elim_trade _
      (forall+ (ii : natlt tile).
        (exists* v.
          m4_pts_to_cell gC #1.0R
            (bid / mcols) (bid % mcols)
            ii tid v));
  };

  M.gpu_matrix_concr sa1; rewrite each M.core sa1 as ar1;
  M.gpu_matrix_concr sa2; rewrite each M.core sa2 as ar2;

  fold barrier_tok tile slA slB ar1 ar2 (2 * mshared) tid;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  ()
}


ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt tile).
      kpre1 comb tile gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    block_setup_tok tile **
    live_c_shmems sh **
    (forall+ (tid : natlt tile).
      kpre1 comb tile gA gB gC eA eB fA fB bid tid)
  ensures
    block_setup_tok tile **
    (forall+ (tid : natlt tile).
      kpre comb tile slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    (forall+ (tid : natlt tile).
      kpost comb tile slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt tile).
      kpost1 comb tile gA gB gC eA eB fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt tile).
      kpost1 comb tile gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  // forevery_flatten #(natlt2 mrows mcols) #_ #(natlt tile)
  //   (fun bid tid -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_tostar #(natlt2 mrows mcols & natlt tile) (fun _tid -> m4_pts_to gA #(1.0R /. mlayout_size lC) eA);

    // (fun (bid, tid) -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout4 mrows   mshared tile tile)
  (#lB : mlayout4 mshared mcols   tile tile)
  (#lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile <= max_threads))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
= {
  nblk = mrows *^ mcols;
  nthr = tile;

  shmems_desc = shmems_desc et tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt tile). kpre1  comb tile gA gB gC eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt tile). kpost1 comb tile gA gB gC eA eB fA fB bid tid);
  setup      = setup    tile comb gA gB gC;
  teardown   = teardown tile comb gA gB gC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile slA slB comb gA gB gC #_ #_ #eC;
  block_teardown = block_teardown tile slA slB comb gA gB gC #_ #_ #eC;

  kpre      = kpre  comb tile slA slB gA gB gC eA eB fA fB;
  kpost     = kpost comb tile slA slB gA gB gC eA eB fA fB;

  f = kf tile slA slB comb gA gB gC;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (lA : mlayout4 mrows   mshared tile tile)
  (lB : mlayout4 mshared mcols   tile tile)
  (lC : mlayout4 mrows   mcols   tile tile)
  {| clayout4 lA, clayout4 lB, clayout4 lC |}
  (gA : gpu_matrix4 et lA)
  (#fA : perm)
  (gB : gpu_matrix4 et lB)
  (#fB : perm)
  (gC : gpu_matrix4 et lC)
  (#eA : ematrix4 et mrows   mshared tile tile)
  (#eB : ematrix4 et mshared mcols   tile tile)
  (#eC : ematrix4 et mrows   mcols   tile tile)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (tile * tile <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  dassert (tile `SZ.gt` 0sz);
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel tile (R.row_major _ _) (R.row_major _ _) comb gA gB gC ());
}
