module Kuiper.Poly.GEMM.SHMem

#lang-pulse

open Kuiper
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.View.TwoTiles { aview_2tile2, mkAIdx, mkCIdx }

module M = Kuiper.Matrix
open Kuiper.Matrix {
  gpu_matrix,
  gpu_matrix_pts_to,
  gpu_matrix_pts_to_cell,
}

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
let barrier_p
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : full_mlayout tile tile) (m1 : gpu_matrix et l1)
  (#l2 : full_mlayout tile tile) (m2 : gpu_matrix et l2)
  : B.barrier_side (tile *^ tile) =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. (tile * tile)) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. (tile * tile)) x)
    else
      (exists* x. gpu_matrix_pts_to_cell m1 (tid / tile) (tid % tile) x) **
      (exists* x. gpu_matrix_pts_to_cell m2 (tid / tile) (tid % tile) x)

let barrier_q
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : full_mlayout tile tile) (m1 : gpu_matrix et l1)
  (#l2 : full_mlayout tile tile) (m2 : gpu_matrix et l2)
  : B.barrier_side (tile *^ tile) =
  fun it tid -> barrier_p m1 m2 (it+1) tid (* flip flop *)

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  (gA |-> Frac (fA /. mlayout_size lC) eA) **
  (gB |-> Frac (fB /. mlayout_size lC) eB) **
  (exists* v.
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      #1.0R
      (tid / tile) (tid % tile) v)

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  (gA |-> Frac (fA /. mlayout_size lC) eA) **
  (gB |-> Frac (fB /. mlayout_size lC) eB) **
  (exists* v.
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      #1.0R
      (tid / tile) (tid % tile) v)

let barrier_tok
  (#et : Type0)
  (tile : valid_tile)
  (l1 l2 : full_mlayout tile tile)
  (ar1 ar2 : gpu_array et (tile * tile))
  (it : nat)
  (tid : natlt (tile *^ tile))
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
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  // ((ar, ()) : c_shmems (shmems_desc et tile))
  // ^ will this work nicely?
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre1 comb tile gA gB gC eA eB fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x) **
  barrier_tok tile slA slB (fst sh) (fst (snd sh)) 0 tid


unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  // (ar : gpu_array et (2 * tile * tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpost1 comb tile gA gB gC eA eB fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. (tile * tile)) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. (tile * tile)) x) **
  barrier_tok tile slA slB (fst sh) (fst (snd sh)) (2 * mshared) tid

(* TODO: Find out where the time is going when checking this function,
it feels a lot slower than the others. *)
inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#eA #eB : ematrix _ _ _)
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : szlt (mrows * mcols))
  (tid : szlt (tile  * tile))
  ()
  norewrite
  requires
    gpu **
    kpre comb tile slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tile slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id (tile * tile) tid **
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

  let gTile = gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  assert (rewrites_to gTile (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)));

  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;
  assert (pure (SZ.v brow == tid / tile));
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  with i0 j0 v0.
    rewrite gpu_matrix_pts_to_cell gTile i0   j0   v0
         as gpu_matrix_pts_to_cell gTile brow bcol v0;

  let mut sum : et = zero;
  let mut bk  : sz = 0sz;

  while (SZ.(!bk <^ mshared))
    invariant live sum
    invariant
      exists* (vbk : SZ.t).
        (bk |-> vbk) **
        B.barrier_tok (barrier_p sa1 sa2) (barrier_q sa1 sa2) (2 * vbk) tid **
        pure (vbk <= mshared)
    invariant
      (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
      (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
  {
    let vbk = !bk;

    gpu_matrix_extract_tile_ro gA tile tile mrow vbk;
    let aTile = gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v vbk);
    assert (rewrites_to aTile (gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v vbk)));
    gpu_matrix_extract_tile_ro gB tile tile vbk mcol;
    let bTile = gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v mcol);
    assert (rewrites_to bTile (gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v mcol)));

    let v1 = M.gpu_matrix_read aTile brow bcol;
    let v2 = M.gpu_matrix_read bTile brow bcol;

    ambig_trade_elim ();
    ambig_trade_elim ();

    rewrite
        (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
        (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x)
      as barrier_p sa1 sa2 (2 * vbk) tid;

    B.barrier_wait ();
    rewrite (barrier_q sa1 sa2 (2 * vbk) tid)
         as (exists* x. gpu_matrix_pts_to_cell sa1 (tid / tile) (tid % tile) x) **
            (exists* x. gpu_matrix_pts_to_cell sa2 (tid / tile) (tid % tile) x);

    (* TEDIOUS *)
    with x.
      rewrite gpu_matrix_pts_to_cell sa1 (tid / tile) (tid % tile) x
           as gpu_matrix_pts_to_cell sa1 brow bcol x;
    with x.
      rewrite gpu_matrix_pts_to_cell sa2 (tid / tile) (tid % tile) x
           as gpu_matrix_pts_to_cell sa2 brow bcol x;

    M.gpu_matrix_write_cell sa1 brow bcol v1;
    M.gpu_matrix_write_cell sa2 brow bcol v2;

    (* TEDIOUS *)
    with x.
      rewrite gpu_matrix_pts_to_cell sa1 brow bcol x
           as gpu_matrix_pts_to_cell sa1 (tid / tile) (tid % tile) x;
    with x.
      rewrite gpu_matrix_pts_to_cell sa2 brow bcol x
           as gpu_matrix_pts_to_cell sa2 (tid / tile) (tid % tile) x;

    rewrite (exists* x. gpu_matrix_pts_to_cell sa1 (tid / tile) (tid % tile) x) **
            (exists* x. gpu_matrix_pts_to_cell sa2 (tid / tile) (tid % tile) x)
         as (barrier_p sa1 sa2 (2 * vbk + 1) tid);
    B.barrier_wait ();
    rewrite (barrier_q sa1 sa2 (2 * vbk + 1) tid)
    as
      (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. (tile * tile)) x) **
      (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. (tile * tile)) x);

    (* At this point the SHMem cache is filled with the submatrices
       and we have RO permission to it. Compute product for our cell in
       the tile and add to sum. *)

    (* Calling the plain old dotproduct matmult here.
       Note: this will generate code like this:

      float_t sum = (float_t)0.0f;
      while (bk < mshared)
      {
        [...]
        float_t sum1 = (float_t)0.0f;
        while (k < tile)
        {
          sum1 += sa1[brow * tile + k] * sa2[k * tile + bcol];
        }
        float_t t = sum1;
        sum += t;
        [...]
      }

      i.e. with an internal sum, that is then added to
      `sum` here. This is accurate according to how we are associating,
      but unidiomatic. This would be gone if matmul_dotprod took
      as an argument a reference into which to add the values.
    *)
    let t = Kuiper.Poly.GEMM.Util.matmul_dotprod sa1 sa2 brow bcol;
    sum := !sum `add` t;

    // What the hell.
    assert (pure (2 * (vbk + 1) == 2 * vbk + 1 + 1));

    (* Move to next tile *)
    bk := vbk +^ 1sz;
  };

  let s = !sum;
  let v0 = M.gpu_matrix_read_cell gTile brow bcol;
  let v1 = comb v0 s;
  M.gpu_matrix_write_cell gTile brow bcol v1;

  with v'.
    rewrite
      M.gpu_matrix_pts_to_cell gTile brow bcol v'
    as
      M.gpu_matrix_pts_to_cell gTile
        (tid / tile) (tid % tile) v';

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
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA #eB #eC : ematrix _ _ _)
  ()
  norewrite
  requires
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> eC)
  ensures
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
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
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    block_setup_tok (tile *^ tile) **
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
      kpre1 comb tile gA gB gC eA eB fA fB bid tid)
  ensures
    block_setup_tok (tile *^ tile) **
    (forall+ (tid : natlt2 tile  tile).
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
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    (forall+ (tid : natlt2 tile  tile).
      kpost comb tile slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt2 tile  tile).
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
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt2 tile  tile).
      kpost1 comb tile gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
  ensures
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> MS.mmcomb comb eC eA eB)
{
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#fA #fB : perm)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> eC))
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> MS.mmcomb comb eC eA eB))
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  shmems_desc = shmems_desc et tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre1  comb tile gA gB gC eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost1 comb tile gA gB gC eA eB fA fB bid tid);
  setup      = setup    tile comb gA gB gC;
  teardown   = teardown tile comb gA gB gC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile slA slB comb gA gB gC #_ #_ #_ #_ #eC;
  block_teardown = block_teardown tile slA slB comb gA gB gC #_ #_ #_ #_ #eC;

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
  (lA : mlayout (mrows   * tile) (mshared * tile))
  (lB : mlayout (mshared * tile) (mcols   * tile))
  (lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols * tile))
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (tile * tile <= max_threads) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  dassert (tile `SZ.gt` 0sz);
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel tile (R.row_major _ _) (R.row_major _ _) comb gA gB gC ());
}
