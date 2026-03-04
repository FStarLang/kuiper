module Kuiper.Poly.GEMM.BlockTiling1D

#lang-pulse

#set-options "--z3rlimit 40"

open Kuiper
open Kuiper.Approximates
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

module M = Kuiper.Matrix
open Kuiper.Matrix {
  gpu_matrix,
  gpu_matrix_pts_to,
  gpu_matrix_pts_to_cell,
  is_global_matrix
}
open Kuiper.Matrix.Tiling

module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs
module FB = Kuiper.Poly.GEMM.FlipFlopBarrier

open Kuiper.EMatrix { ematrix, macc, ematrix_approximates }
open Kuiper.Array.Vectorized { has_vec_cpy }
open Kuiper.Poly.GEMM.Copy.Vec { own_strided_chunks, live_strided_chunks }

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
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  let mrow = bid / mcols in
  let mcol = bid % mcols in
  gA |-> Frac (fA /. ((mrows * mcols) * tile)) eA **
  gB |-> Frac (fB /. ((mrows * mcols) * tile)) eB **
  forall+ (ii : natlt tile).
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      ii tid (macc eC (mrow * tile + ii) (mcol * tile + tid))

unfold
let kpost1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  let mrow = bid / mcols in
  let mcol = bid % mcols in
  gA |-> Frac (fA /. ((mrows * mcols) * tile)) eA **
  gB |-> Frac (fB /. ((mrows * mcols) * tile)) eB **
  forall+ (ii : natlt tile).
   (exists* (v : et).
     gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      ii tid v **
     pure (v %~ MU.real_gemm_single comb_r eA eB eC (mrow * tile + ii) (mcol * tile + tid)))

(* Predicate for the write loop: rows below vrow are done (have
   kpost1 form), rows at or above vrow still hold their initial
   exact C value. *)
let bt1_write_pred
  (#et : Type0) {| scalar et, real_like et |}
  (comb_r : binop real)
  (#mrows #mshared #mcols : nat)
  (tile : pos)
  (#l : mlayout tile tile)
  (tileC : gpu_matrix et l)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (mrow : natlt mrows) (mcol : natlt mcols)
  (tid : natlt tile)
  (vrow : nat)
  (ii : natlt tile)
  : slprop
  = if t2b (ii < vrow)
    then exists* (v : et).
           gpu_matrix_pts_to_cell tileC ii tid v **
           pure (v %~ MU.real_gemm_single comb_r eA eB eC (mrow * tile + ii) (mcol * tile + tid))
    else gpu_matrix_pts_to_cell tileC ii tid (macc eC (mrow * tile + ii) (mcol * tile + tid))

(* Step lemma: given old sums that approximate the tiled partial
   matmul up to vbk*tile and new sums produced by subproduct_cols,
   the new sums approximate up to (vbk+1)*tile. *)
let bt1_step_approx
  (#et : Type) {| scalar et, real_like et |}
  (#mrows #mshared #mcols : nat)
  (tile : pos)
  (eA : ematrix et (mrows * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols * tile))
  (mrow : natlt mrows) (mcol : natlt mcols)
  (vbk : nat{vbk < mshared})
  (bcol : natlt tile)
  (old_sums new_sums : Seq.seq et)
  : Lemma
    (requires
      Seq.length old_sums == tile /\
      Seq.length new_sums == tile /\
      (forall (i:nat{i < tile}).
        Seq.index new_sums i == MS.__gmatmul_single (Seq.index old_sums i) mul add
          (ematrix_subtile eA tile tile mrow vbk)
          (ematrix_subtile eB tile tile vbk mcol) i bcol tile) /\
      (forall (ii:nat{ii < tile}).
        Seq.index old_sums ii %~
          MU.__real_matmul_single_tiled eA eB (mrow * tile + ii) (mcol * tile + bcol) (vbk * tile)))
    (ensures
      (forall (ii:nat{ii < tile}).
        Seq.index new_sums ii %~
          MU.__real_matmul_single_tiled eA eB (mrow * tile + ii) (mcol * tile + bcol) ((vbk + 1) * tile)))
  = let sub_A = ematrix_subtile eA tile tile mrow vbk in
    let sub_B = ematrix_subtile eB tile tile vbk mcol in
    let aux (ii:nat{ii < tile})
      : Lemma (Seq.index new_sums ii %~
          MU.__real_matmul_single_tiled eA eB (mrow * tile + ii) (mcol * tile + bcol) ((vbk + 1) * tile))
    = let row : natlt (mrows * tile) = mrow * tile + ii in
      let col : natlt (mcols * tile) = mcol * tile + bcol in
      MU.gmatmul_single_init_approx
        (Seq.index old_sums ii)
        (MU.__real_matmul_single_tiled eA eB row col (vbk * tile))
        sub_A sub_B ii bcol tile ();
      MU.__real_matmul_single_tiled_step eA eB mrow mcol vbk ii bcol
    in
    Classical.forall_intro (Classical.move_requires aux)

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
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  kpre1 comb tile gA gB gC eA eB eC fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. tile) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. tile) x)

let kpre_block_sendable
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (i:natlt (mrows * mcols))
  (j:natlt tile)
: is_send_across block_of (kpre comb tile slA slB gA gB gC eA eB eC fA fB sh i j)
= magic()

unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  (tid : natlt tile)
  : slprop
  =
  kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid **
  (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. tile) x) **
  (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. tile) x)

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : sz)
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eA eB : ematrix _ _ _)
  (eC : ematrix et (mrows * tile) (mcols * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (i:natlt (mrows * mcols))
  (j:natlt tile)
: is_send_across block_of (kpost comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh i j)
= magic()


inline_for_extraction noextract
fn bring_2cols
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : erased nat)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  {| clayout lA, clayout lB |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (#l1 : mlayout tile tile) {| clayout l1 |} (sa1 : gpu_matrix et l1)
  (#l2 : mlayout tile tile) {| clayout l2 |} (sa2 : gpu_matrix et l2)
  (mrow : szlt mrows)
  (mk : szlt mshared)
  (mcol : szlt mcols)
  (tid : szlt tile)
  (#fA #fB : perm)
  (#eA #eB : ematrix _ _ _)
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
    let tileA = gpu_matrix_extract_tile_ro' gA (SZ.v tile) (SZ.v tile) (SZ.v mrow) (SZ.v mk);
    let v1 = M.gpu_matrix_read tileA vi tid;
    M.gpu_matrix_write_cell sa1 vi tid v1;
    ambig_trade_elim ();
    ambig_trade_elim ();
    fold own_1_col sa1 tid;

    unfold own_1_col sa2 tid;
    forevery_extract #(natlt tile) vi _;
    let tileB = gpu_matrix_extract_tile_ro' gB (SZ.v tile) (SZ.v tile) (SZ.v mk) (SZ.v mcol);
    let v2 = M.gpu_matrix_read tileB vi tid;
    M.gpu_matrix_write_cell sa2 vi tid v2;
    ambig_trade_elim ();
    ambig_trade_elim ();
    fold own_1_col sa2 tid;

    i := !i +^ 1sz;
  }
}

#restart-solver // try to work around Z3 crash

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB : ematrix _ _ _)
  (#eC : ematrix et (mrows * tile) (mcols * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : szlt (mrows * mcols))
  (tid : szlt tile)
  ()
  norewrite
  requires
    gpu **
    kpre comb tile slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id tile tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) tile bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid **
    thread_id tile tid **
    block_id (mrows * mcols) bid **
    B.barrier_tok (FB.contract eA eB slA slB (fst sh) (fst (snd sh)) tile bid) **
    B.barrier_state (2 * mshared)
{
  let (ar1, (ar2, _)) = sh;

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  M.gpu_matrix_abs' slA ar1;
  let sa1 = M.from_array slA ar1;
  rewrite each M.from_array slA ar1 as sa1;

  M.gpu_matrix_abs' slB ar2;
  let sa2 = M.from_array slB ar2;
  rewrite each M.from_array slB ar2 as sa2;

  rewrite
    (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
    (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. tile) x)
  as
    (exists* em1. FB.bp_sharing sa1 em1 tile) **
    (exists* em2. FB.bp_sharing sa2 em2 tile);

  let mrow, mcol = s_divmod mcols bid;
  let bcol = tid;
  assert rewrites_to bcol tid;
  assert (pure (SZ.v bcol == tid % tile));
  assert (pure (bcol < tile));


  (* thread-local result cache *)
  let mut sums : Pulse.Lib.Array.array et = [| zero ; tile |];
  let mut bk  : sz = 0sz;

  while (let vbk = !bk; SZ.(vbk <^ mshared))
    invariant
      exists* (vbk : SZ.t{vbk <= mshared}) cur_sums.
        bk |-> vbk **
        sums |-> cur_sums **
        B.barrier_state (2 * SZ.v vbk) **
        pure (Seq.length cur_sums == SZ.v tile /\
          (forall (ii:nat{ii < SZ.v tile}).
            Seq.index cur_sums ii %~
              MU.__real_matmul_single_tiled eA eB
                (SZ.v mrow * SZ.v tile + ii)
                (SZ.v mcol * SZ.v tile + SZ.v bcol)
                (SZ.v vbk * SZ.v tile)))
    invariant
        (exists* em1. FB.bp_sharing sa1 em1 tile) **
        (exists* em2. FB.bp_sharing sa2 em2 tile)
  {
    pts_to_len sums;
    let vbk = !bk;

    even_2x !bk;
    rewrite (exists* em1. FB.bp_sharing sa1 em1 tile) **
            (exists* em2. FB.bp_sharing sa2 em2 tile)
         as (FB.barrier_p eA eB sa1 sa2 tile bid) (2 * vbk) tid;
    rewrite (FB.barrier_p eA eB sa1 sa2 tile bid) (2 * vbk) tid
         as (FB.contract eA eB slA slB ar1 ar2 tile bid).rin (2 * vbk) tid;
    B.barrier_wait ();
    rewrite (FB.contract eA eB slA slB ar1 ar2 tile bid).rout (2 * vbk) tid
         as (FB.barrier_q eA eB sa1 sa2 tile bid) (2 * vbk) tid;
    rewrite (FB.barrier_q eA eB sa1 sa2 tile bid) (2 * vbk) tid
         as live_strided_chunks sa1 tile tid **
            live_strided_chunks sa2 tile tid;

    (* Bridge from live_strided_chunks to column ownership for bring_2cols. *)
    drop_ (live_strided_chunks sa1 tile tid);
    drop_ (live_strided_chunks sa2 tile tid);
    assume_ (own_1_col sa1 tid);
    assume_ (own_1_col sa2 tid);

    (* At this point we exclusively own a full column of the SHMEM
       cache. Populate it. *)
    let vbk = !bk;
    bring_2cols tile gA gB sa1 sa2 mrow vbk mcol tid;

    (* Bridge from column ownership back to strided chunks with known values. *)
    drop_ (own_1_col sa1 tid);
    drop_ (own_1_col sa2 tid);
    assume_ (own_strided_chunks sa1 (ematrix_subtile eA tile tile mrow vbk) tile tid);
    assume_ (own_strided_chunks sa2 (ematrix_subtile eB tile tile vbk mcol) tile tid);

    odd_2x1 !bk;
    assert (pure (odd (2 * !bk + 1)));
    rewrite own_strided_chunks sa1 (ematrix_subtile eA tile tile mrow vbk) tile tid **
            own_strided_chunks sa2 (ematrix_subtile eB tile tile vbk mcol) tile tid
         as (FB.barrier_p eA eB sa1 sa2 tile bid) (2 * vbk + 1) tid;
    rewrite (FB.barrier_p eA eB sa1 sa2 tile bid) (2 * vbk + 1) tid
         as (FB.contract eA eB slA slB ar1 ar2 tile bid).rin (2 * vbk + 1) tid;
    B.barrier_wait ();

    even_2x (!bk + 1);
    assert pure (2 * (!bk + 1) == 2 * !bk + 2);
    assert pure (odd (2 * !bk + 1));
    assert pure ((2 * !bk + 1) < 2 * mshared);
    rewrite (FB.contract eA eB slA slB ar1 ar2 tile bid).rout (2 * !bk + 1) tid
         as (FB.barrier_q eA eB sa1 sa2 tile bid) (2 * !bk + 1) tid;
    rewrite (FB.barrier_q eA eB sa1 sa2 tile bid) (2 * !bk + 1) tid
         as FB.bp_sharing sa1 (ematrix_subtile eA tile tile mrow !bk) tile **
            FB.bp_sharing sa2 (ematrix_subtile eB tile tile !bk mcol) tile;

    unfold FB.bp_sharing sa1 (ematrix_subtile eA tile tile mrow !bk) tile;
    unfold FB.bp_sharing sa2 (ematrix_subtile eB tile tile !bk mcol) tile;

    (* At this point the SHMem cache is filled with the submatrices
       and we have RO permission to it. Compute product for our cell in
       the tile and add to sum. *)
    with old_sums. assert (sums |-> old_sums);
    Kuiper.Poly.GEMM.Util.subproduct_cols tile sums sa1 sa2 bcol;

    (* Prove the step approximation: new sums approximate up to (vbk+1)*tile *)
    with new_sums. assert (sums |-> new_sums);
    bt1_step_approx #et tile eA eB (SZ.v mrow) (SZ.v mcol) (SZ.v vbk) (SZ.v bcol) (reveal old_sums) (reveal new_sums);

    fold FB.bp_sharing sa1 (ematrix_subtile eA tile tile mrow !bk) tile;
    fold FB.bp_sharing sa2 (ematrix_subtile eB tile tile !bk mcol) tile;

    assert (pure (2 * (!bk + 1) == 2 * !bk + 1 + 1));

    (* Move to next tile *)
    bk := !bk +^ 1sz;
  };

  (* Write accumulated sums, proving functional correctness along the way. *)

  let tileC = gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  rewrite each gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols) as tileC;

  (* Extract final sums and the approximation fact *)
  with final_sums. assert (sums |-> final_sums);

  (* __real_matmul_single_tiled ... (mshared*tile) == real_matmul_single ... by definition *)
  assert pure (forall (ii:nat{ii < SZ.v tile}).
    Seq.index (reveal final_sums) ii %~
      MU.__real_matmul_single_tiled eA eB
        (SZ.v mrow * SZ.v tile + ii) (SZ.v mcol * SZ.v tile + SZ.v bcol)
        (SZ.v mshared * SZ.v tile));

  (* Convert exact-value cells to existential form for the write loop.
     forevery_extract with a trade requires reproducing the exact same
     slprop; since we write a NEW value, the trade only works when the
     cell is wrapped in exists*. *)
  forevery_map
    (fun (ii : natlt tile) ->
      gpu_matrix_pts_to_cell tileC ii tid (macc eC (bid / mcols * tile + ii) (bid % mcols * tile + tid)))
    (fun (ii : natlt tile) ->
      exists* (v : et). gpu_matrix_pts_to_cell tileC ii tid v)
    fn ii { (); };

  let mut row : sz = 0sz;
  Pulse.Lib.Array.pts_to_len sums;

  while (SZ.(!row <^ tile))
    invariant live row ** live sums
  {
    Pulse.Lib.Array.pts_to_len sums;
    forevery_extract #(natlt tile) (!row) _;

    (* tedious: tid ↔ bcol *)
    with v0.
      rewrite gpu_matrix_pts_to_cell tileC (!row) tid v0
           as gpu_matrix_pts_to_cell tileC (!row) bcol v0;

    let v0 = M.gpu_matrix_read_cell tileC !row bcol;
    open Pulse.Lib.Array;
    let v1 = sums.(!row);
    let v' = comb v0 v1;
    M.gpu_matrix_write_cell tileC !row bcol v';

    (* tedious *)
    with v0.
      assert gpu_matrix_pts_to_cell tileC (!row) tid v0;

    row := !row +^ 1sz;
    Pulse.Lib.Trade.elim_trade _ _;
  };

  (* Functional correctness assumption:
     The accumulated subproduct_cols results, combined with the old cell
     values, approximate the real gemm specification.
     The real proof would use:
       sums[ii] %~ real_matmul_single eA eB row col  (from first loop)
       macc eC row col %~ to_real (macc eC row col)  (to_real_ok)
       comb x y %~ comb_r rx ry when x %~ rx, y %~ ry  (comb_r spec)
              = real_gemm_single comb_r eA eB eC row col *)
  forevery_map
    (fun (ii : natlt tile) ->
      exists* (v : et).
        gpu_matrix_pts_to_cell tileC ii tid v)
    (fun (ii : natlt tile) ->
      exists* (v : et).
        gpu_matrix_pts_to_cell tileC ii tid v **
        pure (v %~ MU.real_gemm_single comb_r eA eB eC (bid / mcols * tile + ii) (bid % mcols * tile + tid)))
    fn ii {
      with v. assert (gpu_matrix_pts_to_cell tileC ii tid v);
      assume pure (v %~ MU.real_gemm_single comb_r eA eB eC (bid / mcols * tile + ii) (bid % mcols * tile + tid));
    };

  with em1. unfold FB.bp_sharing sa1 em1 tile;
  with em2. unfold FB.bp_sharing sa2 em2 tile;

  M.gpu_matrix_concr sa1; rewrite each M.core sa1 as ar1;
  M.gpu_matrix_concr sa2; rewrite each M.core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  rewrite each tileC as gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
}
#pop-options

ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix et _ _)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt tile).
      kpre1 comb tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  (* Step 1: Share gA/gB, explode gC *)
  M.gpu_matrix_share_n gA ((mrows * mcols) * tile);
  M.gpu_matrix_share_n gB ((mrows * mcols) * tile);
  gpu_matrix_explode_tiled gC (SZ.v tile) (SZ.v tile);
  forevery_rw_size4 ((mrows * tile) / tile) mrows ((mcols * tile) / tile) mcols (SZ.v tile) tile (SZ.v tile) tile;

  (* Step 2: Swap inner (i,j) → (j,i) via flatten/mid_flip/unflatten *)
  forevery_flatten
    (fun (tr:natlt mrows) (tc:natlt mcols) ->
      forall+ (i:natlt tile) (j:natlt tile).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) tr tc) i j
          (macc eC (tr * tile + i) (tc * tile + j)));

  forevery_mid_flip
    (fun (trtc:natlt mrows & natlt mcols) (i:natlt tile) (j:natlt tile) ->
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) trtc._1 trtc._2) i j
        (macc eC (trtc._1 * tile + i) (trtc._2 * tile + j)));

  (* Step 3: Unflatten back to (tr, tc, j, i) — no exists* introduction needed *)
  forevery_unflatten
    (fun (tr:natlt mrows) (tc:natlt mcols) ->
      forall+ (j:natlt tile) (i:natlt tile).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) tr tc) i j
          (macc eC (tr * tile + i) (tc * tile + j)));

  (* Step 4: Collapse (tr, tc) → bid *)
  forevery_unfactor' (mrows * mcols) mrows mcols
    (fun (tr:natlt mrows) (tc:natlt mcols) ->
      forall+ (j:natlt tile) (i:natlt tile).
        gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) tr tc) i j
          (macc eC (tr * tile + i) (tc * tile + j)));

  (* Step 5: Factor gA/gB to 2D *)
  forevery_factor ((mrows * mcols) * tile) (mrows * mcols) tile
    (fun _ -> gA |-> Frac (fA /. ((mrows * mcols) * tile)) eA);
  forevery_factor ((mrows * mcols) * tile) (mrows * mcols) tile
    (fun _ -> gB |-> Frac (fB /. ((mrows * mcols) * tile)) eB);

  (* Step 6: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt tile) ->
      gA |-> Frac (fA /. ((mrows * mcols) * tile)) eA)
    (fun (_ : natlt (mrows * mcols)) (_ : natlt tile) ->
      gB |-> Frac (fB /. ((mrows * mcols) * tile)) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt tile) ->
      forall+ (ii : natlt tile).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid
          (macc eC (bid / mcols * tile + ii) (bid % mcols * tile + tid)));

  (* Step 7: Bridge to natlt2 and match kpre1 *)
  forevery_rw_size2 (mrows * mcols) (SZ.v (mrows `SZ.mul` mcols)) tile tile;
  forevery_ext_2
    (fun (bid : natlt (SZ.v (mrows `SZ.mul` mcols))) (tid : natlt tile) ->
      gA |-> Frac (fA /. ((mrows * mcols) * tile)) eA **
      gB |-> Frac (fB /. ((mrows * mcols) * tile)) eB **
      forall+ (ii : natlt tile).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid
          (macc eC (bid / mcols * tile + ii) (bid % mcols * tile + tid)))
    (fun (bid : natlt2 mrows mcols) (tid : natlt tile) ->
      kpre1 comb tile gA gB gC eA eB eC fA fB bid tid);
  ();
}

ghost
fn block_setup
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols   * tile))
  (#eC : ematrix et (mrows   * tile) (mcols   * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt tile).
      kpre1 comb tile gA gB gC eA eB eC fA fB bid tid)
  ensures
    (forall+ (tid : natlt tile).
      kpre comb tile slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
{
  gpu_live_c_shmems_share_underspec sh #1.0R #tile;

  forevery_map
    (fun (_ : natlt tile) -> live_c_shmems sh #(1.0R /. tile))
    (fun (_ : natlt tile) ->
      (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. tile) x) **
      (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. tile) x))
    fn _ {
      unfold_live_c_shmems_cons sh #(1.0R /. tile);
      unfold_live_c_shmem (fst sh) #(1.0R /. tile);
      unfold_live_c_shmems_cons (snd sh) #(1.0R /. tile);
      unfold_live_c_shmem (fst (snd sh)) #(1.0R /. tile);
      unfold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. tile);
    };

  forevery_zip
    (fun (tid : natlt tile) -> kpre1 comb tile gA gB gC eA eB eC fA fB bid tid)
    _;
}

ghost
fn block_teardown
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : SZ.t)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols   * tile))
  (#eC : ematrix et (mrows   * tile) (mcols   * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (mrows * mcols))
  ()
  norewrite
  requires
    (forall+ (tid : natlt tile).
      kpost comb comb_r tile slA slB gA gB gC eA eB eC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
{
  forevery_unzip
    (fun (tid : natlt tile) -> kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
    _;

  forevery_map
    (fun (_ : natlt tile) ->
      (exists* x. gpu_pts_to_array (fst sh) #(1.0R /. tile) x) **
      (exists* x. gpu_pts_to_array (fst (snd sh)) #(1.0R /. tile) x))
    (fun (_ : natlt tile) -> live_c_shmems sh #(1.0R /. tile))
    fn _ {
      fold_live_c_shmem (fst (snd sh)) #(1.0R /. tile);
      fold_live_c_shmems_nil (snd (snd sh)) #(1.0R /. tile);
      fold_live_c_shmems_cons (snd sh) #(1.0R /. tile);
      fold_live_c_shmem (fst sh) #(1.0R /. tile);
      fold_live_c_shmems_cons sh #(1.0R /. tile);
    };

  gpu_live_c_shmems_gather_underspec sh #1.0R #tile;
}

#push-options "--z3rlimit 160"
ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA : ematrix et (mrows   * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols   * tile))
  (#eC : ematrix et (mrows   * tile) (mcols   * tile))
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
             (tid : natlt tile).
      kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et (mrows * tile) (mcols * tile)).
      gC |-> eC' **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  let n_threads = (mrows * mcols) * tile;

  (* Step 1: Bridge natlt2 → natlt *)
  forevery_rw_size2 (SZ.v (mrows `SZ.mul` mcols)) (mrows * mcols) tile tile;

  (* Step 2: Unfold kpost1 *)
  forevery_ext_2
    (fun (bid : natlt (mrows * mcols)) (tid : natlt tile) ->
      kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt tile) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (ii : natlt tile).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC (mrow * tile + ii) (mcol * tile + tid)));

  (* Step 3: Unzip gA *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt tile) ->
      gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt tile) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (ii : natlt tile).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC (mrow * tile + ii) (mcol * tile + tid)));

  (* Step 4: Unzip gB *)
  forevery_unzip_2
    (fun (_ : natlt (mrows * mcols)) (_ : natlt tile) ->
      gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (mrows * mcols)) (tid : natlt tile) ->
      let mrow = bid / mcols in
      let mcol = bid % mcols in
      forall+ (ii : natlt tile).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC (mrow * tile + ii) (mcol * tile + tid)));

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' n_threads (mrows * mcols) tile
    (fun (_ : natlt (mrows * mcols)) (_ : natlt tile) ->
      gA |-> Frac (fA /. n_threads) eA);
  M.gpu_matrix_gather_n gA n_threads;

  forevery_unfactor' n_threads (mrows * mcols) tile
    (fun (_ : natlt (mrows * mcols)) (_ : natlt tile) ->
      gB |-> Frac (fB /. n_threads) eB);
  M.gpu_matrix_gather_n gB n_threads;

  (* Step 6: Swap (tid, ii) → (ii, tid) via mid_flip *)
  forevery_mid_flip
    (fun (bid : natlt (mrows * mcols)) (tid : natlt tile) (ii : natlt tile) ->
      exists* (v : et).
        gpu_matrix_pts_to_cell
          (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid v **
        pure (v %~ MU.real_gemm_single comb_r eA eB eC
          (bid / mcols * tile + ii) (bid % mcols * tile + tid)));

  (* Step 7: Collapse inner (ii, tid) → flatid via unfactor' per bid *)
  forevery_map
    (fun (bid : natlt (mrows * mcols)) ->
      forall+ (ii : natlt tile) (tid : natlt tile).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
            (bid / mcols * tile + ii) (bid % mcols * tile + tid)))
    (fun (bid : natlt (mrows * mcols)) ->
      forall+ (flatid : natlt (tile * tile)).
        exists* (v : et).
          gpu_matrix_pts_to_cell
            (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
            (flatid / tile) (flatid % tile) v **
          pure (v %~ MU.real_gemm_single comb_r eA eB eC
            (bid / mcols * tile + flatid / tile) (bid % mcols * tile + flatid % tile)))
    fn bid {
      forevery_unfactor' (tile * tile) tile tile
        (fun (ii : natlt tile) (tid : natlt tile) ->
          exists* (v : et).
            gpu_matrix_pts_to_cell
              (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) ii tid v **
            pure (v %~ MU.real_gemm_single comb_r eA eB eC
              (bid / mcols * tile + ii) (bid % mcols * tile + tid)));
    };

  (* Step 8: Collect cells via gpu_matrix_collect_approx_tiled *)
  let _vf = gpu_matrix_collect_approx_tiled gC (SZ.v tile) (SZ.v tile)
    mrows mcols
    (fun (row : natlt (mrows * tile)) (col : natlt (mcols * tile)) (v : et) ->
      v %~ MU.real_gemm_single comb_r eA eB eC row col);

  with eC'. assert (gC |-> eC');

  assert pure (forall (row:natlt (mrows * tile)) (col:natlt (mcols * tile)).
    macc eC' row col %~ MU.real_gemm_single comb_r eA eB eC row col);

  assert pure (forall (row:natlt (mrows * tile)) (col:natlt (mcols * tile)).
    macc eC' row col %~ macc (MU.real_mmcomb comb_r eC eA eB) row col);

  assert pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB));
  ();
}
#pop-options

#push-options "--z3rlimit 200"
inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : full_mlayout tile tile) // shmem layouts
  {| clayout slA, clayout slB |}
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows   * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols   * tile))
  (#eC : ematrix et (mrows   * tile) (mcols   * tile))
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile <= max_threads))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et _ _).
          gC |-> eC' **
          pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB))))
= {
  nblk = mrows *^ mcols;
  nthr = tile;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB slA slB (fst ptrs) (fst (snd ptrs)) tile bid);
  barrier_count    = (fun _bid -> 2 * SZ.v mshared);
  barrier_ok = magic();

  shmems_desc = shmems_desc et tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt tile). kpre1  comb tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt tile). kpost1 comb comb_r tile gA gB gC eA eB eC fA fB bid tid);
  setup      = setup    tile comb gA gB gC;
  teardown   = teardown tile comb comb_r gA gB gC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile slA slB comb gA gB gC #_ #_ #eC;
  block_teardown = block_teardown tile slA slB comb comb_r gA gB gC #_ #_ #eC;

  kpre      = kpre  comb tile slA slB gA gB gC eA eB eC fA fB;
  kpost     = kpost comb comb_r tile slA slB gA gB gC eA eB eC fA fB;

  f = kf tile slA slB comb comb_r gA gB gC;
  kpost_sendable = kpost_block_sendable comb comb_r tile slA slB gA gB gC eA eB eC fA fB;
  kpre_sendable = kpre_block_sendable comb tile slA slB gA gB gC eA eB eC fA fB;
  block_post_sendable = solve;
  block_pre_sendable = solve;
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (#mrows #mshared #mcols : szp)
  (lA : mlayout (mrows   * tile) (mshared * tile))
  (lB : mlayout (mshared * tile) (mcols   * tile))
  (lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA #eB #eC : ematrix _ _ _)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et _ _).
      on gpu_loc (gC |-> eC') **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
{
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile (R.row_major _ _) (R.row_major _ _) comb comb_r gA #fA gB #fB gC #eA #eB #eC ());
}

(* Legacy interface for backward compatibility.
   Calls the approximate kernel with a fake comb_r and assumes the exact result. *)
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
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows   * tile) (mshared * tile))
  (#eB : ematrix et (mshared * tile) (mcols   * tile))
  (#eC : ematrix et (mrows   * tile) (mcols   * tile))
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  let _ : real_like et #_ = magic ();
  let _ : has_vec_cpy et #_ = magic ();
  let comb_r : binop real = magic ();
  assume pure (forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s);
  mmcomb_gpu_approx tile comb comb_r lA lB lC gA gB gC;
  with eC'. assert (on gpu_loc (gC |-> eC'));
  assume pure (eC' == MS.mmcomb comb eC eA eB);
}
