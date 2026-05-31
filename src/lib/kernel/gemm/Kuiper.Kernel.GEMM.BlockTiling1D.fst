module Kuiper.Kernel.GEMM.BlockTiling1D

#lang-pulse

#set-options "--z3rlimit 40"

open Kuiper
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor.Tiling

open Kuiper.Array2 { array2 }
module M = Kuiper.Array2
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

open Kuiper.EMatrix { ematrix, macc, mkM, ematrix_approximates }

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
  (#l : M.layout tile tile)
  (m : array2 et l)
  (tid : natlt tile)
  : slprop =
  forall+ (ii : natlt tile).
    exists* (x : et).
      Cell m (ii, tid) |-> x

let barrier_p
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : M.layout tile tile) (m1 : array2 et l1)
  (#l2 : M.layout tile tile) (m2 : array2 et l2)
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
  (#l1 : M.layout tile tile) (m1 : array2 et l1)
  (#l2 : M.layout tile tile) (m2 : array2 et l2)
  : B.barrier_side tile =
  fun it tid -> barrier_p m1 m2 (it+1) tid (* flip flop *)

let barrier_contract
  (#et : Type0)
  (tile : valid_tile)
  (* This is defined over the base shared larrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 l2 : M.full_layout tile tile)
  (ar1 ar2 : larray et (tile * tile))
  : B.contract tile =
  {
    rin  = barrier_p (M.from_array l1 ar1) (M.from_array l2 ar2);
    rout = barrier_q (M.from_array l1 ar1) (M.from_array l2 ar2);
  }

(* ---- Barrier transform proof ---- *)

(* Even → odd: distribute fractional whole-array ownership into per-column cells. *)
ghost
fn even_barrier_p_to_q
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : M.layout tile tile) (m1 : array2 et l1)
  (#l2 : M.layout tile tile) (m2 : array2 et l2)
  (it : nat{even it})
  (#_ : squash (SZ.fits (M.layout_size l1)))
  (#_ : squash (SZ.fits (M.layout_size l2)))
  requires
    forall+ (tid : natlt tile). barrier_p m1 m2 it tid
  ensures
    forall+ (tid : natlt tile). barrier_q m1 m2 it tid
{
  assert pure (even it);
  (* barrier_p even = frac shares; barrier_q even = own_1_col *)
  forevery_map
    (fun (tid : natlt tile) -> barrier_p m1 m2 it tid)
    (fun (tid : natlt tile) ->
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. tile) x))
    fn tid {
      rewrite barrier_p m1 m2 it tid
           as (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. tile) x) **
              (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. tile) x);
    };
  forevery_unzip _ _;
  M.gather_n_underspec m1 tile;
  M.gather_n_underspec m2 tile;
  with em1. assert (m1 |-> em1);
  with em2. assert (m2 |-> em2);
  M.ilower m1;
  M.ilower m2;
  forevery_commute (fun (r : natlt tile) (c : natlt tile) -> M.pts_to_cell m1 (r, c) (macc em1 r c));
  forevery_commute (fun (r : natlt tile) (c : natlt tile) -> M.pts_to_cell m2 (r, c) (macc em2 r c));
  forevery_map
    (fun (c : natlt tile) -> forall+ (r : natlt tile). M.pts_to_cell m1 (r, c) (macc em1 r c))
    (fun (c : natlt tile) -> own_1_col m1 c)
    fn c {
      forevery_map
        (fun (r : natlt tile) -> M.pts_to_cell m1 (r, c) (macc em1 r c))
        (fun (r : natlt tile) -> exists* (x : et). Cell m1 (r, c) |-> x)
        fn r { };
      fold own_1_col m1 c;
    };
  forevery_map
    (fun (c : natlt tile) -> forall+ (r : natlt tile). M.pts_to_cell m2 (r, c) (macc em2 r c))
    (fun (c : natlt tile) -> own_1_col m2 c)
    fn c {
      forevery_map
        (fun (r : natlt tile) -> M.pts_to_cell m2 (r, c) (macc em2 r c))
        (fun (r : natlt tile) -> exists* (x : et). Cell m2 (r, c) |-> x)
        fn r { };
      fold own_1_col m2 c;
    };
  forevery_zip
    (fun (tid : natlt tile) -> own_1_col m1 tid)
    (fun (tid : natlt tile) -> own_1_col m2 tid);
  forevery_map
    (fun (tid : natlt tile) -> own_1_col m1 tid ** own_1_col m2 tid)
    (fun (tid : natlt tile) -> barrier_q m1 m2 it tid)
    fn tid {
      rewrite own_1_col m1 tid ** own_1_col m2 tid
           as barrier_q m1 m2 it tid;
    };
}

(* Odd → even: collect per-column cells back to fractional whole-array ownership. *)
ghost
fn odd_barrier_p_to_q
  (#et : Type0)
  (#tile : valid_tile)
  (#l1 : M.layout tile tile) (m1 : array2 et l1)
  (#l2 : M.layout tile tile) (m2 : array2 et l2)
  (it : nat{odd it})
  (#_ : squash (SZ.fits (M.layout_size l1)))
  (#_ : squash (SZ.fits (M.layout_size l2)))
  requires
    forall+ (tid : natlt tile). barrier_p m1 m2 it tid
  ensures
    forall+ (tid : natlt tile). barrier_q m1 m2 it tid
{
  assert pure (odd it);
  (* barrier_p odd = own_1_col; barrier_q odd = frac shares *)
  forevery_map
    (fun (tid : natlt tile) -> barrier_p m1 m2 it tid)
    (fun (tid : natlt tile) -> own_1_col m1 tid ** own_1_col m2 tid)
    fn tid {
      rewrite barrier_p m1 m2 it tid
           as own_1_col m1 tid ** own_1_col m2 tid;
    };
  forevery_unzip _ _;
  (* Unfold own_1_col to nested forall+/exists *)
  forevery_map
    (fun (c : natlt tile) -> own_1_col m1 c)
    (fun (c : natlt tile) -> forall+ (r : natlt tile). exists* (x : et). Cell m1 (r, c) |-> x)
    fn c { unfold own_1_col m1 c };
  forevery_map
    (fun (c : natlt tile) -> own_1_col m2 c)
    (fun (c : natlt tile) -> forall+ (r : natlt tile). exists* (x : et). Cell m2 (r, c) |-> x)
    fn c { unfold own_1_col m2 c };
  (* Commute: forall+ c r -> forall+ r c *)
  forevery_commute (fun (c : natlt tile) (r : natlt tile) -> exists* (x : et). Cell m1 (r, c) |-> x);
  forevery_commute (fun (c : natlt tile) (r : natlt tile) -> exists* (x : et). Cell m2 (r, c) |-> x);
  (* Extract witnesses *)
  let f1 = forevery_exists_2 (fun (r : natlt tile) (c : natlt tile) (x : et) -> Cell m1 (r, c) |-> x);
  let f2 = forevery_exists_2 (fun (r : natlt tile) (c : natlt tile) (x : et) -> Cell m2 (r, c) |-> x);
  (* Construct ematrices from witness functions *)
  let em1 : ematrix et tile tile = mkM f1;
  let em2 : ematrix et tile tile = mkM f2;
  (* Rewrite cells to use macc *)
  forevery_map
    (fun (r : natlt tile) -> forall+ (c : natlt tile). Cell m1 (r, c) |-> f1 r c)
    (fun (r : natlt tile) -> forall+ (c : natlt tile). M.pts_to_cell m1 (r, c) (macc em1 r c))
    fn r {
      forevery_ext
        (fun (c : natlt tile) -> Cell m1 (r, c) |-> f1 r c)
        (fun (c : natlt tile) -> M.pts_to_cell m1 (r, c) (macc em1 r c));
    };
  forevery_map
    (fun (r : natlt tile) -> forall+ (c : natlt tile). Cell m2 (r, c) |-> f2 r c)
    (fun (r : natlt tile) -> forall+ (c : natlt tile). M.pts_to_cell m2 (r, c) (macc em2 r c))
    fn r {
      forevery_ext
        (fun (c : natlt tile) -> Cell m2 (r, c) |-> f2 r c)
        (fun (c : natlt tile) -> M.pts_to_cell m2 (r, c) (macc em2 r c));
    };
  M.iraise m1;
  M.iraise m2;
  M.share_n m1 tile;
  M.share_n m2 tile;
  forevery_zip
    (fun (_ : natlt tile) -> m1 |-> Frac (1.0R /. tile) em1) _;
  forevery_map
    (fun (tid : natlt tile) ->
      m1 |-> Frac (1.0R /. tile) em1 **
      m2 |-> Frac (1.0R /. tile) em2)
    (fun (tid : natlt tile) ->
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. tile) x))
    fn tid { };
  forevery_map
    (fun (tid : natlt tile) ->
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. tile) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. tile) x))
    (fun (tid : natlt tile) -> barrier_q m1 m2 it tid)
    fn tid {
      rewrite
        (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. tile) x) **
        (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. tile) x)
      as
        barrier_q m1 m2 it tid;
    };
}

(* Both helpers have the same pre/postcondition shape (barrier_p → barrier_q),
   so we can define the barrier_transform directly by case-splitting on even/odd.
   We use a regular F* let to avoid Pulse's if/else effect promotion issue. *)
#push-options "--z3rlimit 80"
let barrier_p_to_q_transform
  (#et : Type0)
  (#tile : valid_tile)
  (l1 l2 : M.full_layout tile tile)
  (ar1 ar2 : larray et (tile * tile))
  (#_ : squash (SZ.fits (M.layout_size l1)))
  (#_ : squash (SZ.fits (M.layout_size l2)))
  : B.barrier_transform (barrier_contract tile l1 l2 ar1 ar2)
  = let m1 = M.from_array l1 ar1 in
    let m2 = M.from_array l2 ar2 in
    fun (it : nat) ->
      if even it then
        even_barrier_p_to_q m1 m2 it
      else
        odd_barrier_p_to_q m1 m2 it
#pop-options

(* without shmem ownership *)
unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #k #n : sz)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt tile)
  : slprop
  =
  gA |-> Frac (fA /. ((m * n) * tile)) eA **
  gB |-> Frac (fB /. ((m * n) * tile)) eB **
  forall+ (ii : natlt tile).
   (exists* v.
     M.pts_to_cell
      (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v)

unfold
let kpost1
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #k #n : sz)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : ematrix _ _ _)
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt tile)
  : slprop
  =
  let mrow = bid / n in
  let mcol = bid % n in
  gA |-> Frac (fA /. ((m * n) * tile)) eA **
  gB |-> Frac (fB /. ((m * n) * tile)) eB **
  forall+ (ii : natlt tile).
    exists* (v : et).
      Cell (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) |-> v **
      pure (v %~ MS.gemm_single comb_r rA rB rC (mrow * tile + ii) (mcol * tile + tid))

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #k #n : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (m * n))
  (tid : natlt tile)
  : slprop
  =
  kpre1 comb tile gA gB gC eA eB fA fB bid tid **
  live_c_shmems sh #(1.0R /. tile)

let kpre_block_sendable
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #k #n : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (i:natlt (m * n))
  (j:natlt tile)
: is_send_across block_of (kpre comb tile slA slB gA gB gC eA eB fA fB sh i j)
= solve

unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #k #n : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (eA eB : ematrix _ _ _)
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (m * n))
  (tid : natlt tile)
  : slprop
  =
  kpost1 comb comb_r tile gA gB gC eA eB rA rB rC fA fB bid tid **
  live_c_shmems sh #(1.0R /. tile)

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #k #n : sz)
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et tile))
  (_:squash (c_shmems_inv sh))
  (i:natlt (m * n))
  (j:natlt tile)
: is_send_across block_of (kpost comb comb_r tile slA slB gA gB gC eA eB rA rB rC fA fB sh i j)
= solve

let block_pre_sendable
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #k #n : sz)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (fA fB : perm)
  (bid : natlt (m * n))
: is_send_across gpu_of (forall+ (tid : natlt tile). kpre1 comb tile gA gB gC eA eB fA fB bid tid)
= solve

let block_post_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #k #n : sz)
  (tile : valid_tile)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (eA eB : ematrix _ _ _)
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (fA fB : perm)
  (bid : natlt (m * n))
: is_send_across gpu_of (forall+ (tid : natlt tile). kpost1 comb comb_r tile gA gB gC eA eB rA rB rC fA fB bid tid)
= solve

inline_for_extraction noextract
fn bring_2cols
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (#m #k #n : erased nat)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (#l1 #l2 : M.layout tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (sa1 : array2 et l1) (sa2 : array2 et l2)
  (mm : szlt m)
  (kk : szlt k)
  (nn : szlt n)
  (tid : szlt tile)
  (#fA #fB : perm)
  (#eA #eB : ematrix _ _ _)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  (* Should have stronger spec. *)
  requires
    own_1_col sa1 tid **
    own_1_col sa2 tid
  ensures
    own_1_col sa1 tid **
    own_1_col sa2 tid
{
  let mut i = 0sz;
  while (!i <^ tile)
    invariant live i
    decreases (tile - !i)
  {
    {
      unfold own_1_col sa1 tid;
      forevery_extract #(natlt tile) !i _;
      let tileA = array2_extract_tile_ro' gA (SZ.v tile) (SZ.v tile) (SZ.v mm) (SZ.v kk);
      let v1 = M.read tileA (!i, tid);
      M.write_cell sa1 (!i, tid) v1;
      ambig_trade_elim ();
      ambig_trade_elim ();
      fold own_1_col sa1 tid;
    };

    {
      unfold own_1_col sa2 tid;
      forevery_extract #(natlt tile) !i _;
      let tileB = array2_extract_tile_ro' gB (SZ.v tile) (SZ.v tile) (SZ.v kk) (SZ.v nn);
      let v2 = M.read tileB (!i, tid);
      M.write_cell sa2 (!i, tid) v2;
      ambig_trade_elim ();
      ambig_trade_elim ();
      fold own_1_col sa2 tid;
    };

    i := !i +^ 1sz;
  }
}

inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 : M.layout tile tile) (#l2 : M.layout tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (m1 : array2 et l1)
  (m2 : array2 et l2)
  (j : szlt tile)
  (#acc0 : erased (lseq et tile))
  (#v1 #v2 : ematrix et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f v1 **
    m2 |-> Frac f v2
  requires
    acc |-> acc0
  ensures
    exists* (acc' : lseq et tile).
      acc |-> acc'
{
  pts_to_len acc;
  let mut sk : sz = 0sz;
  while (!sk <^ tile)
    invariant live sk ** live acc
    decreases (tile - !sk)
  {
    let mut i = 0sz;
    (* We can read v2 out of the inner loop, this is extremely
       important for performance. NVCC may realize this is invariant
       across iterations and hoist it out, but don't rely on it. *)
    let v2 = M.read m2 (!sk, j);
    while (!i <^ tile)
      invariant live i ** live acc
      decreases (tile - !i)
    {
      let v1 = M.read m1 (!i, !sk);

      open Pulse.Lib.Array;
      pts_to_len acc;
      acc.(!i) <- acc.(!i) `add` (v1 `mul` v2);
      i := !i +^ 1sz;
    };
    sk := !sk +^ 1sz;
  };
  pts_to_len acc;
}

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
fn kf
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #k #n : sz)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (#fA : perm)
  (gB : array2 et lB)
  (#fB : perm)
  (gC : array2 et lC)
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (_sq : squash (eA %~ rA /\ eB %~ rB))
  (sh : c_shmems (shmems_desc et tile))
  (bid : szlt (m * n))
  (tid : szlt tile)
  ()
  norewrite
  requires
    gpu **
    kpre comb tile slA slB gA gB gC eA eB fA fB sh bid tid **
    thread_id tile tid **
    block_id (m * n) bid **
    B.barrier_tok (barrier_contract tile slA slB (fst sh) (fst (snd sh))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost comb comb_r tile slA slB gA gB gC eA eB rA rB rC fA fB sh bid tid **
    thread_id tile tid **
    block_id (m * n) bid **
    B.barrier_tok (barrier_contract tile slA slB (fst sh) (fst (snd sh))) **
    B.barrier_state (2 * k)
{
  unfold_c_shmems sh #(1.0R /. Real.of_int (v tile)) (`%shmems_desc);
  let (ar1, (ar2, _)) = sh;

  gpu_pts_to_ref ar1;
  gpu_pts_to_ref ar2;

  M.raise' slA ar1;
  let sa1 = M.from_array slA ar1;
  rewrite each M.from_array slA ar1 as sa1;

  M.raise' slB ar2;
  let sa2 = M.from_array slB ar2;
  rewrite each M.from_array slB ar2 as sa2;

  let mrow, mcol = s_divmod n bid;
  let bcol = tid;
  assert rewrites_to bcol tid;

  (* thread-local result cache *)
  let mut sums : Pulse.Lib.Array.array et = [| zero ; tile |];
  let mut bk = 0sz;

  while (!bk <^ k)
    invariant live sums
    invariant live bk ** pure (!bk <= k) ** B.barrier_state (2 * !bk)
    invariant
        (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
        (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. tile) x)
    decreases (k - !bk)
  {
    pts_to_len sums;

    // Odd.. we have to go via this intermediate rewrite.
    rewrite (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
            (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. tile) x)
         as barrier_p sa1 sa2 (2 * !bk) tid;
    rewrite barrier_p sa1 sa2 (2 * !bk) tid
         as (barrier_contract tile slA slB ar1 ar2).rin (2 * !bk) tid;
    B.barrier_wait ();
    rewrite (barrier_contract tile slA slB ar1 ar2).rout (2 * !bk) tid
         as own_1_col sa1 tid ** own_1_col sa2 tid;

    (* At this point we exclusively own a full column of the SHMEM
       cache. Populate it. *)
    bring_2cols tile gA gB sa1 sa2 mrow !bk mcol tid;

    rewrite own_1_col sa1 tid ** own_1_col sa2 tid
        as (barrier_contract tile slA slB ar1 ar2).rin (2 * !bk + 1) tid;
    B.barrier_wait ();
    // Same odd thing.
    rewrite (barrier_contract tile slA slB ar1 ar2).rout (2 * !bk + 1) tid
         as barrier_q sa1 sa2 (2 * !bk + 1) tid;
    rewrite barrier_q sa1 sa2 (2 * !bk + 1) tid
        as (exists* (x : ematrix _ _ _). sa1 |-> Frac (1.0R /. tile) x) **
           (exists* (x : ematrix _ _ _). sa2 |-> Frac (1.0R /. tile) x);

    (* At this point the SHMem cache is filled with the submatrices
       and we have RO permission to it. Compute product for our cell in
       the tile and add to sum. *)
    subproduct_cols tile sums sa1 sa2 bcol;

    (* Move to next tile *)
    bk := !bk +^ 1sz;
  };

  (* Write all the accumulated sums. *)

  let tileC = array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n);
  rewrite each array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n) as tileC;

  let mut row : sz = 0sz;
  pts_to_len sums;
  while (!row <^ tile)
    invariant live row ** live sums
    decreases (tile - !row)
  {
    pts_to_len sums;
    forevery_extract #(natlt tile) (!row) _;

    (* tedious, and very sad that tuple syntax does not work (even if we ascribe
    the components, F* infers the left type to be 'nat', which is wrong. *)
    with v0.
      rewrite M.pts_to_cell tileC (Mktuple2 #(natlt tile) #(natlt tile) !row tid) v0
           as M.pts_to_cell tileC (Mktuple2 #(natlt tile) #(natlt tile) !row bcol) v0;

    let v0 = M.read_cell tileC (!row, bcol);
    open Pulse.Lib.Array;
    let v1 = sums.(!row);
    let v' = comb v0 v1;
    M.write_cell tileC (!row, bcol) v';

    row := !row +^ 1sz;
    Pulse.Lib.Trade.elim_trade _ _;
  };

  M.lower sa1; rewrite each M.core sa1 as ar1;
  M.lower sa2; rewrite each M.core sa2 as ar2;

  rewrite each ar1 as fst sh;
  rewrite each ar2 as fst (snd sh);
  rewrite each tileC as array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n);
  fold_c_shmems sh #(1.0R /. Real.of_int (v tile)) (`%shmems_desc);

  (* Functional correctness assumption (cf. SHMem.fst line 363).
     The accumulated subproduct_cols results, combined with the old cell values,
     approximate the real gemm specification. *)
  forevery_map
    (fun (ii : natlt tile) ->
      exists* (v : et).
        M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, SZ.v tid) v)
    (fun (ii : natlt tile) ->
      exists* (v : et).
        M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, SZ.v tid) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC (bid / n * tile + ii) (bid % n * tile + tid)))
    fn ii {
      with v. assert (M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, SZ.v tid) v);
      assume pure (v %~ MS.gemm_single comb_r rA rB rC (bid / n * tile + ii) (bid % n * tile + tid));
    };
}
#pop-options

ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #k #n : szp)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#eA #eB #eC : ematrix et _ _)
  (#fA #fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt2 m n)
             (tid : natlt tile).
      kpre1 comb tile gA gB gC eA eB fA fB bid tid) **
    emp (* frame *)
{
  (* Step 1: Share gA/gB, explode gC *)
  M.share_n gA ((m * n) * tile);
  M.share_n gB ((m * n) * tile);
  array2_explode_tiled gC tile tile;
  forevery_rw_size4 ((m * tile) / tile) m ((n * tile) / tile) n (SZ.v tile) tile (SZ.v tile) tile;

  (* Step 2: Swap inner (i,j) → (j,i) via flatten/mid_flip/unflatten *)
  forevery_flatten
    (fun (tr:natlt m) (tc:natlt n) ->
      forall+ (i:natlt tile) (j:natlt tile).
        M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) tr tc) (i, j)
          (macc eC (tr * tile + i) (tc * tile + j)));

  forevery_mid_flip
    (fun (trtc:natlt m & natlt n) (i:natlt tile) (j:natlt tile) ->
      M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) trtc._1 trtc._2) (i, j)
        (macc eC (trtc._1 * tile + i) (trtc._2 * tile + j)));

  (* Step 3: Introduce exists* via nested forevery_map *)
  forevery_map
    (fun (trtc:natlt m & natlt n) ->
      forall+ (j:natlt tile) (i:natlt tile).
        M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) trtc._1 trtc._2) (i, j)
          (macc eC (trtc._1 * tile + i) (trtc._2 * tile + j)))
    (fun (trtc:natlt m & natlt n) ->
      forall+ (j:natlt tile) (i:natlt tile).
        exists* v. M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) trtc._1 trtc._2) (i, j) v)
    fn trtc {
      forevery_map_2
        (fun (j:natlt tile) (i:natlt tile) ->
          M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) trtc._1 trtc._2) (i, j)
            (macc eC (trtc._1 * tile + i) (trtc._2 * tile + j)))
        (fun (j:natlt tile) (i:natlt tile) ->
          exists* v. M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) trtc._1 trtc._2) (i, j) v)
        fn j i { (); };
    };

  (* Step 4: Unflatten back to (tr, tc, j, i) *)
  forevery_unflatten
    (fun (tr:natlt m) (tc:natlt n) ->
      forall+ (j:natlt tile) (i:natlt tile).
        exists* v. M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) tr tc) (i, j) v);

  (* Step 5: Collapse (tr, tc) → bid *)
  forevery_unfactor' (m * n) m n
    (fun (tr:natlt m) (tc:natlt n) ->
      forall+ (j:natlt tile) (i:natlt tile).
        exists* v. M.pts_to_cell (array2_subtile gC (SZ.v tile) (SZ.v tile) tr tc) (i, j) v);

  (* Step 6: Factor gA/gB to 2D *)
  forevery_factor ((m * n) * tile) (m * n) tile
    (fun _ -> gA |-> Frac (fA /. ((m * n) * tile)) eA);
  forevery_factor ((m * n) * tile) (m * n) tile
    (fun _ -> gB |-> Frac (fB /. ((m * n) * tile)) eB);

  (* Step 7: Zip gA, gB, gC *)
  forevery_zip3_2
    (fun (_ : natlt (m * n)) (_ : natlt tile) ->
      gA |-> Frac (fA /. ((m * n) * tile)) eA)
    (fun (_ : natlt (m * n)) (_ : natlt tile) ->
      gB |-> Frac (fB /. ((m * n) * tile)) eB)
    (fun (bid : natlt (m * n)) (tid : natlt tile) ->
      forall+ (ii : natlt tile).
        exists* v. M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v);

  (* Step 8: Bridge to natlt2 and match kpre1 *)
  forevery_rw_size2 (m * n) (SZ.v (m `SZ.mul` n)) tile tile;
  forevery_ext_2
    (fun (bid : natlt (SZ.v (m `SZ.mul` n))) (tid : natlt tile) ->
      gA |-> Frac (fA /. ((m * n) * tile)) eA **
      gB |-> Frac (fB /. ((m * n) * tile)) eB **
      forall+ (ii : natlt tile).
        exists* v. M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v)
    (fun (bid : natlt2 m n) (tid : natlt tile) ->
      kpre1 comb tile gA gB gC eA eB fA fB bid tid);
  ();
}

ghost
fn block_setup
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #k #n : szp)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (#fA : perm)
  (gB : array2 et lB)
  (#fB : perm)
  (gC : array2 et lC)
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (#eC : ematrix et (m * tile) (n * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (m * n))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt tile).
      kpre1 comb tile gA gB gC eA eB fA fB bid tid)
  ensures
    (forall+ (tid : natlt tile).
      kpre comb tile slA slB gA gB gC eA eB fA fB sh bid tid) **
    emp (* frame *)
{
  gpu_live_c_shmems_share_underspec sh #1.0R #tile;
  forevery_zip
    (fun (tid : natlt tile) -> kpre1 comb tile gA gB gC eA eB fA fB bid tid)
    _;
}

ghost
fn block_teardown
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #k #n : SZ.t)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (#fA : perm)
  (gB : array2 et lB)
  (#fB : perm)
  (gC : array2 et lC)
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (#eC : ematrix et (m * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (sh : c_shmems (shmems_desc et tile))
  (bid : natlt (m * n))
  ()
  norewrite
  requires
    (forall+ (tid : natlt tile).
      kpost comb comb_r tile slA slB gA gB gC eA eB rA rB rC fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt tile).
      kpost1 comb comb_r tile gA gB gC eA eB rA rB rC fA fB bid tid)
{
  forevery_unzip
    (fun (tid : natlt tile) -> kpost1 comb comb_r tile gA gB gC eA eB rA rB rC fA fB bid tid)
    _;
  gpu_live_c_shmems_gather_underspec sh #1.0R #tile;
}

#push-options "--z3rlimit 160 --fuel 1 --ifuel 1"
ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #k #n : szp)
  (#lA : M.layout (m   * tile) (k * tile))
  (#lB : M.layout (k * tile) (n   * tile))
  (#lC : M.layout (m   * tile) (n   * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA)
  (#fA : perm)
  (gB : array2 et lB)
  (#fB : perm)
  (gC : array2 et lC)
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 m n)
             (tid : natlt tile).
      kpost1 comb comb_r tile gA gB gC eA eB rA rB rC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : ematrix et (m * tile) (n * tile)).
      gC |-> eC' **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  let n_threads = (m * n) * tile;

  (* Step 1: Bridge natlt2 → natlt *)
  forevery_rw_size2 (SZ.v (m `SZ.mul` n)) (m * n) tile tile;

  (* Step 2: Unfold kpost1 *)
  forevery_ext_2
    (fun (bid : natlt (m * n)) (tid : natlt tile) ->
      kpost1 comb comb_r tile gA gB gC eA eB rA rB rC fA fB bid tid)
    (fun (bid : natlt (m * n)) (tid : natlt tile) ->
      let mrow = bid / n in
      let mcol = bid % n in
      gA |-> Frac (fA /. n_threads) eA **
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (ii : natlt tile).
        exists* (v : et).
          M.pts_to_cell
            (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v **
          pure (v %~ MS.gemm_single comb_r rA rB rC (mrow * tile + ii) (mcol * tile + tid)));

  (* Step 3: Unzip gA *)
  forevery_unzip_2
    (fun (_ : natlt (m * n)) (_ : natlt tile) ->
      gA |-> Frac (fA /. n_threads) eA)
    (fun (bid : natlt (m * n)) (tid : natlt tile) ->
      let mrow = bid / n in
      let mcol = bid % n in
      gB |-> Frac (fB /. n_threads) eB **
      forall+ (ii : natlt tile).
        exists* (v : et).
          M.pts_to_cell
            (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v **
          pure (v %~ MS.gemm_single comb_r rA rB rC (mrow * tile + ii) (mcol * tile + tid)));

  (* Step 4: Unzip gB *)
  forevery_unzip_2
    (fun (_ : natlt (m * n)) (_ : natlt tile) ->
      gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (m * n)) (tid : natlt tile) ->
      let mrow = bid / n in
      let mcol = bid % n in
      forall+ (ii : natlt tile).
        exists* (v : et).
          M.pts_to_cell
            (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v **
          pure (v %~ MS.gemm_single comb_r rA rB rC (mrow * tile + ii) (mcol * tile + tid)));

  (* Step 5: Gather gA and gB *)
  forevery_unfactor' n_threads (m * n) tile
    (fun (_ : natlt (m * n)) (_ : natlt tile) ->
      gA |-> Frac (fA /. n_threads) eA);
  M.gather_n gA n_threads;

  forevery_unfactor' n_threads (m * n) tile
    (fun (_ : natlt (m * n)) (_ : natlt tile) ->
      gB |-> Frac (fB /. n_threads) eB);
  M.gather_n gB n_threads;

  (* Step 6: Swap (tid, ii) → (ii, tid) via mid_flip *)
  forevery_mid_flip
    (fun (bid : natlt (m * n)) (tid : natlt tile) (ii : natlt tile) ->
      exists* (v : et).
        M.pts_to_cell
          (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v **
        pure (v %~ MS.gemm_single comb_r rA rB rC
          (bid / n * tile + ii) (bid % n * tile + tid)));

  (* Step 7: Collapse inner (ii, tid) → flatid via unfactor' per bid *)
  forevery_map
    (fun (bid : natlt (m * n)) ->
      forall+ (ii : natlt tile) (tid : natlt tile).
        exists* (v : et).
          M.pts_to_cell
            (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v **
          pure (v %~ MS.gemm_single comb_r rA rB rC
            (bid / n * tile + ii) (bid % n * tile + tid)))
    (fun (bid : natlt (m * n)) ->
      forall+ (flatid : natlt (tile * tile)).
        exists* (v : et).
          M.pts_to_cell
            (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n))
            ((flatid / tile <: natlt tile), (flatid % tile <: natlt tile)) v **
          pure (v %~ MS.gemm_single comb_r rA rB rC (bid / n * tile + flatid / tile) (bid % n * tile + flatid % tile)))
    fn bid {
      forevery_unfactor' (tile * tile) tile tile
        (fun (ii : natlt tile) (tid : natlt tile) ->
          exists* (v : et).
            M.pts_to_cell
              (array2_subtile gC (SZ.v tile) (SZ.v tile) (bid / n) (bid % n)) (ii, tid) v **
            pure (v %~ MS.gemm_single comb_r rA rB rC
              (bid / n * tile + ii) (bid % n * tile + tid)));
    };

  (* Step 8: Collect cells via gpu_matrix_collect_approx_tiled *)
  Kuiper.Tensor.Tiling.CollectApprox.array2_collect_approx_tiled gC (SZ.v tile) (SZ.v tile)
    m n
    (fun (row : natlt (m * tile)) (col : natlt (n * tile)) (v : et) ->
      v %~ MS.gemm_single comb_r rA rB rC row col);

  with eC'. assert (gC |-> eC');

  assert pure (forall (row:natlt (m * tile)) (col:natlt (n * tile)).
    macc eC' row col %~ MS.gemm_single comb_r rA rB rC row col);

  assert pure (forall (row:natlt (m * tile)) (col:natlt (n * tile)).
    macc eC' row col %~ macc (MS.mmcomb comb_r rC rA rB) row col);

  assert pure (eC' %~ MS.mmcomb comb_r rC rA rB);
  ();
}
#pop-options

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (slA slB : M.full_layout tile tile) // shmem layouts
  {| T.ctlayout slA, T.ctlayout slB |}
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #k #n : szp)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (#eC : ematrix et (m * tile) (n * tile))
  (rA : ematrix real (m * tile) (k * tile))
  (rB : ematrix real (k * tile) (n * tile))
  (rC : ematrix real (m * tile) (n * tile))
  (#fA #fB : perm)
  (_ : squash (m * n <= max_blocks
               /\ tile <= max_threads
               /\ eA %~ rA /\ eB %~ rB /\ eC %~ rC))
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : ematrix et (m * tile) (n * tile)).
          gC |-> eC' **
          pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
= {
  nblk = m *^ n;
  nthr = tile;

  barrier_contract = (fun bid ptrs -> barrier_contract tile slA slB (fst ptrs) (fst (snd ptrs)));
  barrier_count    = (fun _bid -> 2 * SZ.v k);
  barrier_ok = (fun _bid ptrs -> barrier_p_to_q_transform slA slB (fst ptrs) (fst (snd ptrs)));

  shmems_desc = shmems_desc et tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt tile). kpre1  comb tile gA gB gC eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt tile). kpost1 comb comb_r tile gA gB gC eA eB rA rB rC fA fB bid tid);
  setup      = setup    tile comb gA gB gC;
  teardown   = teardown tile comb comb_r gA gB gC rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    tile slA slB comb gA gB gC #_ #_ #eC;
  block_teardown = block_teardown tile slA slB comb comb_r gA gB gC #_ #_ #eC rA rB rC;

  kpre      = kpre  comb tile slA slB gA gB gC eA eB fA fB;
  kpost     = kpost comb comb_r tile slA slB gA gB gC eA eB rA rB rC fA fB;

  f = kf tile slA slB comb comb_r gA gB gC rA rB rC () ;
  kpost_sendable = kpost_block_sendable comb comb_r tile slA slB gA gB gC eA eB rA rB rC fA fB;
  kpre_sendable = kpre_block_sendable comb tile slA slB gA gB gC eA eB fA fB;
  block_post_sendable = block_post_sendable comb comb_r tile gA gB gC eA eB rA rB rC fA fB;
  block_pre_sendable = block_pre_sendable comb tile gA gB gC eA eB fA fB;
}

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : M.layout (m * tile) (k * tile))
  (#lB : M.layout (k * tile) (n * tile))
  (#lC : M.layout (m * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : array2 et lA { M.is_global gA })
  (gB : array2 et lB { M.is_global gB })
  (gC : array2 et lC { M.is_global gC })
  (rA  : ematrix real (m * tile) (k * tile))
  (rB  : ematrix real (k * tile) (n * tile))
  (rC  : ematrix real (m * tile) (n * tile))
  (#eA : ematrix et (m * tile) (k * tile))
  (#eB : ematrix et (k * tile) (n * tile))
  (#eC : ematrix et (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  open Kuiper.Tensor.Layout.Alg;
  dassert (tile >^ 0sz);
  launch_sync (mk_kernel tile (l2_row_major _ _) (l2_row_major _ _) comb comb_r gA gB gC rA rB rC ());
}
