module Kuiper.Kernel.GEMM.Tiled

#set-options "--z3rlimit 20"
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Bijection
open Kuiper.Injection

module Chest = Kuiper.Chest
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT

open Kuiper.Shape { shape, abs, conc, all_fit, ( @! ) }
open Kuiper.Chest { chest, chest_slice }

inline_for_extraction noextract
fn tensor_extract_slice_ro'
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : enatlt r) (j : enatlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  returns
    a' : tensor et (tlayout_slice l i j)
  ensures
    factored
      (a' |-> Frac f (chest_slice i j s))
      (a |-> Frac f s) **
    rewrites_to a' (sliceof a i j)
{
  tensor_extract_slice_ro a i j;
  (sliceof a i j);
}

open Kuiper.Shape { ( @| ) }
let chest_flat42
  (#et : Type)
  (#d1 #d2 #d3 #d4 : nat)
  (x : chest4 et d1 d2 d3 d4) : chest2 et (d1 * d3) (d2 * d4) =
  Kuiper.Chest.mk (d1 * d3 @| d2 * d4 @| INil) fun (i, (j, ())) ->
    let i1 : natlt d1 = i / d3 in
    let i2 : natlt d3 = i % d3 in
    let j1 : natlt d2 = j / d4 in
    let j2 : natlt d4 = j % d4 in
    Kuiper.Chest.acc x (i1, (j1, (i2, (j2, ()))))

(* Reading a cell of the flattened (rank-2) view in terms of the underlying
   rank-4 chest. *)
let chest_flat42_cell
  (#et : Type)
  (#d1 #d2 #d3 #d4 : nat)
  (x : chest4 et d1 d2 d3 d4)
  (i1 : natlt d1) (i2 : natlt d3) (j1 : natlt d2) (j2 : natlt d4)
  : Lemma (Chest.acc (chest_flat42 x) (i1 * d3 + i2, (j1 * d4 + j2, ()))
           == Chest.acc x (i1, (j1, (i2, (j2, ())))))
  = ()

(* The (bi,bk) tile of the flattened rank-2 view is exactly the doubly-sliced
   rank-4 chest (both index x at [bi][bk][a][b]). *)
let slice2_eq_subtile
  (#et : Type)
  (#m #k #tile : nat)
  (x : chest4 et m k tile tile)
  (bi : natlt m) (bk : natlt k)
  : Lemma (requires tile > 0)
          (ensures
            chest_slice 0 bk (chest_slice 0 bi x)
            == ematrix_subtile (chest_flat42 x) tile tile bi bk)
  = introduce forall (idx : abs (tile @| tile @| INil)).
      Chest.acc (chest_slice 0 bk (chest_slice 0 bi x)) idx
      == Chest.acc (ematrix_subtile (chest_flat42 x) tile tile bi bk) idx
    with (
      let (a, (b, ())) = idx in
      chest_flat42_cell x bi a bk b
    );
    Chest.lemma_equal_intro
      (chest_slice 0 bk (chest_slice 0 bi x))
      (ematrix_subtile (chest_flat42 x) tile tile bi bk);
    Chest.ext
      (chest_slice 0 bk (chest_slice 0 bi x))
      (ematrix_subtile (chest_flat42 x) tile tile bi bk)

(* Subtiling the flattened views preserves the approximation relation. *)
let subtile_flat_approx
  (#et : Type) {| scalar et, real_like et |}
  (#m #k #tile : nat)
  (eA : chest4 et m k tile tile)
  (rA : chest4 real m k tile tile)
  (bi : natlt m) (bk : natlt k)
  : Lemma (requires eA %~ rA /\ tile > 0)
          (ensures ematrix_subtile (chest_flat42 eA) tile tile bi bk
                   %~ ematrix_subtile (chest_flat42 rA) tile tile bi bk)
  = introduce forall (idx : abs (tile @| tile @| INil)).
      Chest.acc (ematrix_subtile (chest_flat42 eA) tile tile bi bk) idx
      %~ Chest.acc (ematrix_subtile (chest_flat42 rA) tile tile bi bk) idx
    with (
      let (a, (b, ())) = idx in
      chest_flat42_cell eA bi a bk b;
      chest_flat42_cell rA bi a bk b
    )

(* ─── array2 <-> rank-4 tensor "tiling view" reinterpretation ───────────────
   A flat row/col-major (m*tile) x (k*tile) matrix shares its underlying array
   with a rank-4 (m, k, tile, tile) tensor whose layout realizes the tiling.
   These let mmcomb_gpu_approx run the rank-4 kernel over a flat array2. *)

(* Tiling injection: rank-4 (mr,kc,a,b) -> rank-2 flat (mr*tile+a, kc*tile+b). *)
let tile_reshape_f (m k tile : nat)
  : abs (m @| k @| tile @| tile @| INil) -> abs ((m * tile) @| (k * tile) @| INil)
  = fun (mr, (kc, (a, (b, ())))) -> (mr * tile + a, (kc * tile + b, ()))

let tile_reshape_inj (m k tile : nat)
  : (abs (m @| k @| tile @| tile @| INil) @~> abs ((m * tile) @| (k * tile) @| INil))
  = mk_injection (tile_reshape_f m k tile)
      (fun (mr1, (kc1, (a1, (b1, ())))) (mr2, (kc2, (a2, (b2, ())))) -> ())

(* The rank-4 tiled layout over the same underlying array as a flat layout. *)
let tile4_layout (#m #k #tile : nat) (l : tlayout ((m * tile) @| (k * tile) @| INil))
  : layout4 m k tile tile
  = { ulen = l.ulen;
      imap = inj_comp (tile_reshape_inj m k tile) l.imap; }

let tile4_all_fit (m k : nat) (tile : nat)
  : Lemma (requires SZ.fits (m * tile) /\ SZ.fits (k * tile) /\ m > 0 /\ k > 0 /\ tile > 0)
          (ensures all_fit (m @| k @| tile @| tile @| INil))
  = FStar.Math.Lemmas.lemma_mult_le_right tile 1 m;
    FStar.Math.Lemmas.lemma_mult_le_right tile 1 k;
    FStar.Math.Lemmas.lemma_mult_le_left m 1 tile;
    FStar.Math.Lemmas.lemma_mult_le_left k 1 tile

#push-options "--split_queries always --z3rlimit 40"
inline_for_extraction noextract
let c_tile4_layout
  (#m #k : erased pos) (tile : szp)
  (#l : tlayout ((m * tile) @| (k * tile) @| INil))
  {| cc : ctlayout l |}
  : ctlayout (tile4_layout #m #k #tile l)
  = let _ = cc.ulen_fits in
    let _ = cc.all_fit in
    tile4_all_fit m k (SZ.v tile);
    {
      ulen_fits = ();
      all_fit = ();
      cimap = (fun (x : conc (m @| k @| tile @| tile @| INil)) ->
                match x with
                | (mr, (kc, (a, (b, ())))) ->
                  cc.cimap (mr *^ tile +^ a, (kc *^ tile +^ b, ())));
  }
#pop-options

(* The rank-4 chest underlying a flat rank-2 matrix; flattening inverts it. *)
let untile4
  (#et : Type) (#m #k #tile : nat)
  (s : chest2 et (m * tile) (k * tile))
  : chest4 et m k tile tile
  = Chest.mk (m @| k @| tile @| tile @| INil)
      (fun (mr, (kc, (a, (b, ())))) -> Chest.acc s (mr * tile + a, (kc * tile + b, ())))

let flat_untile4
  (#et : Type) (#m #k #tile : nat)
  (s : chest2 et (m * tile) (k * tile))
  : Lemma (requires tile > 0)
          (ensures chest_flat42 (untile4 s) == s)
  = introduce forall (idx : abs ((m * tile) @| (k * tile) @| INil)).
      Chest.acc (chest_flat42 (untile4 #et #m #k #tile s)) idx == Chest.acc s idx
    with (let (i, (j, ())) = idx in ());
    Chest.lemma_equal_intro (chest_flat42 (untile4 #et #m #k #tile s)) s;
    Chest.ext (chest_flat42 (untile4 #et #m #k #tile s)) s

let tile_unreshape_f (m k : nat) (tile : nat{tile > 0})
  : abs ((m * tile) @| (k * tile) @| INil) -> abs (m @| k @| tile @| tile @| INil)
  = fun (i, (j, ())) -> ((i / tile <: natlt m), ((j / tile <: natlt k),
                          ((i % tile <: natlt tile), ((j % tile <: natlt tile), ()))))

let reshape_bij (m k : nat) (tile : nat{tile > 0})
  : (abs (m @| k @| tile @| tile @| INil) =~ abs ((m * tile) @| (k * tile) @| INil))
  = {
      ff = tile_reshape_f m k tile;
      gg = tile_unreshape_f m k tile;
      ff_gg = (fun (i, (j, ())) -> ());
      gg_ff = (fun (mr, (kc, (a, (b, ())))) -> ());
    }

(* Reinterpret a flat rank-2 tensor as its rank-4 tiled view (same memory). *)
ghost
fn tensor_to_tile4
  (#et : Type0) (#m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout ((m * tile) @| (k * tile) @| INil))
  (g2 : tensor et l)
  (#f : perm) (#s : chest2 et (m * tile) (k * tile))
  requires
    g2 |-> Frac f s
  returns
    g4 : tensor et (tile4_layout #m #k #tile l)
  ensures
    g4 |-> Frac f (untile4 #et #m #k #tile s) **
    pure (core g4 == core g2 /\ (is_global g4 <==> is_global g2))
{
  tensor_ilower g2;
  let g4 = from_array (tile4_layout #m #k #tile l) (core g2);
  rewrite each (core g2) as (core g4);
  forevery_ext
    (fun (i : abs ((m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g4) #f (l.imap.f i) (Chest.acc s i))
    (fun (i : abs ((m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g4) #f
         ((tile4_layout #m #k #tile l).imap.f ((reshape_bij m k tile).gg i))
         (Chest.acc (untile4 #et #m #k #tile s) ((reshape_bij m k tile).gg i)));
  forevery_iso_back (reshape_bij m k tile)
    (fun (idx4 : abs (m @| k @| tile @| tile @| INil)) ->
       pts_to_cell (core g4) #f ((tile4_layout #m #k #tile l).imap.f idx4)
         (Chest.acc (untile4 #et #m #k #tile s) idx4));
  tensor_iraise g4;
  g4
}

(* Reinterpret a rank-4 tiled tensor back as its flat rank-2 view. *)
ghost
fn tile4_to_tensor
  (#et : Type0) (#m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout ((m * tile) @| (k * tile) @| INil))
  (g4 : tensor et (tile4_layout #m #k #tile l))
  (#f : perm) (#s4 : chest4 et m k tile tile)
  requires
    g4 |-> Frac f s4
  returns
    g2 : tensor et l
  ensures
    g2 |-> Frac f (chest_flat42 #et #m #k #tile s4) **
    pure (core g2 == core g4 /\ (is_global g2 <==> is_global g4))
{
  tensor_ilower g4;
  let g2 = from_array l (core g4);
  rewrite each (core g4) as (core g2);
  forevery_iso (reshape_bij m k tile)
    (fun (idx4 : abs (m @| k @| tile @| tile @| INil)) ->
       pts_to_cell (core g2) #f ((tile4_layout #m #k #tile l).imap.f idx4)
         (Chest.acc s4 idx4));
  forevery_ext
    (fun (idx2 : abs ((m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g2) #f
         ((tile4_layout #m #k #tile l).imap.f ((reshape_bij m k tile).gg idx2))
         (Chest.acc s4 ((reshape_bij m k tile).gg idx2)))
    (fun (idx2 : abs ((m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g2) #f (l.imap.f idx2)
         (Chest.acc (chest_flat42 #et #m #k #tile s4) idx2));
  tensor_iraise g2;
  g2
}

(* untile4 preserves the approximation relation. *)
let untile4_approx
  (#et : Type) {| scalar et, real_like et |}
  (#m #k #tile : nat)
  (e : chest2 et (m * tile) (k * tile))
  (r : chest2 real (m * tile) (k * tile))
  : Lemma (requires e %~ r)
          (ensures untile4 #et #m #k #tile e %~ untile4 #real #m #k #tile r)
  = introduce forall (idx : abs (m @| k @| tile @| tile @| INil)).
      Chest.acc (untile4 #et #m #k #tile e) idx %~ Chest.acc (untile4 #real #m #k #tile r) idx
    with (let (mr, (kc, (a, (b, ())))) = idx in ())

(* Ownership-only forward reshape (unit-returning, for use under map_loc). *)
ghost
fn tile4_fwd
  (#et : Type0) (#m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout ((m * tile) @| (k * tile) @| INil))
  (g2 : tensor et l)
  (#f : perm) (#s : chest2 et (m * tile) (k * tile))
  requires
    g2 |-> Frac f s
  ensures
    from_array (tile4_layout #m #k #tile l) (core g2) |-> Frac f (untile4 #et #m #k #tile s)
{
  let g4 = tensor_to_tile4 tile g2;
  rewrite (g4 |-> Frac f (untile4 #et #m #k #tile s))
       as (from_array (tile4_layout #m #k #tile l) (core g2) |-> Frac f (untile4 #et #m #k #tile s));
}

(* Ownership-only backward reshape. *)
ghost
fn tile4_bwd
  (#et : Type0) (#m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout ((m * tile) @| (k * tile) @| INil))
  (g4 : tensor et (tile4_layout #m #k #tile l))
  (#f : perm) (#s4 : chest4 et m k tile tile)
  requires
    g4 |-> Frac f s4
  ensures
    from_array l (core g4) |-> Frac f (chest_flat42 #et #m #k #tile s4)
{
  let g2 = tile4_to_tensor tile g4;
  rewrite (g2 |-> Frac f (chest_flat42 #et #m #k #tile s4))
       as (from_array l (core g4) |-> Frac f (chest_flat42 #et #m #k #tile s4));
}

(* Move away somewhere, this is generic. *)
#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn matmul_tiled_dotprod_real
  (#et : Type0) {| scalar et, real_like et |}
  (#m #n #k : sz)
  (#tile : szp)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (#eA #eB : chest4 _ _ _ _ _)
  (rA rB : chest4 _ _ _ _ _)
  (bi : szlt m)
  (bj : szlt n)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB
  requires
    pure (eA %~ rA /\ eB %~ rB)
  returns
    res : et
  ensures
    pure (res %~ MS.matmul_single (chest_flat42 rA) (chest_flat42 rB) (bi * tile + i) (bj * tile + j))
{
  let grow : erased (natlt (m * tile)) = hide (bi * tile + i);
  let gcol : erased (natlt (n * tile)) = hide (bj * tile + j);

  let mut sum : et = zero;
  let mut bk  : szle k = 0sz;

  while (!bk <^ k)
    invariant live bk ** live sum
    invariant pure
      (!sum %~ MS.__matmul_single (chest_flat42 rA) (chest_flat42 rB) grow gcol (SZ.v !bk * tile))
    decreases (k - !bk)
  {
    let abi   = tensor_extract_slice_ro' gA  0 (SZ.v bi);
    let abibk = tensor_extract_slice_ro' abi 0 (SZ.v !bk);

    let bbk   = tensor_extract_slice_ro' gB  0 (SZ.v !bk);
    let bbkbj = tensor_extract_slice_ro' bbk 0 (SZ.v bj);

    let s' = Kuiper.DotProd.matmul_dotprod #_ #_ #_ #_ #tile abibk bbkbj i j;

    ambig_trade_elim ();
    ambig_trade_elim ();
    ambig_trade_elim ();
    ambig_trade_elim ();

    let s = !sum;

    sum := s `add` s';

    (* The (bi,bk)/(bk,bj) tiles seen through the slices coincide with the
       subtiles of the flattened views; subtiling preserves %~. *)
    slice2_eq_subtile eA (SZ.v bi) (SZ.v !bk);
    slice2_eq_subtile eB (SZ.v !bk) (SZ.v bj);
    subtile_flat_approx eA rA (SZ.v bi) (SZ.v !bk);
    subtile_flat_approx eB rB (SZ.v !bk) (SZ.v bj);

    (* s' approximates the real matmul over the subtiles. *)
    MU.__matmul_single_approx_real
      (ematrix_subtile (chest_flat42 eA) tile tile (SZ.v bi) (SZ.v !bk))
      (ematrix_subtile (chest_flat42 eB) tile tile (SZ.v !bk) (SZ.v bj))
      (ematrix_subtile (chest_flat42 rA) tile tile (SZ.v bi) (SZ.v !bk))
      (ematrix_subtile (chest_flat42 rB) tile tile (SZ.v !bk) (SZ.v bj))
      (SZ.v i) (SZ.v j) (SZ.v tile);

    (* The partial real matmul splits over the next tile. *)
    MU.__gmatmul_single_split
      (chest_flat42 rA) (chest_flat42 rB) grow gcol (SZ.v !bk * tile) (SZ.v tile)
      (ematrix_subtile (chest_flat42 rA) tile tile (SZ.v bi) (SZ.v !bk))
      (ematrix_subtile (chest_flat42 rB) tile tile (SZ.v !bk) (SZ.v bj))
      (SZ.v i) (SZ.v j);

    FStar.Math.Lemmas.distributivity_add_left (SZ.v !bk) 1 (SZ.v tile);
    assert (pure ((SZ.v !bk + 1) * SZ.v tile == SZ.v !bk * SZ.v tile + SZ.v tile));

    bk := !bk +^ 1sz;

    ()
  };

  !sum
}
#pop-options

unfold
let kpre
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA eB : chest4 et _ _ _ _)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  : slprop
  =
  let mrow = bid / n in
  let mcol = bid % n in
  let brow = tid / tile in
  let bcol = tid % tile in
  pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
  gA |-> Frac (fA /. ((m * n) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((m * n) * (tile * tile))) eB **
  tensor_pts_to_cell
    gC
    (mrow, (mcol, (brow, (bcol, ()))))
    (Chest.acc eC (mrow, (mcol, (brow, (bcol, ())))))

(* Functional postcondition: the cell contains a value approximating
   MS.gemm_single over external real matrices rA, rB, rC *)
unfold
let kpost
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA eB : chest4 et _ _ _ _)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  =
  let mrow = bid / n in
  let mcol = bid % n in
  let brow = tid / tile in
  let bcol = tid % tile in
  pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
  gA |-> Frac (fA /. ((m * n) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((m * n) * (tile * tile))) eB **
  exists* (v : et).
    tensor_pts_to_cell
      gC
      (mrow, (mcol, (brow, (bcol, ()))))
      v **
    pure (v %~ MS.gemm_single comb_r (chest_flat42 rA) (chest_flat42 rB) (chest_flat42 rC)
          (mrow * tile + brow) (mcol * tile + bcol))

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA eB : chest4 et _ _ _ _)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  (bid : szlt (m * n))
  (tid : szlt (tile * tile))
  ()
  norewrite
  requires
    gpu **
    kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (m * n) bid
  ensures
    gpu **
    kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (m * n) bid
{
  let mrow : szlt m = bid /^ n;
  let mcol : szlt n = bid %^ n;
  let brow : szlt tile = tid /^ tile;
  let bcol : szlt tile = tid %^ tile;

  (* Rewrite kpre's cell indices to use brow/bcol (which equal tid/tile, tid%tile) *)
  rewrite
    tensor_pts_to_cell
      gC
      ((bid / n <: natlt m), ((bid % n <: natlt n), ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))
      (Chest.acc eC ((bid / n <: natlt m), ((bid % n <: natlt n), ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))))
  as
    tensor_pts_to_cell
      gC
      ((mrow <: natlt m), ((mcol <: natlt n), ((brow <: natlt tile), ((bcol <: natlt tile), ()))))
      (Chest.acc eC ((mrow <: natlt m), ((mcol <: natlt n), ((brow <: natlt tile), ((bcol <: natlt tile), ())))));

  let s = matmul_tiled_dotprod_real gA gB rA rB mrow mcol brow bcol;

  let v0 = tensor_read_cell gC (mrow, (mcol, (brow, (bcol, ()))));
  let v1 = comb v0 s;
  tensor_write_cell gC (mrow, (mcol, (brow, (bcol, ())))) v1;

  rewrite
    tensor_pts_to_cell
      gC
      ((mrow <: natlt m), ((mcol <: natlt n), ((brow <: natlt tile), ((bcol <: natlt tile), ()))))
      v1
  as
    tensor_pts_to_cell
      gC
      ((bid / n <: natlt m), ((bid % n <: natlt n), ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))
      v1;

  (* The new cell value v1 = comb v0 s approximates gemm_single over the
     flattened real matrices: v0 is the old cell (approximating rC at this
     position), s approximates the matmul, and comb `approx2` comb_r. *)
  chest_flat42_cell rC (mrow <: natlt m) (brow <: natlt tile) (mcol <: natlt n) (bcol <: natlt tile);

  fold kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid;
  ()
}

(* Reshape the abstract rank-4 index into a pair of pairs (pure tuple
   manipulation, no arithmetic). *)
let abs_pairpair_bij (m n p q : nat)
  : (abs (m @| n @| p @| q @| INil) =~ ((natlt m & natlt n) & (natlt p & natlt q))) =
  {
    ff = (fun (a, (b, (c, (d, ())))) -> ((a, b), (c, d)));
    gg = (fun ((a, b), (c, d)) -> (a, (b, (c, (d, ())))));
    ff_gg = (fun ((a, b), (c, d)) -> ());
    gg_ff = (fun (a, (b, (c, (d, ())))) -> ());
  }

(* Full reshape: abstract rank-4 index <-> (block index, thread index).
   gg (bid, tid) = (bid/n, (bid%n, (tid/tile, (tid%tile, ())))), exactly the
   per-thread cell index used by kpre/kpost. *)
let tile_idx_bij (m n tile : nat)
  : (abs (m @| n @| tile @| tile @| INil) =~ (natlt (m * n) & natlt (tile * tile))) =
  bij_comp (abs_pairpair_bij m n tile tile)
           (bij_prod (bij_nat_prod #m #n) (bij_nat_prod #tile #tile))

ghost
fn setup
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest4 et m k tile tile)
  (eB : chest4 et k n tile tile)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC)
  ensures
    (forall+ (bid : natlt (m *^ n))
             (tid : natlt (tile *^ tile)).
      kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp (* frame *)
{
  let n_threads : nat = (m * n) * (tile * tile);

  (* Step 1: Share gA/gB across all threads, explode gC into per-cell. *)
  tensor_share_n gA n_threads;
  tensor_share_n gB n_threads;
  tensor_explode gC;

  (* Step 2: Reshape gC's per-cell forall from abs d4 to (bid, tid). *)
  forevery_iso (tile_idx_bij m n tile)
    (fun (idx : abs (m @| n @| tile @| tile @| INil)) ->
       tensor_pts_to_cell gC idx (Chest.acc eC idx));
  forevery_unflatten' _;
  forevery_ext_2 _
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
       tensor_pts_to_cell gC
         ((bid / n <: natlt m), ((bid % n <: natlt n),
           ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))
         (Chest.acc eC
           ((bid / n <: natlt m), ((bid % n <: natlt n),
             ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))));

  (* Step 3: Factor shared inputs into (bid, tid). *)
  forevery_factor n_threads (m * n) (tile * tile) (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (m * n) (tile * tile) (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  (* Step 4: Duplicate the pure approximation facts into the forall+. *)
  forevery_intro_pure_2 (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
    eA %~ rA /\ eB %~ rB /\ eC %~ rC);

  (* Step 5: Zip all four kpre components together. *)
  forevery_zip4_2
    #(natlt (m * n)) #(natlt (tile * tile))
    (fun bid tid -> pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC))
    (fun bid tid -> gA |-> Frac (fA /. n_threads) eA)
    (fun bid tid -> gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
       tensor_pts_to_cell gC
         ((bid / n <: natlt m), ((bid % n <: natlt n),
           ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))
         (Chest.acc eC
           ((bid / n <: natlt m), ((bid % n <: natlt n),
             ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))));

  (* Final ext match + size equality. *)
  forevery_ext_2 _
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
       kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  forevery_rw_size2
    (m * n) (SZ.v (m *^ n))
    (tile * tile) (SZ.v (tile *^ tile));
  ();
}

#push-options "--z3rlimit 80 --ifuel 5"
ghost
fn teardown
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest4 et m k tile tile)
  (eB : chest4 et k n tile tile)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (m *^ n))
             (tid : natlt (tile *^ tile)).
      kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : chest4 et m n tile tile).
      gC |-> eC' **
      pure (chest_flat42 eC' %~
              MS.mmcomb comb_r (chest_flat42 rC) (chest_flat42 rA) (chest_flat42 rB)))
{
  forevery_rw_size2
    (SZ.v (m *^ n)) (m * n)
    (SZ.v (tile *^ tile)) (tile * tile);

  (* Step 1: Unzip kpost's four components (kpost is unfold). *)
  forevery_unzip_2
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) -> pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC)) _;
  forevery_unzip_2
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) -> gA |-> Frac (fA /. ((m * n) * (tile * tile))) eA) _;
  forevery_unzip_2
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) -> gB |-> Frac (fB /. ((m * n) * (tile * tile))) eB) _;
  drop_ (forall+ (bid : natlt (m * n)) (tid : natlt (tile * tile)). pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC));

  (* Step 2: Gather gA and gB. *)
  forevery_unfactor' ((m * n) * (tile * tile)) (m * n) (tile * tile)
    (fun (_ : natlt (m * n)) (_ : natlt (tile * tile)) -> gA |-> Frac (fA /. ((m * n) * (tile * tile))) eA);
  forevery_unfactor' ((m * n) * (tile * tile)) (m * n) (tile * tile)
    (fun (_ : natlt (m * n)) (_ : natlt (tile * tile)) -> gB |-> Frac (fB /. ((m * n) * (tile * tile))) eB);
  tensor_gather_n gA ((m * n) * (tile * tile));
  tensor_gather_n gB ((m * n) * (tile * tile));

  (* Step 3: Collect the gC per-cell existentials into a single chest. *)
  let vf : (natlt (m * n) -> natlt (tile * tile) -> GTot et) =
    forevery_exists_2
      (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) (v : et) ->
        tensor_pts_to_cell gC
          ((bid / n <: natlt m), ((bid % n <: natlt n),
            ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))) v **
        pure (v %~ MS.gemm_single comb_r (chest_flat42 rA) (chest_flat42 rB) (chest_flat42 rC)
                ((bid / n) * tile + (tid / tile)) ((bid % n) * tile + (tid % tile))));

  let eC' : chest4 et m n tile tile =
    Chest.mk (m @| n @| tile @| tile @| INil)
      (fun (idx : abs (m @| n @| tile @| tile @| INil)) ->
        let (bid, tid) = (tile_idx_bij m n tile).ff idx in
        vf bid tid);

  (* Extract the per-cell approximation facts into one pure proposition. *)
  forevery_extract_pure_2
    #(natlt (m * n)) #(natlt (tile * tile))
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC
        ((bid / n <: natlt m), ((bid % n <: natlt n),
          ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))) (vf bid tid) **
      pure (vf bid tid %~ MS.gemm_single comb_r (chest_flat42 rA) (chest_flat42 rB) (chest_flat42 rC)
              ((bid / n) * tile + (tid / tile)) ((bid % n) * tile + (tid % tile))))
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
      vf bid tid %~ MS.gemm_single comb_r (chest_flat42 rA) (chest_flat42 rB) (chest_flat42 rC)
        ((bid / n) * tile + (tid / tile)) ((bid % n) * tile + (tid % tile)))
    fn bid tid { (); };

  (* Drop the per-cell pure, keeping only ownership. *)
  forevery_map_2
    #(natlt (m * n)) #(natlt (tile * tile))
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC
        ((bid / n <: natlt m), ((bid % n <: natlt n),
          ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))) (vf bid tid) **
      pure (vf bid tid %~ MS.gemm_single comb_r (chest_flat42 rA) (chest_flat42 rB) (chest_flat42 rC)
              ((bid / n) * tile + (tid / tile)) ((bid % n) * tile + (tid % tile))))
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC
        ((bid / n <: natlt m), ((bid % n <: natlt n),
          ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))) (vf bid tid))
    fn bid tid { () };

  (* Reshape the (bid, tid) ownership back into abs-indexed cells and implode. *)
  forevery_ext_2 _
    (fun (bid : natlt (m * n)) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((tile_idx_bij m n tile).gg (bid, tid))
        (Chest.acc eC' ((tile_idx_bij m n tile).gg (bid, tid))));
  forevery_flatten'
    (fun (xy : natlt (m * n) & natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((tile_idx_bij m n tile).gg xy)
        (Chest.acc eC' ((tile_idx_bij m n tile).gg xy)));
  forevery_iso_back (tile_idx_bij m n tile)
    (fun (idx : abs (m @| n @| tile @| tile @| INil)) ->
      tensor_pts_to_cell gC idx (Chest.acc eC' idx));
  tensor_implode gC;

  (* Final matrix-level approximation. *)
  assert pure (chest_flat42 eC' %~
                 MS.mmcomb comb_r (chest_flat42 rC) (chest_flat42 rA) (chest_flat42 rB));
  ();
}
#pop-options

(* No op *)
ghost
fn block_setup
  (nblk : nat)
  (nthr : nat)
  (#p : natlt nblk -> slprop)
  (bid : natlt nblk)
  norewrite
  requires
    p bid
  ensures
    p bid **
    emp
{
  ();
}

(* Sendability helpers — standalone definitions so the typeclass resolver
   runs in a minimal context with an explicit goal type.
   kpre/kpost are block_of-sendable (all components are global matrices, hence
   gpu_of-sendable, and gpu_of implies block_of).
   block_pre/post wrap kpre/kpost in forall+, needing gpu_of directly. *)
let kpre_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (eA : chest4 et m k tile tile)
  (eB : chest4 et k n tile tile)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  : is_send_across block_of (kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

let kpost_block_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (eA : chest4 et m k tile tile)
  (eB : chest4 et k n tile tile)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  (bid : natlt (m * n))
  (tid : natlt (tile * tile))
  : is_send_across block_of (kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

let block_pre_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (eA : chest4 et m k tile tile)
  (eB : chest4 et k n tile tile)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  (bid : natlt (m * n))
  : is_send_across gpu_of
      (forall+ (tid : natlt (tile * tile)).
        kpre comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

let block_post_gpu_sendable
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (eA : chest4 et m k tile tile)
  (eB : chest4 et k n tile tile)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  (bid : natlt (m * n))
  : is_send_across gpu_of
      (forall+ (tid : natlt (tile * tile)).
        kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (tile : valid_tile)
  (#_ : squash (m * n <= max_blocks))
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  (#lC : layout4 m n tile tile)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (eA : chest4 et m k tile tile)
  (eB : chest4 et k n tile tile)
  (eC : chest4 et m n tile tile)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (rC : chest4 real m n tile tile)
  (fA fB : perm)
  : kernel_desc_m_n
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC ** pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC))
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : chest4 et m n tile tile).
          gC |-> eC' **
          pure (chest_flat42 eC' %~
                  MS.mmcomb comb_r (chest_flat42 rC) (chest_flat42 rA) (chest_flat42 rB))))
= {
  nblk = m *^ n;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  =
    (fun bid -> forall+ (tid : natlt2 tile tile). kpre  comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  block_post =
    (fun bid -> forall+ (tid : natlt2 tile tile). kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  setup     = setup    comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  teardown  = teardown comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  block_frame    = (fun _bid -> emp);
  block_setup    = block_setup (m * n) (tile * tile);
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  kpost     = kpost comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  f = kf comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  kpre_sendable       = kpre_block_sendable     comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  kpost_sendable      = kpost_block_sendable    comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  block_pre_sendable  = block_pre_gpu_sendable  comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  block_post_sendable = block_post_gpu_sendable comb comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
}
#pop-options

(* ─── glue: run the rank-4 kernel over a flat array2 ──────────────────────── *)

(* Forward: flat array2 ownership becomes its rank-4 tiled view. *)
ghost
fn array2_to_tile4_ow
  (#et : Type0) (#m #k : nat) (tile : nat{tile > 0})
  (#l : layout2 (m * tile) (k * tile))
  (gA : array2 et l)
  (#f : perm) (#s : chest2 et (m * tile) (k * tile))
  requires
    gA |-> Frac f s
  ensures
    from_array (tile4_layout #m #k #tile l) (core (gA)) |-> Frac f (untile4 #et #m #k #tile s)
{
  tile4_fwd tile (gA);
}

(* Backward: rank-4 tiled view ownership becomes the flat array2. *)
ghost
fn tile4_to_array2_ow
  (#et : Type0) (#m #k : nat) (tile : nat{tile > 0})
  (#l : layout2 (m * tile) (k * tile))
  (gA : array2 et l)
  (#f : perm) (#s4 : chest4 et m k tile tile)
  requires
    from_array (tile4_layout #m #k #tile l) (core (gA)) |-> Frac f s4
  ensures
    gA |-> Frac f (chest_flat42 #et #m #k #tile s4)
{
  tile4_bwd tile (from_array (tile4_layout #m #k #tile l) (core (gA)));
  rewrite
    (from_array l (core (from_array (tile4_layout #m #k #tile l) (core (gA))))
      |-> Frac f (chest_flat42 #et #m #k #tile s4))
  as
    (gA |-> Frac f (chest_flat42 #et #m #k #tile s4));
}

(* Backward for the (existential) output, restoring the array2 result. *)
ghost
fn tile4_to_array2_gc
  (#et : Type0) {| scalar et, real_like et |}
  (comb_r : binop real)
  (#m #n #k : nat) (tile : nat{tile > 0})
  (#l : layout2 (m * tile) (n * tile))
  (gC : array2 et l)
  (rA : chest2 real (m * tile) (k * tile))
  (rB : chest2 real (k * tile) (n * tile))
  (rC : chest2 real (m * tile) (n * tile))
  (#f : perm)
  requires
    (exists* (s4 : chest4 et m n tile tile).
      from_array (tile4_layout #m #n #tile l) (core (gC)) |-> Frac f s4 **
      pure (chest_flat42 #et #m #n #tile s4 %~ MS.mmcomb comb_r rC rA rB))
  ensures
    (exists* (eC' : chest2 et (m * tile) (n * tile)).
      gC |-> Frac f eC' **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  tile4_to_array2_ow tile gC;
  ()
}

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  (#lC : layout2 (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (rA : chest2 real (m * tile) (k * tile))
  (rB : chest2 real (k * tile) (n * tile))
  (rC : chest2 real (m * tile) (n * tile))
  (#eA : chest2 et (m * tile) (k * tile))
  (#eB : chest2 et (k * tile) (n * tile))
  (#eC : chest2 et (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 et (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  (* Reinterpret each flat array2 as its rank-4 tiled tensor (same memory). *)
  map_loc gpu_loc (fun () -> array2_to_tile4_ow (SZ.v tile) gA);
  map_loc gpu_loc (fun () -> array2_to_tile4_ow (SZ.v tile) gB);
  map_loc gpu_loc (fun () -> array2_to_tile4_ow (SZ.v tile) gC);

  (* Approximation facts in the rank-4 (untiled) world. *)
  untile4_approx eA rA;
  untile4_approx eB rB;
  untile4_approx eC rC;
  flat_untile4 #real #(SZ.v m) #(SZ.v k) #(SZ.v tile) rA;
  flat_untile4 #real #(SZ.v k) #(SZ.v n) #(SZ.v tile) rB;
  flat_untile4 #real #(SZ.v m) #(SZ.v n) #(SZ.v tile) rC;
  flat_untile4 #et #(SZ.v m) #(SZ.v k) #(SZ.v tile) eA;
  flat_untile4 #et #(SZ.v k) #(SZ.v n) #(SZ.v tile) eB;

  launch_sync
    (mk_kernel comb comb_r tile
       #_ #_ #_ #_
       #(c_tile4_layout #(SZ.v m) #(SZ.v k) tile)
       #(c_tile4_layout #(SZ.v k) #(SZ.v n) tile)
       #(c_tile4_layout #(SZ.v m) #(SZ.v n) tile)
       (from_array (tile4_layout #m #k #tile lA) (core (gA)))
       (from_array (tile4_layout #k #n #tile lB) (core (gB)))
       (from_array (tile4_layout #m #n #tile lC) (core (gC)))
       (untile4 eA) (untile4 eB) (untile4 eC)
       (untile4 rA) (untile4 rB) (untile4 rC) fA fB);

  (* Restore the flat array2 views. *)
  map_loc gpu_loc (fun () -> tile4_to_array2_ow (SZ.v tile) gA);
  map_loc gpu_loc (fun () -> tile4_to_array2_ow (SZ.v tile) gB);
  map_loc gpu_loc (fun () -> tile4_to_array2_gc comb_r (SZ.v tile) gC rA rB rC);
  ();
}
