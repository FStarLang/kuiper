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
open Pulse.Lib.Trade

module Chest = Kuiper.Chest
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SZ = Kuiper.SizeT
module C = Kuiper.Matrix.Casts

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

(* Move away somewhere, this is generic. *)
(* General (fused-map, multi-type) tiled dot-product cell: reads A-cells of type
   [ta] and B-cells of type [tb], maps each into the accumulation type [tacc] via
   [mapA]/[mapB], accumulates in [tacc], and approximates the mapped real matmul
   (the real maps [mapA_r]/[mapB_r] are applied inside [chest_map]). *)
#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn gmatmul_tiled_dotprod_real
  (#ta #tb #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (mapA_r mapB_r : real -> real)
  (#m #n #k : sz)
  (#tile : szp)
  (#lA : layout4 m k tile tile)
  (#lB : layout4 k n tile tile)
  {| ctlayout lA, ctlayout lB |}
  (gA : tensor ta lA)
  (gB : tensor tb lB)
  (#eA : chest4 ta _ _ _ _)
  (#eB : chest4 tb _ _ _ _)
  (rA : chest4 real m k tile tile)
  (rB : chest4 real k n tile tile)
  (bi : szlt m)
  (bj : szlt n)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (eA %~ rA /\ eB %~ rB)
  returns
    res : tacc
  ensures
    pure (res %~ MS.matmul_single
                   (Chest.chest_map mapA_r (chest_flat42 rA))
                   (Chest.chest_map mapB_r (chest_flat42 rB))
                   (bi * tile + i) (bj * tile + j))
{
  let grow : erased (natlt (m * tile)) = hide (bi * tile + i);
  let gcol : erased (natlt (n * tile)) = hide (bj * tile + j);

  let mut sum : tacc = zero;
  let mut bk  : szle k = 0sz;

  while (!bk <^ k)
    invariant live bk ** live sum
    invariant pure
      (!sum %~ MS.__matmul_single
                 (Chest.chest_map mapA_r (chest_flat42 rA))
                 (Chest.chest_map mapB_r (chest_flat42 rB)) grow gcol (SZ.v !bk * tile))
    decreases (k - !bk)
  {
    let abi   = tensor_extract_slice_ro' gA  0 (SZ.v bi);
    let abibk = tensor_extract_slice_ro' abi 0 (SZ.v !bk);

    let bbk   = tensor_extract_slice_ro' gB  0 (SZ.v !bk);
    let bbkbj = tensor_extract_slice_ro' bbk 0 (SZ.v bj);

    let s' = Kuiper.DotProd.gmatmul_dotprod mapA mapB #_ #_ #tile abibk bbkbj i j;

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

    (* Mapping the subtiles into [tacc] preserves the approximation. *)
    MU.chest_map_approx mapA mapA_r
      (ematrix_subtile (chest_flat42 eA) tile tile (SZ.v bi) (SZ.v !bk))
      (ematrix_subtile (chest_flat42 rA) tile tile (SZ.v bi) (SZ.v !bk));
    MU.chest_map_approx mapB mapB_r
      (ematrix_subtile (chest_flat42 eB) tile tile (SZ.v !bk) (SZ.v bj))
      (ematrix_subtile (chest_flat42 rB) tile tile (SZ.v !bk) (SZ.v bj));

    (* s' approximates the real matmul over the mapped subtiles. *)
    MU.__matmul_single_approx_real
      (Chest.chest_map mapA (ematrix_subtile (chest_flat42 eA) tile tile (SZ.v bi) (SZ.v !bk)))
      (Chest.chest_map mapB (ematrix_subtile (chest_flat42 eB) tile tile (SZ.v !bk) (SZ.v bj)))
      (Chest.chest_map mapA_r (ematrix_subtile (chest_flat42 rA) tile tile (SZ.v bi) (SZ.v !bk)))
      (Chest.chest_map mapB_r (ematrix_subtile (chest_flat42 rB) tile tile (SZ.v !bk) (SZ.v bj)))
      (SZ.v i) (SZ.v j) (SZ.v tile);

    (* The partial mapped-real matmul splits over the next tile. *)
    MU.__gmatmul_single_split
      (Chest.chest_map mapA_r (chest_flat42 rA)) (Chest.chest_map mapB_r (chest_flat42 rB))
      grow gcol (SZ.v !bk * tile) (SZ.v tile)
      (Chest.chest_map mapA_r (ematrix_subtile (chest_flat42 rA) tile tile (SZ.v bi) (SZ.v !bk)))
      (Chest.chest_map mapB_r (ematrix_subtile (chest_flat42 rB) tile tile (SZ.v !bk) (SZ.v bj)))
      (SZ.v i) (SZ.v j);

    FStar.Math.Lemmas.distributivity_add_left (SZ.v !bk) 1 (SZ.v tile);
    assert (pure ((SZ.v !bk + 1) * SZ.v tile == SZ.v !bk * SZ.v tile + SZ.v tile));

    bk := !bk +^ 1sz;

    ()
  };

  !sum
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
(* ══════════════════════════════════════════════════════════════════════════
   BATCHED (rank-3) tiled GEMM.  A single launch computes [batch] independent
   tiled GEMMs.  We reinterpret the whole rank-3 operand as a rank-5 tensor
   (batch @| m @| k @| tile @| tile) ONCE at entry, then each thread slices its
   page (dim 0) to obtain a rank-4 page-view and reuses the existing rank-4
   per-page dot-product [gmatmul_tiled_dotprod_real].

   All the rank-5 view/reshape infrastructure below mirrors the rank-4
   infrastructure above, with a [batch] index prepended.
   ══════════════════════════════════════════════════════════════════════════ *)

(* ─── rank-5 chest flattening (batch prepended to chest_flat42) ───────────── *)
let chest_flat53
  (#et : Type)
  (#batch #d1 #d2 #d3 #d4 : nat)
  (x : chest (batch @| d1 @| d2 @| d3 @| d4 @| INil) et)
  : chest3 et batch (d1 * d3) (d2 * d4) =
  Kuiper.Chest.mk (batch @| (d1 * d3) @| (d2 * d4) @| INil) fun (p, (i, (j, ()))) ->
    let i1 : natlt d1 = i / d3 in
    let i2 : natlt d3 = i % d3 in
    let j1 : natlt d2 = j / d4 in
    let j2 : natlt d4 = j % d4 in
    Kuiper.Chest.acc x (p, (i1, (j1, (i2, (j2, ())))))

let chest_flat53_cell
  (#et : Type)
  (#batch #d1 #d2 #d3 #d4 : nat)
  (x : chest (batch @| d1 @| d2 @| d3 @| d4 @| INil) et)
  (p : natlt batch)
  (i1 : natlt d1) (i2 : natlt d3) (j1 : natlt d2) (j2 : natlt d4)
  : Lemma (Chest.acc (chest_flat53 x) (p, (i1 * d3 + i2, (j1 * d4 + j2, ())))
           == Chest.acc x (p, (i1, (j1, (i2, (j2, ()))))))
  = ()

(* Slicing the page of the rank-3 flat view coincides with flattening the
   rank-4 page-slice of the rank-5 tensor.  This lets us reduce rank-5
   obligations to the existing rank-4 lemmas. *)
let slice_page_flat53
  (#et : Type)
  (#batch #d1 #d2 #d3 #d4 : nat)
  (x : chest (batch @| d1 @| d2 @| d3 @| d4 @| INil) et)
  (p : natlt batch)
  : Lemma (requires d3 > 0 /\ d4 > 0)
          (ensures slice_page (chest_flat53 x) p
                   == chest_flat42 (chest_slice 0 p x))
  = introduce forall (idx : abs ((d1 * d3) @| (d2 * d4) @| INil)).
      Chest.acc (slice_page (chest_flat53 x) p) idx
      == Chest.acc (chest_flat42 (chest_slice 0 p x)) idx
    with (let (i, (j, ())) = idx in ());
    Chest.lemma_equal_intro
      (slice_page (chest_flat53 x) p)
      (chest_flat42 (chest_slice 0 p x));
    Chest.ext
      (slice_page (chest_flat53 x) p)
      (chest_flat42 (chest_slice 0 p x))

(* ─── array3 <-> rank-5 tensor "tiling view" reinterpretation ─────────────── *)

let tile5_reshape_f (batch m k tile : nat)
  : abs (batch @| m @| k @| tile @| tile @| INil) -> abs (batch @| (m * tile) @| (k * tile) @| INil)
  = fun (p, (mr, (kc, (a, (b, ()))))) -> (p, (mr * tile + a, (kc * tile + b, ())))

let tile5_reshape_inj (batch m k tile : nat)
  : (abs (batch @| m @| k @| tile @| tile @| INil) @~> abs (batch @| (m * tile) @| (k * tile) @| INil))
  = mk_injection (tile5_reshape_f batch m k tile)
      (fun (p1, (mr1, (kc1, (a1, (b1, ()))))) (p2, (mr2, (kc2, (a2, (b2, ()))))) -> ())

let tile5_layout (#batch #m #k #tile : nat)
  (l : tlayout (batch @| (m * tile) @| (k * tile) @| INil))
  : tlayout (batch @| m @| k @| tile @| tile @| INil)
  = { ulen = l.ulen;
      imap = inj_comp (tile5_reshape_inj batch m k tile) l.imap; }

let tile5_all_fit (batch m k : nat) (tile : nat)
  : Lemma (requires SZ.fits batch /\ SZ.fits (m * tile) /\ SZ.fits (k * tile) /\ m > 0 /\ k > 0 /\ tile > 0)
          (ensures all_fit (batch @| m @| k @| tile @| tile @| INil))
  = FStar.Math.Lemmas.lemma_mult_le_right tile 1 m;
    FStar.Math.Lemmas.lemma_mult_le_right tile 1 k;
    FStar.Math.Lemmas.lemma_mult_le_left m 1 tile;
    FStar.Math.Lemmas.lemma_mult_le_left k 1 tile

#push-options "--split_queries always --z3rlimit 40"
inline_for_extraction noextract
let c_tile5_layout
  (#batch : erased pos) (#m #k : erased pos) (tile : szp)
  (#l : tlayout (batch @| (m * tile) @| (k * tile) @| INil))
  {| cc : ctlayout l |}
  : ctlayout (tile5_layout #batch #m #k #tile l)
  = let _ = cc.ulen_fits in
    let _ = cc.all_fit in
    tile5_all_fit batch m k (SZ.v tile);
    {
      ulen_fits = ();
      all_fit = ();
      cimap = (fun (x : conc (batch @| m @| k @| tile @| tile @| INil)) ->
                match x with
                | (p, (mr, (kc, (a, (b, ()))))) ->
                  cc.cimap (p, (mr *^ tile +^ a, (kc *^ tile +^ b, ()))));
  }
#pop-options

let untile5
  (#et : Type) (#batch #m #k #tile : nat)
  (s : chest3 et batch (m * tile) (k * tile))
  : chest (batch @| m @| k @| tile @| tile @| INil) et
  = Chest.mk (batch @| m @| k @| tile @| tile @| INil)
      (fun (p, (mr, (kc, (a, (b, ()))))) -> Chest.acc s (p, (mr * tile + a, (kc * tile + b, ()))))

let flat_untile5
  (#et : Type) (#batch #m #k #tile : nat)
  (s : chest3 et batch (m * tile) (k * tile))
  : Lemma (requires tile > 0)
          (ensures chest_flat53 (untile5 s) == s)
  = introduce forall (idx : abs (batch @| (m * tile) @| (k * tile) @| INil)).
      Chest.acc (chest_flat53 (untile5 #et #batch #m #k #tile s)) idx == Chest.acc s idx
    with (let (p, (i, (j, ()))) = idx in ());
    Chest.lemma_equal_intro (chest_flat53 (untile5 #et #batch #m #k #tile s)) s;
    Chest.ext (chest_flat53 (untile5 #et #batch #m #k #tile s)) s

let tile5_unreshape_f (batch m k : nat) (tile : nat{tile > 0})
  : abs (batch @| (m * tile) @| (k * tile) @| INil) -> abs (batch @| m @| k @| tile @| tile @| INil)
  = fun (p, (i, (j, ()))) -> (p, ((i / tile <: natlt m), ((j / tile <: natlt k),
                          ((i % tile <: natlt tile), ((j % tile <: natlt tile), ())))))

let reshape_bij5 (batch m k : nat) (tile : nat{tile > 0})
  : (abs (batch @| m @| k @| tile @| tile @| INil) =~ abs (batch @| (m * tile) @| (k * tile) @| INil))
  = {
      ff = tile5_reshape_f batch m k tile;
      gg = tile5_unreshape_f batch m k tile;
      ff_gg = (fun (p, (i, (j, ()))) -> ());
      gg_ff = (fun (p, (mr, (kc, (a, (b, ()))))) -> ());
    }

ghost
fn tensor_to_tile5
  (#et : Type0) (#batch #m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout (batch @| (m * tile) @| (k * tile) @| INil))
  (g3 : tensor et l)
  (#f : perm) (#s : chest3 et batch (m * tile) (k * tile))
  requires
    g3 |-> Frac f s
  returns
    g5 : tensor et (tile5_layout #batch #m #k #tile l)
  ensures
    g5 |-> Frac f (untile5 #et #batch #m #k #tile s) **
    pure (core g5 == core g3 /\ (is_global g5 <==> is_global g3))
{
  tensor_ilower g3;
  let g5 = from_array (tile5_layout #batch #m #k #tile l) (core g3);
  rewrite each (core g3) as (core g5);
  forevery_ext
    (fun (i : abs (batch @| (m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g5) #f (l.imap.f i) (Chest.acc s i))
    (fun (i : abs (batch @| (m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g5) #f
         ((tile5_layout #batch #m #k #tile l).imap.f ((reshape_bij5 batch m k tile).gg i))
         (Chest.acc (untile5 #et #batch #m #k #tile s) ((reshape_bij5 batch m k tile).gg i)));
  forevery_iso_back (reshape_bij5 batch m k tile)
    (fun (idx5 : abs (batch @| m @| k @| tile @| tile @| INil)) ->
       pts_to_cell (core g5) #f ((tile5_layout #batch #m #k #tile l).imap.f idx5)
         (Chest.acc (untile5 #et #batch #m #k #tile s) idx5));
  tensor_iraise g5;
  g5
}

ghost
fn tile5_to_tensor
  (#et : Type0) (#batch #m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout (batch @| (m * tile) @| (k * tile) @| INil))
  (g5 : tensor et (tile5_layout #batch #m #k #tile l))
  (#f : perm) (#s5 : chest (batch @| m @| k @| tile @| tile @| INil) et)
  requires
    g5 |-> Frac f s5
  returns
    g3 : tensor et l
  ensures
    g3 |-> Frac f (chest_flat53 #et #batch #m #k #tile s5) **
    pure (core g3 == core g5 /\ (is_global g3 <==> is_global g5))
{
  tensor_ilower g5;
  let g3 = from_array l (core g5);
  rewrite each (core g5) as (core g3);
  forevery_iso (reshape_bij5 batch m k tile)
    (fun (idx5 : abs (batch @| m @| k @| tile @| tile @| INil)) ->
       pts_to_cell (core g3) #f ((tile5_layout #batch #m #k #tile l).imap.f idx5)
         (Chest.acc s5 idx5));
  forevery_ext
    (fun (idx3 : abs (batch @| (m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g3) #f
         ((tile5_layout #batch #m #k #tile l).imap.f ((reshape_bij5 batch m k tile).gg idx3))
         (Chest.acc s5 ((reshape_bij5 batch m k tile).gg idx3)))
    (fun (idx3 : abs (batch @| (m * tile) @| (k * tile) @| INil)) ->
       pts_to_cell (core g3) #f (l.imap.f idx3)
         (Chest.acc (chest_flat53 #et #batch #m #k #tile s5) idx3));
  tensor_iraise g3;
  g3
}

let untile5_approx
  (#et : Type) {| scalar et, real_like et |}
  (#batch #m #k #tile : nat)
  (e : chest3 et batch (m * tile) (k * tile))
  (r : chest3 real batch (m * tile) (k * tile))
  : Lemma (requires e %~ r)
          (ensures untile5 #et #batch #m #k #tile e %~ untile5 #real #batch #m #k #tile r)
  = introduce forall (idx : abs (batch @| m @| k @| tile @| tile @| INil)).
      Chest.acc (untile5 #et #batch #m #k #tile e) idx %~ Chest.acc (untile5 #real #batch #m #k #tile r) idx
    with (let (p, (mr, (kc, (a, (b, ()))))) = idx in ())

ghost
fn tile5_fwd
  (#et : Type0) (#batch #m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout (batch @| (m * tile) @| (k * tile) @| INil))
  (g3 : tensor et l)
  (#f : perm) (#s : chest3 et batch (m * tile) (k * tile))
  requires
    g3 |-> Frac f s
  ensures
    from_array (tile5_layout #batch #m #k #tile l) (core g3) |-> Frac f (untile5 #et #batch #m #k #tile s)
{
  let g5 = tensor_to_tile5 tile g3;
  rewrite (g5 |-> Frac f (untile5 #et #batch #m #k #tile s))
       as (from_array (tile5_layout #batch #m #k #tile l) (core g3) |-> Frac f (untile5 #et #batch #m #k #tile s));
}

ghost
fn tile5_bwd
  (#et : Type0) (#batch #m #k : nat) (tile : nat{tile > 0})
  (#l : tlayout (batch @| (m * tile) @| (k * tile) @| INil))
  (g5 : tensor et (tile5_layout #batch #m #k #tile l))
  (#f : perm) (#s5 : chest (batch @| m @| k @| tile @| tile @| INil) et)
  requires
    g5 |-> Frac f s5
  ensures
    from_array l (core g5) |-> Frac f (chest_flat53 #et #batch #m #k #tile s5)
{
  let g3 = tile5_to_tensor tile g5;
  rewrite (g3 |-> Frac f (chest_flat53 #et #batch #m #k #tile s5))
       as (from_array l (core g5) |-> Frac f (chest_flat53 #et #batch #m #k #tile s5));
}


(* ══════════════════════════════════════════════════════════════════════════
   SECTION C — the batched kernel and entry point.

   This is the ONLY kernel description in this module.  The rank-2 (non-batched)
   entry [gmmcomb_gpu_approx] below is derived from this batched entry at
   [batch = 1] (see SECTION D), so there is no separate rank-2 kernel.

   Each thread first fixes its page (dim 0), slices the rank-5 operands down to a
   rank-4 page view, and reuses [gmatmul_tiled_dotprod_real].
   ══════════════════════════════════════════════════════════════════════════ *)

(* Slicing preserves the approximation relation (cellwise). *)
let chest_slice_approx
  (#et : Type) {| scalar et, real_like et |}
  (#r : nat) (#d : shape r)
  (i : natlt r) (j : natlt (d @! i))
  (e : chest d et) (rr : chest d real)
  : Lemma (requires e %~ rr)
          (ensures chest_slice i j e %~ chest_slice i j rr)
  = introduce forall (idx : abs (Kuiper.Shape.modulo_i i d)).
      Chest.acc (chest_slice i j e) idx %~ Chest.acc (chest_slice i j rr) idx
    with ()

(* A cell of the batched combined spec over flattened rank-5 chests equals the
   per-page rank-4 gemm_single cell over the flattened page-slice.  Reduces the
   rank-5 obligation to the rank-4 world via [slice_page_flat53]. *)
let bmmcomb_flat_cell
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real)
  (#batch #m #n #k #tile : nat)
  (rA5 : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB5 : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC5 : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (page : natlt batch) (row : natlt (m * tile)) (col : natlt (n * tile))
  : Lemma (requires tile > 0)
          (ensures
            Chest.acc (MS.gbmmcomb mapA_r mapB_r comb_r (chest_flat53 rC5) (chest_flat53 rA5) (chest_flat53 rB5))
                      (page, (row, (col, ())))
            == MS.ggemm_single mapA_r mapB_r comb_r
                 (chest_flat42 #real #m #k #tile #tile (chest_slice 0 page rA5))
                 (chest_flat42 #real #k #n #tile #tile (chest_slice 0 page rB5))
                 (chest_flat42 #real #m #n #tile #tile (chest_slice 0 page rC5))
                 row col)
  = slice_page_flat53 rA5 page;
    slice_page_flat53 rB5 page;
    slice_page_flat53 rC5 page

(* Universally-quantified form of [bmmcomb_flat_cell]: reduces EVERY cell of the
   batched combined spec (over flattened rank-5 chests) to its per-page rank-4
   [ggemm_single] cell.  Used by [bteardown] to bridge from the ambient per-cell
   [ggemm_single] facts (over page-sliced rank-4 chests) to the rank-5 [gbmmcomb]
   matrix-level approximation. *)
let bmmcomb_flat_all
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real)
  (#batch #m #n #k #tile : nat)
  (rA5 : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB5 : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC5 : chest (batch @| m @| n @| tile @| tile @| INil) real)
  : Lemma (requires tile > 0)
          (ensures
            forall (page : natlt batch) (row : natlt (m * tile)) (col : natlt (n * tile)).
              Chest.acc (MS.gbmmcomb mapA_r mapB_r comb_r (chest_flat53 rC5) (chest_flat53 rA5) (chest_flat53 rB5))
                        (page, (row, (col, ())))
              == MS.ggemm_single mapA_r mapB_r comb_r
                   (chest_flat42 #real #m #k #tile #tile (chest_slice 0 page rA5))
                   (chest_flat42 #real #k #n #tile #tile (chest_slice 0 page rB5))
                   (chest_flat42 #real #m #n #tile #tile (chest_slice 0 page rC5))
                   row col)
  = introduce
      forall (page : natlt batch) (row : natlt (m * tile)) (col : natlt (n * tile)).
        Chest.acc (MS.gbmmcomb mapA_r mapB_r comb_r (chest_flat53 rC5) (chest_flat53 rA5) (chest_flat53 rB5))
                  (page, (row, (col, ())))
        == MS.ggemm_single mapA_r mapB_r comb_r
             (chest_flat42 #real #m #k #tile #tile (chest_slice 0 page rA5))
             (chest_flat42 #real #k #n #tile #tile (chest_slice 0 page rB5))
             (chest_flat42 #real #m #n #tile #tile (chest_slice 0 page rC5))
             row col
      with bmmcomb_flat_cell mapA_r mapB_r comb_r rA5 rB5 rC5 page row col

(* ─── batched pre/post conditions (rank-5 cell) ───────────────────────────── *)

unfold
let bkpre
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  (gA : tensor ta lA)
  (gB : tensor tb lB)
  (gC : tensor tc lC)
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  (bid : natlt (batch * (m * n)))
  (tid : natlt (tile * tile))
  : slprop
  =
  let page = bid % batch in
  let mrow = (bid / batch) / n in
  let mcol = (bid / batch) % n in
  let brow = tid / tile in
  let bcol = tid % tile in
  pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC /\
        MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
  gA |-> Frac (fA /. ((batch * (m * n)) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((batch * (m * n)) * (tile * tile))) eB **
  tensor_pts_to_cell
    gC
    (page, (mrow, (mcol, (brow, (bcol, ())))))
    (Chest.acc eC (page, (mrow, (mcol, (brow, (bcol, ()))))))

unfold
let bkpost
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  (gA : tensor ta lA)
  (gB : tensor tb lB)
  (gC : tensor tc lC)
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  (bid : natlt (batch * (m * n)))
  (tid : natlt (tile * tile))
  =
  let page = bid % batch in
  let mrow = (bid / batch) / n in
  let mcol = (bid / batch) % n in
  let brow = tid / tile in
  let bcol = tid % tile in
  pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC /\
        MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
  gA |-> Frac (fA /. ((batch * (m * n)) * (tile * tile))) eA **
  gB |-> Frac (fB /. ((batch * (m * n)) * (tile * tile))) eB **
  exists* (v : tc).
    tensor_pts_to_cell
      gC
      (page, (mrow, (mcol, (brow, (bcol, ())))))
      v **
    pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
            (chest_flat42 #real #m #k #tile #tile (chest_slice 0 page rA))
            (chest_flat42 #real #k #n #tile #tile (chest_slice 0 page rB))
            (chest_flat42 #real #m #n #tile #tile (chest_slice 0 page rC))
            (mrow * tile + brow) (mcol * tile + bcol))

(* ─── batched kernel body ─────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn bkf
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor ta lA)
  (gB : tensor tb lB)
  (gC : tensor tc lC)
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  (bid : szlt (batch * (m * n)))
  (tid : szlt (tile * tile))
  ()
  norewrite
  requires
    gpu **
    bkpre mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (batch * (m * n)) bid
  ensures
    gpu **
    bkpost mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (batch * (m * n)) bid
{
  let page : szlt batch = bid %^ batch;
  let rest = bid /^ batch;
  let mrow : szlt m = rest /^ n;
  let mcol : szlt n = rest %^ n;
  let brow : szlt tile = tid /^ tile;
  let bcol : szlt tile = tid %^ tile;

  (* Rewrite bkpre's cell index into decoded-variable form. *)
  rewrite
    tensor_pts_to_cell
      gC
      ((bid % batch <: natlt batch),
        (((bid / batch) / n <: natlt m),
          (((bid / batch) % n <: natlt n),
            ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))))
      (Chest.acc eC
        ((bid % batch <: natlt batch),
          (((bid / batch) / n <: natlt m),
            (((bid / batch) % n <: natlt n),
              ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))))
  as
    tensor_pts_to_cell
      gC
      ((page <: natlt batch),
        ((mrow <: natlt m),
          ((mcol <: natlt n),
            ((brow <: natlt tile), ((bcol <: natlt tile), ())))))
      (Chest.acc eC
        ((page <: natlt batch),
          ((mrow <: natlt m),
            ((mcol <: natlt n),
              ((brow <: natlt tile), ((bcol <: natlt tile), ()))))));

  (* Slice out the [page]-th rank-4 page views of A and B (read-only). *)
  tensor_extract_slice_ro gA 0 (SZ.v page);
  tensor_extract_slice_ro gB 0 (SZ.v page);

  (* Per-page operands still approximate their real counterparts. *)
  chest_slice_approx 0 (SZ.v page) eA rA;
  chest_slice_approx 0 (SZ.v page) eB rB;

  let s = gmatmul_tiled_dotprod_real
            mapA mapB mapA_r mapB_r
            (sliceof gA 0 (SZ.v page)) (sliceof gB 0 (SZ.v page))
            (chest_slice 0 (SZ.v page) rA) (chest_slice 0 (SZ.v page) rB)
            mrow mcol brow bcol;

  (* Restore A and B. *)
  elim_trade
    (sliceof gA 0 (SZ.v page)
      |-> Frac (fA /. ((batch * (m * n)) * (tile * tile))) (chest_slice 0 (SZ.v page) eA))
    (gA |-> Frac (fA /. ((batch * (m * n)) * (tile * tile))) eA);
  elim_trade
    (sliceof gB 0 (SZ.v page)
      |-> Frac (fB /. ((batch * (m * n)) * (tile * tile))) (chest_slice 0 (SZ.v page) eB))
    (gB |-> Frac (fB /. ((batch * (m * n)) * (tile * tile))) eB);

  let v0 = tensor_read_cell gC (page, (mrow, (mcol, (brow, (bcol, ())))));
  let v1 = comb v0 s;
  tensor_write_cell gC (page, (mrow, (mcol, (brow, (bcol, ())))) ) v1;

  (* Rewrite the cell index back to the arithmetic form used by bkpost. *)
  rewrite
    tensor_pts_to_cell
      gC
      ((page <: natlt batch),
        ((mrow <: natlt m),
          ((mcol <: natlt n),
            ((brow <: natlt tile), ((bcol <: natlt tile), ())))))
      v1
  as
    tensor_pts_to_cell
      gC
      ((bid % batch <: natlt batch),
        (((bid / batch) / n <: natlt m),
          (((bid / batch) % n <: natlt n),
            ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))))
      v1;

  (* The new cell value approximates ggemm_single over the per-page real flat
     views: v0 approximates rC at this cell, s approximates the mapped matmul, and
     comb `approx2` comb_r. *)
  chest_slice_approx 0 (SZ.v page) eC rC;
  chest_flat42_cell (chest_slice 0 (SZ.v page) rC)
    (mrow <: natlt m) (brow <: natlt tile) (mcol <: natlt n) (bcol <: natlt tile);
  chest_flat42_cell (chest_slice 0 (SZ.v page) eC)
    (mrow <: natlt m) (brow <: natlt tile) (mcol <: natlt n) (bcol <: natlt tile);

  fold bkpost mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid;
  ()
}

(* ─── batched index bijection (abstract rank-5 <-> (block, thread)) ────────── *)

(* Pure tuple rearrangement: bring the page index to the innermost block slot
   (page-minor) and split thread dims off. *)
let abs_page_bij (batch m n p q : nat)
  : (abs (batch @| m @| n @| p @| q @| INil)
     =~ (((natlt m & natlt n) & natlt batch) & (natlt p & natlt q)))
  = {
      ff = (fun (pg, (mr, (mc, (a, (b, ()))))) -> (((mr, mc), pg), (a, b)));
      gg = (fun (((mr, mc), pg), (a, b)) -> (pg, (mr, (mc, (a, (b, ()))))));
      ff_gg = (fun (((mr, mc), pg), (a, b)) -> ());
      gg_ff = (fun (pg, (mr, (mc, (a, (b, ()))))) -> ());
    }

(* Identity bijection witnessing commutativity of the block-count product. *)
let bij_comm_size (a b : nat) : (natlt (a * b) =~ natlt (b * a)) =
  {
    ff = (fun (x : natlt (a * b)) -> (x <: natlt (b * a)));
    gg = (fun (x : natlt (b * a)) -> (x <: natlt (a * b)));
    ff_gg = (fun x -> ());
    gg_ff = (fun x -> ());
  }

(* Full reshape: abstract rank-5 index <-> (block index, thread index).
   The block index is page-minor: bid = (mrow*n+mcol)*batch + page, so that
   gg (bid, tid) = (bid%batch, ((bid/batch)/n, ((bid/batch)%n,
                    (tid/tile, (tid%tile, ()))))). *)
let btile_idx_bij (batch m n tile : nat)
  : (abs (batch @| m @| n @| tile @| tile @| INil)
     =~ (natlt (batch * (m * n)) & natlt (tile * tile)))
  = bij_comp (abs_page_bij batch m n tile tile)
       (bij_prod
          (bij_comp
             (bij_comp (bij_prod (bij_nat_prod #m #n) (bij_self (natlt batch)))
                       (bij_nat_prod #(m * n) #batch))
             (bij_comm_size (m * n) batch))
          (bij_nat_prod #tile #tile))

(* ─── batched setup / teardown (ForEvery distribution) ────────────────────── *)

ghost
fn bsetup
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#_ : squash (batch * (m * n) <= max_blocks))
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor ta lA)
  (gB : tensor tb lB)
  (gC : tensor tc lC)
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC /\
          MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r)
  ensures
    (forall+ (bid : natlt (batch *^ (m *^ n)))
             (tid : natlt (tile *^ tile)).
      bkpre mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp
{
  let n_threads : nat = (batch * (m * n)) * (tile * tile);

  tensor_share_n gA n_threads;
  tensor_share_n gB n_threads;
  tensor_explode gC;

  forevery_iso (btile_idx_bij batch m n tile)
    (fun (idx : abs (batch @| m @| n @| tile @| tile @| INil)) ->
       tensor_pts_to_cell gC idx (Chest.acc eC idx));
  forevery_unflatten' _;
  forevery_ext_2 _
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
       tensor_pts_to_cell gC
         ((bid % batch <: natlt batch),
           (((bid / batch) / n <: natlt m),
             (((bid / batch) % n <: natlt n),
               ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))))
         (Chest.acc eC
           ((bid % batch <: natlt batch),
             (((bid / batch) / n <: natlt m),
               (((bid / batch) % n <: natlt n),
                 ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))))));

  forevery_factor n_threads (batch * (m * n)) (tile * tile)
    (fun _ -> gA |-> Frac (fA /. n_threads) eA);
  forevery_factor n_threads (batch * (m * n)) (tile * tile)
    (fun _ -> gB |-> Frac (fB /. n_threads) eB);

  forevery_intro_pure_2 (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
    eA %~ rA /\ eB %~ rB /\ eC %~ rC /\
    MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r);

  forevery_zip4_2
    #(natlt (batch * (m * n))) #(natlt (tile * tile))
    (fun bid tid -> pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC /\
                          MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r))
    (fun bid tid -> gA |-> Frac (fA /. n_threads) eA)
    (fun bid tid -> gB |-> Frac (fB /. n_threads) eB)
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
       tensor_pts_to_cell gC
         ((bid % batch <: natlt batch),
           (((bid / batch) / n <: natlt m),
             (((bid / batch) % n <: natlt n),
               ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))))
         (Chest.acc eC
           ((bid % batch <: natlt batch),
             (((bid / batch) / n <: natlt m),
               (((bid / batch) % n <: natlt n),
                 ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ())))))));

  forevery_ext_2 _
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
       bkpre mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  forevery_rw_size2
    (batch * (m * n)) (SZ.v (batch *^ (m *^ n)))
    (tile * tile) (SZ.v (tile *^ tile));
  ();
}

#push-options "--z3rlimit 150 --ifuel 5"
ghost
fn bteardown
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#_ : squash (batch * (m * n) <= max_blocks))
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor ta lA)
  (gB : tensor tb lB)
  (gC : tensor tc lC)
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (batch *^ (m *^ n)))
             (tid : natlt (tile *^ tile)).
      bkpost mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid) **
    emp
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    (exists* (eC' : chest (batch @| m @| n @| tile @| tile @| INil) tc).
      gC |-> eC' **
      pure (chest_flat53 eC' %~
              MS.gbmmcomb mapA_r mapB_r comb_r (chest_flat53 rC) (chest_flat53 rA) (chest_flat53 rB)))
{
  forevery_rw_size2
    (SZ.v (batch *^ (m *^ n))) (batch * (m * n))
    (SZ.v (tile *^ tile)) (tile * tile);

  forevery_unzip_2
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) -> pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC /\ MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r)) _;
  forevery_unzip_2
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) -> gA |-> Frac (fA /. ((batch * (m * n)) * (tile * tile))) eA) _;
  forevery_unzip_2
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) -> gB |-> Frac (fB /. ((batch * (m * n)) * (tile * tile))) eB) _;
  drop_ (forall+ (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)). pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC /\ MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r));

  forevery_unfactor' ((batch * (m * n)) * (tile * tile)) (batch * (m * n)) (tile * tile)
    (fun (_ : natlt (batch * (m * n))) (_ : natlt (tile * tile)) -> gA |-> Frac (fA /. ((batch * (m * n)) * (tile * tile))) eA);
  forevery_unfactor' ((batch * (m * n)) * (tile * tile)) (batch * (m * n)) (tile * tile)
    (fun (_ : natlt (batch * (m * n))) (_ : natlt (tile * tile)) -> gB |-> Frac (fB /. ((batch * (m * n)) * (tile * tile))) eB);
  tensor_gather_n gA ((batch * (m * n)) * (tile * tile));
  tensor_gather_n gB ((batch * (m * n)) * (tile * tile));

  let vf : (natlt (batch * (m * n)) -> natlt (tile * tile) -> GTot tc) =
    forevery_exists_2
      (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) (v : tc) ->
        tensor_pts_to_cell gC
          ((bid % batch <: natlt batch),
            (((bid / batch) / n <: natlt m),
              (((bid / batch) % n <: natlt n),
                ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))) v **
        pure (v %~ MS.ggemm_single mapA_r mapB_r comb_r
                (chest_flat42 #real #m #k #tile #tile (chest_slice 0 (bid % batch) rA))
                (chest_flat42 #real #k #n #tile #tile (chest_slice 0 (bid % batch) rB))
                (chest_flat42 #real #m #n #tile #tile (chest_slice 0 (bid % batch) rC))
                (((bid / batch) / n) * tile + (tid / tile)) (((bid / batch) % n) * tile + (tid % tile))));

  let eC' : chest (batch @| m @| n @| tile @| tile @| INil) tc =
    Chest.mk (batch @| m @| n @| tile @| tile @| INil)
      (fun (idx : abs (batch @| m @| n @| tile @| tile @| INil)) ->
        let (bid, tid) = (btile_idx_bij batch m n tile).ff idx in
        vf bid tid);

  forevery_extract_pure_2
    #(natlt (batch * (m * n))) #(natlt (tile * tile))
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC
        ((bid % batch <: natlt batch),
          (((bid / batch) / n <: natlt m),
            (((bid / batch) % n <: natlt n),
              ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))) (vf bid tid) **
      pure (vf bid tid %~ MS.ggemm_single mapA_r mapB_r comb_r
              (chest_flat42 #real #m #k #tile #tile (chest_slice 0 (bid % batch) rA))
              (chest_flat42 #real #k #n #tile #tile (chest_slice 0 (bid % batch) rB))
              (chest_flat42 #real #m #n #tile #tile (chest_slice 0 (bid % batch) rC))
              (((bid / batch) / n) * tile + (tid / tile)) (((bid / batch) % n) * tile + (tid % tile))))
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
      vf bid tid %~ MS.ggemm_single mapA_r mapB_r comb_r
        (chest_flat42 #real #m #k #tile #tile (chest_slice 0 (bid % batch) rA))
        (chest_flat42 #real #k #n #tile #tile (chest_slice 0 (bid % batch) rB))
        (chest_flat42 #real #m #n #tile #tile (chest_slice 0 (bid % batch) rC))
        (((bid / batch) / n) * tile + (tid / tile)) (((bid / batch) % n) * tile + (tid % tile)))
    fn bid tid { (); };

  forevery_map_2
    #(natlt (batch * (m * n))) #(natlt (tile * tile))
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC
        ((bid % batch <: natlt batch),
          (((bid / batch) / n <: natlt m),
            (((bid / batch) % n <: natlt n),
              ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))) (vf bid tid) **
      pure (vf bid tid %~ MS.ggemm_single mapA_r mapB_r comb_r
              (chest_flat42 #real #m #k #tile #tile (chest_slice 0 (bid % batch) rA))
              (chest_flat42 #real #k #n #tile #tile (chest_slice 0 (bid % batch) rB))
              (chest_flat42 #real #m #n #tile #tile (chest_slice 0 (bid % batch) rC))
              (((bid / batch) / n) * tile + (tid / tile)) (((bid / batch) % n) * tile + (tid % tile))))
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC
        ((bid % batch <: natlt batch),
          (((bid / batch) / n <: natlt m),
            (((bid / batch) % n <: natlt n),
              ((tid / tile <: natlt tile), ((tid % tile <: natlt tile), ()))))) (vf bid tid))
    fn bid tid { () };

  forevery_ext_2 _
    (fun (bid : natlt (batch * (m * n))) (tid : natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((btile_idx_bij batch m n tile).gg (bid, tid))
        (Chest.acc eC' ((btile_idx_bij batch m n tile).gg (bid, tid))));
  forevery_flatten'
    (fun (xy : natlt (batch * (m * n)) & natlt (tile * tile)) ->
      tensor_pts_to_cell gC ((btile_idx_bij batch m n tile).gg xy)
        (Chest.acc eC' ((btile_idx_bij batch m n tile).gg xy)));
  forevery_iso_back (btile_idx_bij batch m n tile)
    (fun (idx : abs (batch @| m @| n @| tile @| tile @| INil)) ->
      tensor_pts_to_cell gC idx (Chest.acc eC' idx));
  tensor_implode gC;

  (* Final batched matrix-level approximation, reduced cellwise to the per-page
     rank-4 ggemm_single facts already established. *)
  bmmcomb_flat_all mapA_r mapB_r comb_r rA rB rC;
  assert pure (chest_flat53 eC' %~
                 MS.gbmmcomb mapA_r mapB_r comb_r (chest_flat53 rC) (chest_flat53 rA) (chest_flat53 rB));
  ();
}
#pop-options

(* ─── batched sendability helpers ─────────────────────────────────────────── *)

let bkpre_block_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  (bid : natlt (batch * (m * n)))
  (tid : natlt (tile * tile))
  : is_send_across block_of (bkpre mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

let bkpost_block_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  (bid : natlt (batch * (m * n)))
  (tid : natlt (tile * tile))
  : is_send_across block_of (bkpost mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

let bblock_pre_gpu_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  (bid : natlt (batch * (m * n)))
  : is_send_across gpu_of
      (forall+ (tid : natlt (tile * tile)).
        bkpre mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

let bblock_post_gpu_sendable
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  (bid : natlt (batch * (m * n)))
  : is_send_across gpu_of
      (forall+ (tid : natlt (tile * tile)).
        bkpost mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid)
  = solve

(* ─── batched kernel descriptor ───────────────────────────────────────────── *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let bmk_kernel
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (tile : valid_tile)
  (#_ : squash (batch * (m * n) <= max_blocks))
  (#lA : tlayout (batch @| m @| k @| tile @| tile @| INil))
  (#lB : tlayout (batch @| k @| n @| tile @| tile @| INil))
  (#lC : tlayout (batch @| m @| n @| tile @| tile @| INil))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor ta lA { is_global gA })
  (gB : tensor tb lB { is_global gB })
  (gC : tensor tc lC { is_global gC })
  (eA : chest (batch @| m @| k @| tile @| tile @| INil) ta)
  (eB : chest (batch @| k @| n @| tile @| tile @| INil) tb)
  (eC : chest (batch @| m @| n @| tile @| tile @| INil) tc)
  (rA : chest (batch @| m @| k @| tile @| tile @| INil) real)
  (rB : chest (batch @| k @| n @| tile @| tile @| INil) real)
  (rC : chest (batch @| m @| n @| tile @| tile @| INil) real)
  (fA fB : perm)
  : kernel_desc_m_n
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC **
        pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC /\
              MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r))
      (gA |-> Frac fA eA ** gB |-> Frac fB eB **
        (exists* (eC' : chest (batch @| m @| n @| tile @| tile @| INil) tc).
          gC |-> eC' **
          pure (chest_flat53 eC' %~
                  MS.gbmmcomb mapA_r mapB_r comb_r (chest_flat53 rC) (chest_flat53 rA) (chest_flat53 rB))))
= {
  nblk = batch *^ (m *^ n);
  nthr = tile *^ tile;

  frame = emp;
  block_pre  =
    (fun bid -> forall+ (tid : natlt2 tile tile). bkpre  mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  block_post =
    (fun bid -> forall+ (tid : natlt2 tile tile). bkpost mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB bid tid);
  setup     = bsetup    mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  teardown  = bteardown mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  block_frame    = (fun _bid -> emp);
  block_setup    = block_setup (batch * (m * n)) (tile * tile);
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = bkpre  mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  kpost     = bkpost mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  f = bkf mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;

  kpre_sendable       = bkpre_block_sendable     mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  kpost_sendable      = bkpost_block_sendable    mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  block_pre_sendable  = bblock_pre_gpu_sendable  mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
  block_post_sendable = bblock_post_gpu_sendable mapA mapB comb mapA_r mapB_r comb_r tile gA gB gC eA eB eC rA rB rC fA fB;
}
#pop-options

(* ─── glue: run the rank-5 kernel over a flat array3 ──────────────────────── *)

ghost
fn array3_to_tile5_ow
  (#et : Type0) (#batch #m #k : nat) (tile : nat{tile > 0})
  (#l : layout3 batch (m * tile) (k * tile))
  (gA : array3 et l)
  (#f : perm) (#s : chest3 et batch (m * tile) (k * tile))
  requires
    gA |-> Frac f s
  ensures
    from_array (tile5_layout #batch #m #k #tile l) (core (gA)) |-> Frac f (untile5 #et #batch #m #k #tile s)
{
  tile5_fwd tile (gA);
}

ghost
fn tile5_to_array3_ow
  (#et : Type0) (#batch #m #k : nat) (tile : nat{tile > 0})
  (#l : layout3 batch (m * tile) (k * tile))
  (gA : array3 et l)
  (#f : perm) (#s5 : chest (batch @| m @| k @| tile @| tile @| INil) et)
  requires
    from_array (tile5_layout #batch #m #k #tile l) (core (gA)) |-> Frac f s5
  ensures
    gA |-> Frac f (chest_flat53 #et #batch #m #k #tile s5)
{
  tile5_bwd tile (from_array (tile5_layout #batch #m #k #tile l) (core (gA)));
  rewrite
    (from_array l (core (from_array (tile5_layout #batch #m #k #tile l) (core (gA))))
      |-> Frac f (chest_flat53 #et #batch #m #k #tile s5))
  as
    (gA |-> Frac f (chest_flat53 #et #batch #m #k #tile s5));
}

ghost
fn tile5_to_array3_gc
  (#tc : Type0) {| scalar tc, real_like tc |}
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real)
  (#batch #m #n #k : nat) (tile : nat{tile > 0})
  (#l : layout3 batch (m * tile) (n * tile))
  (gC : array3 tc l)
  (rA : chest3 real batch (m * tile) (k * tile))
  (rB : chest3 real batch (k * tile) (n * tile))
  (rC : chest3 real batch (m * tile) (n * tile))
  (#f : perm)
  requires
    (exists* (s5 : chest (batch @| m @| n @| tile @| tile @| INil) tc).
      from_array (tile5_layout #batch #m #n #tile l) (core (gC)) |-> Frac f s5 **
      pure (chest_flat53 #tc #batch #m #n #tile s5 %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB))
  ensures
    (exists* (eC' : chest3 tc batch (m * tile) (n * tile)).
      gC |-> Frac f eC' **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB))
{
  tile5_to_array3_ow tile gC;
  ()
}

(* ─── batched entry point ─────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn gbmmcomb_gpu_approx
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch (m * tile) (k * tile))
  (#lB : layout3 batch (k * tile) (n * tile))
  (#lC : layout3 batch (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array3 ta lA { is_global gA })
  (gB : array3 tb lB { is_global gB })
  (gC : array3 tc lC { is_global gC })
  (rA : chest3 real batch (m * tile) (k * tile))
  (rB : chest3 real batch (k * tile) (n * tile))
  (rC : chest3 real batch (m * tile) (n * tile))
  (#eA : chest3 ta batch (m * tile) (k * tile))
  (#eB : chest3 tb batch (k * tile) (n * tile))
  (#eC : chest3 tc batch (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (bsize_req batch m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest3 tc batch (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB))
{
  (* Reinterpret each flat array3 as its rank-5 tiled tensor (same memory). *)
  map_loc gpu_loc (fun () -> array3_to_tile5_ow (SZ.v tile) gA);
  map_loc gpu_loc (fun () -> array3_to_tile5_ow (SZ.v tile) gB);
  map_loc gpu_loc (fun () -> array3_to_tile5_ow (SZ.v tile) gC);

  (* Approximation facts in the rank-5 (untiled) world. *)
  untile5_approx eA rA;
  untile5_approx eB rB;
  untile5_approx eC rC;
  flat_untile5 #real #(SZ.v batch) #(SZ.v m) #(SZ.v k) #(SZ.v tile) rA;
  flat_untile5 #real #(SZ.v batch) #(SZ.v k) #(SZ.v n) #(SZ.v tile) rB;
  flat_untile5 #real #(SZ.v batch) #(SZ.v m) #(SZ.v n) #(SZ.v tile) rC;
  flat_untile5 #ta #(SZ.v batch) #(SZ.v m) #(SZ.v k) #(SZ.v tile) eA;
  flat_untile5 #tb #(SZ.v batch) #(SZ.v k) #(SZ.v n) #(SZ.v tile) eB;

  launch_sync
    (bmk_kernel mapA mapB comb mapA_r mapB_r comb_r tile
       #_ #_ #_ #_
       #(c_tile5_layout #(SZ.v batch) #(SZ.v m) #(SZ.v k) tile)
       #(c_tile5_layout #(SZ.v batch) #(SZ.v k) #(SZ.v n) tile)
       #(c_tile5_layout #(SZ.v batch) #(SZ.v m) #(SZ.v n) tile)
       (from_array (tile5_layout #batch #m #k #tile lA) (core (gA)))
       (from_array (tile5_layout #batch #k #n #tile lB) (core (gB)))
       (from_array (tile5_layout #batch #m #n #tile lC) (core (gC)))
       (untile5 eA) (untile5 eB) (untile5 eC)
       (untile5 rA) (untile5 rB) (untile5 rC) fA fB);

  (* Restore the flat array3 views. *)
  map_loc gpu_loc (fun () -> tile5_to_array3_ow (SZ.v tile) gA);
  map_loc gpu_loc (fun () -> tile5_to_array3_ow (SZ.v tile) gB);
  map_loc gpu_loc (fun () -> tile5_to_array3_gc mapA_r mapB_r comb_r (SZ.v tile) gC rA rB rC);
  ();
}

inline_for_extraction noextract
fn bmmcomb_gpu_approx
  (tile : valid_tile)
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { comb `approx2` comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch (m * tile) (k * tile))
  (#lB : layout3 batch (k * tile) (n * tile))
  (#lC : layout3 batch (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array3 et lA { is_global gA })
  (gB : array3 et lB { is_global gB })
  (gC : array3 et lC { is_global gC })
  (rA : chest3 real batch (m * tile) (k * tile))
  (rB : chest3 real batch (k * tile) (n * tile))
  (rC : chest3 real batch (m * tile) (n * tile))
  (#eA : chest3 et batch (m * tile) (k * tile))
  (#eB : chest3 et batch (k * tile) (n * tile))
  (#eC : chest3 et batch (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (bsize_req batch m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest3 et batch (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.bmmcomb comb_r rC rA rB))
{
  gbmmcomb_gpu_approx tile (fun (x:et) -> x) (fun (x:et) -> x) comb
    (fun (r:real) -> r) (fun (r:real) -> r) comb_r gA gB gC rA rB rC;
  MS.gbmmcomb_id comb_r rC rA rB;
  ()
}



(* [bsize_req] at batch one follows from the rank-2 size requirement. *)
let size_req_bsize1 (m n k tile : nat)
  : Lemma (requires m * n <= max_blocks) (ensures bsize_req 1 m n k tile)
  = let _ = max_blocks_explicit in ()

(* ─── rank-2 entry point (batch-one specialization of gbmmcomb) ───────────── *)

inline_for_extraction noextract
fn gmmcomb_gpu_approx
  (tile : valid_tile)
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { comb `approx2` comb_r })
  (#m #n #k : szp)
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  (#lC : layout2 (m * tile) (n * tile))
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 ta lA { is_global gA })
  (gB : array2 tb lB { is_global gB })
  (gC : array2 tc lC { is_global gC })
  (rA : chest2 real (m * tile) (k * tile))
  (rB : chest2 real (k * tile) (n * tile))
  (rC : chest2 real (m * tile) (n * tile))
  (#eA : chest2 ta (m * tile) (k * tile))
  (#eB : chest2 tb (k * tile) (n * tile))
  (#eC : chest2 tc (m * tile) (n * tile))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 tc (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gmmcomb mapA_r mapB_r comb_r rC rA rB))
{
  let afA : squash (all_fit ((m * tile) @| (k * tile) @| INil)) = C.layout2_all_fit lA;
  let afB : squash (all_fit ((k * tile) @| (n * tile) @| INil)) = C.layout2_all_fit lB;
  let afC : squash (all_fit ((m * tile) @| (n * tile) @| INil)) = C.layout2_all_fit lC;
  size_req_bsize1 (SZ.v m) (SZ.v n) (SZ.v k) (SZ.v tile);

  (* cast_in: relayout the rank-2 ownership to its batch-one rank-3 view. *)
  map_loc gpu_loc (fun () -> C.t2_to_t3n (m * tile) (k * tile) afA gA);
  map_loc gpu_loc (fun () -> C.t2_to_t3n (k * tile) (n * tile) afB gB);
  map_loc gpu_loc (fun () -> C.t2_to_t3n (m * tile) (n * tile) afC gC);

  (* carry the approximation facts to the rank-3 chests. *)
  MU.c2_to_c3_approx (m * tile) (k * tile) afA eA rA;
  MU.c2_to_c3_approx (k * tile) (n * tile) afB eB rB;
  MU.c2_to_c3_approx (m * tile) (n * tile) afC eC rC;

  gbmmcomb_gpu_approx tile mapA mapB comb mapA_r mapB_r comb_r
    #1sz #m #n #k
    #(C.l2_to_l3n #(m * tile) #(k * tile) #lA)
    #(C.l2_to_l3n #(k * tile) #(n * tile) #lB)
    #(C.l2_to_l3n #(m * tile) #(n * tile) #lC)
    (relay gA (C.l2_to_l3n #(m * tile) #(k * tile) #lA))
    (relay gB (C.l2_to_l3n #(k * tile) #(n * tile) #lB))
    (relay gC (C.l2_to_l3n #(m * tile) #(n * tile) #lC))
    (C.c2_to_c3n (m * tile) (k * tile) afA rA)
    (C.c2_to_c3n (k * tile) (n * tile) afB rB)
    (C.c2_to_c3n (m * tile) (n * tile) afC rC)
    #(C.c2_to_c3n (m * tile) (k * tile) afA eA)
    #(C.c2_to_c3n (k * tile) (n * tile) afB eB)
    #(C.c2_to_c3n (m * tile) (n * tile) afC eC)
    #fA #fB;

  (* restore the flat rank-2 views of A and B. *)
  map_loc gpu_loc (fun () -> C.t3_to_t2n_ow (m * tile) (k * tile) afA gA);
  map_loc gpu_loc (fun () -> C.t3_to_t2n_ow (k * tile) (n * tile) afB gB);

  (* cast_out for C: lower the batched result to the rank-2 gmmcomb post. *)
  with eC3'. assert (on gpu_loc (relay gC (C.l2_to_l3n #(m * tile) #(n * tile) #lC) |-> eC3'));
  map_loc gpu_loc (fun () -> C.t3_to_t2n (m * tile) (n * tile) afC gC);
  MU.c3_to_c2_approx (m * tile) (n * tile) afC eC3'
    (MS.gbmmcomb mapA_r mapB_r comb_r
      (C.c2_to_c3n (m * tile) (n * tile) afC rC)
      (C.c2_to_c3n (m * tile) (k * tile) afA rA)
      (C.c2_to_c3n (k * tile) (n * tile) afB rB));
  MU.batch1_gmmcomb mapA_r mapB_r comb_r
    (m * tile) (k * tile) (n * tile) afC afA afB rC rA rB;
  ();
}

(* Approximate tiled GEMM: result matrix approximates MS.mmcomb over
   external real matrices rA, rB, rC related by %~ to eA, eB, eC. *)
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
    pure (size_req m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 et (m * tile) (n * tile)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB))
{
  gmmcomb_gpu_approx tile (fun (x:et) -> x) (fun (x:et) -> x) comb
    (fun (r:real) -> r) (fun (r:real) -> r) comb_r gA gB gC rA rB rC;
  MS.gmmcomb_id comb_r rC rA rB;
  ()
}
