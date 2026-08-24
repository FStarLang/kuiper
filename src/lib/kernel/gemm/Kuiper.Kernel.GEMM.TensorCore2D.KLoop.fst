module Kuiper.Kernel.GEMM.TensorCore2D.KLoop

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 120"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module Chest = Kuiper.Chest
module MU = Kuiper.Kernel.GEMM.Util

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Subproducts
#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
// Convert a block-local tile coordinate to the corresponding global offset.
let tile_offset
  (outer tile : pos)
  (#_ : squash (tile /? outer))
  (block inner global : nat)
  (#_ : squash (global == block * (outer / tile) + inner))
  (_ : unit)
  : Lemma (tile * global == outer * block + inner * tile)
= calc (==) {
    tile * global;
    == {}
    tile * (block * (outer / tile) + inner);
    == { Math.Lemmas.distributivity_add_right tile (block * (outer / tile)) inner }
    tile * (block * (outer / tile)) + tile * inner;
    == { Math.Lemmas.paren_mul_right tile block (outer / tile);
         Math.Lemmas.swap_mul tile block;
         Math.Lemmas.paren_mul_right block tile (outer / tile) }
    block * (tile * (outer / tile)) + tile * inner;
    == { Math.Lemmas.lemma_div_exact outer tile }
    block * outer + tile * inner;
    == { Math.Lemmas.swap_mul block outer; Math.Lemmas.swap_mul tile inner }
    outer * block + inner * tile;
  }

let zero_tile_offset (tile : pos) (i : nat) (_ : unit)
  : Lemma (tile * i == tile * i + 0 * tile)
= ()

// A nested subtile is the corresponding cell of the directly tiled matrix.
let nested_subtile_tiled_cell
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (outer_rows : pos { outer_rows /? rows })
  (outer_cols : pos { outer_cols /? cols })
  (inner_rows : pos { inner_rows /? outer_rows /\ inner_rows /? rows })
  (inner_cols : pos { inner_cols /? outer_cols /\ inner_cols /? cols })
  (outer_row : natlt (rows / outer_rows))
  (outer_col : natlt (cols / outer_cols))
  (inner_row : natlt (outer_rows / inner_rows))
  (inner_col : natlt (outer_cols / inner_cols))
  (global_row : natlt (rows / inner_rows))
  (global_col : natlt (cols / inner_cols))
  (#_ : squash (inner_rows * global_row ==
                outer_rows * outer_row + inner_row * inner_rows))
  (#_ : squash (inner_cols * global_col ==
                outer_cols * outer_col + inner_col * inner_cols))
  (outer : chest2 et outer_rows outer_cols {
    outer == ematrix_subtile em outer_rows outer_cols outer_row outer_col })
  (_ : unit)
  : Lemma (
      ematrix_subtile outer inner_rows inner_cols inner_row inner_col
      == acc2 (ematrix_tiled em inner_rows inner_cols) global_row global_col)
= Math.Lemmas.swap_mul inner_rows global_row;
  Math.Lemmas.swap_mul outer_rows outer_row;
  Math.Lemmas.swap_mul inner_cols global_col;
  Math.Lemmas.swap_mul outer_cols outer_col;
  macc_ematrix_tiled em inner_rows inner_cols global_row global_col;
  Kuiper.Chest.ext
    (ematrix_subtile outer inner_rows inner_cols inner_row inner_col)
    (acc2 (ematrix_tiled em inner_rows inner_cols) global_row global_col)

// One tile product advances the tiled partial matmul by exactly one step.
let partial_matmul_step
  (#rows #shared #cols #tm #tk #tn : nat)
  (z : chest2 real tm tn)
  (a : chest2 (chest2 real tm tk) rows shared)
  (b : chest2 (chest2 real tk tn) shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (vk : natlt shared)
  (rAcc : chest2 real tm tn {
    rAcc == __gmatmul_single z matmul matplus a b row col vk })
  (aTile : chest2 real tm tk { aTile == acc2 a row vk })
  (bTile : chest2 real tk tn { bTile == acc2 b vk col })
  (_ : unit)
  : Lemma (
      rAcc `matplus` matmul aTile bTile
      == __gmatmul_single z matmul matplus a b row col (vk + 1))
= let lhs : chest2 real tm tn = rAcc `matplus` matmul aTile bTile in
  let rhs : chest2 real tm tn =
    __gmatmul_single z matmul matplus a b row col (vk + 1) in
  let aux (i : natlt tm) (j : natlt tn) :
    Lemma (acc2 lhs i j == acc2 rhs i j) =
    calc (==) {
      acc2 lhs i j;
      == {}
      acc2 (__gmatmul_single z matmul matplus a b row col vk
        `matplus` matmul (acc2 a row vk) (acc2 b vk col)) i j;
      == { __gmatmul_single_lemma
             z matmul matplus a b row col (vk + 1) }
      acc2 (__gmatmul_single z matmul matplus
        a b row col (vk + 1)) i j;
      == {}
      acc2 rhs i j;
    } in
  Classical.forall_intro_2 aux;
  assert (Kuiper.EMatrix.equal lhs rhs)
#pop-options

// Expose the opaque invariant at one concrete index.
let kacc_inv_eq
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (mrA : chest2 real m k)
  (mrB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn))
  (gwRow : natlt (m/(wm*tm)))
  (gwCol : natlt (n/(wn*tn)))
  (vk : nat { vk <= k / bk })
: Lemma (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk
           == __gmatmul_single rAcc0 matmul matplus
                (ematrix_tiled mrA (wm*tm) bk) (ematrix_tiled mrB bk (wn*tn))
                gwRow gwCol vk)
= reveal_opaque (`%kacc_inv)
    (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk)

// Isolate the per-tile advance from kf's large SMT context.
#push-options "--z3rlimit 100"
noextract
ghost fn advance_kloop_invariant
  (#et:Type0) {| scalar et, real_like et |}
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm/(wm*tm)))
  (warpCol : natlt (bn/(wn*tn)))
  (gwRow : natlt (m/(wm*tm)) { gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : natlt (n/(wn*tn)) { gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (vk : natlt (k / bk))
  // These matrices are already map-applied at the call site.
  (mrA : chest2 real m k)
  (mrB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn) { rAcc0 == const _ 0.0R })
  (rAcc  : chest2 real (wm*tm) (wn*tn))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  // Pulse reads the wrapper's result type without rechecking the raw term.
  (#_ : squash (rAcc ==
          kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk))
  (mrA_sub : chest2 real bm bk { mrA_sub == ematrix_subtile mrA bm bk mrow vk })
  (mrB_sub : chest2 real bk bn { mrB_sub == ematrix_subtile mrB bk bn vk mcol })
  (accFrags : array (fragment et FragAcc tm tn tk FragLAcc)
              { Pulse.Lib.Array.length accFrags == wm*wn })
  requires
    fragarrayAcc_approximates wm wn accFrags
      (rAcc `matplus`
        matmul (ematrix_subtile mrA_sub (wm*tm) bk warpRow 0)
               (ematrix_subtile mrB_sub bk (wn*tn) 0 warpCol))
  ensures
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol (vk + 1))
{
  // Expose the opaque invariant only at the two concrete indices involved.
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk;
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol (vk + 1);

  tile_offset bm (wm*tm) mrow warpRow gwRow ();
  tile_offset bn (wn*tn) mcol warpCol gwCol ();
  zero_tile_offset bk vk ();
  nested_subtile_tiled_cell mrA
    bm bk (wm*tm) bk mrow vk warpRow 0 gwRow vk mrA_sub ();
  nested_subtile_tiled_cell mrB
    bk bn bk (wn*tn) vk mcol 0 warpCol vk gwCol mrB_sub ();
  partial_matmul_step rAcc0
    (ematrix_tiled mrA (wm*tm) bk)
    (ematrix_tiled mrB bk (wn*tn))
    gwRow gwCol vk rAcc
    (ematrix_subtile mrA_sub (wm*tm) bk warpRow 0)
    (ematrix_subtile mrB_sub bk (wn*tn) 0 warpCol) ();

  rewrite_fragarrayAcc wm wn accFrags
    (rAcc `matplus`
      matmul (ematrix_subtile mrA_sub (wm*tm) bk warpRow 0)
             (ematrix_subtile mrB_sub bk (wn*tn) 0 warpCol))
    (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol (vk + 1));
}
#pop-options

// Wraps the per-k-tile compute step (subproducts_tc_2d) TOGETHER with the
// ghost accumulator advance, exposing a pre/post stated ONLY in the clean
// [kacc_inv] (ematrix_tiled) form.  The map-aware generalization made
// subproducts' postcondition mention [ematrix_subtile (chest_map mapX ..)],
// which drives the [chest_map_subtile_comm] SMTPat.  Framing that call in kf's
// large ambient context (barrier / shared-memory / bp_sharing predicates, plus
// many ematrix_subtile terms) caused a multi-GB Z3 E-matching blowup.  By
// isolating subproducts' framing to this helper's SMALL context -- and keeping
// [chest_map]-subtile terms entirely OUT of kf (the helper's pre/post use
// [kacc_inv], which unfolds to [ematrix_tiled]-based __gmatmul_single, never
// [ematrix_subtile (chest_map ..)]) -- the SMTPat can no longer explode.
#push-options "--z3rlimit 100"
#restart-solver
inline_for_extraction noextract
fn ktile_advance
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c, real_like et_ab, real_like et_c |}
  // NOTE: m n k are [szp] (sizet), NOT [nat].  kf's dims are sizet; the
  // sz->nat coercion [sizet_to_nat] is GTot, so taking [nat] params here would
  // make kf's stateful call to this fn GHOST (Pulse "Application of a stateful
  // computation cannot have a ghost effect").  subproducts_tc_2d uses szp/szlt
  // for the same reason.  Coercions to nat happen inside slprops / ghost-fn
  // arguments below, where GTot is fine.
  (m n k : szp)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (aFrags     : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags     : array (fragment et_ab FragB tm tn tk FragLRM))
  (accFrags   : array (fragment et_c FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm*wn))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#eA_sub : chest2 et_ab bm bk)
  (#eB_sub : chest2 et_ab bk bn)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (vk : szlt (k / bk))
  (rA_sub : chest2 real bm bk
            { eA_sub %~ rA_sub /\ rA_sub == ematrix_subtile rA bm bk mrow vk })
  (rB_sub : chest2 real bk bn
            { eB_sub %~ rB_sub /\ rB_sub == ematrix_subtile rB bk bn vk mcol })
  (emA emB : et_ab -> et_ab)
  (mapA mapB : real -> real)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (rAcc0 : chest2 real (wm*tm) (wn*tn) { rAcc0 == const _ 0.0R })
  (warpRow : szlt (bm/(wm*tm)))
  (warpCol : szlt (bn/(wn*tn)))
  // NOTE: gwRow/gwCol are [enatlt] (ERASED) to match kf's erased locals.  If
  // they were informative [natlt], passing kf's erased gwRow/gwCol here would
  // require a [reveal], which gives this STATEFUL application a ghost effect
  // (Pulse "Application of a stateful computation cannot have a ghost effect").
  (gwRow : enatlt (m/(wm*tm)) { gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : enatlt (n/(wn*tn)) { gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (#fA #fB : perm)
  ()
  preserves gpu
  preserves sA |-> Frac fA eA_sub
  preserves sB |-> Frac fB eB_sub
  preserves live aFrags ** live bFrags
  requires
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol vk)
  ensures
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol (vk + 1))
{
  with rAcc. assert fragarrayAcc_approximates wm wn accFrags rAcc;

  subproducts_tc_2d bm bn bk tm tn tk wm wn aFrags bFrags accFrags
    sA sB
    rA_sub rB_sub
    emA emB mapA mapB
    rAcc
    warpRow warpCol;

  // advance_kloop_invariant needs its subtile args in [ematrix_subtile (chest_map
  // ..) ..] form, but kf supplies them as [chest_map .. (ematrix_subtile ..)].
  // Bridge with the commutation lemma explicitly (its SMTPat was removed to avoid
  // an E-matching blowup, so it must be invoked by hand here).
  chest_map_subtile_comm mapA rA bm bk mrow vk;
  chest_map_subtile_comm mapB rB bk bn vk mcol;

  advance_kloop_invariant
    m n k
    bm bn bk tm tn tk wm wn
    mrow mcol
    warpRow warpCol
    gwRow gwCol
    vk
    (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
    rAcc0 rAcc
    (Chest.chest_map mapA rA_sub) (Chest.chest_map mapB rB_sub)
    accFrags;
}
#pop-options
