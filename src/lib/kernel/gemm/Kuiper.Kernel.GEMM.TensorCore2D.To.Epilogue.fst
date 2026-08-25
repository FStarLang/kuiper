module Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue

open Kuiper.EMatrix.Tiling
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

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

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueLoopStep

(* (i * n + j) / n == i and (i * n + j) % n == j, when j < n. Kept as a
   top-level pure lemma so the nonlinear division/modulo facts type-check in
   a minimal context: inside [epilogue_to]'s large ambient proof state (with
   many size_layout SMTPats already in scope) Z3 does not reliably close
   this goal when it is asserted inline.
   TODO: consider upstreaming to the Kuiper library and reusing from
   Kuiper.Kernel.GEMM.BlockTiling2D's near-identical [epilogue_tile_div_mod]
   if a third call site appears. *)
let div_mod_of_mul_add (n : pos) (i : nat) (j : natlt n)
  : Lemma ((i * n + j) / n == i /\ (i * n + j) % n == j)
  = FStar.Math.Lemmas.lemma_div_plus j i n;
    FStar.Math.Lemmas.small_div j n;
    FStar.Math.Lemmas.lemma_mod_plus j i n;
    FStar.Math.Lemmas.small_mod j n

(* [x / wn * wn + x % wn == x].  Needed in a *type-level* position (it is what
   makes the reindexed [output_fragment_post] argument of the [forevery_ext]
   below well-typed, and what makes the two indexed slprops equal), so it has
   to be a trigger rather than a lemma call.  The pattern is the exact shape of
   the goal, so it cannot fire spuriously. *)
private let div_mul_mod_id_pat (a : nat) (b : pos)
  : Lemma (a / b * b + a % b == a)
    [SMTPat (a / b * b + a % b)]
= FStar.Math.Lemmas.euclidean_division_definition a b;
  FStar.Math.Lemmas.swap_mul b (a / b)

#push-options ""
inline_for_extraction noextract
fn epilogue_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd,
     scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n : szp)
  (gC : array2 et_cd (rm m n))
  (#fC : perm)
  (#eC : chest2 et_cd m n)
  (#rC : chest2 real m n)
  (gD : array2 et_cd (rm m n))
  (d : epilogue_dims m n)
  (sh : c_shmems
    (shmems_desc_to et_ab et_acc d.bm d.bn d.bk d.tm d.tn d.nthr))
  (accFrags : array
    (fragment et_acc FragAcc d.tm d.tn d.tk FragLAcc))
  (rAcc : chest2 real (d.wm * d.tm) (d.wn * d.tn))
  (bid : szlt (m / d.bm * (n / d.bn)))
  (tid : szlt d.nthr)
  (#_ : squash (Pulse.Lib.Array.length accFrags == d.wm * d.wn))
  norewrite
  requires
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      d.bm d.bn d.bk d.tm d.tn d.tk d.wm d.wn d.nthr
      sh accFrags rAcc tid **
    output_lane_live gD d.bm d.bn d.tm d.tn d.wm d.wn bid tid
  ensures
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      d.bm d.bn d.bk d.tm d.tn d.tk d.wm d.wn d.nthr
      sh accFrags rAcc tid **
    output_lane_approximates
      gD d.bm d.bn d.tm d.tn d.wm d.wn bid tid
      (epilogue_warp_output comb_r rC
        d.bm d.bn d.tm d.tn d.wm d.wn bid tid rAcc)
{
  let bm = d.bm;
  let bn = d.bn;
  let bk = d.bk;
  let tm = d.tm;
  let tn = d.tn;
  let tk = d.tk;
  let wm = d.wm;
  let wn = d.wn;
  let nthr = d.nthr;
  rewrite each d.bm as bm;
  rewrite each d.bn as bn;
  rewrite each d.bk as bk;
  rewrite each d.tm as tm;
  rewrite each d.tn as tn;
  rewrite each d.tk as tk;
  rewrite each d.wm as wm;
  rewrite each d.wn as wn;
  rewrite each d.nthr as nthr;
  assert pure (constraints bm bn bk tm tn tk wm wn);
  assert pure (bm /?+ m /\ bn /?+ n);
  assert pure (SZ.fits (bm * bk) /\ SZ.fits (bk * bn));
  assert pure (SZ.fits (wm * wn));
  assert pure (SZ.fits ((nthr / warp_size) * tm * tn));
  assert pure (SZ.fits (tm * tn + warp_size));
  unfold epilogue_frame #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC fC eC rC
    bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;
  let wid = tid /^ warp_size;
  let lane = tid %^ warp_size;
  let mrow = bid /^ (n /^ bn);
  let mcol = bid %^ (n /^ bn);
  let warpRow = wid /^ (bn /^ (wn *^ tn));
  let warpCol = wid %^ (bn /^ (wn *^ tn));
  let rCWarp =
    ematrix_subtile
      (ematrix_subtile rC bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;

  unfold output_lane_live gD bm bn tm tn wm wn bid tid;
  forevery_flatten _;
  forevery_iso
    (Kuiper.Bijection.bij_nat_prod #wm #wn)
    (fun (xy : (natlt wm & natlt wn)) ->
      live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid (tid / warp_size) xy._1 xy._2)
        (tid % warp_size));
  rewrite each (tid / warp_size) as wid;
  rewrite each (tid % warp_size) as lane;
  let output_live =
    (fun (idx : natlt (wm * wn)) ->
      live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (idx / wn) (idx % wn))
        lane);
  forevery_ext
    (fun (idx : natlt (wm * wn)) ->
      live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (idx / wn) (idx % wn))
        lane)
    output_live;
  forevery_map
    #(natlt (wm * wn))
    output_live
    (fun idx ->
      output_fragment_state_at
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) 0 idx)
    fn idx {
      rewrite output_live idx as
        live_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid (idx / wn) (idx % wn))
          lane;
      Kuiper.Conditional.if_intro_false
        (output_fragment_post
          gD bm bn tm tn wm wn bid wid lane
          (chest_comb comb_r rCWarp rAcc) idx);
      Kuiper.Conditional.if_rewrite_bool
        false (idx < 0)
        (output_fragment_post
          gD bm bn tm tn wm wn bid wid lane
          (chest_comb comb_r rCWarp rAcc) idx);
      Kuiper.Conditional.if_intro_true
        (live_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid (idx / wn) (idx % wn))
          lane);
      Kuiper.Conditional.if_rewrite_bool
        true (not (idx < 0))
        (live_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid (idx / wn) (idx % wn))
          lane);
      fold if_else_ (idx < 0)
        (output_fragment_post
          gD bm bn tm tn wm wn bid wid lane
          (chest_comb comb_r rCWarp rAcc) idx)
        (live_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid (idx / wn) (idx % wn))
          lane);
      fold output_fragment_state_at
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) 0 idx;
    };
  fold output_epilogue_state
    gD bm bn tm tn wm wn bid wid lane
    (chest_comb comb_r rCWarp rAcc) 0;
  fold epilogue_frame #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC fC eC rC
    bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;

  let mut idx : szle (wm * wn) = 0sz;
  while (!idx <^ wm *^ wn)
    invariant live idx
    invariant
      epilogue_frame #et_ab #et_cd #et_acc
        #_ #_ #_ #_ #_
        #m #n gC fC eC rC
        bm bn bk tm tn tk wm wn nthr
        sh accFrags rAcc tid
    invariant
      output_epilogue_state
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) !idx
    decreases (wm * wn - !idx)
  {
    let before = !idx;
    epilogue_loop_step comb comb_r gC gD
      bm bn bk tm tn tk wm wn nthr
      sh accFrags rAcc bid tid wid lane rCWarp idx before;
    let next = !idx;
    assert pure (SZ.v next == SZ.v before + 1);
    rewrite each (SZ.v before + 1) as (SZ.v next);
  };

  rewrite each !idx as (wm *^ wn);
  unfold output_epilogue_state
    gD bm bn tm tn wm wn bid wid lane
    (chest_comb comb_r rCWarp rAcc) (wm * wn);
  forevery_map
    #(natlt (wm * wn))
    (fun (fi : natlt (wm * wn)) ->
      output_fragment_state_at
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (wm * wn) fi)
    (fun (fi : natlt (wm * wn)) ->
      output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) fi)
    fn fi {
      unfold output_fragment_state_at
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (wm * wn) fi;
      unfold if_else_ (fi < wm * wn)
        (output_fragment_post
          gD bm bn tm tn wm wn bid wid lane
          (chest_comb comb_r rCWarp rAcc) fi)
        (live_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid (fi / wn) (fi % wn))
          lane);
      Kuiper.Conditional.if_rewrite_bool
        (fi < wm * wn) true
        (output_fragment_post
          gD bm bn tm tn wm wn bid wid lane
          (chest_comb comb_r rCWarp rAcc) fi);
      Kuiper.Conditional.if_elim_true
        (output_fragment_post
          gD bm bn tm tn wm wn bid wid lane
          (chest_comb comb_r rCWarp rAcc) fi);
      Kuiper.Conditional.if_rewrite_bool
        (not (fi < wm * wn)) false
        (live_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid (fi / wn) (fi % wn))
          lane);
      Kuiper.Conditional.if_elim_false
        (live_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid (fi / wn) (fi % wn))
          lane);
    };
  forevery_ext
    (fun (fi : natlt (wm * wn)) ->
      output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) fi)
    (fun (fi : natlt (wm * wn)) ->
      output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc)
        (fi / wn * wn + fi % wn));
  forevery_iso_back
    (Kuiper.Bijection.bij_nat_prod #wm #wn)
    (fun xy ->
      output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc)
        (xy._1 * wn + xy._2));
  forevery_map
    #(natlt wm & natlt wn)
    (fun (xy : (natlt wm & natlt wn)) ->
      output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc)
        (xy._1 * wn + xy._2))
    (fun (xy : (natlt wm & natlt wn)) ->
      exists* (eD : chest2 et_cd tm tn).
        own_lane_cells
          (output_fragment gD bm bn tm tn wm wn
            bid wid xy._1 xy._2)
          eD lane **
        pure (eD %~
          ematrix_subtile
            (chest_comb comb_r rCWarp rAcc)
            tm tn xy._1 xy._2))
    fn xy {
      unfold output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc)
        (xy._1 * wn + xy._2);
      div_mod_of_mul_add wn xy._1 xy._2;
      assert pure (
        (xy._1 * wn + xy._2) / wn == xy._1);
      assert pure (
        (xy._1 * wn + xy._2) % wn == xy._2);
      rewrite each
        ((xy._1 * wn + xy._2) / wn)
      as xy._1;
      rewrite each
        ((xy._1 * wn + xy._2) % wn)
      as xy._2;
    };
  forevery_unflatten
    (fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eD : chest2 et_cd tm tn).
        own_lane_cells
          (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
          eD lane **
        pure (eD %~
          ematrix_subtile
            (chest_comb comb_r rCWarp rAcc)
            tm tn mi nj));
  rewrite each rCWarp as
    ematrix_subtile
      (ematrix_subtile rC bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
  rewrite each (SZ.v mrow) as (SZ.v bid / (SZ.v n / SZ.v bn));
  rewrite each (SZ.v mcol) as (SZ.v bid % (SZ.v n / SZ.v bn));
  rewrite each (SZ.v warpRow) as
    ((SZ.v tid / warp_size) / (SZ.v bn / (SZ.v wn * SZ.v tn)));
  rewrite each (SZ.v warpCol) as
    ((SZ.v tid / warp_size) % (SZ.v bn / (SZ.v wn * SZ.v tn)));
  rewrite each (SZ.v wid) as (SZ.v tid / warp_size);
  rewrite each (SZ.v lane) as (SZ.v tid % warp_size);
  fold output_lane_approximates
    gD bm bn tm tn wm wn bid tid
    (chest_comb comb_r
      (ematrix_subtile
        (ematrix_subtile rC bm bn
          (bid / (n / bn)) (bid % (n / bn)))
        (wm * tm) (wn * tn)
        ((tid / warp_size) / (bn / (wn * tn)))
        ((tid / warp_size) % (bn / (wn * tn))))
      rAcc);
  rewrite
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid
  as
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      d.bm d.bn d.bk d.tm d.tn d.tk d.wm d.wn d.nthr
      sh accFrags rAcc tid;
  rewrite
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn
            (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
        rAcc)
  as
    output_lane_approximates gD
      d.bm d.bn d.tm d.tn d.wm d.wn bid tid
      (epilogue_warp_output comb_r rC
        d.bm d.bn d.tm d.tn d.wm d.wn bid tid rAcc);
}
#pop-options
