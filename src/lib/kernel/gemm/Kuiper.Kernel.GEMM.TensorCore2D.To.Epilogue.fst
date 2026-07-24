module Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue

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

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
module BW = Kuiper.Barrier.Warp

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

let output_fragment_post
  (#et : Type0) {| scalar et, real_like et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : natlt warp_size)
  (rD : chest2 real (wm * tm) (wn * tn))
  (idx : natlt (wm * wn))
  : slprop
= exists* (eD : chest2 et tm tn).
    own_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (idx / wn) (idx % wn))
      eD lane **
    pure (eD %~ ematrix_subtile rD tm tn (idx / wn) (idx % wn))

let if_else_ (b : bool) (p q : slprop) : slprop =
  if_ b p ** if_ (not b) q

ghost
fn warp_emp_transform ()
  requires forall+ (_ : natlt BW.warp_size). emp
  ensures forall+ (_ : natlt BW.warp_size). emp
{}

  ghost
  fn forevery_extract_replace_eqtype
    (#a : eqtype)
    (z : a)
    (p1 p2 : a -> slprop)
    (#_ : squash (forall x. x =!= z ==> p1 x == p2 x))
    requires forall+ (x : a). p1 x
    ensures
      p1 z **
      (p2 z @==> forall+ (x : a). p2 x)
  {
    forevery_extract_if_eqtype z p1;
    intro_trade #emp_inames
      (p2 z)
      (forall+ (x : a). p2 x)
      (forall+ (x : a). if x = z then emp else p1 x)
      fn _ {
        forevery_map #a
          (fun x -> if x = z then emp else p1 x)
          (fun x -> if x = z then emp else p2 x)
          fn x {
            let is_z = x = z;
            if is_z {
              rewrite
                (if x = z then emp else p1 x)
              as emp;
              rewrite emp as
                (if x = z then emp else p2 x);
            } else {
              rewrite
                (if x = z then emp else p1 x)
              as p1 x;
              rewrite p1 x as p2 x;
              rewrite p2 x as
                (if x = z then emp else p2 x);
            }
          };
        forevery_unextract_if_eqtype z p2;
      };
  }

  ghost
  fn forevery_insert_replace_eqtype
    (#a : eqtype)
    (z : a)
    (p1 p2 : a -> slprop)
    (#_ : squash (forall x. x =!= z ==> p1 x == p2 x))
    requires
      p2 z **
      (forall+ (x : a { x =!= z }). p1 x)
    ensures forall+ (x : a). p2 x
  {
    forevery_map
      #(x : a { x =!= z })
      p1 p2
      fn x {
        rewrite p1 x as p2 x;
      };
    forevery_insert
      #a
      #(fun x -> x =!= z)
      p2 z;
    forevery_refine_ext
      #a
      #(fun x -> x =!= z \/ z == x)
      (fun _ -> True)
      p2;
  }

  let output_fragment_state_at
    (#et : Type0) {| scalar et, real_like et |}
    (#m #n : nat)
    (gD : array2 et (rm m n))
    (bm bn tm tn wm wn : pos)
    (#_ : squash (bm /?+ m /\ bn /?+ n /\
                  wm * tm /?+ bm /\ wn * tn /?+ bn))
    (bid : natlt (m / bm * (n / bn)))
    (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
    (lane : natlt warp_size)
    (rD : chest2 real (wm * tm) (wn * tn))
    (done : natle (wm * wn))
    (idx : natlt (wm * wn))
    : slprop
  = if_else_ (idx < done)
      (output_fragment_post
        gD bm bn tm tn wm wn bid wid lane rD idx)
      (live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (idx / wn) (idx % wn))
        lane)

  let output_epilogue_state
  (#et : Type0) {| scalar et, real_like et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : natlt warp_size)
  (rD : chest2 real (wm * tm) (wn * tn))
  (done : natle (wm * wn))
  : slprop
= forall+ (idx : natlt (wm * wn)).
    output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane rD done idx

ghost
fn output_epilogue_extract_step
  (#et : Type0) {| scalar et, real_like et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : natlt warp_size)
  (rD : chest2 real (wm * tm) (wn * tn))
  (idx_ref : ref (szle (wm * wn)))
  (done : szle (wm * wn) { SZ.v done < wm * wn })
  preserves idx_ref |-> done
  requires
    output_epilogue_state
      gD bm bn tm tn wm wn bid wid lane rD (SZ.v done)
  ensures
    output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane rD
      (SZ.v done) (SZ.v done) **
    (output_fragment_state_at
        gD bm bn tm tn wm wn bid wid lane rD
        (SZ.v done + 1) (SZ.v done)
      @==>
      forall+ (idx : natlt (wm * wn)).
        output_fragment_state_at
          gD bm bn tm tn wm wn bid wid lane rD (SZ.v done + 1) idx)
{
  unfold output_epilogue_state
    gD bm bn tm tn wm wn bid wid lane rD (SZ.v done);
  forevery_extract_replace_eqtype
    #(natlt (wm * wn))
    (SZ.v done)
    (output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane rD (SZ.v done))
    (output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane rD (SZ.v done + 1));
}

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
  (#_ : squash (eC %~ rC))
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn nthr : szp{
    constraints bm bn bk tm tn tk wm wn /\
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (SZ.fits (tm * tn + warp_size)))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  preserves
    gpu **
    thread_id nthr tid **
    gC |-> Frac fC eC **
    fragarrayAcc_approximates wm wn accFrags rAcc
  requires
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    output_lane_live gD bm bn tm tn wm wn bid tid
  ensures
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn
            (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
        rAcc)
{
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

  with eScratch.
    unfold scratch_tile_live bm bn bk tm tn nthr sh tid;
  let sTile = scratch_tile_st bm bn bk tm tn nthr sh wid;

  let mut idx : szle (wm * wn) = 0sz;
  while (!idx <^ wm *^ wn)
    invariant live idx
    invariant
      exists* (eScratchLoop : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh
          (SZ.v tid / warp_size)
          |-> Frac (1.0R /. warp_size) eScratchLoop
    invariant
      output_epilogue_state
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) !idx
    invariant
      fragarrayAcc_approximates wm wn accFrags rAcc
    decreases (wm * wn - !idx)
  {
    with vidx. assert idx |-> vidx;
    output_epilogue_extract_step
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) idx vidx;
    unfold output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v vidx) (SZ.v vidx);
    unfold if_else_ (SZ.v vidx < SZ.v vidx)
      (output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (SZ.v vidx))
      (live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (SZ.v vidx / wn) (SZ.v vidx % wn))
        lane);
    Kuiper.Conditional.if_rewrite_bool
      (SZ.v vidx < SZ.v vidx) false
      (output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (SZ.v vidx));
    Kuiper.Conditional.if_elim_false
      (output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (SZ.v vidx));
    Kuiper.Conditional.if_rewrite_bool
      (not (SZ.v vidx < SZ.v vidx)) true
      (live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (SZ.v vidx / wn) (SZ.v vidx % wn))
        lane);
    Kuiper.Conditional.if_elim_true
      (live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (SZ.v vidx / wn) (SZ.v vidx % wn))
        lane);

    unfold fragarrayAcc_approximates wm wn accFrags rAcc;
    with eAccFrags. assert accFrags `array_fragment_pts_to` eAccFrags;
    array_fragment_pts_to_ref accFrags;
    array_fragment_extract_ro accFrags vidx;

    rewrite each
      scratch_tile bm bn bk tm tn nthr sh (SZ.v tid / warp_size)
    as sTile;
    mma_store accFrags.(!idx) sTile;

    BW.warp_barrier_wait ()
      (fun _ -> emp)
      (fun _ -> emp)
      warp_emp_transform;

    let rCFrag =
      ematrix_subtile rCWarp tm tn (SZ.v vidx / wn) (SZ.v vidx % wn);
    let rAccFrag =
      ematrix_subtile rAcc tm tn (SZ.v vidx / wn) (SZ.v vidx % wn);

    assert pure (Seq.Base.index eAccFrags vidx %~ rAccFrag);
    epilogue_fragment_from_warp comb comb_r gC
      bm bn tm tn wm wn mrow mcol warpRow warpCol bid wid
      #_ #_ #_ #rC #_
      sTile #(Seq.Base.index eAccFrags vidx) #rAccFrag #_
      gD !idx lane;
    rewrite each sTile as
      scratch_tile bm bn bk tm tn nthr sh (SZ.v tid / warp_size);

    with eOut.
      assert own_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (SZ.v vidx / wn) (SZ.v vidx % wn))
        eOut lane;
    fold output_fragment_post
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) vidx;
    Kuiper.Conditional.if_intro_true
      (output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (SZ.v vidx));
    Kuiper.Conditional.if_rewrite_bool
      true (SZ.v vidx < SZ.v vidx + 1)
      (output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (SZ.v vidx));
    Kuiper.Conditional.if_intro_false
      (live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (SZ.v vidx / wn) (SZ.v vidx % wn))
        lane);
    Kuiper.Conditional.if_rewrite_bool
      false (not (SZ.v vidx < SZ.v vidx + 1))
      (live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (SZ.v vidx / wn) (SZ.v vidx % wn))
        lane);
    fold if_else_ (SZ.v vidx < SZ.v vidx + 1)
      (output_fragment_post
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (SZ.v vidx))
      (live_lane_cells
        (output_fragment gD bm bn tm tn wm wn
          bid wid (SZ.v vidx / wn) (SZ.v vidx % wn))
        lane);
    fold output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc)
      (SZ.v vidx + 1) (SZ.v vidx);
    Pulse.Lib.Trade.elim_trade
      (output_fragment_state_at
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc)
        (SZ.v vidx + 1) (SZ.v vidx))
      (forall+ (fi : natlt (wm * wn)).
        output_fragment_state_at
          gD bm bn tm tn wm wn bid wid lane
          (chest_comb comb_r rCWarp rAcc) (SZ.v vidx + 1) fi);
    ambig_trade_elim ();
    fold fragarrayAcc_approximates wm wn accFrags rAcc;

    fold output_epilogue_state
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v vidx + 1);
    with vnext. assert idx |-> vnext;
    idx := sz_succ !idx;
    rewrite each
      (SZ.v vnext + 1)
    as
      (SZ.v (sz_succ vnext));
  };

  rewrite each !idx as (wm *^ wn);
  fold scratch_tile_live bm bn bk tm tn nthr sh tid;
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
}
