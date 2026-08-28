module Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueLoopStep

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Epilogue
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

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

(* [output_fragment_state_at] mentions [done] only through the boolean
   [idx < done], so bumping [done] by one leaves every index other than [done]
   itself unchanged.  Discharged here, in a small context: inferred inline as
   the [#_ : squash (forall x. x =!= z ==> p1 x == p2 x)] witness of the
   [forevery_extract_replace_eqtype] call below, the query times out. *)
let output_fragment_state_at_bump
  (#et : Type0) {| scalar et, real_like et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#pf : squash (bm /?+ m /\ bn /?+ n /\
                 wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : natlt warp_size)
  (rD : chest2 real (wm * tm) (wn * tn))
  (done : nat { done < wm * wn })
  : Lemma
    (forall (x : natlt (wm * wn)).
       x =!= done ==>
       output_fragment_state_at gD bm bn tm tn wm wn bid wid lane rD done x
       == output_fragment_state_at gD bm bn tm tn wm wn bid wid lane rD (done + 1) x)
= introduce forall (x : natlt (wm * wn)).
      x =!= done ==>
      output_fragment_state_at #_ #_ #_ #_ #_ gD bm bn tm tn wm wn #pf bid wid lane rD done x
      == output_fragment_state_at #_ #_ #_ #_ #_ gD bm bn tm tn wm wn #pf bid wid lane rD (done + 1) x
  with introduce _ ==> _
  with begin
    assert ((x < done) == (x < done + 1))
  end

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
  output_fragment_state_at_bump
    gD bm bn tm tn wm wn bid wid lane rD (SZ.v done);
  forevery_extract_replace_eqtype
    #(natlt (wm * wn))
    (SZ.v done)
    (output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane rD (SZ.v done))
    (output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane rD (SZ.v done + 1));
}

(* The epilogue-state step below needs 4x the file's default rlimit. *)
#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn epilogue_loop_step
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
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (SZ.fits (wm * wn)))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (SZ.fits (tm * tn + warp_size)))
  (sh : c_shmems
    (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (accFrags : array
    (fragment et_acc FragAcc tm tn tk FragLAcc))
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (wid : szlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : szlt warp_size)
  (#_ : squash (
    SZ.v wid == SZ.v tid / warp_size /\
    SZ.v lane == SZ.v tid % warp_size))
  (rCWarp : chest2 real (wm * tm) (wn * tn))
  (#_ : squash (
    rCWarp == epilogue_warp_input rC bm bn tm tn wm wn bid tid))
  (idx : ref (szle (wm * wn)))
  (done : szle (wm * wn) { SZ.v done < wm * wn })
  norewrite
  requires
    idx |-> done **
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      bm bn bk tm tn tk wm wn nthr
      sh accFrags rAcc tid **
    output_epilogue_state
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc)
      (SZ.v done)
  ensures
    (exists* (next : szle (wm * wn)).
      idx |-> next **
      pure (SZ.v next == SZ.v done + 1)) **
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      bm bn bk tm tn tk wm wn nthr
      sh accFrags rAcc tid **
    output_epilogue_state
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc)
      (SZ.v done + 1)
{
  assert pure (constraints bm bn bk tm tn tk wm wn);
  assert pure (bm /?+ m /\ bn /?+ n);
  assert pure (SZ.fits (bm * bk) /\ SZ.fits (bk * bn));
  assert pure (SZ.fits ((nthr / warp_size) * tm * tn));
  assert pure (SZ.fits (tm * tn + warp_size));
  unfold epilogue_frame #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC fC eC rC
    bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;

  let mrow = bid /^ (n /^ bn);
  let mcol = bid %^ (n /^ bn);
  let warpRow = wid /^ (bn /^ (wn *^ tn));
  let warpCol = wid %^ (bn /^ (wn *^ tn));
  assert pure (
    rCWarp ==
    ematrix_subtile
      (ematrix_subtile rC bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol);
  output_epilogue_extract_step
    gD bm bn tm tn wm wn bid wid lane
    (chest_comb comb_r rCWarp rAcc) idx done;
  unfold output_fragment_state_at
    gD bm bn tm tn wm wn bid wid lane
    (chest_comb comb_r rCWarp rAcc) (SZ.v done) (SZ.v done);
  unfold if_else_ (SZ.v done < SZ.v done)
    (output_fragment_post
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v done))
    (live_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (SZ.v done / wn) (SZ.v done % wn))
      lane);
  Kuiper.Conditional.if_rewrite_bool
    (SZ.v done < SZ.v done) false
    (output_fragment_post
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v done));
  Kuiper.Conditional.if_elim_false
    (output_fragment_post
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v done));
  Kuiper.Conditional.if_rewrite_bool
    (not (SZ.v done < SZ.v done)) true
    (live_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (SZ.v done / wn) (SZ.v done % wn))
      lane);
  Kuiper.Conditional.if_elim_true
    (live_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (SZ.v done / wn) (SZ.v done % wn))
      lane);

  unfold fragarrayAcc_approximates wm wn accFrags rAcc;
  with eAccFrags. assert accFrags `array_fragment_pts_to` eAccFrags;
  array_fragment_pts_to_ref accFrags;
  array_fragment_extract_ro accFrags done;

  with eScratch.
    unfold scratch_tile_live bm bn bk tm tn nthr sh tid;
  let sTile = scratch_tile_st bm bn bk tm tn nthr sh wid;
  rewrite each
    scratch_tile bm bn bk tm tn nthr sh (SZ.v tid / warp_size)
  as sTile;
  mma_store accFrags.(!idx) #_
    #(strided_row_major_subtile
      (scratch_layout tm tn nthr)
      #_ #(strided_row_major_l2_row_major
        #((SZ.v nthr / warp_size) * SZ.v tm) #(SZ.v tn) #_ #_)
      (SZ.v tm) (SZ.v tn) (SZ.v wid) 0)
    sTile;

  let rCFrag =
    ematrix_subtile rCWarp tm tn
      (SZ.v done / wn) (SZ.v done % wn);
  let rAccFrag =
    ematrix_subtile rAcc tm tn
      (SZ.v done / wn) (SZ.v done % wn);
  let eAccFrag : chest2 et_acc tm tn =
    Seq.Base.index eAccFrags done;

  assert pure (eAccFrag %~ rAccFrag);
  epilogue_fragment_from_warp comb comb_r gC
    bm bn tm tn wm wn
    mrow mcol warpRow warpCol bid wid
    #_ #_ #_ #rC #_
    #_ #(c_subtile_layout
      (scratch_layout tm tn nthr)
      #(Kuiper.Tensor.Layout.Alg.c_l2_row_major
        ((SZ.v nthr / warp_size) * SZ.v tm) tn)
      (SZ.v tm) (SZ.v tn) (SZ.v wid) 0 #_ #_ #_ #_)
    sTile #eAccFrag #rAccFrag #_
    gD !idx lane;
  rewrite each sTile as
    scratch_tile bm bn bk tm tn nthr sh (SZ.v tid / warp_size);

  with eOut.
    assert own_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (SZ.v done / wn) (SZ.v done % wn))
      eOut lane;
  chest_comb_subtile comb_r rCWarp rAcc tm tn
    (SZ.v done / SZ.v wn) (SZ.v done % SZ.v wn);
  fold output_fragment_post
    gD bm bn tm tn wm wn bid wid lane
    (chest_comb comb_r rCWarp rAcc) done;
  Kuiper.Conditional.if_intro_true
    (output_fragment_post
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v done));
  Kuiper.Conditional.if_rewrite_bool
    true (SZ.v done < SZ.v done + 1)
    (output_fragment_post
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v done));
  Kuiper.Conditional.if_intro_false
    (live_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (SZ.v done / wn) (SZ.v done % wn))
      lane);
  Kuiper.Conditional.if_rewrite_bool
    false (not (SZ.v done < SZ.v done + 1))
    (live_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (SZ.v done / wn) (SZ.v done % wn))
      lane);
  fold if_else_ (SZ.v done < SZ.v done + 1)
    (output_fragment_post
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc) (SZ.v done))
    (live_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (SZ.v done / wn) (SZ.v done % wn))
      lane);
  fold output_fragment_state_at
    gD bm bn tm tn wm wn bid wid lane
    (chest_comb comb_r rCWarp rAcc)
    (SZ.v done + 1) (SZ.v done);
  Pulse.Lib.Trade.elim_trade
    (output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc)
      (SZ.v done + 1) (SZ.v done))
    (forall+ (fi : natlt (wm * wn)).
      output_fragment_state_at
        gD bm bn tm tn wm wn bid wid lane
        (chest_comb comb_r rCWarp rAcc) (SZ.v done + 1) fi);
  ambig_trade_elim ();
  fold fragarrayAcc_approximates wm wn accFrags rAcc;
  fold scratch_tile_live bm bn bk tm tn nthr sh tid;
  fold epilogue_frame #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC fC eC rC
    bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;

  fold output_epilogue_state
    gD bm bn tm tn wm wn bid wid lane
    (chest_comb comb_r rCWarp rAcc) (SZ.v done + 1);
  idx := sz_succ !idx;
}

#pop-options
