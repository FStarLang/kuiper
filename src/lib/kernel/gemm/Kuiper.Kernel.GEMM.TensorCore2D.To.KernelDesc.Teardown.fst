module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Teardown

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
module RO = Kuiper.TensorRO
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.TensorCore

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module VG = Kuiper.Array2.Vectorized.Group

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

#push-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

let in_lane_covers_all
  (w : pos)
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  : Lemma (exists lane. in_lane w rows cols lane ij)
= let lane : natlt warp_size =
    VG.group_of w cols ij._1 ij._2 % warp_size in
  assert (in_lane w rows cols lane ij);
  ()

let in_lane_no_overlap
  (w : pos)
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  (lane1 lane2 : natlt warp_size)
  : Lemma
      (requires in_lane w rows cols lane1 ij /\ in_lane w rows cols lane2 ij)
      (ensures lane1 == lane2)
= ()

let lane_coincide
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (lane : natlt warp_size)
  (em1 em2 : chest2 et rows cols)
  : prop
= forall (i : natlt rows) (j : natlt cols).
    in_lane (chunk et #_ #hvc) rows cols lane (i, j) ==>
      acc2 em1 i j == acc2 em2 i j

ghost
fn own_lane_cells_rw
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (lane : natlt warp_size)
  (em1 em2 : chest2 et rows cols)
  (#_ : squash (lane_coincide lane em1 em2))
  requires own_lane_cells m em1 lane
  ensures own_lane_cells m em2 lane
{
  unfold own_lane_cells m em1 lane;
  forevery_map
    #(ij : (natlt rows & natlt cols){
       in_lane (chunk et #_ #hvc) rows cols lane ij})
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em1 ij._1 ij._2))
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em2 ij._1 ij._2))
    fn ij {
      rewrite
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em1 ij._1 ij._2)
      as
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em2 ij._1 ij._2);
    };
  fold own_lane_cells m em2 lane;
}

ghost
fn join_array2_from_lane_cells
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (#em : chest2 et rows cols)
  requires forall+ (lane : natlt warp_size). own_lane_cells m em lane
  ensures m |-> em
{
  forevery_map
    (fun lane -> own_lane_cells m em lane)
    (fun lane ->
      forall+ (ij : (natlt rows & natlt cols){
        in_lane (chunk et #_ #hvc) rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    fn lane { unfold own_lane_cells m em lane };
  forevery_join_or_n
    (fun (lane : natlt warp_size) ij ->
       in_lane (chunk et #_ #hvc) rows cols lane ij)
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  Classical.forall_intro (in_lane_covers_all (chunk et #_ #hvc) rows cols);
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap (chunk et #_ #hvc) rows cols ij lane1));
  forevery_refine_ext #_
    #(fun (ij : natlt rows & natlt cols) ->
      exists lane. in_lane (chunk et #_ #hvc) rows cols lane ij)
    (fun _ -> True) _;
  forevery_unflatten' _;
  tensor_iraise2 m;
}

ghost
fn join_lane_cells_approximates
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (r : chest2 real rows cols)
  requires
    forall+ (lane : natlt warp_size).
      exists* (em : chest2 et rows cols).
        own_lane_cells m em lane ** pure (em %~ r)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
{
  let ff = forevery_exists #(natlt warp_size)
    (fun lane em -> own_lane_cells m em lane ** pure (em %~ r));
  let em' : chest2 et rows cols =
    mk2 (fun i j ->
      let lane : natlt warp_size =
        VG.group_of (chunk et #_ #hvc) cols i j % warp_size in
      acc2 (ff lane) i j);
  forevery_unzip
    (fun lane -> own_lane_cells m (ff lane) lane)
    (fun lane -> pure (ff lane %~ r));
  forevery_elim_pure (fun lane -> ff lane %~ r);
  assert pure (em' %~ r);
  forevery_map
    (fun lane -> own_lane_cells m (ff lane) lane)
    (fun lane -> own_lane_cells m em' lane)
    fn lane {
      assert pure (lane_coincide lane (ff lane) em');
      own_lane_cells_rw m lane (ff lane) em';
    };
  join_array2_from_lane_cells m;
}

ghost
fn array2_untile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? cols})
  {| enumerable (natlt (rows / trows)),
     enumerable (natlt (cols / tcols)) |}
  (r : chest2 real rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (#_ : squash (SZ.fits (rows / trows)))
  (#_ : squash (SZ.fits (cols / tcols)))
  requires
    forall+ (tr : natlt (rows / trows))
             (tc : natlt (cols / tcols)).
      exists* (em : chest2 et trows tcols).
        array2_subtile m trows tcols tr tc |-> em **
        pure (em %~ ematrix_subtile r trows tcols tr tc)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
{
  let ff = forevery_exists_2
    #(natlt (rows / trows)) #_ #(natlt (cols / tcols)) #_
    (fun tr tc (em : chest2 et trows tcols) ->
      array2_subtile m trows tcols tr tc |-> em **
      pure (em %~ ematrix_subtile r trows tcols tr tc));
  forevery_extract_pure_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      ff tr tc %~ ematrix_subtile r trows tcols tr tc)
    fn tr tc { () };
  assert pure (forall (tr : natlt (rows / trows))
                      (tc : natlt (cols / tcols)).
    ff tr tc %~ ematrix_subtile r trows tcols tr tc);
  forevery_map_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc)
    fn tr tc { () };
  array2_untile' m trows tcols ff;
  assert pure (ematrix_from_tiles trows tcols ff %~ r);
}

#push-options "--split_queries no --z3rlimit 30"

ghost
fn gather_warp
  (#et_cd : Type0) {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (#m #n : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (#_ : squash (SZ.fits ((rm m n).ulen)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (wid : natlt (nthr / warp_size))
  (rWarp : chest2 real (wm * tm) (wn * tn))
  requires
    forall+ (lane : natlt warp_size).
      output_lane_approximates
        gD bm bn tm tn wm wn bid (wid * warp_size + lane)
        rWarp
  ensures
    exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
      warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn) wid |-> eWarp **
      pure (eWarp %~ rWarp)
{
  assert pure (wid < bm / (wm * tm) * (bn / (wn * tn)));
  assert pure (wid / (bn / (wn * tn)) < bm / (wm * tm));
  assert pure (wid % (bn / (wn * tn)) < bn / (wn * tn));
  Math.Lemmas.lemma_div_exact (wm * tm) tm;
  Math.Lemmas.lemma_div_exact (wn * tn) tn;
  assert pure ((wm * tm) / tm == wm);
  assert pure ((wn * tn) / tn == wn);
  forevery_map
    (fun (lane : natlt warp_size) ->
      output_lane_approximates gD bm bn tm tn wm wn bid
        (wid * warp_size + lane) rWarp)
    (fun (lane : natlt warp_size) ->
      forall+ (mi : natlt wm) (nj : natlt wn).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn lane {
      assert pure ((wid * warp_size + lane) / warp_size == wid);
      assert pure ((wid * warp_size + lane) % warp_size == lane);
      unfold output_lane_approximates gD bm bn tm tn wm wn bid
        (wid * warp_size + lane) rWarp;
      rewrite each ((wid * warp_size + lane) / warp_size) as wid;
      rewrite each ((wid * warp_size + lane) % warp_size) as lane;
    };
  forevery_commute
    (fun (lane : natlt warp_size) (mi : natlt wm) ->
      forall+ (nj : natlt wn).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  forevery_mid_flip
    (fun (mi : natlt wm) (lane : natlt warp_size) (nj : natlt wn) ->
      exists* (eFrag : chest2 et_cd tm tn).
        own_lane_cells
          (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
          eFrag lane **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  forevery_map_2
    (fun mi nj ->
      forall+ (lane : natlt warp_size).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        output_fragment gD bm bn tm tn wm wn bid wid mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn mi nj {
      join_lane_cells_approximates
        (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
        (ematrix_subtile rWarp tm tn mi nj);
    };
  let dWarp =
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
  forevery_map_2
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        output_fragment gD bm bn tm tn wm wn bid wid mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn mi nj {
      rewrite each output_fragment gD bm bn tm tn wm wn bid wid mi nj
        as array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj;
    };
  forevery_rw_size2 wm ((wm * tm) / tm) wn ((wn * tn) / tn)
    #(fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eFrag : chest2 et_cd tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  array2_untile_approximates dWarp (SZ.v tm) (SZ.v tn) rWarp;
  rewrite each dWarp as
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
}

#pop-options

ghost
fn gather_block
  (#et_cd : Type0) {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (#m #n : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (SZ.fits ((rm m n).ulen)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (rBlock : chest2 real bm bn)
  requires
    forall+ (wid : natlt (nthr / warp_size)).
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn) wid |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn))))
  ensures
    exists* (eBlock : chest2 et_cd bm bn).
      block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
      pure (eBlock %~ rBlock)
{
  forevery_map
    #(natlt (nthr / warp_size))
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn) wid |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))))
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn)
          ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
            wid % (bn / (wn * tn))) |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))))
    fn wid {
      FStar.Math.Lemmas.euclidean_division_definition wid (bn / (wn * tn));
      rewrite each wid as
        ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
          wid % (bn / (wn * tn)));
    };
  forevery_factor'
    (nthr / warp_size)
    (bm / (wm * tm))
    (bn / (wn * tn))
    (fun wr wc ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc) |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn) wr wc));
  let dBlock = block_tile gD (SZ.v bm) (SZ.v bn) bid;
  rewrite each block_tile gD (SZ.v bm) (SZ.v bn) bid as dBlock;
  forevery_map_2
    #(natlt (bm / (wm * tm)))
    #(natlt (bn / (wn * tn)))
    (fun wr wc ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile dBlock (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc) |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn) wr wc))
    (fun wr wc ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        array2_subtile dBlock (wm * tm) (wn * tn) wr wc |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn) wr wc))
    fn wr wc {
      assert pure (
        (wr * (bn / (wn * tn)) + wc) / (bn / (wn * tn)) == wr);
      assert pure (
        (wr * (bn / (wn * tn)) + wc) % (bn / (wn * tn)) == wc);
      rewrite each
        warp_tile dBlock (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc)
      as array2_subtile dBlock (wm * tm) (wn * tn) wr wc;
    };
  array2_untile_approximates dBlock (wm * tm) (wn * tn) rBlock;
  rewrite each dBlock as block_tile gD (SZ.v bm) (SZ.v bn) bid;
}

ghost
fn gather_output
  (#et_cd : Type0) {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_approximates gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB)
{
  forevery_map
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_approximates gD bm bn tm tn wm wn bid tid
          (ematrix_subtile
            (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
              bm bn (bid / (n / bn)) (bid % (n / bn)))
            (wm * tm) (wn * tn)
            ((tid / warp_size) / (bn / (wn * tn)))
            ((tid / warp_size) % (bn / (wn * tn)))))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    fn bid {
      forevery_ext
        (fun (tid : natlt nthr) ->
          output_lane_approximates gD bm bn tm tn wm wn bid tid
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              ((tid / warp_size) / (bn / (wn * tn)))
              ((tid / warp_size) % (bn / (wn * tn)))))
        (fun (tid : natlt nthr) ->
          output_lane_approximates gD bm bn tm tn wm wn bid
            ((tid / warp_size) * warp_size + tid % warp_size)
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              ((tid / warp_size) / (bn / (wn * tn)))
              ((tid / warp_size) % (bn / (wn * tn)))));
      forevery_factor' nthr (nthr / warp_size) warp_size
        (fun wid lane ->
          output_lane_approximates gD bm bn tm tn wm wn bid
            (wid * warp_size + lane)
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))));
      forevery_map
        (fun (wid : natlt (nthr / warp_size)) ->
          forall+ (lane : natlt warp_size).
            output_lane_approximates gD bm bn tm tn wm wn bid
              (wid * warp_size + lane)
              (ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        (fun (wid : natlt (nthr / warp_size)) ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        fn wid {
          gather_warp gD bm bn bk tm tn tk wm wn
            nblk nthr bid wid
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn))));
        };
      gather_block gD bm bn bk tm tn tk wm wn
        nblk nthr bid
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn (bid / (n / bn)) (bid % (n / bn)));
    };
  forevery_map
    #(natlt nblk)
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          ((bid / (n / bn)) * (n / bn) + bid % (n / bn)) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    fn bid {
      FStar.Math.Lemmas.euclidean_division_definition bid (n / bn);
      rewrite each bid as
        ((bid / (n / bn)) * (n / bn) + bid % (n / bn));
    };
  forevery_factor' nblk (m / bm) (n / bn)
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc));
  forevery_map_2
    #(natlt (m / bm)) #(natlt (n / bn))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        array2_subtile gD (SZ.v bm) (SZ.v bn) br bc |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc))
    fn br bc {
      assert pure ((br * (n / bn) + bc) / (n / bn) == br);
      assert pure ((br * (n / bn) + bc) % (n / bn) == bc);
      rewrite each block_tile gD (SZ.v bm) (SZ.v bn)
        (br * (n / bn) + bc)
      as array2_subtile gD (SZ.v bm) (SZ.v bn) br bc;
    };
  array2_untile_approximates gD (SZ.v bm) (SZ.v bn)
    (MS.mmcomb comb_r rC rA rB);
}

#pop-options
#push-options "--split_queries always"

let teardown_inputs_pre
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB) (eB : chest2 et_ab k n)
  (#lC : RO.vlayout2 m n)
  (gC : RO.roarray2 et_cd lC) (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  : slprop
= (forall+ (bid : natlt nblk) (tid : natlt nthr).
    kpost1_to comb_r gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn fA fB fC rA rB rC
      nblk nthr bid tid) **
  pure (SZ.fits ((rm m n).ulen))

let teardown_inputs_post
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB) (eB : chest2 et_ab k n)
  (#lC : RO.vlayout2 m n)
  (gC : RO.roarray2 et_cd lC) (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  : slprop
= gA |-> Frac fA eA **
  gB |-> Frac fB eB **
  gC |-> Frac fC eC **
  (forall+ (bid : natlt nblk) (tid : natlt nthr).
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (ematrix_subtile
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn (bid / (n / bn)) (bid % (n / bn)))
        (wm * tm) (wn * tn)
        ((tid / warp_size) / (bn / (wn * tn)))
        ((tid / warp_size) % (bn / (wn * tn))))) **
  pure (SZ.fits ((rm m n).ulen))

ghost
fn gather_kernel_outputs
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (#lC : RO.vlayout2 m n)
  (gC : RO.roarray2 et_cd lC)
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    teardown_inputs_pre comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
  ensures
    teardown_inputs_post comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
{
  unfold teardown_inputs_pre comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
  forevery_map_2
    #(natlt nblk)
    #(natlt nthr)
    (fun bid tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))))
    fn bid tid {
      unfold kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid;
    };
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_gather_n gA (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_gather_n gB (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  RO.tensor_gather_n gC (nblk * nthr);
  fold teardown_inputs_post comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
}

#pop-options

ghost
fn teardown_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (#lC : RO.vlayout2 m n)
  (gC : RO.roarray2 et_cd lC)
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> Frac fC eC **
    (exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB))
{
  fold teardown_inputs_pre comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
  gather_kernel_outputs comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC ();
  unfold teardown_inputs_post comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
  gather_output comb_r gD
    bm bn bk tm tn tk wm wn nblk nthr rA rB rC ();
}
