module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

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



let in_lane_covers_all
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  : Lemma (exists lane. in_lane rows cols lane ij)
= let lane : natlt warp_size =
    (ij._1 * cols + ij._2) % warp_size in
  assert (in_lane rows cols lane ij);
  ()

let in_lane_no_overlap
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  (lane1 lane2 : natlt warp_size)
  : Lemma
      (requires in_lane rows cols lane1 ij /\ in_lane rows cols lane2 ij)
      (ensures lane1 == lane2)
= ()

ghost
fn split_array2_into_lane_cells
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#em : chest2 et rows cols)
  requires m |-> em
  ensures forall+ (lane : natlt warp_size). own_lane_cells m em lane
{
  tensor_ilower2 m;
  forevery_flatten _;
  Classical.forall_intro (in_lane_covers_all rows cols);
  forevery_refine_ext #_ #(fun _ -> True)
    (fun (ij : natlt rows & natlt cols) -> exists lane. in_lane rows cols lane ij) _;
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap rows cols ij lane1));
  forevery_split_or_n _ _;
  forevery_map
    (fun lane ->
      forall+ (ij : (natlt rows & natlt cols){
        in_lane rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2)
          (acc2 em ij._1 ij._2))
    (fun lane -> own_lane_cells m em lane)
    fn lane { fold own_lane_cells m em lane };
}

let lane_coincide
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (lane : natlt warp_size)
  (em1 em2 : chest2 et rows cols)
  : prop
= forall (i : natlt rows) (j : natlt cols).
    in_lane rows cols lane (i, j) ==> acc2 em1 i j == acc2 em2 i j

ghost
fn own_lane_cells_rw
  (#et : Type0) {| scalar et |}
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
    #(ij : (natlt rows & natlt cols){in_lane rows cols lane ij})
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
  (#et : Type0) {| scalar et |}
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
      forall+ (ij : (natlt rows & natlt cols){in_lane rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    fn lane { unfold own_lane_cells m em lane };
  forevery_join_or_n
    (fun (lane : natlt warp_size) ij -> in_lane rows cols lane ij)
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  Classical.forall_intro (in_lane_covers_all rows cols);
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap rows cols ij lane1));
  forevery_refine_ext #_
    #(fun (ij : natlt rows & natlt cols) ->
      exists lane. in_lane rows cols lane ij)
    (fun _ -> True) _;
  forevery_unflatten' _;
  tensor_iraise2 m;
}

ghost
fn join_lane_cells_approximates
  (#et : Type0) {| scalar et, real_like et |}
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
      let lane : natlt warp_size = (i * cols + j) % warp_size in
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

#push-options "--z3rlimit 100 --fuel 1 --ifuel 1 --split_queries no"
ghost
fn split_output_to_lanes
  (#et : Type0) {| scalar et |}
  (#m #n : szp)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  requires live gD
  ensures
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_live gD bm bn tm tn wm wn bid tid
{
  with (eD : chest2 _ _ _). assert gD |-> eD;
  array2_tile gD (SZ.v bm) (SZ.v bn) #eD #1.0R;
  forevery_unfactor' nblk (m / bm) (n / bn) _;

  forevery_map
    (fun (bid : natlt nblk) ->
      array2_subtile gD (SZ.v bm) (SZ.v bn)
        (bid / (n / bn)) (bid % (n / bn))
        |-> ematrix_subtile eD bm bn
              (bid / (n / bn)) (bid % (n / bn)))
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_live gD bm bn tm tn wm wn bid tid)
    fn bid {
      rewrite each
        array2_subtile gD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn))
      as block_tile gD (SZ.v bm) (SZ.v bn) bid;
      let dBlock = block_tile gD (SZ.v bm) (SZ.v bn) bid;
      let eBlock =
        ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn));
      array2_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn)
        #(ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn)))
        #1.0R;
      rewrite each
        block_tile gD (SZ.v bm) (SZ.v bn) bid
        as dBlock;
      rewrite each
        ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn))
        as eBlock;
      forevery_unfactor'
        (bm / (wm * tm) * (bn / (wn * tn)))
        (bm / (wm * tm))
        (bn / (wn * tn))
        _;
      forevery_map
        (fun (wid : natlt (
          bm / (wm * tm) * (bn / (wn * tn)))) ->
          array2_subtile dBlock
            (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))
          |-> ematrix_subtile eBlock (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn))))
        (fun (wid : natlt (
          bm / (wm * tm) * (bn / (wn * tn)))) ->
          forall+ (lane : natlt warp_size)
                   (mi : natlt wm)
                   (nj : natlt wn).
            live_lane_cells
              (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
              lane)
        fn wid {
          rewrite each
            array2_subtile dBlock
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))
          as warp_tile dBlock
            (wm * tm) (wn * tn) wid;
          let dWarp = warp_tile dBlock
            (wm * tm) (wn * tn) wid;
          let eWarp =
            ematrix_subtile eBlock
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)));
          array2_tile
            (warp_tile dBlock (wm * tm) (wn * tn) wid)
            (SZ.v tm) (SZ.v tn)
            #(ematrix_subtile eBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn))))
            #1.0R;
          rewrite each
            warp_tile dBlock (wm * tm) (wn * tn) wid
            as dWarp;
          rewrite each
            ematrix_subtile eBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))
            as eWarp;
          forevery_rw_size2
            ((wm * tm) / tm) wm
            ((wn * tn) / tn) wn
            #(fun (mi : natlt ((wm * tm) / tm))
                  (nj : natlt ((wn * tn) / tn)) ->
              array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                |-> ematrix_subtile eWarp tm tn mi nj);

          forevery_map_2
            (fun (mi : natlt wm) (nj : natlt wn) ->
              array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                |-> ematrix_subtile eWarp tm tn mi nj)
            (fun (mi : natlt wm) (nj : natlt wn) ->
              forall+ (lane : natlt warp_size).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            fn mi nj {
              rewrite each
                array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                as output_fragment gD bm bn tm tn wm wn bid wid mi nj;
              split_array2_into_lane_cells
                (output_fragment gD bm bn tm tn wm wn bid wid mi nj);
              forevery_map
                (fun lane ->
                  own_lane_cells
                    (output_fragment gD bm bn tm tn wm wn
                      bid wid mi nj)
                    (ematrix_subtile eWarp tm tn mi nj)
                    lane)
                (fun lane ->
                  live_lane_cells
                    (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                    lane)
                fn lane {
                  fold live_lane_cells
                    (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                    lane;
                };
            };

          forevery_map
            (fun (mi : natlt wm) ->
              forall+ (nj : natlt wn) (lane : natlt warp_size).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            (fun (mi : natlt wm) ->
              forall+ (lane : natlt warp_size) (nj : natlt wn).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            fn mi { forevery_commute _ };
          forevery_commute
            (fun (mi : natlt wm) (lane : natlt warp_size) ->
              forall+ (nj : natlt wn).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane);
        };

      forevery_unfactor' nthr
        (bm / (wm * tm) * (bn / (wn * tn))) warp_size _;
      forevery_map
        (fun (tid : natlt nthr) ->
          forall+ (mi : natlt wm) (nj : natlt wn).
            live_lane_cells
              (output_fragment gD bm bn tm tn wm wn
                bid (tid / warp_size) mi nj)
              (tid % warp_size))
        (fun tid ->
          output_lane_live gD bm bn tm tn wm wn bid tid)
        fn tid {
          fold output_lane_live gD bm bn tm tn wm wn bid tid;
        };
    };
}
#pop-options

#push-options "--split_queries no"
ghost
fn setup_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
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
    gA |-> Frac fA eA ** pure (eA %~ rA) **
    gB |-> Frac fB eB ** pure (eB %~ rB) **
    gC |-> Frac fC eC ** pure (eC %~ rC) **
    live gD
  ensures
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid) **
    pure (SZ.fits ((rm m n).ulen))
{
  tensor_share_n gA (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_share_n gB (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_share_n gC (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  split_output_to_lanes gD bm bn tm tn wm wn nblk nthr;

  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_live gD bm bn tm tn wm wn bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_live gD bm bn tm tn wm wn bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    _;
  forevery_rw_size2
    (SZ.v nblk) (m / bm * (n / bn))
    (SZ.v nthr)
      (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
    #(fun (bid : natlt nblk) (tid : natlt nthr) ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_live gD bm bn tm tn wm wn bid tid);
  forevery_map_2
    #(natlt (m / bm * (n / bn)))
    #(natlt (
      bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_live gD bm bn tm tn wm wn bid tid)
    (fun bid tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    fn bid tid {
      fold kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid;
    };
  forevery_rw_size2
    (m / bm * (n / bn)) (SZ.v nblk)
    (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
      (SZ.v nthr)
    #(fun bid tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid);
}
#pop-options

#push-options "--split_queries no"
ghost
fn split_scratch_to_threads
  (#et_ab #et_acc : Type0)
  {| sized et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  requires live_c_shmem (fst (snd (snd sh)))
  ensures forall+ (tid : natlt nthr).
    scratch_tile_live bm bn bk tm tn nthr sh tid
{
  let (_, (_, (sarAcc, _))) = sh;
  unfold_live_c_shmem sarAcc;
  with vAcc. assert sarAcc |-> vAcc;
  gpu_pts_to_ref sarAcc;
  tensor_abs' (scratch_layout tm tn nthr) sarAcc;
  let sAcc = scratch_matrix bm bn bk tm tn nthr sh;
  rewrite each from_array (scratch_layout tm tn nthr) sarAcc as sAcc;
  array2_tile sAcc (SZ.v tm) (SZ.v tn);
  forevery_rw_size2
    (((nthr / warp_size) * tm) / tm) (nthr / warp_size)
    (tn / tn) 1
    #(fun
        (wid : natlt (((nthr / warp_size) * tm) / tm))
        (tc : natlt (tn / tn)) ->
      array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc
        |-> ematrix_subtile
              (from_seq (scratch_layout tm tn nthr) vAcc)
              tm tn wid tc);
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (tc : natlt 1).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc
          |-> ematrix_subtile
                (from_seq (scratch_layout tm tn nthr) vAcc)
                tm tn wid tc)
    (fun (wid : natlt (nthr / warp_size)) ->
      array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0
        |-> ematrix_subtile
              (from_seq (scratch_layout tm tn nthr) vAcc)
              tm tn wid 0)
    fn wid {
      forevery_singleton_elim #(natlt 1)
        (fun (tc : natlt 1) ->
          array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc
            |-> ematrix_subtile
                  (from_seq (scratch_layout tm tn nthr) vAcc)
                  tm tn wid tc);
    };

  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0
        |-> ematrix_subtile
              (from_seq (scratch_layout tm tn nthr) vAcc)
              tm tn wid 0)
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (lane : natlt warp_size).
        scratch_tile_live bm bn bk tm tn nthr sh
          (wid * warp_size + lane))
    fn wid {
      rewrite each array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0
        as scratch_tile bm bn bk tm tn nthr sh wid;
      tensor_share_n
        (scratch_tile bm bn bk tm tn nthr sh wid)
        warp_size;
      forevery_map
        #(natlt warp_size)
        (fun lane ->
          scratch_tile bm bn bk tm tn nthr sh wid
            |-> Frac (1.0R /. warp_size)
              (ematrix_subtile
                (from_seq (scratch_layout tm tn nthr) vAcc)
                tm tn wid 0))
        (fun lane ->
          scratch_tile_live bm bn bk tm tn nthr sh
            (wid * warp_size + lane))
        fn lane {
          assert pure (
            (wid * warp_size + lane) / warp_size == wid);
          rewrite each scratch_tile bm bn bk tm tn nthr sh wid
            as scratch_tile bm bn bk tm tn nthr sh
              ((wid * warp_size + lane) / warp_size);
          fold scratch_tile_live bm bn bk tm tn nthr sh
            (wid * warp_size + lane);
        };
    };
  forevery_unfactor' nthr (nthr / warp_size) warp_size _;
  forevery_map
    #(natlt nthr)
    (fun tid ->
      scratch_tile_live bm bn bk tm tn nthr sh
        ((tid / warp_size) * warp_size + tid % warp_size))
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid)
    fn tid {
      unfold scratch_tile_live bm bn bk tm tn nthr sh
        ((tid / warp_size) * warp_size + tid % warp_size);
      assert pure (
        (tid / warp_size) * warp_size + tid % warp_size == tid);
      rewrite each scratch_tile bm bn bk tm tn nthr sh
        (((tid / warp_size) * warp_size + tid % warp_size) / warp_size)
        as scratch_tile bm bn bk tm tn nthr sh (tid / warp_size);
      fold scratch_tile_live bm bn bk tm tn nthr sh tid;
    };
}

ghost
fn gather_scratch_from_threads
  (#et_ab #et_acc : Type0)
  {| sized et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  requires forall+ (tid : natlt nthr).
    scratch_tile_live bm bn bk tm tn nthr sh tid
  ensures live_c_shmem (fst (snd (snd sh)))
{
  forevery_map
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid)
    (fun tid ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
          |-> Frac (1.0R /. warp_size) eAcc)
    fn tid {
      unfold scratch_tile_live bm bn bk tm tn nthr sh tid;
    };
  forevery_factor' nthr (nthr / warp_size) warp_size
    (fun wid _lane ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh wid
          |-> Frac (1.0R /. warp_size) eAcc);
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (_lane : natlt warp_size).
        exists* (eAcc : chest2 et_acc tm tn).
          scratch_tile bm bn bk tm tn nthr sh wid
            |-> Frac (1.0R /. warp_size) eAcc)
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh wid |-> eAcc)
    fn wid {
      tensor_gather_n_underspec
        (scratch_tile bm bn bk tm tn nthr sh wid)
        warp_size;
    };

  let sarAcc = fst (snd (snd sh));
  let sAcc = scratch_matrix bm bn bk tm tn nthr sh;
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh wid |-> eAcc)
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0 |-> eAcc)
    fn wid {
      rewrite each scratch_tile bm bn bk tm tn nthr sh wid
        as array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0;
    };
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0 |-> eAcc)
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (tc : natlt 1).
        exists* (eAcc : chest2 et_acc tm tn).
          array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc |-> eAcc)
    fn wid {
      forevery_singleton_intro #(natlt 1)
        (fun tc ->
          exists* (eAcc : chest2 et_acc tm tn).
            array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc |-> eAcc);
    };
  forevery_rw_size2
    (nthr / warp_size) (((nthr / warp_size) * tm) / tm)
    1 (tn / tn)
    #(fun (wid : natlt (nthr / warp_size)) (tc : natlt 1) ->
      exists* (eAcc : chest2 et_acc tm tn).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc |-> eAcc);
  array2_untile_underspec sAcc (SZ.v tm) (SZ.v tn);
  with eAcc. assert sAcc |-> eAcc;
  tensor_concr sAcc;
  rewrite each core sAcc as sarAcc;
  with vAcc. assert sarAcc |-> vAcc;
  fold_live_c_shmem sarAcc;
  rewrite each sarAcc as fst (snd (snd sh));
}
#pop-options

#push-options "--split_queries no"
ghost
fn block_setup_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd, scalar et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
  ensures
    (forall+ (tid : natlt nthr).
      kpre_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid) **
    emp
{
  unfold_live_c_shmems_cons sh #1.0R;
  unfold_live_c_shmems_cons (snd sh) #1.0R;
  unfold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  unfold_live_c_shmems_nil (snd (snd (snd sh))) #1.0R;

  gpu_live_c_shmem_share_underspec (fst sh) #1.0R #nthr;
  gpu_live_c_shmem_share_underspec (fst (snd sh)) #1.0R #nthr;
  split_scratch_to_threads bm bn bk tm tn nthr sh;

  forevery_zip #(natlt nthr)
    (fun tid ->
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr))
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid);
  forevery_zip #(natlt nthr)
    (fun tid -> live_c_shmem (fst sh) #(1.0R /. nthr))
    (fun tid ->
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid);
  forevery_map
    (fun tid ->
      live_c_shmem (fst sh) #(1.0R /. nthr) **
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid)
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid)
    fn tid {
      fold shared_thread_live bm bn bk tm tn nthr sh tid;
    };
  forevery_zip #(natlt nthr)
    (fun tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid);
  forevery_map
    (fun tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid **
      shared_thread_live bm bn bk tm tn nthr sh tid)
    (fun tid ->
      kpre_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid)
    fn tid {
      fold kpre_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid;
    };
}
#pop-options

ghost
fn block_teardown_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd, scalar et_acc |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid) **
    emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
{
  forevery_map
    (fun tid ->
      kpost_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid)
    (fun tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid **
      shared_thread_live bm bn bk tm tn nthr sh tid)
    fn tid {
      unfold kpost_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid;
    };
  forevery_unzip #(natlt nthr)
    (fun tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid);

  forevery_map
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid)
    (fun tid ->
      live_c_shmem (fst sh) #(1.0R /. nthr) **
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid)
    fn tid {
      unfold shared_thread_live bm bn bk tm tn nthr sh tid;
    };
  forevery_unzip #(natlt nthr)
    (fun _ -> live_c_shmem (fst sh) #(1.0R /. nthr))
    (fun tid ->
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid);
  forevery_unzip #(natlt nthr)
    (fun _ -> live_c_shmem (fst (snd sh)) #(1.0R /. nthr))
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid);

  gpu_live_c_shmem_gather_underspec (fst sh) #1.0R #nthr;
  gpu_live_c_shmem_gather_underspec (fst (snd sh)) #1.0R #nthr;
  gather_scratch_from_threads bm bn bk tm tn nthr sh;

  fold_live_c_shmems_nil (snd (snd (snd sh))) #1.0R;
  fold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  fold_live_c_shmems_cons (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;
}
