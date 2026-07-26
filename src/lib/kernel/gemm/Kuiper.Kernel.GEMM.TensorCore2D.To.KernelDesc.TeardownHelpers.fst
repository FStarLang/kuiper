module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownHelpers

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Kernel.GEMM.Tiled.Common.Vec

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

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
