module Kuiper.Kernel.BatchedGEMM

#lang-pulse
open Kuiper
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix { ematrix }
open Kuiper.Chest
open Pulse.Lib.Trade
open Kuiper.Tensor
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module P  = Kuiper.Kernel.GEMM.Naive2

inline_for_extraction noextract
fn batched_gemm_f32
  (batch rows shared cols : szp)
  (a : tensor f32 (l3_batched_row_major batch rows shared) { is_global a })
  (b : tensor f32 (l3_batched_row_major batch shared cols) { is_global b })
  (#sa : erased (EMatrix3.t f32 batch rows shared))
  (#sb : erased (EMatrix3.t f32 batch shared cols))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> Frac fA sa) **
    on gpu_loc (b |-> Frac fB sb)
  requires
    pure (
      rows * cols <= max_blocks * max_threads  /\
      SZ.fits (batch * rows * cols)
    )
  returns
    out : tensor f32 (l3_batched_row_major batch rows cols)
  ensures
    on gpu_loc (out |-> batched_matmul sa sb) **
    pure (is_global out)
{
  tensor_pts_to_ref_located a;
  tensor_pts_to_ref_located b;
  (* Allocate output Array3 on GPU *)
  let out = alloc0 #f32 (batch *^ rows *^ cols) (l3_batched_row_major batch rows cols);
  with sc0. assert on gpu_loc (out |-> sc0);

  (* Batch loop. Invariant: all pages < vi have been overwritten with the
     matmul of the corresponding input pages. *)
  let mut idx = 0sz;

  while (!idx <^ batch)
    invariant
      exists* (vi : sz) (sc : EMatrix3.t f32 batch rows cols).
        idx |-> vi **
        on gpu_loc (out |-> sc) **
        pure (
          SZ.v vi <= SZ.v batch /\
          rows * cols <= max_blocks * max_threads /\
          (forall (k:nat). k < SZ.v vi ==>
             EMatrix3.slice_page sc k ==
             MS.matmul (EMatrix3.slice_page sa k) (EMatrix3.slice_page sb k))
        )
  {
    let i = !idx;

    map_loc gpu_loc (fun () -> tensor_extract_slice_ro a 0 i);
    map_loc gpu_loc (fun () -> tensor_extract_slice_ro b 0 i);

    (* This one is not RO. *)
    with sc. assert on gpu_loc (out |-> sc);
    map_loc gpu_loc (fun () -> tensor_extract_slice out 0 i);

    (* 4. Launch verified GEMM on the pages *)
    P.mmcomb_gpu_exact MS.comb2
      #_ #_ #_
      #_ #_ #_
      #(T.ctlayout_slice _ 0 (SZ.v i)) // should not be needed
      #(T.ctlayout_slice _ 0 (SZ.v i)) // should not be needed
      #(T.ctlayout_slice _ 0 (SZ.v i)) // should not be needed
      (sliceof a 0 (SZ.v i))
      (sliceof b 0 (SZ.v i))
      (sliceof out 0 (SZ.v i));

    (* 4. Restore out: recombine page with its forall*/trade wand. The page
          now holds `mmcomb comb2 (slice_page sc i) ... == matmul ...`
          (by MS.matmul_is_gemm SMTPat). *)
    with eC. assert on gpu_loc (sliceof out 0 (SZ.v i) |-> eC);
    open Kuiper.Index;
    assert pure (modulo_i 0 (batch @| rows @| cols @| INil) == (rows @| cols @| INil));
    assert
      on gpu_loc
        (forall* (s' : chest (modulo_i 0 (batch @| rows @| cols @| INil)) f32).
          sliceof out 0 (SZ.v i) |-> s' @==>
          out |-> chest_update_slice 0 i sc s');

    map_loc gpu_loc
      #(sliceof out 0 (SZ.v i) |-> eC **
        (forall* (s' : chest (modulo_i 0 (batch @| rows @| cols @| INil)) f32).
          sliceof out 0 (SZ.v i) |-> s' @==>
          out |-> chest_update_slice 0 i sc s'))
      #(out |-> EMatrix3.upd_page sc i eC)
      fn () {
        // FIXME: somehow need the cast
        elim_forall (eC <: chest (modulo_i 0 (batch @| rows @| cols @| INil)) f32);
        elim_trade _ _;
        ()
      };

    (* 5. Restore a+b: recombine pages with trades, then elim trades *)
    map_loc gpu_loc (fun () ->
      elim_trade (sliceof a 0 (SZ.v i) |-> Frac fA (chest_slice 0 i sa)) (a |-> Frac fA sa)
    );
    map_loc gpu_loc (fun () ->
      elim_trade (sliceof b 0 (SZ.v i) |-> Frac fB (chest_slice 0 i sb)) (b |-> Frac fB sb)
    );

    idx := !idx +^ 1sz;
  };

  (* Exit: vi == batch, so every page of sc matches matmul of input pages.
     Conclude sc == batched_matmul sa sb by ematrix3 extensionality. *)
  with sc. assert on gpu_loc (out |-> sc);
  assert pure (EMatrix3.equal sc (batched_matmul sa sb));
  out
}
