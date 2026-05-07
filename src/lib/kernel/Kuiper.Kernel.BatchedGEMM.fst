module Kuiper.Kernel.BatchedGEMM

#lang-pulse
open Kuiper
open Kuiper.Array3
module Array3 = Kuiper.Array3
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT
open Kuiper.EMatrix { ematrix }
open Pulse.Lib.Trade
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module P  = Kuiper.Kernel.GEMM.Naive2

inline_for_extraction noextract
fn batched_gemm_f32
  (batch rows shared cols : szp)
  (a : Array3.t f32 (l3_batched_row_major batch rows shared) { Array3.is_global a })
  (b : Array3.t f32 (l3_batched_row_major batch shared cols) { Array3.is_global b })
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
    out : Array3.t f32 (l3_batched_row_major batch rows cols)
  ensures
    on gpu_loc (out |-> batched_matmul sa sb) **
    pure (Array3.is_global out)
{
  Array3.pts_to_ref_located a;
  Array3.pts_to_ref_located b;
  (* Allocate output Array3 on GPU *)
  let out = Array3.alloc0 #f32 batch rows cols (l3_batched_row_major batch rows cols);
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

    map_loc gpu_loc (fun () -> Array3.extract_page_ro a i);
    map_loc gpu_loc (fun () -> Array3.extract_page_ro b i);

    (* This one is not RO. *)
    with sc. assert on gpu_loc (out |-> sc);
    map_loc gpu_loc (fun () -> Array3.extract_page out i);

    (* 4. Launch verified GEMM on the pages *)
    P.mmcomb_gpu_exact MS.comb2
      #_ #_ #_
      #_ #_ #_
      #(T.ctlayout_slice _ 0sz i) // should not be needed
      #(T.ctlayout_slice _ 0sz i) // should not be needed
      #(T.ctlayout_slice _ 0sz i) // should not be needed
      (Array3.page a (SZ.v i))
      (Array3.page b (SZ.v i))
      (Array3.page out (SZ.v i));

    (* 4. Restore out: recombine page with its forall*/trade wand. The page
          now holds `mmcomb comb2 (slice_page sc i) ... == matmul ...`
          (by MS.matmul_is_gemm SMTPat). *)
    with eC. assert on gpu_loc (Array3.page out (SZ.v i) |-> eC);
    map_loc gpu_loc
      #(Array3.page out (SZ.v i) |-> eC **
        (forall* (s' : ematrix f32 rows cols).
          Array3.page out (SZ.v i) |-> s' @==>
          out |-> EMatrix3.upd_page sc i s'))
      #(out |-> EMatrix3.upd_page sc i eC)
      fn () {
        elim_forall eC;
        elim_trade _ _;
      };

    (* 5. Restore a+b: recombine pages with trades, then elim trades *)
    map_loc gpu_loc (fun () ->
        elim_trade
          (Array3.page a (SZ.v i) |-> Frac fA (EMatrix3.slice_page sa (SZ.v i)))
          (a |-> Frac fA sa));
    map_loc gpu_loc (fun () ->
        elim_trade
          (Array3.page b (SZ.v i) |-> Frac fB (EMatrix3.slice_page sb (SZ.v i)))
          (b |-> Frac fB sb));

    idx := !idx +^ 1sz;
  };

  (* Exit: vi == batch, so every page of sc matches matmul of input pages.
     Conclude sc == batched_matmul sa sb by ematrix3 extensionality. *)
  with sc. assert on gpu_loc (out |-> sc);
  assert pure (EMatrix3.equal sc (batched_matmul sa sb));
  out
}
