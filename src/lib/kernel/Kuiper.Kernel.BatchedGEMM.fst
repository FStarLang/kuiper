module Kuiper.Kernel.BatchedGEMM

#lang-pulse
open Kuiper
open Kuiper.Array3
module Array3 = Kuiper.Array3
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout
module SZ = Kuiper.SizeT
open Kuiper.EMatrix { ematrix }
open Pulse.Lib.Trade
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module P  = Kuiper.Kernel.GEMM.Naive2

inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (batch rows shared cols : szp)
  (#la : Array3.layout batch rows shared)
  (#lb : Array3.layout batch shared cols)
  (#lc : Array3.layout batch rows cols)
  {| ctlayout la, ctlayout lb, ctlayout lc |}
  (a : Array3.t et la { Array3.is_global a })
  (b : Array3.t et lb { Array3.is_global b })
  (c : Array3.t et lc { Array3.is_global c })
  (#sa : erased (EMatrix3.t et batch rows shared))
  (#sb : erased (EMatrix3.t et batch shared cols))
  (#sc : erased (EMatrix3.t et batch rows cols))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> Frac fA sa) **
    on gpu_loc (b |-> Frac fB sb)
  requires
    pure (
      rows * cols <= max_blocks * max_threads /\
      SZ.fits (batch * rows * cols)
    ) ** 
    on gpu_loc (c |-> sc)
  ensures
    on gpu_loc (c |-> MS.bmmcomb comb sc sa sb)
{
  Array3.pts_to_ref_located a;
  Array3.pts_to_ref_located b;
  Array3.pts_to_ref_located c;

  (* Batch loop. Invariant: all pages < vi have been overwritten with the
     matmul of the corresponding input pages. *)
  let mut idx = 0sz;
  while (!idx <^ batch)
    invariant
      exists* (vi : sz) (sc' : EMatrix3.t et batch rows cols).
        idx |-> vi **
        on gpu_loc (c |-> sc') **
        pure (
          SZ.v vi <= SZ.v batch /\
          rows * cols <= max_blocks * max_threads /\
          (forall (k:natlt batch). 
            if (k < SZ.v vi) then
              EMatrix3.slice_page sc' k ==
              MS.mmcomb comb (EMatrix3.slice_page sc k) (EMatrix3.slice_page sa k) (EMatrix3.slice_page sb k) 
            else EMatrix3.slice_page sc' k == EMatrix3.slice_page sc k)
        )
  {
    let i = !idx;
    
    map_loc gpu_loc (fun () -> Array3.extract_page_ro a i);
    map_loc gpu_loc (fun () -> Array3.extract_page_ro b i);

    (* This one is not RO. *)
    with sc'. assert on gpu_loc (c |-> sc');
    map_loc gpu_loc (fun () -> Array3.extract_page c i);

    (* 4. Launch verified GEMM on the pages *)
    P.mmcomb_gpu_exact comb
      #rows #cols #shared
      #_ #_ #_
      #(T.ctlayout_slice _ 0sz i) // should not be needed
      #(T.ctlayout_slice _ 0sz i) // should not be needed
      #(T.ctlayout_slice _ 0sz i) // should not be needed
      (Array3.page a (SZ.v i))
      (Array3.page b (SZ.v i))
      (Array3.page c (SZ.v i));

    with eC. assert on gpu_loc (Array3.page c (SZ.v i) |-> eC);
    map_loc gpu_loc
      #(Array3.page c (SZ.v i) |-> eC **
        (forall* (s' : ematrix et rows cols).
          Array3.page c (SZ.v i) |-> s' @==>
          c |-> EMatrix3.upd_page sc' i s'))
      #(c |-> EMatrix3.upd_page sc' i eC)
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

  (* Exit: vi == batch, so every page of sc matches mmcomb of input pages.
     Conclude sc' == mmcomb comb sa sb sc by ematrix3 extensionality. *)
  with sc'. assert on gpu_loc (c |-> sc');
  assert pure (forall (k:nat). k < SZ.v batch ==>
             EMatrix3.slice_page sc' k ==
             MS.mmcomb comb (EMatrix3.slice_page sc k) 
              (EMatrix3.slice_page sa k) (EMatrix3.slice_page sb k));
  assert pure (EMatrix3.equal sc' (MS.bmmcomb comb sc sa sb));
}
