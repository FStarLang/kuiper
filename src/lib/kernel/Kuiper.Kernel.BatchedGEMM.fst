module Kuiper.Kernel.BatchedGEMM

#lang-pulse
open Kuiper
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix { ematrix }
open Kuiper.Chest
open Pulse.Lib.Trade
open Kuiper.Tensor
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module P  = Kuiper.Kernel.GEMM.Naive2


inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (batch rows shared cols : szp)
  (#la : tlayout (batch @| rows @| shared @| INil))
  (#lb : tlayout (batch @| shared @| cols @| INil))
  (#lc : tlayout (batch @| rows @| cols @| INil))
  {| ctlayout la, ctlayout lb, ctlayout lc |}
  (a : tensor et la { is_global a })
  (b : tensor et lb { is_global b })
  (c : tensor et lc { is_global c })
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
      rows * cols <= max_blocks * max_threads  /\
      SZ.fits (batch * rows * cols)
    ) **
    on gpu_loc (c |-> sc)
  ensures
    on gpu_loc (c |-> MS.bmmcomb comb sc sa sb)
{
  tensor_pts_to_ref_located a;
  tensor_pts_to_ref_located b;
  tensor_pts_to_ref_located c;

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

    map_loc gpu_loc (fun () -> tensor_extract_slice_ro a 0 i);
    map_loc gpu_loc (fun () -> tensor_extract_slice_ro b 0 i);

    (* This one is not RO. *)
    with sc'. assert on gpu_loc (c |-> sc');
    map_loc gpu_loc (fun () -> tensor_extract_slice c 0 i);

    (* 4. Launch verified GEMM on the pages *)
    P.mmcomb_gpu_exact comb
      (sliceof a 0 (SZ.v i))
      (sliceof b 0 (SZ.v i))
      (sliceof c 0 (SZ.v i));


    //   P.mmcomb_gpu_exact comb
    //     #rows #cols #shared
    //     #_ #_ #_
    //     #(T.ctlayout_slice _ 0sz i) // should not be needed
    //     #(T.ctlayout_slice _ 0sz i) // should not be needed
    //     #(T.ctlayout_slice _ 0sz i) // should not be needed
    //     (Array3.page a (SZ.v i))
    //     (Array3.page b (SZ.v i))
    //     (Array3.page c (SZ.v i));
    with eC. assert on gpu_loc (sliceof c 0 (SZ.v i) |-> eC);
    open Kuiper.Shape;
    assert pure (modulo_i 0 (batch @| rows @| cols @| INil) == (rows @| cols @| INil));
    assert
      on gpu_loc
        (forall* (s' : chest (modulo_i 0 (batch @| rows @| cols @| INil)) et).
          sliceof c 0 (SZ.v i) |-> s' @==>
          c |-> chest_update_slice 0 i sc' s');

    map_loc gpu_loc
      #(sliceof c 0 (SZ.v i) |-> eC **
        (forall* (s' : chest (modulo_i 0 (batch @| rows @| cols @| INil)) et).
          sliceof c 0 (SZ.v i) |-> s' @==>
          c |-> chest_update_slice 0 i sc' s'))
      #(c |-> EMatrix3.upd_page sc' i eC)
      fn () {
        // FIXME: somehow need the cast
        elim_forall (eC <: chest (modulo_i 0 (batch @| rows @| cols @| INil)) et);
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

  (* Exit: vi == batch, so every page of sc matches mmcomb of input pages.
     Conclude sc' == mmcomb comb sa sb sc by ematrix3 extensionality. *)
  with sc'. assert on gpu_loc (c |-> sc');
  assert pure (forall (k:nat). k < SZ.v batch ==>
             EMatrix3.slice_page sc' k ==
             MS.mmcomb comb (EMatrix3.slice_page sc k)
              (EMatrix3.slice_page sa k) (EMatrix3.slice_page sb k));
  assert pure (EMatrix3.equal sc' (MS.bmmcomb comb sc sa sb));
}
