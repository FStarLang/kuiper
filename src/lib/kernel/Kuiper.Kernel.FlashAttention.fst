module Kuiper.Kernel.FlashAttention

(*

Remaining admits/magics — and why I couldn't remove them in this session

Each requires substantive Kuiper-library work, not 5-minute Pulse fiddling:

  1. fa_kf body (admit ()): kpre_fa's existentials exists* lOt gOt. live gOt don't carry ctlayout lOt — Pulse fails  typeclass resolution for ctlayout (M.row_layout (fa_gS bc br sh) (v tid)) and the matching constraints. I tried passing  #(ctlayout_slice _ 0sz tid) explicitly and even marking fa_lS/kpre_fa as unfold; that caused Frac (fK /. br) to lose its   non-zero refinement on br when it was unfolded into the kdesc record literal at multiple call sites. Real fix: thread  per-thread layouts as concrete block_frame fields rather than slprop existentials.
  2. block_setup_fa / block_teardown_fa bodies (admit ()): The shmem decomposition can be done by mirroring  Kuiper.Kernel.HReduce.Block.block_setup_block, but the heart of the proof — splitting gO/gl/gm into per-thread strided  sub-tiles {rows i*br + tid} — is not supported by the current Kuiper.Array2.Strided machinery, which only handles  contiguous subtile_layout sub-tiles. New library helpers (a tlayout_slice-based strided-row-extract, or a Cell-to-array2   glue primitive) are needed.
  3. Four *_sendable fields (magic ()): solve diverges when trying to construct is_send_across for kpre_fa's slprop.  is_send_across_exists and is_send_across_star instances exist, but the recursion through nested existentials and through   live (which itself unfolds to exists* y. pts_to) hangs (timed out at 700 s). Likely needs explicit hand-written  witnesses or a tactic that unfolds kpre_fa step-by-step.

Recommended next steps for a follow-up session:

  - First restructure kpre_fa to take per-thread layout records as explicit parameters (eliminates blocker #1 and likely  simplifies #3 by making the slprop concrete).
  - Then add a strided-row-extract helper to Kuiper.Array2.Strided to unblock #2.
  - After both, the sendable proofs should fall out via solve.

*)


#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Array
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Tiling
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l2_row_major }

module M = Kuiper.Array2 
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
module Array1 = Kuiper.Array1
open Kuiper.Array1

open Kuiper.Kernel.FlashAttention.KernelDesc

inline_for_extraction noextract
fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (#lKj #lVj: M.layout bc d)
  (#lSt: layout bc)
  (#lQit #lOit: layout d)
  {| ctlayout lSt, ctlayout lKj, ctlayout lVj, ctlayout lQit, ctlayout lOit |}
  (gKj: M.array2 et lKj) 
  (gVj: M.array2 et lVj)
  (gSt: array1 et lSt)
  (gQit: array1 et lQit)
  (gOit: array1 et lOit)
  (glit gmit: ref et)
  (#eKj #eVj: ematrix et bc d)
  (#vQit #vOit: erased (lseq et d))
  (#vlit #vmit: erased et)
  (#fKj #fVj #fQit: perm)
  requires 
    gOit |-> vOit ** glit |-> vlit ** gmit |-> vmit
  preserves 
    (gKj |-> Frac fKj eKj) ** (gVj |-> Frac fVj eVj) ** (gQit |-> Frac fQit vQit) **
    live gSt
  ensures 
    live gOit ** live glit ** live gmit // No functional spec
{
  let row_m_prev = !gmit;
  let row_l_prev = !glit;
  let mut row_m: et = neg infinity;
  let mut y: szle bc = 0sz;
  
  while (!y <^ bc) 
    invariant live y ** live row_m ** live gSt
    decreases (bc - !y)
  {
	  let mut sum: et = zero;
    let mut x: szle d = 0sz;
    while (!x <^ d)
      invariant live x ** live sum
      decreases (d - !x)
    {
      assert pure (!x <^ d);
      let vx = !x; let vy = !y;
      let vq: et = read gQit vx;
      let vk: et = M.read gKj ((vy <: sz), (vx <: sz));
      sum := !sum `add` (vq `mul` vk);
      x := !x +^ 1sz;
    };
    // TODO: add softmax scale factor
    // sum := !sum * alpha;

    let vy = !y;
    gSt.(vy) <- !sum;
    row_m := fmax !row_m !sum;

    y := !y +^ 1sz;
  };

  let mut row_l: et = zero;
  y := 0sz;
  while (!y <^ bc)
    invariant live y ** live row_l ** live gSt
    decreases (bc - !y)
  {
    let vy = !y;
    let vs: et = (fexp gSt.(vy)) `sub` !row_m;
    gSt.(vy) <- vs;
    row_l := !row_l `add` vs;

    y := !y +^ 1sz;
  };

  let row_m_new = fmax row_m_prev !row_m;
  let row_l_new = row_l_prev `mul` (fexp (row_m_prev `sub` row_m_new)) `add` (!row_l `mul` (fexp (!row_m `sub` row_m_new)));

  let mut x: sz = 0sz;
  while (!x <^ d) 
    invariant live x ** live gOit
    decreases (d - !x) 
  {
    let mut pv: et = zero;
    y := 0sz;
    while (!y <^ bc)
      invariant live y ** live pv
      decreases (bc - !y)
    {
      let vx = !x; let vy = !y;
      let vs: et = gSt.(vy);
      let vv: et = M.read gVj ((vy <: sz), (vx <: sz));
      pv := !pv `add` (vs `mul` vv);

      y := !y +^ 1sz;
    };

    let vx = !x;
    let vo: et = gOit.(vx);
    let vo: et = (vo `mul` row_l_prev `mul` (fexp (row_m_prev `sub` row_m_new))) `div` row_l_new;
    let vo: et = vo `add` ((fexp (!row_m `sub` row_m_new)) `mul` !pv);

    gOit.(vx) <- vo;

    x := !x +^ 1sz;
  };

  glit := row_l_new;
  gmit := row_m_new;

  ()
}

// flash attention kernel executed by each thread (no shared memory caching)
inline_for_extraction noextract 
fn flashattention_kf_no_smem (#et : Type0) {| scalar et, floating et |}
  (n d: szp)
  (bc br: szp { bc /? n /\ br /? n })
  (lSt: layout bc)
  (lK lV lQ: M.layout n d)
  (lOt: M.layout (n /^ br) d)
  (llt lmt: layout (n /^ br))
  {| ctlayout lSt, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lOt, ctlayout llt, ctlayout lmt |}
  (gSt: array1 et lSt)
  (gK: M.array2 et lK) 
  (gV: M.array2 et lV)
  (gQ: M.array2 et lQ)
  (gOt: M.array2 et lOt)
  (glt: array1 et llt)
  (gmt: array1 et lmt)
  (eK eV eQ: ematrix et n d)
  (#fK #fV #fQ: perm)
  (tid: sz { tid <^ br /\ tid <^ bc }) // TODO: impossible to materialize tid in a kernel unless br = bc
  preserves 
    gpu ** 
    kpre_post_inner_fa n d bc br lSt lK lV lQ lOt llt lmt gSt gK gV gQ gOt glt gmt eK eV eQ #fK #fV #fQ
{
  let tc = n /^ bc;
  let tr = n /^ br;
  let mut j: szle tc = 0sz;

  while (!j <^ tc) 
    invariant live j ** live gSt ** live gOt ** live glt ** live gmt
    decreases (tc - !j)
  {
    let gKj = array2_extract_tile_ro' gK (SZ.v bc) (SZ.v d) (SZ.v !j) 0;
    let gVj = array2_extract_tile_ro' gV (SZ.v bc) (SZ.v d) (SZ.v !j) 0;

    let mut i: szle tr = 0sz;
    while (!i <^ tr)
      invariant live i ** live gSt ** live gOt ** live glt ** live gmt
      decreases (tr - !i)
    {
      with eOt. assert gOt |-> eOt;
      with eSt. assert gSt |-> eSt;
      with vlt. assert glt |-> vlt;
      with vmt. assert gmt |-> vmt;
      let ii = !i;
      let qi = br *^ ii +^ tid;

      M.extract_row_ro gQ qi;
      let gQit = M.row gQ (SZ.v qi);
      M.extract_row gOt ii #1.0R #eOt; // gO has already been split into per-thread chunks
      let gOit = M.row gOt (SZ.v ii);

      extract_cell glt ii #1.0R #vlt;
      array1_cell_to_ref glt ii;
      let glit = get_ref_of_array_cell glt ii;
      assert rewrites_to glit (ref_of_array_cell glt ii);
      
      extract_cell gmt ii #1.0R #vmt;
      array1_cell_to_ref gmt ii;
      let gmit = get_ref_of_array_cell gmt ii;
      assert rewrites_to gmit (ref_of_array_cell gmt ii);

      flashattention_tile bc br d
        #_ #_ #_ #_ #_
        #_ #_ #_ #(ctlayout_slice _ (SZ.v 0sz) (SZ.v qi)) #(ctlayout_slice _ (SZ.v 0sz) (SZ.v ii))
        gKj gVj gSt (M.row gQ (SZ.v qi)) (M.row gOt (SZ.v ii)) glit gmit;

      array1_cell_from_ref glt ii;
      array1_cell_from_ref gmt ii;
      restore_cell glt ii;
      restore_cell gmt ii;

      M.restore_row gQ (SZ.v qi);
      with (eOit: lseq _ _). assert ((M.row gOt ((SZ.v ii) <: natlt n)) <: (array1 et (M.row_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit);
      elim_forall (eOit);
      Trade.elim_trade (((M.row gOt ((SZ.v ii) <: natlt n)) <: (array1 et (M.row_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit)) _;

      i := !i +^ 1sz; 
    };
    
    Trade.elim_trade (gKj |-> Frac fK (ematrix_subtile eK (SZ.v bc) (SZ.v d) (SZ.v !j) 0)) _;
    Trade.elim_trade (gVj |-> Frac fV (ematrix_subtile eV (SZ.v bc) (SZ.v d) (SZ.v !j) 0)) _;

    j := !j +^ 1sz;
  }
}