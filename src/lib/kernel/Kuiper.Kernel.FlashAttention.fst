module Kuiper.Kernel.FlashAttention

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Array
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Tiling
open Kuiper.EMatrix

module M = Kuiper.Array2 
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
open Kuiper.Array1

inline_for_extraction noextract
fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (#lKj #lVj: M.layout bc d)
  (#lSt: layout bc)
  (#lQit #lOit: layout d)
  {| ctlayout lSt, ctlayout lKj, ctlayout lVj |} // ctlayout lQit, ctlayout lOit  <--- TODO: cant infer for row? (just commenting it out to make below typecheck)
  (gKj: M.array2 et lKj) 
  (gVj: M.array2 et lVj)
  (gSt: array1 et lSt)
  (gQit: array1 et lQit)
  (gOit: array1 et lOit)
  (glit gmit: ref et)
  (eKj eVj: ematrix et bc d)
  (vQit vOit: erased (lseq et d))
  (vlit vmit: erased et)
  (#fKj #fVj #fQit: perm)
  requires 
    gOit |-> vOit ** glit |-> vlit ** gmit |-> vmit
  preserves 
    (gKj |-> Frac fKj eKj) ** (gVj |-> Frac fVj eVj) ** (gQit |-> Frac fQit vQit) **
    live gSt
  ensures 
    live gOit ** live glit ** live gmit // No functional spec
{
  assume pure (ctlayout lQit);
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
  (tid: sz { tid <^ br /\ tid <^ bc }) // TODO: impossible to materialize tid in a kernel unless br = bc
  (#fK #fV #fQ: perm)
  preserves 
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    live gSt ** live gOt ** live glt ** live gmt// No functional spec; note that O, l, m would have preconditions here though. S does not
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
      let qi = SZ.v (br *^ !i +^ tid);
      let oi = SZ.v !i;

      M.extract_row_ro gQ qi;
      let gQit = M.row gQ qi;
      M.extract_row gOt oi #1.0R #eOt; // gO has already been split into per-thread chunks
      let gOit = M.row gOt oi;

      explode glt #1.0R #vlt;
      forevery_extract' #(natlt (SZ.v (n /^ br))) oi _;
      array1_cell_to_ref glt oi;
      let glit = ref_of_array_cell glt oi;
      
      explode gmt #1.0R #vmt;
      forevery_extract' #(natlt (SZ.v (n /^ br))) oi _;
      array1_cell_to_ref gmt oi;
      let gmit = ref_of_array_cell gmt oi;

      with eKj. assert gKj |-> eKj;
      with eVj. assert gVj |-> eVj;

      flashattention_tile bc br d
        #_ #_ #_ #_ #_
        #_ #_ #_ // #_ #_ TODO: can't infer ctlayout for row layout?
        gKj gVj gSt (M.row gQ qi) (M.row gOt oi) (ref_of_array_cell glt oi) (ref_of_array_cell gmt oi)
        eKj eVj (ematrix_row eQ qi) (ematrix_row eOt oi) (vlt @! oi) (vmt @! oi);

      admit ();

      M.restore_row gQ qi;
      elim_forall (ematrix_row eOt oi);
      Trade.elim_trade (gOit |-> Frac 1.0R (ematrix_row eOt oi)) _;
      i := !i +^ 1sz;
    };
    
    Trade.elim_trade (gKj |-> Frac fK (ematrix_subtile eK (SZ.v bc) (SZ.v d) (SZ.v !j) 0)) _;
    Trade.elim_trade (gVj |-> Frac fV (ematrix_subtile eV (SZ.v bc) (SZ.v d) (SZ.v !j) 0)) _;

    j := !j +^ 1sz;
  }
}

(*

open Kuiper.Tensor.Layout.Alg

let flashattention_tile_f32 =
  flashattention_tile #f32
  32sz 32sz 128sz
  (l2_row_major _ _)
  (l2_row_major _ _)
  (l2_row_major _ _)
  (l2_row_major _ _)
  (l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
*)