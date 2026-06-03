module Kuiper.Kernel.FlashAttention

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Array
open Kuiper.Tensor.Layout
open Kuiper.EMatrix

module M = Kuiper.Array2 
open Kuiper.Array1

inline_for_extraction noextract
fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (lKj lVj: M.layout bc d)
  (lSt: layout bc)
  (lQit lOit: layout d)
  {| ctlayout lKj, ctlayout lVj, ctlayout lSt, ctlayout lQit, ctlayout lOit |}
  (gKj: M.array2 et lKj) 
  (gVj: M.array2 et lVj)
  (gSt: array1 et lSt)
  (gQit: array1 et lQit)
  (gOit: array1 et lOit)
  (glt gmt: ref et)
  (eKj: ematrix et bc d)
  (eVj: ematrix et bc d)
  (vQit vOit: erased (lseq et d))
  (vlt vmt: erased et)
  (#fKj #fVj #fQit: perm)
  requires 
    gOit |-> vOit
  preserves 
    (gKj |-> Frac fKj eKj) ** (gVj |-> Frac fVj eVj) ** (gQit |-> Frac fQit vQit) ** glt |-> vlt ** gmt |-> vmt **
    live gSt
  ensures 
    live gOit // No functional spec
{
  let row_m_prev = !gmt;
  let row_l_prev = !glt;
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
    let vs: et = (exp gSt.(vy)) `sub` !row_m;
    gSt.(vy) <- vs;
    row_l := !row_l `add` vs;

    y := !y +^ 1sz;
  };

  let row_m_new = fmax row_m_prev !row_m;
  let row_l_new = row_l_prev `mul` (exp (row_m_prev `sub` row_m_new)) `add` (!row_l `mul` (exp (!row_m `sub` row_m_new)));

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
    let vo: et = (vo `mul` row_l_prev `mul` (exp (row_m_prev `sub` row_m_new))) `div` row_l_new;
    let vo: et = vo `add` ((exp (!row_m `sub` row_m_new)) `mul` !pv);

    gOit.(vx) <- vo;

    x := !x +^ 1sz;
  }
}

(*
// Which elements each thread owns in Q and O
let ttile_j (#et : Type0) {| scalar et, floating et |}
  (N d: szp)
  (bc br: szp)
  (i j: szp)

// Which elements each thread owns (at the start of each iteration of i loop)

// flash attention kernel executed by each thread (no shared memory caching)
inline_for_extraction noextract 
fn flashattention_kf_no_smem (#et : Type0) {| scalar et, floating et |}
  (N d: szp)
  (bc br: szp)
  (lSt: layout bc)
  (lK lV lQ lO: M.layout N d)
  {| ctlayout lSt, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO |}
  (gSt: array1 et lSt)
  (gK: M.array2 et lK) 
  (gV: M.array2 et lV)
  (gQ: M.array2 et lQ)
  (gO: M.array2 et lO)
  (gl gm: larray et N)
  (eK eV eQ eO: ematrix et N d)
  (vl vm: erased (lseq et N))
  (tid: sz { tid <^ br /\ tid <^ bc }) // TODO: impossible to materialize tid in a kernel unless br = bc
  requires 
    gOi |-> eOi
  preserves 
    gKj |-> eKj ** gVj |-> eVj ** gQi |-> eQi ** gl |-> vl ** gm |-> vm **
    live gS
  ensures 
    live gOi // No functional spec
{

}

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