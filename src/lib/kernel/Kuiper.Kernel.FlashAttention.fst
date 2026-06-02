module Kuiper.Kernel.FlashAttention

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Array
open Kuiper.Tensor.Layout

module M = Kuiper.Array2 


open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.EMatrix
open Kuiper.Index
open Kuiper.Array1

fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (#lS: M.layout br bc)  {| ctlayout lS |}
  (#lKj lVj: M.layout bc d) {| ctlayout lKj, ctlayout lVj |}
  (#lQi lOi: M.layout br d) {| ctlayout lQi, ctlayout lOi |}
  (gS: M.array2 et lS)
  (gKj: M.array2 et lKj) 
  (gVj: M.array2 et lVj)
  (gQi: M.array2 et lQi)
  (gOi: M.array2 et lOi)
  (gl gm: ref et)
  (eKj: ematrix et bc d)
  (eVj: ematrix et bc d)
  (eQi: ematrix et br d)
  (eOi: ematrix et br d)
  (vl vm: et)
  (tid: sz { tid <^ br /\ tid <^ bc }) // TODO: impossible to materialize tid in a kernel unless br = bc
  requires 
    gOi |-> eOi
  preserves 
    gKj |-> eKj ** gVj |-> eVj ** gQi |-> eQi ** gl |-> vl ** gm |-> vm **
    (exists* (eS: ematrix et br bc). gS |-> eS)
  ensures 
    (exists* (eOi': ematrix et br d). gOi |-> eOi')
{
  let row_m_prev = !gm;
  let row_l_prev = !gl;
  let mut row_m: et = neg infinity;
  let mut y: szle bc = 0sz;
  
  while (!y <^ bc) 
    invariant live y ** live row_m ** live gS
    decreases (bc - !y)
  {
	  let mut sum: et = zero;
    let mut x: szle d = 0sz;
    while (!x <^ d) 
      invariant live x ** live sum
      decreases (d - !x)
    {
      assert pure (tid <^ br);
      assert pure (!x <^ d);
      let vx = !x; let vy = !y;
      let vq: et = M.read gQi ((tid <: sz), (vx <: sz));
      let vk: et = M.read gKj ((vy <: sz), (vx <: sz));
      sum := !sum `add` (vq `mul` vk);
      x := !x +^ 1sz;
    };
    // TODO: add softmax scale factor
    // sum := !sum * alpha;

    let wtf2 = !y;
    let idxs: M.raw_cit = ((tid <: sz), (wtf2 <: sz));
    M.write gS idxs !sum;
    row_m := fmax !row_m !sum;
    
    y := !y +^ 1sz;
  }; 

  let mut row_l: et = zero;
  y := 0sz;
  while (!y <^ bc)
    invariant live y ** live row_l ** live gS
    decreases (bc - !y)
  {
    let wtf2 = !y;
    let idxs: M.raw_cit = ((tid <: sz), (wtf2 <: sz));
    let vs: et = exp ((M.read gS idxs) `sub` !row_m);
    M.write gS idxs vs;
    row_l := !row_l `add` vs;

    y := !y +^ 1sz;
  };

  let row_m_new = fmax row_m_prev !row_m;
  let row_l_new = row_l_prev `mul` (exp (row_m_prev `sub` row_m_new)) `add` (!row_l `mul` (exp (!row_m `sub` row_m_new)));

  let mut x: sz = 0sz;
  while (!x <^ d) 
    invariant live x ** live gOi
    decreases (d - !x) 
  {
    let mut pv: et = zero;
    y := 0sz;    
    while (!y <^ bc) 
      invariant live y ** live pv 
      decreases (bc - !y)
    {
      let wtf1 = !x; let wtf2 = !y;
      let idxs: M.raw_cit = ((tid <: sz), (wtf2 <: sz));
      let idxv: M.raw_cit = ((wtf2 <: sz), (wtf1 <: sz));
      let vs: et = M.read gS idxs;
      let vv: et = M.read gVj idxv;
      pv := !pv `add` (vs `mul` vv);

      y := !y +^ 1sz;
    };

    let wtf1 = !x;
    let idxo: M.raw_cit = ((tid <: sz), (wtf1 <: sz));
    let vo: et = M.read gOi idxo;
    let vo: et = (vo `mul` row_l_prev `mul` (exp (row_m_prev `sub` row_m_new))) `div` row_l_new;
    let vo: et = vo `add` ((exp (!row_m `sub` row_m_new)) `mul` !pv);

    x := !x +^ 1sz;
  }
}
  
(*

qs:
* cit
* syntax for add, mul, etc.

*)