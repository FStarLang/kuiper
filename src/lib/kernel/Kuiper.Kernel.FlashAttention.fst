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

inline_for_extraction noextract
fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (lS: M.layout br bc)
  (lKj lVj: M.layout bc d)
  (lQi lOi: M.layout br d)
  {| ctlayout lS, ctlayout lKj, ctlayout lVj, ctlayout lQi, ctlayout lOi |}
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
  (vl vm: erased et)
  (tid: sz { tid <^ br /\ tid <^ bc }) // TODO: impossible to materialize tid in a kernel unless br = bc
  requires
    gOi |-> eOi
  preserves
    gKj |-> eKj ** gVj |-> eVj ** gQi |-> eQi ** gl |-> vl ** gm |-> vm **
    live gS
  ensures
    live gOi // No functional spec
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

    let vy = !y;
    M.write gS ((tid <: sz), (vy <: sz)) !sum;
    row_m := fmax !row_m !sum;

    y := !y +^ 1sz;
  };

  let mut row_l: et = zero;
  y := 0sz;
  while (!y <^ bc)
    invariant live y ** live row_l ** live gS
    decreases (bc - !y)
  {
    let vy = !y;
    let vs: et = exp ((M.read gS ((tid <: sz), (vy <: sz))) `sub` !row_m);
    M.write gS ((tid <: sz), (vy <: sz)) vs;
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
      let vx = !x; let vy = !y;
      let vs: et = M.read gS ((tid <: sz), (vy <: sz));
      let vv: et = M.read gVj ((vy <: sz), (vx <: sz));
      pv := !pv `add` (vs `mul` vv);

      y := !y +^ 1sz;
    };

    let vx = !x;
    let vo: et = M.read gOi ((tid <: sz), (vx <: sz));
    let vo: et = (vo `mul` row_l_prev `mul` (exp (row_m_prev `sub` row_m_new))) `div` row_l_new;
    let vo: et = vo `add` ((exp (!row_m `sub` row_m_new)) `mul` !pv);

    M.write gOi ((tid <: sz), (vx <: sz)) vo;

    x := !x +^ 1sz;
  }
}

open Kuiper.Tensor.Layout.Alg

let flashattention_f32 =
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
