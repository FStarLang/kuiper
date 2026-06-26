module Kuiper.Kernel.FlashAttention

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Bijection

inline_for_extraction noextract
fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (lS: layout2 br bc)
  (lKj lVj: layout2 bc d)
  (lQi lOi: layout2 br d)
  {| ctlayout lS, ctlayout lKj, ctlayout lVj, ctlayout lQi, ctlayout lOi |}
  (gS: array2 et lS)
  (gKj: array2 et lKj)
  (gVj: array2 et lVj)
  (gQi: array2 et lQi)
  (gOi: array2 et lOi)
  (gl gm: ref et)
  (eKj: chest2 et bc d)
  (eVj: chest2 et bc d)
  (eQi: chest2 et br d)
  (eOi: chest2 et br d)
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
      let vq: et = tensor_read gQi (cidx2 tid vx);
      let vk: et = tensor_read gKj (cidx2 vy vx);
      sum := !sum `add` (vq `mul` vk);
      x := !x +^ 1sz;
    };
    // TODO: add softmax scale factor
    // sum := !sum * alpha;

    let vy = !y;
    tensor_write gS (cidx2 tid vy) !sum;
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
    let vs: et = fexp ((tensor_read gS (cidx2 tid vy)) `sub` !row_m);
    tensor_write gS (cidx2 tid vy) vs;
    row_l := !row_l `add` vs;

    y := !y +^ 1sz;
  };

  let row_m_new = fmax row_m_prev !row_m;
  let row_l_new = row_l_prev `mul` (fexp (row_m_prev `sub` row_m_new)) `add` (!row_l `mul` (fexp (!row_m `sub` row_m_new)));

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
      let vs: et = tensor_read gS (cidx2 tid vy);
      let vv: et = tensor_read gVj (cidx2 vy vx);
      pv := !pv `add` (vs `mul` vv);

      y := !y +^ 1sz;
    };

    let vx = !x;
    let vo: et = tensor_read gOi (cidx2 tid vx);
    let vo: et = (vo `mul` row_l_prev `mul` (fexp (row_m_prev `sub` row_m_new))) `div` row_l_new;
    let vo: et = vo `add` ((fexp (!row_m `sub` row_m_new)) `mul` !pv);

    tensor_write gOi (cidx2 tid vx) vo;

    x := !x +^ 1sz;
  }
}

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
