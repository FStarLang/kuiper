module Kuiper.Kernel.UGS.Epilogue

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Float.Casts.Base
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

(* Proof of [epilogue_cell_approx]: walks [epilogue_cell]'s definition
   step by step, calling the matching [Kuiper.Approximates.Base] algebra
   lemma at each step (each of which also has an [SMTPat]-triggered
   variant, so most of these calls are redundant with automatic firing,
   but are spelled out explicitly for robustness/readability, mirroring
   this codebase's existing style e.g. in [Kuiper.Kernel.RowSoftmax]). *)
let epilogue_cell_approx (gate_h up_h : half) (gR uR : real)
  : Lemma (requires gate_h %~ gR /\ up_h %~ uR)
          (ensures epilogue_cell gate_h up_h %~ real_swiglu gR uR)
  = let g : float = cast_f16_to_f32 gate_h in
    cast_f16_to_f32_ok gate_h gR;
    assert (v_approximates (zero #float) 0.0R);
    assert (v_approximates (one #float) 1.0R);
    sub_approx (zero #float) g 0.0R gR;
    let negG : float = zero `sub` g in
    exp_approx negG (0.0R -. gR);
    assert (exp (0.0R -. gR) >. 0.0R);
    a_add (one #float) (fexp negG) 1.0R (exp (0.0R -. gR));
    let denom : float = add (one #float) (fexp negG) in
    div_approx (one #float) denom 1.0R (1.0R +. exp (0.0R -. gR));
    let sgm : float = div (one #float) denom in
    a_mul g sgm gR (1.0R /. (1.0R +. exp (0.0R -. gR)));
    let silu_f : float = mul g sgm in
    cast_f32_to_f16_ok silu_f (gR *. (1.0R /. (1.0R +. exp (0.0R -. gR))));
    let silu_h : half = cast_f32_to_f16 silu_f in
    a_mul silu_h up_h (gR *. (1.0R /. (1.0R +. exp (0.0R -. gR)))) uR

(* If [n > 0] and [k < m * n] then [k / n < m]. (Same shape as
   [Kuiper.Kernel.UGS.Round]'s private helper of the same purpose; kept
   local here since that one isn't exposed outside its own module.) *)
let lemma_div_bound (k m n : nat)
  : Lemma (requires n > 0 /\ k < m * n)
          (ensures k / n < m)
  = if m = 0 then ()
    else begin
      Math.small_div (n - 1) n;
      Math.lemma_div_plus (n - 1) (m - 1) n;
      Math.lemma_div_le k (m * n - 1) n
    end

(* Proof of [epilogue_matrix_approx]: for each output cell, extract the
   corresponding gate/up approximation facts, feed them to
   [epilogue_cell_approx], and collect the result. *)
let epilogue_matrix_approx
  (#m #n : nat)
  (vGate vUp : chest2 half m n)
  (rGate rUp : chest2 real m n)
  (eOut : chest2 half m n)
  : Lemma (requires
             vGate %~ rGate /\
             vUp %~ rUp /\
             (forall (r : natlt m) (c : natlt n).
                acc2 eOut r c ==
                  epilogue_cell (acc2 vGate r c) (acc2 vUp r c)))
          (ensures eOut %~ chest_comb real_swiglu rGate rUp)
  = introduce forall (r : natlt m) (c : natlt n).
      acc2 eOut r c %~ acc2 (chest_comb real_swiglu rGate rUp) r c
    with (
      assert (acc2 vGate r c %~ acc2 rGate r c);
      assert (acc2 vUp r c %~ acc2 rUp r c);
      epilogue_cell_approx
        (acc2 vGate r c) (acc2 vUp r c)
        (acc2 rGate r c) (acc2 rUp r c)
    );
    lemma_approximates_intro eOut (chest_comb real_swiglu rGate rUp)

(* [converted_before n2 k r c] holds iff the row-major-flattened index of
   cell [(r,c)] (in the [n]-wide output) is strictly less than the linear
   loop counter [k], i.e. iff cell [(r,c)] has already been overwritten by
   the epilogue by the time the loop's counter reaches [k]. *)
let converted_before (n k r c : nat) : bool =
  r * n + c < k

let converted_before_last (m n r c : nat)
  : Lemma (requires n > 0 /\ r < m /\ c < n)
          (ensures converted_before n (m * n) r c == true)
  = Math.lemma_mult_le_left n r (m - 1);
    Math.distributivity_sub_right n m 1

let epilogue_final_forall
  (m n : nat)
  (vOut vGate vUp vOut0 : chest2 half m n)
  : Lemma
      (requires
        (forall (r : natlt m) (c : natlt n).
           acc2 vOut r c ==
             (if converted_before n (m * n) r c
              then epilogue_cell (acc2 vGate r c) (acc2 vUp r c)
              else acc2 vOut0 r c)))
      (ensures
        forall (r : natlt m) (c : natlt n).
          acc2 vOut r c ==
            epilogue_cell (acc2 vGate r c) (acc2 vUp r c))
  = introduce forall (r : natlt m) (c : natlt n).
      acc2 vOut r c == epilogue_cell (acc2 vGate r c) (acc2 vUp r c)
    with converted_before_last m n r c

inline_for_extraction noextract
fn epilogue
  (#m #n : szp)
  (#_ : squash (SZ.fits (m * n)))
  (gGate : array2 half (rm m n) { is_global gGate })
  (gUp : array2 half (rm m n) { is_global gUp })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#fGate #fUp : perm)
  (#vGate #vUp #vOut0 : chest2 half m n)
  requires
    gpu **
    gGate |-> Frac fGate vGate **
    gUp |-> Frac fUp vUp **
    gOut |-> vOut0
  ensures
    gpu **
    gGate |-> Frac fGate vGate **
    gUp |-> Frac fUp vUp **
    (exists* (vOut : chest2 half m n).
      gOut |-> vOut **
      pure (
        forall (r : natlt m) (c : natlt n).
          acc2 vOut r c ==
            epilogue_cell (acc2 vGate r c) (acc2 vUp r c)))
{
  let mut k : sz = 0sz;
  let kend = m *^ n;

  while (!k <^ kend)
    invariant
      gpu **
      gGate |-> Frac fGate vGate **
      gUp |-> Frac fUp vUp **
      live k **
      pure (!k <= kend) **
      (exists* (vOut : chest2 half m n).
        gOut |-> vOut **
        pure (
          forall (r : natlt m) (c : natlt n).
            acc2 vOut r c ==
              (if converted_before n !k r c
               then epilogue_cell (acc2 vGate r c) (acc2 vUp r c)
               else acc2 vOut0 r c)))
    decreases (kend - !k)
  {
    let kv = !k;
    assert pure (kv < kend);
    assert pure (kv < m * n);
    let ri : sz = kv /^ n;
    let ci : szlt n = kv %^ n;
    lemma_div_bound kv m n;
    assert pure (ri < m);
    let ri : szlt m = ri;
    let gate_h = tensor_read gGate (cidx2 ri ci);
    let up_h = tensor_read gUp (cidx2 ri ci);
    let y = epilogue_cell gate_h up_h;
    tensor_write gOut (cidx2 ri ci) y;
    k := kv +^ 1sz;
    dassert (!k <=^ kend);
  };

  with vOut. assert (gOut |-> vOut);
  epilogue_final_forall m n vOut vGate vUp vOut0;
}
