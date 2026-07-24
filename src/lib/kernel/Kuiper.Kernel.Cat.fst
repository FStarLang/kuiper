module Kuiper.Kernel.Cat

(* Implementation of the verified torch.cat kernel; see Kuiops.Cat.fsti for the
   public spec (abs_cat / cat_chest) and the cat_gpu signature. *)

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Shareable

open Kuiper.Kernel.TMap

module SZ = Kuiper.SizeT

(* ----------------------------------------------------------------------- *)
(* Index machinery: the cat kernel reads, for each output coordinate `i`, the
   coordinate `j` along `dim` (via `abs_get_at`) and rebuilds the input index by
   narrowing across `dim` (via `abs_narrow`, copying the off-`dim` coordinates).
   This lemma re-expresses the abstract spec `abs_cat` -- defined through the
   `abs_bring_forward_bij` split -- in those coordinate terms, so the concrete
   `conc_get_at`/`conc_narrow` used in `fcat` (both of which commute with the
   abstract ops by construction) discharge the per-element contract. Crucially,
   it never mentions `conc (modulo_i dim d)`, the type karamel cannot extract. *)

(* `abs_get_at` reads the same coordinate the bijection peels off along `dim`. *)
let rec get_at_is_ff (#r : nat) (dim : natlt r) (d : shape r) (x : abs d)
  : Lemma (ensures abs_get_at #r #d dim x == fst ((abs_bring_forward_bij dim d).ff x))
          (decreases dim)
  = let (h, t) = x <: natlt (d @! 0) & abs (tail d) in
    if dim = 0 then ()
    else get_at_is_ff #(r-1) (dim - 1) (tail d) t

(* `abs_narrow` into `d1` reinserts `v` along `dim` over the off-`dim` remainder
   that the bijection on `d2` peels off (the remainders coincide via the
   `modulo_i` equality). *)
let rec narrow_is_gg (#r : nat) (dim : natlt r) (d1 d2 : shape r)
  (v : natlt (d1 @! dim)) (x : abs d2 { modulo_i dim d1 == modulo_i dim d2 })
  : Lemma (ensures abs_narrow dim d1 d2 v x ==
                   (abs_bring_forward_bij dim d1).gg (v, snd ((abs_bring_forward_bij dim d2).ff x)))
          (decreases dim)
  = let (h, t) = x <: natlt (d2 @! 0) & abs (tail d2) in
    if dim = 0 then (modulo_zero d1; modulo_zero d2)
    else (modulo_succ dim d1; modulo_succ dim d2;
          narrow_is_gg #(r-1) (dim - 1) (tail d1) (tail d2) v t)

let cat_via_narrow (#et: Type0) (#r : nat)
  (dim : natlt r) (dA dB dout : shape r)
  (eA : chest dA et) (eB : chest dB et)
  (na : nat { na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (x : abs dout)
  : Lemma (ensures
      (let j = abs_get_at #r #dout dim x in
       abs_cat dim dA dB dout eA eB na pf_sz pfA pfB x ==
         (if j < na then acc eA (abs_narrow dim dA dout j x)
          else acc eB (abs_narrow dim dB dout (j - na) x))))
  = get_at_is_ff dim dout x;
    let (j, m) = (abs_bring_forward_bij dim dout).ff x in
    if j < na then narrow_is_gg dim dA dout j x
    else narrow_is_gg dim dB dout (j - na) x

(* ----------------------------------------------------------------------- *)
(* The per-element value relation (links abs_cat to the map's contract). *)

let vcat (#et: Type0) (#r : nat)
  (dim : natlt r) (dA dB dout : shape r)
  (eA : chest dA et) (eB : chest dB et)
  (na : nat { na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (x : abs dout) (_old : et) (o : et)
  : prop
  = o == abs_cat dim dA dB dout eA eB na pf_sz pfA pfB x

(* ----------------------------------------------------------------------- *)
(* The two read-only inputs, held with split fractions of `fr` (so the frame is
   shareable across threads), exactly as gather holds its input + index. *)

unfold
let cat_frame (#et : Type0) (#r : erased nat) (dA dB : shape r)
  (#lA : tlayout dA) (#lB : tlayout dB) {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (eA : chest dA et)
  (eB : chest dB et)
  (fA fB : perm) (fr : perm) : slprop =
    (tensor_pts_to gA #(fA *. fr) eA) **
    (tensor_pts_to gB #(fB *. fr) eB)

instance cat_frame_shareable
  (#et : Type0) (#r : erased nat) (dA dB : shape r)
  (#lA : tlayout dA) (#lB : tlayout dB) {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (eA : chest dA et)
  (eB : chest dB et)
  (fA fB : perm) :
  shareable (cat_frame dA dB #lA #lB gA gB eA eB fA fB) =
    double_shareable
      (fun fr -> tensor_pts_to gA #fr eA)
      (fun fr -> tensor_pts_to gB #fr eB)
      fA fB

(* ----------------------------------------------------------------------- *)
(* Per-element function: read the right input for one output coordinate. *)

inline_for_extraction noextract
fn fcat
  (#et : Type0) (#r : nat)
  (dim : natlt r) (dimsz : szlt r { SZ.v dimsz == dim })
  (dA dB dout : shape r)
  (cdA : cshape dA) (cdB : cshape dB) (cdout : cshape dout)
  (#lA : tlayout dA) (#lB : tlayout dB) {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (na : sz { SZ.v na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (eA : chest dA et)
  (eB : chest dB et)
  (#fA #fB : perm)
  (#fr : perm) (i : conc dout) (x : et)
norewrite
preserves
  gpu **
  (cat_frame dA dB #lA #lB gA gB eA eB fA fB fr)
returns
  res : et
ensures
  pure (vcat dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB (up i) x res)
{
  cat_via_narrow dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB (up i);
  let jc = conc_get_at cdout dimsz i;
  let b = jc <^ na;
  if b {
    let ia = conc_narrow dimsz cdA cdout () jc () i;
    let v = tensor_read gA ia;
    v
  } else {
    let ib = conc_narrow dimsz cdB cdout () (jc -^ na) () i;
    let v = tensor_read gB ib;
    v
  }
}

(* ----------------------------------------------------------------------- *)
(* The kernel: map over the output, reading from the two inputs. *)

inline_for_extraction noextract
fn cat_gpu
  (#et : Type0) (#r : nat)
  (dA dB dout : shape r)
  (cdA : cshape dA) (cdB : cshape dB) (cdout : cshape dout)
  (dim : natlt r) (dimsz : szlt r { SZ.v dimsz == dim })
  (na : sz { SZ.v na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (#lA : tlayout dA) (#lB : tlayout dB) (#lOut : tlayout dout)
  {| ctlayout lA, ctlayout lB, ctlayout lOut |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (gOut : tensor et lOut {is_global gOut})
  (n : sz {SZ.v n == sizeof dout /\ n <= max_blocks * max_threads /\ n > 0})
  (eA : chest dA et)
  (eB : chest dB et)
  (#fA #fB : perm)
  preserves cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)
  requires on gpu_loc (live gOut)
  ensures
    on gpu_loc (gOut |-> cat_chest dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB)
{
  with eOut. assert on gpu_loc (gOut |-> eOut);
  launch_sync
    (kmap cdout
      (cat_frame dA dB #lA #lB gA gB eA eB fA fB)
      #(cat_frame_shareable dA dB #lA #lB gA gB eA eB fA fB)
      (vcat dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB)
      (fcat dim dimsz dA dB dout cdA cdB cdout #lA #lB gA gB na pf_sz pfA pfB eA eB)
      n gOut #eOut #_ #1.0R);
  with eOut'. assert on gpu_loc (gOut |-> eOut');
  assert pure (Kuiper.Chest.equal eOut'
                 (cat_chest dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB));
}
