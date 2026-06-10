module Kuiper.Spec.Softmax

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common

(* [seq_fold_left (+.)] commutes with pointwise division by a constant, scaling
   both the elements and the initial accumulator.  Generalizing over [acc] is
   what makes the recursion go through (the tail fold carries [acc +. hd], not
   [0.0R]).  Mirrors [lemma_seq_fold_left_distrib_mul] in Kuiper.Math.OnlineSoftmax. *)
let rec fold_div_scale (acc : real) (k : real { k =!= 0.0R }) (s : Seq.seq real)
  : Lemma (ensures seq_fold_left (+.) (acc /. k) (seq_map (fun (e:real) -> e /. k) s)
                   == seq_fold_left (+.) acc s /. k)
          (decreases Seq.length s)
  = let f : real -> real = fun (e:real) -> e /. k in
    let s_mapped = seq_map f s in
    match view_seq s with
    | SNil ->
      assert (Seq.equal s Seq.empty);
      assert (Seq.equal s_mapped Seq.empty)
    | SCons hd tl ->
      assert (Seq.equal s_mapped (Seq.cons (hd /. k) (seq_map f tl)));
      calc (==) {
        seq_fold_left (+.) (acc /. k) s_mapped;
        == { }
        seq_fold_left (+.) ((acc /. k) +. (hd /. k)) (seq_map f tl);
        == { }
        seq_fold_left (+.) ((acc +. hd) /. k) (seq_map f tl);
        == { fold_div_scale (acc +. hd) k tl }
        seq_fold_left (+.) (acc +. hd) tl /. k;
        == { }
        seq_fold_left (+.) acc s /. k;
      }

let rsum_div_scale (s : Seq.seq real) (k : real { k =!= 0.0R })
  : Lemma (ensures rsum (seq_map (fun w -> w /. k) s) == rsum s /. k)
  = fold_div_scale 0.0R k s

(* The shifted exps are the unshifted exps divided by [exp c]. *)
let shift_denom (r0 : Seq.seq real) (c : real)
  : Lemma (rsum (seq_map (fun z -> exp (z -. c)) r0)
           == rsum (seq_map exp r0) /. exp c)
  = assert (Seq.equal (seq_map (fun z -> exp (z -. c)) r0)
                      (seq_map (fun w -> w /. exp c) (seq_map exp r0)));
    rsum_div_scale (seq_map exp r0) (exp c)

let div_cancel_aux (a b c : real{b =!= 0.0R /\ c =!= 0.0R})
  : Lemma ((a /. c) /. (b /. c) == a /. b)
  = ()

(* The pointwise softmax value is unchanged by the shift. *)
let softmax_shift (r0 : Seq.seq real) (c : real)
  : Lemma (ensures softmax_real (seq_map (fun x -> x -. c) r0)
                   == softmax_real r0)
  = if len r0 > 0 then (
      let exps = seq_map exp r0 in
      let exps' = seq_map (fun x -> exp (x -. c)) r0 in
      sum_non_zero exps 0.0R;
      shift_denom r0 c;
      let lhs = softmax_real (seq_map (fun x -> x -. c) r0) in
      let rhs = softmax_real r0 in
      let aux (i : nat { i < Seq.length r0 }) : Lemma (lhs @! i == rhs @! i) =
        calc (==) {
          lhs @! i;
          == {}
          exp ((r0 @! i) -. c) /. rsum (seq_map exp (seq_map (fun x -> x -. c) r0));
          == {
            assert
              seq_map exp (seq_map (fun x -> x -. c) r0)
              `Seq.equal`
              seq_map (fun x -> exp (x -. c)) r0
          }
          exp ((r0 @! i) -. c) /. rsum (seq_map (fun x -> exp (x -. c)) r0);
          == {}
          exp ((r0 @! i) -. c) /. (rsum exps /. exp c);
          == { exp_sub c (r0 @! i) }
          (exp (r0 @! i) /. exp c) /. (rsum exps /. exp c);
          == { div_cancel_aux (exp (r0 @! i)) (rsum exps) (exp c) }
          exp (r0 @! i) /. rsum exps;
          == {}
          rhs @! i;
        }
      in
      Classical.forall_intro aux;
      assert Seq.equal lhs rhs;
      ()
  ) else (
    assert softmax_real r0 `Seq.equal` seq![];
    assert softmax_real (seq_map (fun x -> x -. c) r0) `Seq.equal` seq![];
    ()
  )
