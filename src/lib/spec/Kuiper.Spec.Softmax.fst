module Kuiper.Spec.Softmax

#lang-pulse
open Kuiper
open Kuiper.Chest
open Kuiper.Seq.Common

let chest1_to_seq_map (#a #b:Type) (#n:nat) (f:a->b) (c:chest1 a n)
  : Lemma (chest1_to_seq (chest_map f c) == seq_map f (chest1_to_seq c))
  = Seq.lemma_eq_elim (chest1_to_seq (chest_map f c)) (seq_map f (chest1_to_seq c))

let lem_softmax_real_to_seq #n (s : chest1 real n)
  : Lemma (chest1_to_seq (softmax_real s) == softmax_real_seq (chest1_to_seq s))
          [SMTPat (chest1_to_seq (softmax_real s))]
  = chest1_to_seq_map exp s;
    Seq.lemma_eq_elim (chest1_to_seq (softmax_real s))
                      (softmax_real_seq (chest1_to_seq s))

let chest1_roundtrip (#a:Type) (#n:nat) (c : chest1 a n)
  : Lemma (seq_to_chest1 (chest1_to_seq c) == c)
  = lemma_equal_intro (seq_to_chest1 (chest1_to_seq c)) c;
    ext (seq_to_chest1 (chest1_to_seq c)) c


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
  = (* fold_div_scale gives the result starting from accumulator [0.0R /. k];
       bridge that to [rsum] (which folds from [0.0R]) via [0.0R /. k == 0.0R]. *)
    fold_div_scale 0.0R k s;
    calc (==) {
      rsum (seq_map (fun w -> w /. k) s);
      == { (* rsum = seq_fold_left (+.) 0.0R, and 0.0R == 0.0R /. k *) }
      seq_fold_left (+.) (0.0R /. k) (seq_map (fun (e:real) -> e /. k) s);
      == { fold_div_scale 0.0R k s }
      seq_fold_left (+.) 0.0R s /. k;
      == { }
      rsum s /. k;
    }

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
let softmax_shift_seq #n (r0 : lseq real n) (c : real)
  : Lemma (ensures softmax_real_seq (seq_map (fun x -> x -. c) r0)
                   == softmax_real_seq r0)
  = if len r0 > 0 then (
      let exps = seq_map exp r0 in
      let exps' = seq_map (fun x -> exp (x -. c)) r0 in
      sum_non_zero exps 0.0R;
      shift_denom r0 c;
      let lhs = softmax_real_seq (seq_map (fun x -> x -. c) r0) in
      let rhs = softmax_real_seq r0 in
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
    assert softmax_real_seq r0 `Seq.equal` seq![];
    assert softmax_real_seq (seq_map (fun x -> x -. c) r0) `Seq.equal` seq![];
    ()
  )

let softmax_shift #n (r0 : chest1 real n) (c : real)
  : Lemma (ensures softmax_real (chest_map (fun x -> x -. c) r0)
                   == softmax_real r0)
  = let cm = chest_map (fun x -> x -. c) r0 in
    chest1_to_seq_map (fun x -> x -. c) r0;
    softmax_shift_seq (chest1_to_seq r0) c;
    chest1_roundtrip (softmax_real cm);
    chest1_roundtrip (softmax_real r0)
