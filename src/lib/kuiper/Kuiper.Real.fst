module Kuiper.Real

open FStar.Real
open Kuiper.Seq.Common
open Kuiper.Common
open FStar.Functions

let rsum_append (s1 s2 : Seq.seq real)
  : Lemma (ensures rsum (s1 @+ s2) == rsum s1 +. rsum s2)
          [SMTPat (rsum (s1 @+ s2))]
  = calc (==) {
      rsum (s1 @+ s2);
      == {}
      seq_fold_left (+.) 0.0R (s1 @+ s2);
      == { lemma_seq_fold_left_sum 0.0R (+.) s1 s2 }
      seq_fold_left (+.) 0.0R s1 +. seq_fold_left (+.) 0.0R s2;
    }

let lem_rmax_comm (x: real) (y: real)
  : Lemma (ensures rmax x y == rmax y x)
  = ()

let lem_rmax_assoc (x: real) (y: real) (z: real)
  : Lemma (ensures rmax x (rmax y z) == rmax (rmax x y) z)
  = ()

(* Real square root: an abstract primitive whose defining property is assumed
   for non-negative inputs (FStar.Real only provides sqrt_2). *)
assume val realsqrt0 : real -> real
let realsqrt = realsqrt0
let realsqrt_nonneg_sq x = admit ()

let rec sum_non_zero
    (s : Seq.seq real { forall (i:natlt (Seq.length s)). Seq.index s i >. 0.0R })
    (acc : real)
  : Lemma (requires Seq.length s > 0)
          (ensures seq_fold_left (+.) acc s >. acc)
          (decreases Seq.length s)
          [SMTPat (seq_fold_left (+.) acc s)]
  = if Seq.length s = 1 then ()
    else
      let SCons hd tl = view_seq s in
      sum_non_zero tl (acc +. hd <: real)
