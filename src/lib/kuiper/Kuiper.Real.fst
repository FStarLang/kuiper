module Kuiper.Real

open FStar.Real
include FStar.Real
open Kuiper.Seq.Common
open Kuiper.Common
open FStar.Functions

type rpos = x:real{x >. 0.0R}

(* A proxy, see https://github.com/FStarLang/FStar/issues/4244 *)
assume val rexp0 : real -> rpos

(* This module was a pain to write. See also
https://github.com/FStarLang/FStar/issues/4245.  Many mysterious failures and
brittle proofs. *)

let rexp = rexp0

let rexp0_injective (x y : real)
  : Lemma (ensures rexp0 x == rexp0 y ==> x == y)
  = admit() (* basic property, assumed *)

let rexp0_surjective (x : rpos)
  : Lemma (ensures exists (y:real). rexp0 y == x)
  = admit() (* basic property, assumed *)

let rlog0 : f : ((x:rpos) -> real) {is_bij f /\ f `is_inverse_of` rexp0 /\ rexp0 `is_inverse_of` f} =
  Classical.forall_intro_2 rexp0_injective;
  Classical.forall_intro rexp0_surjective;
  FStar.Functions.inverse_of_bij #real #rpos rexp0

let rlog = rlog0

let rexp_positive (x : real)
  : Lemma (ensures rexp x >. 0.0R)
          [SMTPat (rexp x)]
  = ()

let rexp_base () : Lemma (rexp 0.0R == 1.0R)
  = admit() (* basic property, assumed *)

let exp_add (x y : real)
  : Lemma (ensures rexp (x +. y) == rexp x *. rexp y)
          [SMTPat (rexp (x +. y))]
  = admit() (* basic property, assumed *)

let log_exp (x : real)
  : Lemma (ensures rlog (rexp x) == x)
          [SMTPat (rlog (rexp x))]
  = assert (rlog0 `is_inverse_of` rexp0)

let exp_log (x : real{x >. 0.0R})
  : Lemma (ensures rexp (rlog x) == x)
          [SMTPat (rexp (rlog x))]
  = assert (rlog0 `is_inverse_of` rexp0)

let exp_sub (x y : real)
  : Lemma (ensures rexp (x -. y) == rexp x /. rexp y)
          [SMTPat (rexp (x -. y))]
  = exp_add x (0.0R -. y);
    exp_add (0.0R -. y) y;
    rexp_base ();
    ()

let log_mul (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures rlog (x *. y) == rlog x +. rlog y)
          [SMTPat (rlog (x *. y))]
  = assert (rexp (rlog x +. rlog y) == rexp (rlog x) *. rexp (rlog y));
    ()

let log_div (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures rlog (x /. y) == rlog x -. rlog y)
          [SMTPat (rlog (x /. y))]
  = log_mul x (1.0R /. y);
    assert (rexp (rlog x -. rlog y) == rexp (rlog x) /. rexp (rlog y));
    ()

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
