module KPlay.Proof.Tightness

module P = KPlay.SharedSpec
module Root = FStar.Math.Sqrt
module BF16 = Kuiper.BFloat16
module A = Kuiper.Approximates.Base
module Scalar = Kuiper.Scalars.Base

let witness (#h #v:pos) (#state:Type0)
  (epsilon:Root.rpos)
  (hidden weight:P.vector BF16.t h) (embedding:P.matrix BF16.t v h)
  (before:state)
  : GTot (result:P.logical_outputs h v state{
      P.logical_postcondition epsilon hidden weight embedding before result}) =
  P.final_norm_projection_satisfies epsilon hidden weight embedding before;
  P.final_norm_projection epsilon hidden weight embedding before

let unique (#h #v:pos) (#state:Type0)
  (epsilon:Root.rpos)
  (hidden weight:P.vector BF16.t h) (embedding:P.matrix BF16.t v h)
  (before:state) (left right:P.logical_outputs h v state)
  : Lemma
      (requires P.logical_postcondition epsilon hidden weight embedding before left /\
                P.logical_postcondition epsilon hidden weight embedding before right)
      (ensures (
        let (left_logits, left_hidden, left_state) = left in
        let (right_logits, right_hidden, right_state) = right in
        (forall (i:P.index v). left_logits i == right_logits i) /\
        (forall (i:P.index h). left_hidden i == right_hidden i) /\
        left_state == right_state)) =
  ()

let full_contract (#h #v:pos) (#state #score:Type0)
  {| Scalar.scalar score, A.real_like score |}
  (epsilon:Root.rpos)
  (hidden weight:P.vector BF16.t h) (embedding:P.matrix BF16.t v h)
  (before:state) (logits:P.vector score v)
  (observed_hidden:P.vector BF16.t h) (after:state)
  : Lemma (
      let (model_logits, model_hidden, model_state) =
        P.final_norm_projection epsilon hidden weight embedding before in
      P.postcondition epsilon hidden weight embedding before
        logits observed_hidden after <==>
      ((forall (i:P.index v). A.v_approximates (logits i) (model_logits i)) /\
       (forall (i:P.index h). observed_hidden i == model_hidden i) /\
       after == model_state)) =
  ()
