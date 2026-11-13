module KPlay.Proof.Tightness

module P = KPlay.SharedSpec
module A = Kuiper.Approximates.Base
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64
module C = FStar.Classical

let probability_witness (#p:P.model_profile) (boundary:P.selected_boundary p)
  (expert:P.expert_index p) (slot:P.slot_index p)
  : value:F32.t{
      A.v_approximates value ((P.route_result boundary).weights expert slot)} =
  let owner = P.slot_owner boundary (P.token_count p) expert slot in
  let _ = A.a0 #F32.t in
  if owner < 0 then F.zero #F32.t
  else (
    let q = boundary.selected_probability owner in
    let denominator = F.add (F.zero #F32.t) q in
    A.a_add (F.zero #F32.t) q 0.0R (boundary.selected_model owner);
    P.normalization_approx boundary owner denominator;
    F.div q denominator
  )

let witness (#p:P.model_profile) (boundary:P.selected_boundary p)
  : result:P.public_outputs p{P.route_post boundary result} =
  {
    expert_demand = (fun e -> I32.int_to_t (P.assigned_count boundary (P.token_count p) e));
    token_indices = (fun e c -> I64.int_to_t (P.slot_owner boundary (P.token_count p) e c));
    probabilities = probability_witness boundary;
    survival = (fun t e -> I32.int_to_t (P.prefix_survival boundary (P.token_count p) t e))
  }

let full_contract (#p:P.model_profile) (boundary:P.selected_boundary p)
  (out:P.public_outputs p)
  : Lemma (
      P.route_post boundary out <==>
      ((forall (e:P.expert_index p).
          I32.v (out.expert_demand e) == P.assigned_count boundary (P.token_count p) e) /\
       (forall (e:P.expert_index p) (c:P.slot_index p).
          I64.v (out.token_indices e c) == P.slot_owner boundary (P.token_count p) e c /\
          A.v_approximates (out.probabilities e c)
            (P.slot_probability boundary (P.token_count p) e c)) /\
       (forall (t:P.token_index p) (e:P.expert_index p).
          I32.v (out.survival t e) == P.prefix_survival boundary (P.token_count p) t e))) =
  ()

let exact_outputs_unique (#p:P.model_profile) (boundary:P.selected_boundary p)
  (left right:P.public_outputs p)
  : Lemma
      (requires P.route_post boundary left /\ P.route_post boundary right)
      (ensures
        (forall (e:P.expert_index p). left.expert_demand e == right.expert_demand e) /\
        (forall (e:P.expert_index p) (c:P.slot_index p).
          left.token_indices e c == right.token_indices e c) /\
        (forall (t:P.token_index p) (e:P.expert_index p).
          left.survival t e == right.survival t e)) =
  let demand (e:P.expert_index p)
    : Lemma (left.expert_demand e == right.expert_demand e) =
    I32.v_inj (left.expert_demand e) (right.expert_demand e)
  in
  let owner (e:P.expert_index p) (c:P.slot_index p)
    : Lemma (left.token_indices e c == right.token_indices e c) =
    I64.v_inj (left.token_indices e c) (right.token_indices e c)
  in
  let survives (t:P.token_index p) (e:P.expert_index p)
    : Lemma (left.survival t e == right.survival t e) =
    I32.v_inj (left.survival t e) (right.survival t e)
  in
  C.forall_intro demand;
  C.forall_intro_2 owner;
  C.forall_intro_2 survives

let all_targets_unique (#p:P.model_profile)
  (left right:P.selected_boundary p)
  : Lemma
      (requires P.same_native_rows left right)
      (ensures P.state_equal (P.route_result left) (P.route_result right)) =
  P.prefix_native_extensional left right (P.token_count p)
