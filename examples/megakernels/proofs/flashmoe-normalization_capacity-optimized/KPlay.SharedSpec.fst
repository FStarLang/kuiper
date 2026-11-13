module KPlay.SharedSpec

open FStar.Real
open Kuiper.Common
module C = FStar.Classical
module I32 = FStar.Int32
module I64 = FStar.Int64
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base
module R = Kuiper.Real

(* Authority: reference_route in base.py and the supplied input-domain.txt.
   This is the stage AFTER native FP32 softmax, not an ideal-real Model.forward.
   Every definition is specification-only; there is no physical layout or ABI. *)

noextract
type model_profile = | Smoke | Full

inline_for_extraction noextract
let token_count (p:model_profile) : n:pos{n <= 1024} =
  match p with | Smoke -> 256 | Full -> 1024

inline_for_extraction noextract
let hidden_count (p:model_profile) : pos =
  match p with | Smoke -> 512 | Full -> 5120

inline_for_extraction noextract
let expert_count (p:model_profile) : pos =
  match p with | Smoke -> 8 | Full -> 16

inline_for_extraction noextract
let capacity (p:model_profile) : pos =
  match p with | Smoke -> 32 | Full -> 64

noextract type token_index (p:model_profile) = natlt (token_count p)
noextract type hidden_index (p:model_profile) = natlt (hidden_count p)
noextract type expert_index (p:model_profile) = natlt (expert_count p)
noextract type slot_index (p:model_profile) = natlt (capacity p)
noextract type prefix_length (p:model_profile) = n:nat{n <= token_count p}
noextract type positive_real = r:real{r >. 0.0R}

(* Shapes and BF16 dtype are intrinsic here. Contiguity and a common device
   are producer representation obligations, not restrictions on stage layout. *)
inline_for_extraction noextract
let input_domain
  (p:model_profile)
  (tokens:token_index p -> hidden_index p -> BF16.t)
  (router_weights:hidden_index p -> expert_index p -> BF16.t)
  : prop =
  (forall t h. F.is_finite (tokens t h) /\
     F.lte (F.abs (tokens t h)) (F.of_int #BF16.t 32L)) /\
  (forall h e. F.is_finite (router_weights h e) /\
     F.lte (F.abs (router_weights h e)) (F.of_int #BF16.t 4L))

inline_for_extraction noextract
let ordered (x:F32.t) : prop = ~(F.is_nan x)

noextract
type native_row (n:pos) =
  row:(natlt n -> F32.t){forall e. ordered (row e)}

(* These are native comparison laws, never order laws about approximation. *)
inline_for_extraction noextract
let native_order2 (x y:F32.t)
  : Lemma
      (requires ordered x /\ ordered y)
      (ensures
        (F.lt x y <==> ~(F.lte y x)) /\
        (F.lt y x <==> ~(F.lte x y)) /\
        (F.lte x y <==> F.lt x y \/ F.eq x y) /\
        (F.lte y x <==> F.lt y x \/ F.eq y x) /\
        (F.eq x y <==> F.eq y x))
  =
  F.eq_spec x y;
  F.eq_spec y x;
  F.lte_is_lt_or_eq x y;
  F.lte_is_lt_or_eq y x;
  F.negate_lt_is_lte x y;
  F.negate_lt_is_lte y x

inline_for_extraction noextract
let native_order3 (x y z:F32.t)
  : Lemma
      (requires ordered x /\ ordered y /\ ordered z)
      (ensures
        (F.lte x y /\ F.lt y z ==> F.lt x z) /\
        (F.lt x y /\ F.lte y z ==> F.lt x z) /\
        (F.lte x y /\ F.lte y z ==> F.lte x z))
  =
  native_order2 x y;
  native_order2 y z;
  native_order2 x z;
  F.eq_spec x y;
  F.eq_spec y z;
  F.eq_spec x z;
  if F.lt x y && F.lt y z then F32.lt_transitive x y z;
  if F.lt y z && F.lt z x then F32.lt_transitive y z x;
  if F.lt z x && F.lt x y then F32.lt_transitive z x y

inline_for_extraction noextract
let rec selection_prefix (#n:pos) (row:native_row n)
  (k:pos{k <= n}) : Tot (natlt k) (decreases k) =
  if k = 1 then 0
  else
    let previous = selection_prefix row (k - 1) in
    if F.lt (row previous) (row (k - 1)) then k - 1 else previous

inline_for_extraction noextract
let reference_selection (#n:pos) (row:native_row n) : natlt n =
  selection_prefix row n

inline_for_extraction noextract
let rec selection_prefix_law (#n:pos) (row:native_row n)
  (k:pos{k <= n}) (e:natlt k)
  : Lemma
      (ensures
        F.lte (row e) (row (selection_prefix row k)) /\
        (e < selection_prefix row k ==>
           F.lt (row e) (row (selection_prefix row k))))
      (decreases k)
  =
  if k = 1 then native_order2 (row 0) (row 0)
  else (
    let previous = selection_prefix row (k - 1) in
    if e < k - 1 then (
      selection_prefix_law row (k - 1) e;
      native_order3 (row e) (row previous) (row (k - 1))
    ) else (
      native_order2 (row previous) (row (k - 1));
      native_order2 (row (k - 1)) (row (k - 1))
    )
  )

inline_for_extraction noextract
let native_selection_law (#n:pos) (row:native_row n) (e:natlt n)
  : Lemma
      (F.lte (row e) (row (reference_selection row)) /\
       (e < reference_selection row ==>
          F.lt (row e) (row (reference_selection row))))
  = selection_prefix_law row n e

inline_for_extraction noextract
let rec selection_prefix_extensional (#n:pos) (a b:native_row n)
  (k:pos{k <= n})
  : Lemma
      (requires forall e. a e == b e)
      (ensures selection_prefix a k == selection_prefix b k)
      (decreases k)
  =
  if k = 1 then () else selection_prefix_extensional a b (k - 1)

inline_for_extraction noextract
let native_tie_law (#n:pos) (row:native_row n) (e:natlt n)
  : Lemma
      (requires F.eq (row e) (row (reference_selection row)))
      (ensures reference_selection row <= e)
  =
  native_selection_law row e;
  native_order2 (row e) (row (reference_selection row))

(* A stable descending argsort is a permutation ordered by native value,
   then original index. This includes ties caused by FP32 rounding. *)
inline_for_extraction noextract
let native_stable_argsort (#n:pos) (row:native_row n)
  (order position:natlt n -> natlt n) : prop =
  (forall e. order (position e) == e) /\
  (forall k. position (order k) == k) /\
  (forall (i j:natlt n). i < j ==>
     F.lt (row (order j)) (row (order i)) \/
     (F.eq (row (order j)) (row (order i)) /\ order i < order j))

inline_for_extraction noextract
let stable_argsort_gather_law (#n:pos) (row:native_row n)
  (order position:natlt n -> natlt n)
  : Lemma
      (requires native_stable_argsort row order position)
      (ensures
        order 0 == reference_selection row /\
        row (order 0) == row (reference_selection row))
  =
  let selected = reference_selection row in
  native_selection_law row (order 0);
  native_order2 (row selected) (row (order 0));
  if position selected = 0 then ()
  else (
    assert (0 < position selected);
    assert (order (position selected) == selected)
  )

(* The producer passes ITS stored probability rows, from FP32 scores and
   native softmax on the original BF16 operands. It does not substitute a
   different original input or recompute the selection in ideal reals.
   Non-NaN rows follow from that finite bounded producer domain. Zero and
   underflowed nonselected entries are allowed. No real-order premise occurs.

   A positive model is attached to the SAME gathered stored value. It can
   come from the producer's algebraic softmax proof at this native index.
   It is not a canonical interpretation of FP32 and never chooses an expert. *)
noextract noeq
type selected_boundary (p:model_profile) = {
  probability_rows: token_index p -> native_row (expert_count p);
  selected_expert: t:token_index p ->
    Tot (e:expert_index p{e == reference_selection (probability_rows t)});
  selected_probability: t:token_index p ->
    Tot (q:F32.t{q == probability_rows t (selected_expert t)});
  selected_model: t:token_index p ->
    Tot (r:positive_real{A.v_approximates (selected_probability t) r})
}

inline_for_extraction noextract
let same_native_rows (#p:model_profile) (a b:selected_boundary p) : prop =
  forall t e. a.probability_rows t e == b.probability_rows t e

inline_for_extraction noextract
let boundary_selection_extensional (#p:model_profile)
  (a b:selected_boundary p) (t:token_index p)
  : Lemma
      (requires same_native_rows a b)
      (ensures a.selected_expert t == b.selected_expert t /\
               a.selected_probability t == b.selected_probability t)
  =
  selection_prefix_extensional
    (a.probability_rows t) (b.probability_rows t) (expert_count p)

(* A constructive witness adapter, not a new softmax axiom. Its premise is
   the ordinary cellwise algebraic softmax postcondition on the actual
   stored rows. Any score models supplied here belong to the same producer
   run. This helper imposes no agreement between native and real rankings. *)
inline_for_extraction noextract
let rec exponential_sum (#n:pos) (scores:natlt n -> real)
  (k:nat{k <= n})
  : Tot (s:real{s >=. 0.0R /\ (k > 0 ==> s >. 0.0R)}) (decreases k) =
  if k = 0 then 0.0R
  else (
    R.exp_positive (scores (k - 1));
    exponential_sum scores (k - 1) +. R.exp (scores (k - 1))
  )

inline_for_extraction noextract
let softmax_model (#n:pos) (scores:natlt n -> real) (e:natlt n)
  : positive_real =
  R.exp_positive (scores e);
  R.exp (scores e) /. exponential_sum scores n

inline_for_extraction noextract
let boundary_from_softmax_models (p:model_profile)
  (rows:token_index p -> native_row (expert_count p))
  (scores:token_index p -> expert_index p -> real)
  (_:squash (forall t e.
       A.v_approximates (rows t e) (softmax_model (scores t) e)))
  : selected_boundary p =
  {
    probability_rows = rows;
    selected_expert = (fun t -> reference_selection (rows t));
    selected_probability = (fun t -> rows t (reference_selection (rows t)));
    selected_model =
      (fun t -> softmax_model (scores t) (reference_selection (rows t)))
  }

inline_for_extraction noextract
let selected_set_sum (selected:positive_real) : positive_real =
  0.0R +. selected

inline_for_extraction noextract
let normalized_probability (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) : real =
  b.selected_model t /. selected_set_sum (b.selected_model t)

inline_for_extraction noextract
let singleton_sum_approx (#p:model_profile) (b:selected_boundary p)
  (t:token_index p)
  : Lemma (A.v_approximates (b.selected_probability t)
             (selected_set_sum (b.selected_model t)))
  = ()

(* The denominator is the selected-set SUM, not a literal one and not the
   sum of the full expert row. Any realization of that sum may use this law. *)
inline_for_extraction noextract
let normalization_approx (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) (selected_sum:F32.t)
  : Lemma
      (requires A.v_approximates selected_sum
                  (selected_set_sum (b.selected_model t)))
      (ensures A.v_approximates
                 (F.div (b.selected_probability t) selected_sum)
                 (normalized_probability b t))
  =
  A.div_approx (b.selected_probability t) selected_sum
    (b.selected_model t) (selected_set_sum (b.selected_model t))

inline_for_extraction noextract
let normalization_model_independent (r s:positive_real)
  : Lemma (r /. selected_set_sum r == s /. selected_set_sum s)
  = ()

inline_for_extraction noextract
let rec assigned_count (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (e:expert_index p)
  : Tot (k:nat{k <= n}) (decreases n) =
  if n = 0 then 0
  else assigned_count b (n - 1) e +
       (if b.selected_expert (n - 1) = e then 1 else 0)

inline_for_extraction noextract
let rec slot_owner (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (e:expert_index p) (c:slot_index p)
  : Tot (owner:int{-1 <= owner /\ owner < n}) (decreases n) =
  if n = 0 then -1
  else
    let t = n - 1 in
    if b.selected_expert t = e && assigned_count b t e = c
    then t
    else slot_owner b t e c

inline_for_extraction noextract
let slot_probability (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (e:expert_index p) (c:slot_index p) : real =
  let owner = slot_owner b n e c in
  if owner < 0 then 0.0R else normalized_probability b owner

inline_for_extraction noextract
let prefix_survival (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (t:token_index p) (e:expert_index p)
  : k:nat{k <= 1} =
  if t < n && b.selected_expert t = e && assigned_count b t e < capacity p
  then 1 else 0

inline_for_extraction noextract
let rec slot_characterization (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (e:expert_index p) (c:slot_index p)
  : Lemma
      (ensures
        (slot_owner b n e c == -1 <==> assigned_count b n e <= c) /\
        (slot_owner b n e c >= 0 ==>
          b.selected_expert (slot_owner b n e c) == e /\
          assigned_count b (slot_owner b n e c) e == c))
      (decreases n)
  =
  if n = 0 then () else slot_characterization b (n - 1) e c

inline_for_extraction noextract
let rec assigned_count_monotone (#p:model_profile) (b:selected_boundary p)
  (first:prefix_length p) (last:prefix_length p{first <= last})
  (e:expert_index p)
  : Lemma
      (ensures assigned_count b first e <= assigned_count b last e)
      (decreases (last - first))
  =
  if last = first then ()
  else assigned_count_monotone b first (last - 1) e

inline_for_extraction noextract
let assignment_rank_strict (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) (n:prefix_length p) (e:expert_index p)
  : Lemma
      (requires t < n /\ b.selected_expert t == e)
      (ensures assigned_count b t e < assigned_count b n e)
  = assigned_count_monotone b (t + 1) n e

inline_for_extraction noextract
let retained_slot_law (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (t:token_index p) (e:expert_index p)
  : Lemma
      (requires t < n /\ b.selected_expert t == e /\
                assigned_count b t e < capacity p)
      (ensures slot_owner b n e (assigned_count b t e) == t /\
               slot_probability b n e (assigned_count b t e) ==
                 normalized_probability b t)
  =
  let rank = assigned_count b t e in
  assignment_rank_strict b t n e;
  slot_characterization b n e rank;
  let owner = slot_owner b n e rank in
  assert (owner >= 0);
  if owner < t then assignment_rank_strict b owner t e
  else if t < owner then assignment_rank_strict b t owner e

(* Exact correspondence with assignments * (inclusive cumsum <= capacity). *)
inline_for_extraction noextract
let survival_reference_law (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (t:token_index p) (e:expert_index p)
  : Lemma
      (prefix_survival b n t e ==
        (if t < n then
           (if b.selected_expert t = e then 1 else 0) *
           (if assigned_count b (t + 1) e <= capacity p then 1 else 0)
         else 0))
  = ()

inline_for_extraction noextract
let occupied_slot_survives (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (e:expert_index p) (c:slot_index p)
  : Lemma
      (requires slot_owner b n e c >= 0)
      (ensures prefix_survival b n (slot_owner b n e c) e == 1)
  = slot_characterization b n e c

noextract noeq
type routing_state (p:model_profile) = {
  counts: expert_index p -> nat;
  owners: expert_index p -> slot_index p -> int;
  weights: expert_index p -> slot_index p -> real;
  survival_cells: token_index p -> expert_index p -> nat
}

inline_for_extraction noextract
let prefix_state (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) : routing_state p =
  {
    counts = assigned_count b n;
    owners = slot_owner b n;
    weights = slot_probability b n;
    survival_cells = prefix_survival b n
  }

inline_for_extraction noextract
let state_equal (#p:model_profile) (a b:routing_state p) : prop =
  (forall e. a.counts e == b.counts e) /\
  (forall e c. a.owners e c == b.owners e c /\
               a.weights e c == b.weights e c) /\
  (forall t e. a.survival_cells t e == b.survival_cells t e)

inline_for_extraction noextract
let initial_state (p:model_profile) : routing_state p =
  {
    counts = (fun _ -> 0);
    owners = (fun _ _ -> -1);
    weights = (fun _ _ -> 0.0R);
    survival_cells = (fun _ _ -> 0)
  }

inline_for_extraction noextract
let initial_prefix_law (#p:model_profile) (b:selected_boundary p)
  : Lemma (state_equal (initial_state p) (prefix_state b 0))
  = ()

inline_for_extraction noextract
let count_step (#p:model_profile) (b:selected_boundary p) (t:token_index p)
  (before:expert_index p -> nat) (e:expert_index p) : nat =
  before e + (if b.selected_expert t = e then 1 else 0)

inline_for_extraction noextract
let slot_step (#a:Type) (#p:model_profile)
  (b:selected_boundary p) (t:token_index p)
  (before_count:expert_index p -> nat)
  (before:expert_index p -> slot_index p -> a) (value:a)
  (e:expert_index p) (c:slot_index p) : a =
  if b.selected_expert t = e && before_count e < capacity p &&
     c = before_count e
  then value else before e c

inline_for_extraction noextract
let survival_step (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) (before_count:expert_index p -> nat)
  (before:token_index p -> expert_index p -> nat)
  (i:token_index p) (e:expert_index p) : nat =
  if i = t then
    (if b.selected_expert t = e && before_count e < capacity p then 1 else 0)
  else before i e

inline_for_extraction noextract
let prefix_transition (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) (before:routing_state p) : routing_state p =
  {
    counts = count_step b t before.counts;
    owners = slot_step b t before.counts before.owners t;
    weights = slot_step b t before.counts before.weights
                (normalized_probability b t);
    survival_cells = survival_step b t before.counts before.survival_cells
  }

inline_for_extraction noextract
let count_step_law (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) (e:expert_index p)
  : Lemma (assigned_count b (t + 1) e ==
             count_step b t (assigned_count b t) e)
  = ()

inline_for_extraction noextract
let slot_step_law (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) (e:expert_index p) (c:slot_index p)
  : Lemma
      (slot_owner b (t + 1) e c ==
         slot_step b t (assigned_count b t) (slot_owner b t) t e c /\
       slot_probability b (t + 1) e c ==
         slot_step b t (assigned_count b t) (slot_probability b t)
           (normalized_probability b t) e c)
  = ()

inline_for_extraction noextract
let survival_step_law (#p:model_profile) (b:selected_boundary p)
  (t i:token_index p) (e:expert_index p)
  : Lemma (prefix_survival b (t + 1) i e ==
             survival_step b t (assigned_count b t) (prefix_survival b t) i e)
  = ()

inline_for_extraction noextract
let prefix_transition_law (#p:model_profile) (b:selected_boundary p)
  (t:token_index p) (before:routing_state p)
  : Lemma
      (requires state_equal before (prefix_state b t))
      (ensures state_equal (prefix_transition b t before)
                 (prefix_state b (t + 1)))
  =
  C.forall_intro (count_step_law b t);
  C.forall_intro_2 (slot_step_law b t);
  C.forall_intro_2 (survival_step_law b t)

inline_for_extraction noextract
let rec prefix_run (#p:model_profile) (b:selected_boundary p)
  (first:prefix_length p) (last:prefix_length p{first <= last})
  (before:routing_state p) : Tot (routing_state p) (decreases (last - first)) =
  if last = first then before
  else prefix_transition b (last - 1) (prefix_run b first (last - 1) before)

inline_for_extraction noextract
let rec prefix_composition_law (#p:model_profile) (b:selected_boundary p)
  (first middle last:prefix_length p{first <= middle /\ middle <= last})
  (before:routing_state p)
  : Lemma
      (ensures prefix_run b first last before ==
        prefix_run b middle last (prefix_run b first middle before))
      (decreases (last - middle))
  =
  if last = middle then ()
  else prefix_composition_law b first middle (last - 1) before

inline_for_extraction noextract
let rec prefix_run_law (#p:model_profile) (b:selected_boundary p)
  (first:prefix_length p) (last:prefix_length p{first <= last})
  (before:routing_state p)
  : Lemma
      (requires state_equal before (prefix_state b first))
      (ensures state_equal (prefix_run b first last before) (prefix_state b last))
      (decreases (last - first))
  =
  if last = first then ()
  else (
    prefix_run_law b first (last - 1) before;
    prefix_transition_law b (last - 1) (prefix_run b first (last - 1) before)
  )

inline_for_extraction noextract
let transition_model_independent (#p:model_profile)
  (a b:selected_boundary p) (t:token_index p)
  (before_a before_b:routing_state p)
  : Lemma
      (requires a.selected_expert t == b.selected_expert t /\
                state_equal before_a before_b)
      (ensures state_equal (prefix_transition a t before_a)
                           (prefix_transition b t before_b))
  = normalization_model_independent (a.selected_model t) (b.selected_model t)

(* All logical outputs depend only on the actual native rows, not on the
   choice of positive approximation witnesses or redundant boundary fields. *)
inline_for_extraction noextract
let rec prefix_native_extensional (#p:model_profile)
  (a b:selected_boundary p) (n:prefix_length p)
  : Lemma
      (requires same_native_rows a b)
      (ensures state_equal (prefix_state a n) (prefix_state b n))
      (decreases n)
  =
  if n = 0 then ()
  else (
    let t = n - 1 in
    prefix_native_extensional a b t;
    boundary_selection_extensional a b t;
    transition_model_independent a b t (prefix_state a t) (prefix_state b t);
    prefix_transition_law a t (prefix_state a t);
    prefix_transition_law b t (prefix_state b t)
  )

noextract noeq
type public_outputs (p:model_profile) = {
  expert_demand: expert_index p -> I32.t;
  token_indices: expert_index p -> slot_index p -> I64.t;
  probabilities: expert_index p -> slot_index p -> F32.t;
  survival: token_index p -> expert_index p -> I32.t
}

inline_for_extraction noextract
let state_representation (#p:model_profile)
  (state:routing_state p) (out:public_outputs p) : prop =
  (forall e. I32.v (out.expert_demand e) == state.counts e) /\
  (forall e c. I64.v (out.token_indices e c) == state.owners e c /\
     A.v_approximates (out.probabilities e c) (state.weights e c)) /\
  (forall t e. I32.v (out.survival t e) == state.survival_cells t e)

inline_for_extraction noextract
let route_result (#p:model_profile) (b:selected_boundary p) : routing_state p =
  prefix_state b (token_count p)

inline_for_extraction noextract
let route_post (#p:model_profile) (b:selected_boundary p)
  (out:public_outputs p) : prop =
  state_representation (route_result b) out

inline_for_extraction noextract
let signed_public_bounds (#p:model_profile) (b:selected_boundary p)
  (n:prefix_length p) (e:expert_index p) (c:slot_index p)
  : Lemma
      (I32.fits (assigned_count b n e) /\
       I64.fits (slot_owner b n e c))
  = ()

inline_for_extraction noextract
let final_prefix_law (#p:model_profile) (b:selected_boundary p)
  (state:routing_state p) (out:public_outputs p)
  : Lemma
      (requires state_equal state (prefix_state b (token_count p)) /\
                state_representation state out)
      (ensures route_post b out)
  = ()

inline_for_extraction noextract
let route_post_native_extensional (#p:model_profile)
  (a b:selected_boundary p) (out:public_outputs p)
  : Lemma
      (requires same_native_rows a b /\ route_post a out)
      (ensures route_post b out)
  =
  prefix_native_extensional a b (token_count p);
  final_prefix_law b (route_result a) out

inline_for_extraction noextract
let completed_run_law (#p:model_profile) (b:selected_boundary p)
  (out:public_outputs p)
  : Lemma
      (requires state_representation
          (prefix_run b 0 (token_count p) (initial_state p)) out)
      (ensures route_post b out)
  =
  initial_prefix_law b;
  prefix_run_law b 0 (token_count p) (initial_state p);
  final_prefix_law b
    (prefix_run b 0 (token_count p) (initial_state p)) out
