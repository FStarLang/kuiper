module KPlay.Functional

#lang-pulse

open Kuiper
module Shared = KPlay.SharedSpec
module S = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module Cast = FStar.Int.Cast
module C = FStar.Classical

#set-options "--fuel 2 --ifuel 1 --z3rlimit 80"

inline_for_extraction noextract
let weight_correct (numerator best:f32)
  : Lemma (normalized_weight_post numerator best
      (selected_weight_f32 numerator best)) =
  let approx (nr:real) (br:real{br =!= 0.0R})
    : Lemma (
        v_approximates numerator nr /\ v_approximates best br ==>
        v_approximates (selected_weight_f32 numerator best)
          (selected_weight_real nr br)) =
    if v_approximates numerator nr /\ v_approximates best br then
      selected_weight_real_model numerator best nr br
  in
  C.forall_intro_2 approx

inline_for_extraction noextract
let count_step
  (#tokens:nat) (selected:lseq I32.t tokens) (expert:nat)
  (token:nat{token < tokens})
  : Lemma (
      prefix_count selected expert (token + 1) ==
        prefix_count selected expert token +
          (if I32.v (S.index selected token) = expert then 1 else 0)) =
  ()

inline_for_extraction noextract
let flat_index_distinct
  (width:pos) (row other:nat) (column other_column:nat)
  : Lemma
      (requires column < width /\ other_column < width)
      (ensures
        (row * width + column == other * width + other_column) <==>
        (row == other /\ column == other_column)) =
  if row < other then begin
    assert (row * width + column < (row + 1) * width);
    assert ((row + 1) * width <= other * width)
  end else if other < row then begin
    assert (other * width + other_column < (other + 1) * width);
    assert ((other + 1) * width <= row * width)
  end

inline_for_extraction noextract
let nonnegative_i32_to_sz (x:I32.t{0 <= I32.v x})
  : r:sz{SZ.v r == I32.v x} =
  SZ.uint32_to_sizet (Cast.int32_to_uint32 x)

inline_for_extraction noextract
let positive_i64_to_sz (x:I64.t{0 < I64.v x /\ I64.v x <= 2147483647})
  : r:sz{SZ.v r == I64.v x} =
  SZ.uint32_to_sizet (Cast.int64_to_uint32 x)

inline_for_extraction noextract
let token_to_i64 (x:sz)
  : r:I64.t{I64.v r == SZ.v x} =
  Cast.uint32_to_int64 (SZ.sizet_to_u32 x)

inline_for_extraction noextract
let prefix_initial
  (p:Shared.model_profile)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  : Lemma (
      native_prefix p rows selected best 0
        (S.create (Shared.expert_count p) 0l)
        (S.create (slot_length p) (-1L))
        (S.create (slot_length p) (zero #f32))
        (S.create (row_length p) 0l)) =
  ()

inline_for_extraction noextract
let prefix_finished
  (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  (demand:lseq I32.t (Shared.expert_count p))
  (indices:lseq I64.t (slot_length p))
  (weights:lseq f32 (slot_length p))
  (flags:lseq I32.t (row_length p))
  : Lemma
      (requires
        input_representation p b rows selected best /\
        native_prefix p rows selected best (Shared.token_count p)
          demand indices weights flags)
      (ensures semantic_post p b rows selected best demand indices weights flags) =
  prefix_refines_shared p b rows selected best (Shared.token_count p)
    demand indices weights flags;
  Shared.final_prefix_law b (Shared.route_result b)
    (logical_output p demand indices weights flags)

inline_for_extraction noextract
let slot_advance_unchanged
  (p:Shared.model_profile)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  (token:Shared.token_index p) (expert:Shared.expert_index p)
  (slot:nat) (token_id:I64.t) (probability:f32)
  : Lemma
      (requires
        native_slot_post p rows selected best token expert slot token_id probability /\
        (I32.v (S.index selected token) <> expert \/
          slot <> prefix_count selected expert token))
      (ensures
        native_slot_post p rows selected best (token + 1) expert slot
          token_id probability) =
  count_step selected expert token

inline_for_extraction noextract
let prefix_advance
  (p:Shared.model_profile)
  (rows:lseq f32 (row_length p))
  (selected:lseq I32.t (Shared.token_count p))
  (best:lseq f32 (Shared.token_count p))
  (token:Shared.token_index p) (chosen:Shared.expert_index p)
  (demand:lseq I32.t (Shared.expert_count p))
  (indices:lseq I64.t (slot_length p))
  (weights:lseq f32 (slot_length p))
  (flags:lseq I32.t (row_length p))
  (new_count:I32.t) (new_id:I64.t) (weight:f32)
  : Lemma
      (requires
        native_prefix p rows selected best token demand indices weights flags /\
        I32.v (S.index selected token) == chosen /\
        I32.v new_count == prefix_count selected chosen token + 1 /\
        I64.v new_id == token /\
        normalized_weight_post
          (S.index rows (token * Shared.expert_count p + chosen))
          (S.index best token) weight)
      (ensures (
        let rank = prefix_count selected chosen token in
        let keep = rank < Shared.capacity p in
        native_prefix p rows selected best (token + 1)
          (S.upd demand chosen new_count)
          (if keep then S.upd indices (chosen * Shared.capacity p + rank) new_id
            else indices)
          (if keep then S.upd weights (chosen * Shared.capacity p + rank) weight
            else weights)
          (if keep then S.upd flags (token * Shared.expert_count p + chosen) 1l
            else flags))) =
  let rank = prefix_count selected chosen token in
  let keep = rank < Shared.capacity p in
  let d = S.upd demand chosen new_count in
  let ids = if keep then S.upd indices (chosen * Shared.capacity p + rank) new_id
    else indices in
  let probs = if keep then S.upd weights (chosen * Shared.capacity p + rank) weight
    else weights in
  let survival = if keep then S.upd flags (token * Shared.expert_count p + chosen) 1l
    else flags in
  let demand_ok (expert:Shared.expert_index p)
    : Lemma (
        I32.v (S.index d expert) == prefix_count selected expert (token + 1)) =
    count_step selected expert token
  in
  C.forall_intro demand_ok;
  let slot_ok (expert:Shared.expert_index p) (slot:Shared.slot_index p)
    : Lemma (
        native_slot_post p rows selected best (token + 1) expert slot
          (S.index ids (expert * Shared.capacity p + slot))
          (S.index probs (expert * Shared.capacity p + slot))) =
    if keep then begin
      flat_index_distinct (Shared.capacity p) chosen expert rank slot;
      if expert = chosen && slot = rank then
        count_step selected chosen token
      else
        slot_advance_unchanged p rows selected best token expert slot
          (S.index indices (expert * Shared.capacity p + slot))
          (S.index weights (expert * Shared.capacity p + slot))
    end else
      slot_advance_unchanged p rows selected best token expert slot
        (S.index indices (expert * Shared.capacity p + slot))
        (S.index weights (expert * Shared.capacity p + slot))
  in
  C.forall_intro_2 slot_ok;
  let survival_ok (t:Shared.token_index p) (expert:Shared.expert_index p)
    : Lemma (
        I32.v (S.index survival (t * Shared.expert_count p + expert)) ==
          (if t < token + 1 && retained selected (Shared.capacity p) t expert
           then 1 else 0)) =
    flat_index_distinct (Shared.expert_count p) token t chosen expert;
    if t = token then begin
      if expert = chosen then ()
      else assert (not (retained selected (Shared.capacity p) t expert))
    end
  in
  C.forall_intro_2 survival_ok

inline_for_extraction noextract
fn fill
  (#a:Type0) (n:sz) (output:larray a (SZ.v n)) (value:a)
  (#before:erased (lseq a (SZ.v n)))
  preserves gpu
  requires output |-> before
  ensures output |-> S.create (SZ.v n) value
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{SZ.v j <= SZ.v n}) (values:lseq a (SZ.v n)).
      i |-> j ** output |-> values **
      pure (forall (k:nat{k < SZ.v j}). S.index values k == value)
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    output.(j) <- value;
    i := !i +^ 1sz;
  };
  with values. assert output |-> values;
  assert pure (S.equal values (S.create (SZ.v n) value));
  S.lemma_eq_elim values (S.create (SZ.v n) value);
}

inline_for_extraction noextract
fn routing_kernel
  (#p:erased Shared.model_profile)
  (#b:erased (Shared.selected_boundary p))
  (token_count:sz{SZ.v token_count == Shared.token_count p})
  (expert_count:sz{SZ.v expert_count == Shared.expert_count p})
  (capacity:I64.t{I64.v capacity == Shared.capacity p})
  (normalized_rows:larray f32 (row_length p))
  (selected_experts:larray I32.t (Shared.token_count p))
  (best_values:larray f32 (Shared.token_count p))
  (expert_demand:larray I32.t (Shared.expert_count p))
  (token_indices:larray I64.t (slot_length p))
  (probabilities:larray f32 (slot_length p))
  (survival:larray I32.t (row_length p))
  (#rows_perm #selected_perm #best_perm:perm)
  (#rows:erased (lseq f32 (row_length p)))
  (#selected:erased (lseq I32.t (Shared.token_count p)))
  (#best:erased (lseq f32 (Shared.token_count p)))
  (#demand_before:erased (lseq I32.t (Shared.expert_count p)))
  (#indices_before:erased (lseq I64.t (slot_length p)))
  (#weights_before:erased (lseq f32 (slot_length p)))
  (#flags_before:erased (lseq I32.t (row_length p)))
  preserves gpu **
    normalized_rows |-> Frac rows_perm (reveal rows) **
    selected_experts |-> Frac selected_perm (reveal selected) **
    best_values |-> Frac best_perm (reveal best)
  requires pure (input_representation p b rows selected best) **
    expert_demand |-> demand_before **
    token_indices |-> indices_before **
    probabilities |-> weights_before **
    survival |-> flags_before
  ensures exists*
    (demand:lseq I32.t (Shared.expert_count p))
    (indices:lseq I64.t (slot_length p))
    (weights:lseq f32 (slot_length p))
    (flags:lseq I32.t (row_length p)).
    expert_demand |-> demand **
    token_indices |-> indices **
    probabilities |-> weights **
    survival |-> flags **
    pure (semantic_post p b rows selected best demand indices weights flags)
{
  profile_bounds p;
  let cap = positive_i64_to_sz capacity;
  let slot_count = expert_count *^ cap;
  let flag_count = token_count *^ expert_count;
  fill expert_count expert_demand 0l;
  fill slot_count token_indices (-1L);
  fill slot_count probabilities (zero #f32);
  fill flag_count survival 0l;
  prefix_initial p rows selected best;

  // The retained fusion preserves each expert's token order and scalar rounding.
  let mut token = 0sz;
  while (!token <^ token_count)
    invariant exists*
      (t:sz{SZ.v t <= Shared.token_count p})
      (demand:lseq I32.t (Shared.expert_count p))
      (indices:lseq I64.t (slot_length p))
      (weights:lseq f32 (slot_length p))
      (flags:lseq I32.t (row_length p)).
      token |-> t **
      expert_demand |-> demand **
      token_indices |-> indices **
      probabilities |-> weights **
      survival |-> flags **
      pure (native_prefix p rows selected best (SZ.v t) demand indices weights flags)
    decreases (SZ.v token_count - SZ.v !token)
  {
    let t = !token;
    with demand indices weights flags.
      assert expert_demand |-> demand ** token_indices |-> indices **
        probabilities |-> weights ** survival |-> flags;
    let selected_expert = selected_experts.(t);
    assert pure (I32.v selected_expert == (reveal b).selected_expert (SZ.v t));
    assert pure (0 <= I32.v selected_expert /\
      I32.v selected_expert < Shared.expert_count p);
    let expert = nonnegative_i32_to_sz selected_expert;
    let row_index = t *^ expert_count +^ expert;
    let numerator = normalized_rows.(row_index);
    let best_value = best_values.(t);
    let selected_sum = add (zero #f32) best_value;
    let weight = div numerator selected_sum;
    weight_correct numerator best_value;

    let count = expert_demand.(expert);
    let new_count = I32.add count 1l;
    let slot = I32.sub new_count 1l;
    let slot_index = nonnegative_i32_to_sz slot;
    let keep = slot_index <^ cap;
    let new_id = token_to_i64 t;
    prefix_advance p rows selected best (SZ.v t) (SZ.v expert)
      demand indices weights flags new_count new_id weight;
    expert_demand.(expert) <- new_count;
    token := t +^ 1sz;
    if keep {
      let destination = expert *^ cap +^ slot_index;
      token_indices.(destination) <- new_id;
      probabilities.(destination) <- weight;
      survival.(row_index) <- 1l;
    }
  };
  with demand indices weights flags.
    assert expert_demand |-> demand ** token_indices |-> indices **
      probabilities |-> weights ** survival |-> flags;
  assert pure (native_prefix p rows selected best (Shared.token_count p)
    demand indices weights flags);
  prefix_finished p b rows selected best demand indices weights flags;
  assert pure (semantic_post p b rows selected best demand indices weights flags);
}

inline_for_extraction noextract
instance send_global_array_contents
  (#a:Type0) (arr:array a{is_global_array arr}) (#f:perm) (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (#p:erased Shared.model_profile)
  (#b:erased (Shared.selected_boundary p))
  (token_count:sz{SZ.v token_count == Shared.token_count p})
  (expert_count:sz{SZ.v expert_count == Shared.expert_count p})
  (capacity:I64.t{I64.v capacity == Shared.capacity p})
  (normalized_rows:larray f32 (row_length p){is_global_array normalized_rows})
  (selected_experts:larray I32.t (Shared.token_count p){is_global_array selected_experts})
  (best_values:larray f32 (Shared.token_count p){is_global_array best_values})
  (expert_demand:larray I32.t (Shared.expert_count p){is_global_array expert_demand})
  (token_indices:larray I64.t (slot_length p){is_global_array token_indices})
  (probabilities:larray f32 (slot_length p){is_global_array probabilities})
  (survival:larray I32.t (row_length p){is_global_array survival})
  (#rows_perm #selected_perm #best_perm:perm)
  (#rows:erased (lseq f32 (row_length p)))
  (#selected:erased (lseq I32.t (Shared.token_count p)))
  (#best:erased (lseq f32 (Shared.token_count p)))
  preserves cpu **
    on gpu_loc (
      normalized_rows |-> Frac rows_perm (reveal rows) **
      selected_experts |-> Frac selected_perm (reveal selected) **
      best_values |-> Frac best_perm (reveal best))
  requires pure (input_representation p b rows selected best) **
    (exists*
      (demand_before:lseq I32.t (Shared.expert_count p))
      (indices_before:lseq I64.t (slot_length p))
      (weights_before:lseq f32 (slot_length p))
      (flags_before:lseq I32.t (row_length p)).
      on gpu_loc (
        expert_demand |-> demand_before **
        token_indices |-> indices_before **
        probabilities |-> weights_before **
        survival |-> flags_before))
  ensures exists*
    (demand:lseq I32.t (Shared.expert_count p))
    (indices:lseq I64.t (slot_length p))
    (weights:lseq f32 (slot_length p))
    (flags:lseq I32.t (row_length p)).
    on gpu_loc (
      expert_demand |-> demand **
      token_indices |-> indices **
      probabilities |-> weights **
      survival |-> flags) **
    pure (semantic_post p b rows selected best demand indices weights flags)
{
  with demand_before indices_before weights_before flags_before.
    assert on gpu_loc (
      expert_demand |-> demand_before **
      token_indices |-> indices_before **
      probabilities |-> weights_before **
      survival |-> flags_before);
  launch_kernel_1 (fun () ->
    routing_kernel #p #b token_count expert_count capacity
      normalized_rows selected_experts best_values
      expert_demand token_indices probabilities survival
      #rows_perm #selected_perm #best_perm #rows #selected #best
      #demand_before #indices_before #weights_before #flags_before);
}
