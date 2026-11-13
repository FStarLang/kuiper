module KPlay.Functional

#lang-pulse

open Kuiper
module Shared = KPlay.SharedSpec
module I32 = FStar.Int32
module I64 = FStar.Int64
module U64 = FStar.UInt64
module Cast = FStar.Int.Cast
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module G = FStar.Ghost
module C = FStar.Classical

#set-options "--z3rlimit 80"

inline_for_extraction noextract
let u64_as_size (v:u64{U64.v v < 4294967296})
  : r:sz{SZ.v r == U64.v v} =
  SZ.uint32_to_sizet (Cast.uint64_to_uint32 v)

inline_for_extraction noextract
let i64_as_size (v:i64{0 <= I64.v v /\ I64.v v < 4294967296})
  : r:sz{SZ.v r == I64.v v} =
  SZ.uint32_to_sizet (Cast.int64_to_uint32 v)

inline_for_extraction noextract
let i32_as_size (v:i32{0 <= I32.v v})
  : r:sz{SZ.v r == I32.v v} =
  SZ.uint32_to_sizet (Cast.int32_to_uint32 v)

inline_for_extraction noextract
let count_as_i32 (v:i64{0 <= I64.v v /\ I64.v v <= 1024})
  : r:i32{I32.v r == I64.v v} =
  Cast.int64_to_int32 v

inline_for_extraction noextract
let incoming_count_bounds
  (p:Shared.model_profile) (x:reference_input p) (t:Shared.token_index p)
  (before:physical_state p (t + 1))
  : Lemma
      (requires state_representation p (t + 1) before (Shared.prefix_state x t))
      (ensures forall (e:Shared.expert_index p).
        0 <= I64.v (Seq.index before.counts e) /\
        I64.v (Seq.index before.counts e) <= t)
  = ()

inline_for_extraction noextract
let packed_offset_injective
  (width:pos) (a b:nat) (i j:nat{i < width /\ j < width})
  : Lemma (a * width + i == b * width + j <==> a == b /\ i == j)
  = ()

inline_for_extraction noextract
let updated_state
  (p:Shared.model_profile)
  (token:i64{0 <= I64.v token /\ I64.v token < Shared.token_count p})
  (chosen:Shared.expert_index p) (weight:f32)
  (before:physical_state p (I64.v token + 1))
  (rank:i64{0 < I64.v rank})
  : physical_state p (I64.v token + 1) =
  let counts = Seq.upd before.counts chosen rank in
  if I64.v rank <= Shared.capacity p then
    let off = slot_offset p chosen (I64.v rank - 1) in
    {
      counts = counts;
      indices = Seq.upd before.indices off token;
      weights = Seq.upd before.weights off weight;
      flags = Seq.upd before.flags
        (survival_offset p (I64.v token + 1) (I64.v token) chosen) 1l
    }
  else {
    counts = counts;
    indices = before.indices;
    weights = before.weights;
    flags = before.flags
  }

inline_for_extraction noextract
let updated_state_is_native
  (p:Shared.model_profile)
  (token:i64{0 <= I64.v token /\ I64.v token < Shared.token_count p})
  (chosen:Shared.expert_index p)
  (row:lseq f32 (Shared.expert_count p)) (weight:f32)
  (before:physical_state p (I64.v token + 1))
  (rank:i64{0 < I64.v rank})
  : Lemma
      (requires
        I64.v rank == I64.v (Seq.index before.counts chosen) + 1 /\
        weight == selected_weight (Seq.index row chosen))
      (ensures
        native_step p (I64.v token) chosen row before
          (updated_state p token chosen weight before rank))
  =
  let after = updated_state p token chosen weight before rank in
  let count_cell (e:Shared.expert_index p)
    : Lemma (
        I64.v (Seq.index after.counts e) ==
          I64.v (Seq.index before.counts e) + (if e = chosen then 1 else 0))
    = ()
  in
  C.forall_intro count_cell;
  let slot_cell (e:Shared.expert_index p) (j:Shared.slot_index p)
    : Lemma (
        let off = slot_offset p e j in
        if e = chosen && I64.v (Seq.index before.counts e) = j then
          I64.v (Seq.index after.indices off) == I64.v token /\
          Seq.index after.weights off == selected_weight (Seq.index row chosen)
        else
          Seq.index after.indices off == Seq.index before.indices off /\
          Seq.index after.weights off == Seq.index before.weights off)
    =
    if I64.v rank <= Shared.capacity p then
      packed_offset_injective (Shared.capacity p) e chosen j (I64.v rank - 1)
    else ()
  in
  C.forall_intro_2 slot_cell;
  let flag_cell (u:nat{u < I64.v token + 1}) (e:Shared.expert_index p)
    : Lemma (
        let off = survival_offset p (I64.v token + 1) u e in
        if u = I64.v token && e = chosen &&
           I64.v (Seq.index before.counts e) < Shared.capacity p then
          I32.v (Seq.index after.flags off) == 1
        else Seq.index after.flags off == Seq.index before.flags off)
    =
    packed_offset_injective (Shared.expert_count p) u (I64.v token) e chosen
  in
  C.forall_intro_2 flag_cell

inline_for_extraction noextract
fn copy_array
  (#a:Type0) (n:sz)
  (input output:larray a (SZ.v n))
  (#values:erased (lseq a (SZ.v n)))
  (#old:erased (lseq a (SZ.v n)))
  preserves gpu ** input |-> values
  requires output |-> old
  ensures output |-> values
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (v:lseq a (SZ.v n)).
      i |-> j ** output |-> v **
      pure (forall (k:nat{k < SZ.v j}). Seq.index v k == Seq.index values k)
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    let value = input.(j);
    output.(j) <- value;
    i := j +^ 1sz;
  };
  with v. assert output |-> v;
  assert pure (Seq.equal v values);
}

inline_for_extraction noextract
fn write_retained
  (#p:erased Shared.model_profile)
  (experts:sz{SZ.v experts == Shared.expert_count p})
  (capacity:sz{SZ.v capacity == Shared.capacity p})
  (token:i64{0 <= I64.v token /\ I64.v token < Shared.token_count p})
  (chosen:sz{SZ.v chosen < Shared.expert_count p})
  (weight:f32) (rank:i64{0 < I64.v rank /\ I64.v rank <= 1024})
  (#before:erased (physical_state p (I64.v token + 1)))
  (indices:larray i64 (slot_length p))
  (weights:larray f32 (slot_length p))
  (flags:larray i32 (survival_length p (I64.v token + 1)))
  preserves gpu
  requires
    indices |-> (G.reveal before).indices **
    weights |-> (G.reveal before).weights **
    flags |-> (G.reveal before).flags
  ensures
    indices |-> (updated_state p token (SZ.v chosen) weight before rank).indices **
    weights |-> (updated_state p token (SZ.v chosen) weight before rank).weights **
    flags |-> (updated_state p token (SZ.v chosen) weight before rank).flags
{
  let rank_size = i64_as_size rank;
  if (rank_size <=^ capacity) {
    let destination = chosen *^ capacity +^ (rank_size -^ 1sz);
    indices.(destination) <- token;
    weights.(destination) <- weight;
    let token_size = i64_as_size token;
    let flag_offset = token_size *^ experts +^ chosen;
    flags.(flag_offset) <- 1l;
  }
}

inline_for_extraction noextract
fn write_demand
  (#p:erased Shared.model_profile)
  (token:i64{0 <= I64.v token /\ I64.v token < Shared.token_count p})
  (#after:erased (physical_state p (I64.v token + 1)))
  (counts:larray i64 (Shared.expert_count p))
  (demand:larray i32 (demand_length (I64.v token)))
  (#old:erased (lseq i32 (demand_length (I64.v token))))
  preserves gpu ** counts |-> (G.reveal after).counts
  requires pure (forall (e:Shared.expert_index p).
    0 <= I64.v (Seq.index (G.reveal after).counts e) /\
    I64.v (Seq.index (G.reveal after).counts e) <= 1024)
  requires demand |-> old
  ensures exists* (values:lseq i32 (demand_length (I64.v token))).
    demand |-> values **
    pure (demand_representation p (I64.v token) values after)
{
  if (I64.eq token 255L) {
    let mut i = 0sz;
    while (!i <^ 8sz)
      invariant exists* (j:sz{j <= 8})
        (v:lseq i32 (demand_length (I64.v token))).
        i |-> j ** demand |-> v **
        pure (forall (e:nat{e < SZ.v j}).
          I32.v (Seq.index v e) ==
            I64.v (Seq.index (G.reveal after).counts e))
      decreases (8 - SZ.v !i)
    {
      let j = !i;
      let count = counts.(j);
      let value = count_as_i32 count;
      demand.(j) <- value;
      i := j +^ 1sz;
    };
    with values. assert demand |-> values;
    introduce exists* values. demand |-> values **
      pure (demand_representation p (I64.v token) values after)
      with values;
  } else {
    introduce exists* values. demand |-> values **
      pure (demand_representation p (I64.v token) values after)
      with old;
  }
}

inline_for_extraction noextract
fn routing_kernel
  (#p:erased Shared.model_profile)
  (#x:erased (reference_input p))
  (#selected:erased (lseq i32 1))
  (#row:erased (lseq f32 (Shared.expert_count p)))
  (experts:u64{U64.v experts == Shared.expert_count p})
  (capacity:u64{U64.v capacity == Shared.capacity p})
  (input_selected:larray i32 1)
  (input_row:larray f32 (Shared.expert_count p))
  (input_counts:larray i64 (Shared.expert_count p))
  (input_indices:larray i64 (slot_length p))
  (input_weights:larray f32 (slot_length p))
  (token:i64{0 <= I64.v token /\ I64.v token < Shared.token_count p})
  (input_flags:larray i32 (survival_length p (I64.v token + 1)))
  (#before:erased (physical_state p (I64.v token + 1)))
  (#old:erased (physical_state p (I64.v token + 1)))
  (#old_demand:erased (lseq i32 (demand_length (I64.v token))))
  (output_counts:larray i64 (Shared.expert_count p))
  (output_indices:larray i64 (slot_length p))
  (output_weights:larray f32 (slot_length p))
  (output_flags:larray i32 (survival_length p (I64.v token + 1)))
  (output_demand:larray i32 (demand_length (I64.v token)))
  preserves gpu **
    input_selected |-> selected **
    input_row |-> row **
    input_counts |-> (G.reveal before).counts **
    input_indices |-> (G.reveal before).indices **
    input_weights |-> (G.reveal before).weights **
    input_flags |-> (G.reveal before).flags
  requires pure (input_representation p x (I64.v token) selected row before)
  requires
    output_counts |-> (G.reveal old).counts **
    output_indices |-> (G.reveal old).indices **
    output_weights |-> (G.reveal old).weights **
    output_flags |-> (G.reveal old).flags **
    output_demand |-> old_demand
  ensures
    exists* (after:physical_state p (I64.v token + 1))
      (demand:lseq i32 (demand_length (I64.v token))).
    output_counts |-> after.counts **
    output_indices |-> after.indices **
    output_weights |-> after.weights **
    output_flags |-> after.flags **
    output_demand |-> demand **
    pure (semantic_post p x (I64.v token) row before after demand)
{
  approved_profile_fits_native p (I64.v token);
  incoming_count_bounds p x (I64.v token) before;
  let e_size = u64_as_size experts;
  let c_size = u64_as_size capacity;
  let t_size = i64_as_size token;
  let slots = e_size *^ c_size;
  let cells = (t_size +^ 1sz) *^ e_size;
  copy_array e_size input_counts output_counts;
  copy_array slots input_indices output_indices;
  copy_array slots input_weights output_weights;
  copy_array cells input_flags output_flags;
  let selected_value = input_selected.(0sz);
  let chosen = i32_as_size selected_value;
  let probability = input_row.(chosen);
  let selected_sum = add (zero #f32) probability;
  let weight = div probability selected_sum;
  let previous_count = output_counts.(chosen);
  let rank = I64.add previous_count 1L;
  output_counts.(chosen) <- rank;
  write_retained e_size c_size token chosen weight rank #before
    output_indices output_weights output_flags;
  let after = G.hide (updated_state p token (SZ.v chosen) weight before rank);
  updated_state_is_native p token (SZ.v chosen) row weight before rank;
  write_demand token #after output_counts output_demand;
  with demand. assert output_demand |-> demand;
  observed_step_refines_shared p x (I64.v token) selected row before after demand;
  introduce exists* (demand:lseq i32 (demand_length (I64.v token))).
    output_counts |-> (G.reveal after).counts **
    output_indices |-> (G.reveal after).indices **
    output_weights |-> (G.reveal after).weights **
    output_flags |-> (G.reveal after).flags **
    output_demand |-> demand **
    pure (semantic_post p x (I64.v token) row before after demand)
    with demand;
  introduce exists* (after:physical_state p (I64.v token + 1)).
    exists* (demand:lseq i32 (demand_length (I64.v token))).
    output_counts |-> after.counts **
    output_indices |-> after.indices **
    output_weights |-> after.weights **
    output_flags |-> after.flags **
    output_demand |-> demand **
    pure (semantic_post p x (I64.v token) row before after demand)
    with after;
}

instance send_global_array_contents
  (#a:Type0) (arr:array a{is_global_array arr}) (#f:perm) (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (#p:erased Shared.model_profile)
  (#x:erased (reference_input p))
  (#selected:erased (lseq i32 1))
  (#row:erased (lseq f32 (Shared.expert_count p)))
  (experts:u64{
    U64.v experts == Shared.expert_count p /\
    U64.v experts <= 2147483648})
  (capacity:u64{
    U64.v capacity == Shared.capacity p /\ slot_length p < 4294967296})
  (input_selected:larray i32 1{is_global_array input_selected})
  (input_row:larray f32 (Shared.expert_count p){is_global_array input_row})
  (input_counts:larray i64 (Shared.expert_count p){is_global_array input_counts})
  (input_indices:larray i64 (slot_length p){is_global_array input_indices})
  (input_weights:larray f32 (slot_length p){is_global_array input_weights})
  (token:i64{
    0 <= I64.v token /\ I64.v token < Shared.token_count p /\
    survival_length p (I64.v token + 1) < 4294967296})
  (input_flags:larray i32 (survival_length p (I64.v token + 1)){
    is_global_array input_flags})
  (#before:erased (physical_state p (I64.v token + 1)))
  (output_counts:larray i64 (Shared.expert_count p){is_global_array output_counts})
  (output_indices:larray i64 (slot_length p){is_global_array output_indices})
  (output_weights:larray f32 (slot_length p){is_global_array output_weights})
  (output_flags:larray i32 (survival_length p (I64.v token + 1)){
    is_global_array output_flags})
  (output_demand:larray i32 (demand_length (I64.v token)){
    is_global_array output_demand})
  preserves cpu **
    on gpu_loc (
      input_selected |-> selected **
      input_row |-> row **
      input_counts |-> (G.reveal before).counts **
      input_indices |-> (G.reveal before).indices **
      input_weights |-> (G.reveal before).weights **
      input_flags |-> (G.reveal before).flags)
  requires pure (input_representation p x (I64.v token) selected row before)
  requires
    exists* (old:physical_state p (I64.v token + 1))
      (old_demand:lseq i32 (demand_length (I64.v token))).
    on gpu_loc (
      output_counts |-> old.counts **
      output_indices |-> old.indices **
      output_weights |-> old.weights **
      output_flags |-> old.flags **
      output_demand |-> old_demand)
  ensures
    exists* (after:physical_state p (I64.v token + 1))
      (demand:lseq i32 (demand_length (I64.v token))).
    on gpu_loc (
      output_counts |-> after.counts **
      output_indices |-> after.indices **
      output_weights |-> after.weights **
      output_flags |-> after.flags **
      output_demand |-> demand) **
    pure (semantic_post p x (I64.v token) row before after demand)
{
  with old old_demand.
    assert on gpu_loc (
      output_counts |-> old.counts **
      output_indices |-> old.indices **
      output_weights |-> old.weights **
      output_flags |-> old.flags **
      output_demand |-> old_demand);
  launch_kernel_1 (fun () ->
    routing_kernel #p #x #selected #row experts capacity
      input_selected input_row input_counts input_indices input_weights
      token input_flags #before #old #old_demand
      output_counts output_indices output_weights output_flags output_demand);
}
