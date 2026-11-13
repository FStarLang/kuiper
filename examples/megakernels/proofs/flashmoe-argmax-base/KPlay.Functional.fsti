module KPlay.Functional

#lang-pulse

open Kuiper

module Shared = KPlay.SharedSpec
module Seq = FStar.Seq
module FP = Kuiper.Floating
module A = Kuiper.Approximates.Base
module I32 = FStar.Int32
module I64 = FStar.Int64
module U64 = FStar.UInt64
module KA = Kuiper.Array
module PA = Pulse.Lib.Array

inline_for_extraction noextract
unfold let row_values (e:pos) : Type0 =
  r:lseq f32 e {
    forall (j:Shared.index e).
      FP.kind #f32 (Seq.index r j) == FP.Finite /\
      FP.lte #f32 FP.zero (Seq.index r j) /\
      FP.lte #f32 (Seq.index r j) FP.one
  }

inline_for_extraction noextract
unfold let selected_values (e:pos) : Type0 =
  v:lseq i32 1 {
    0 <= I32.v (Seq.index v 0) /\ I32.v (Seq.index v 0) < e
  }

inline_for_extraction noextract
noeq type routing_state (e:pos) (slots history:nat) = {
  counts: lseq i64 e;
  indices: lseq i64 slots;
  probabilities: lseq f32 slots;
  survival: lseq i32 history
}

inline_for_extraction noextract
let same_state (#e:pos) (#slots #history:nat)
  (before after:routing_state e slots history) : prop =
  after.counts == before.counts /\
  after.indices == before.indices /\
  after.probabilities == before.probabilities /\
  after.survival == before.survival

inline_for_extraction noextract
let logical_input (#e:pos) (row:row_values e)
  : Shared.probabilities 1 e =
  fun _ j -> Seq.index row j

(* Routing buffers are frame state, not extra shared selection state. *)
inline_for_extraction noextract
let logical_state (#e:pos) (#slots #history:nat)
  (row:row_values e) (_frame:routing_state e slots history)
  : Shared.probabilities 1 e =
  logical_input row

inline_for_extraction noextract
let selected_index (#e:pos) (selected:selected_values e)
  : Shared.index e =
  I32.v (Seq.index selected 0)

(* The physical row and index jointly represent the logical gather output. *)
inline_for_extraction noextract
let logical_output (#e:pos)
  (selected:selected_values e) (copied_row:row_values e)
  : Shared.outputs 1 e = {
  selected_expert = (fun _ -> selected_index selected);
  selected_probability =
    (fun _ -> A.to_real #f32 (Seq.index copied_row (selected_index selected)))
}

inline_for_extraction noextract
let semantic_post (#e:pos) (#slots #history:nat)
  (before:row_values e) (state_before:routing_state e slots history)
  (selected:selected_values e)
  (after:row_values e) (state_after:routing_state e slots history) : prop =
  after == before /\
  same_state state_before state_after /\
  Shared.post
    (logical_state before state_before)
    (logical_state after state_after)
    (logical_output selected after) /\
  A.v_approximates #f32
    (Seq.index after (selected_index selected))
    ((logical_output selected after).selected_probability 0)

inline_for_extraction noextract
let stable_maximum_unique (#e:pos) (row:row_values e) (chosen:Shared.index e)
  : Lemma
      (requires Shared.stable_maximum (logical_input row 0) e chosen)
      (ensures chosen == Shared.top1_index (logical_input row 0))
  = let p = logical_input row 0 in
    Shared.best_prefix_correct p e;
    let best = Shared.top1_index p in
    Shared.comparison (p chosen) (p best)

inline_for_extraction noextract
let copied_selection_refines (#e:pos) (#slots #history:nat)
  (row:row_values e) (state:routing_state e slots history)
  (selected:selected_values e)
  : Lemma
      (requires
        Shared.stable_maximum
          (logical_input row 0) e (selected_index selected))
      (ensures semantic_post row state selected row state)
  = stable_maximum_unique row (selected_index selected);
    A.to_real_ok #f32 (Seq.index row (selected_index selected))

inline_for_extraction noextract
let semantic_post_is_stable (#e:pos) (#slots #history:nat)
  (before:row_values e) (state_before:routing_state e slots history)
  (selected:selected_values e)
  (after:row_values e) (state_after:routing_state e slots history)
  : Lemma
      (requires semantic_post before state_before selected after state_after)
      (ensures
        Shared.stable_maximum
          (logical_input before 0) e (selected_index selected) /\
        A.v_approximates #f32
          (Seq.index after (selected_index selected))
          (A.to_real #f32
            (Seq.index before (Shared.top1_index (logical_input before 0)))))
  = Shared.best_prefix_correct (logical_input before 0) e

inline_for_extraction noextract
let batch_input (#s:nat) (#e:pos)
  (rows:Shared.index s -> row_values e) : Shared.probabilities s e =
  fun t j -> Seq.index (rows t) j

inline_for_extraction noextract
let batch_output (#s:nat) (#e:pos)
  (selected:Shared.index s -> selected_values e)
  (rows:Shared.index s -> row_values e) : Shared.outputs s e = {
  selected_expert = (fun t -> selected_index (selected t));
  selected_probability =
    (fun t -> A.to_real #f32 (Seq.index (rows t) (selected_index (selected t))))
}

(* The base ABI is per-row. This lifting also covers an empty batch, with
   no row calls, and permits the survival extent to vary between calls. *)
inline_for_extraction noextract
let batch_refines_shared (#s:nat) (#e:pos) (#slots:nat)
  (history:Shared.index s -> nat)
  (before after:Shared.index s -> row_values e)
  (state_before state_after:
    (t:Shared.index s -> routing_state e slots (history t)))
  (selected:Shared.index s -> selected_values e)
  : Lemma
      (requires
        forall (t:Shared.index s).
          semantic_post (before t) (state_before t)
            (selected t) (after t) (state_after t))
      (ensures
        Shared.post (batch_input before) (batch_input after)
          (batch_output selected after))
  = let cells (t:Shared.index s)
      : Lemma (
          forall (j:Shared.index e).
            batch_input after t j == batch_input before t j)
      = assert (semantic_post (before t) (state_before t)
          (selected t) (after t) (state_after t));
        assert (after t == before t)
    in
    let outputs (t:Shared.index s)
      : Lemma (
          (batch_output selected after).selected_expert t ==
            Shared.top1_index (batch_input before t) /\
          (batch_output selected after).selected_probability t ==
            A.to_real #f32
              (batch_input before t (Shared.top1_index (batch_input before t))))
      = assert (semantic_post (before t) (state_before t)
          (selected t) (after t) (state_after t));
        assert (Shared.post
          (logical_input (before t)) (logical_input (after t))
          (logical_output (selected t) (after t)));
        assert_norm (batch_input before t == logical_input (before t) 0)
    in
    FStar.Classical.forall_intro cells;
    FStar.Classical.forall_intro outputs

(* A logical row need not fit one physical index. A finite chain of calls
   represents its index by local choices, not by truncating the global index. *)
inline_for_extraction noextract
noeq type call_observation (width:pos) = {
  before_row: row_values width;
  after_row: row_values width;
  chosen: selected_values width;
  before_frame: routing_state width width width;
  after_frame: routing_state width width width
}

inline_for_extraction noextract
let call_post (#width:pos) (c:call_observation width) : prop =
  semantic_post c.before_row c.before_frame c.chosen c.after_row c.after_frame

inline_for_extraction noextract
let gathered (#width:pos) (c:call_observation width) : Shared.probability =
  Seq.index c.after_row (selected_index c.chosen)

inline_for_extraction noextract
noeq type selection_chain : pos -> Type0 =
  | One : call_observation 1 -> selection_chain 1
  | Next : #n:pos -> selection_chain n -> call_observation 2 -> selection_chain (n + 1)

inline_for_extraction noextract
let rec chain_input (#e:pos) (c:selection_chain e)
  : Tot (Shared.row e) (decreases e) =
  match c with
  | One leaf -> fun _ -> Seq.index leaf.before_row 0
  | Next #n previous step ->
      fun j -> if j < n then chain_input previous j else Seq.index step.before_row 1

inline_for_extraction noextract
let rec chain_state (#e:pos) (c:selection_chain e)
  : Tot (Shared.row e) (decreases e) =
  match c with
  | One leaf -> fun _ -> Seq.index leaf.after_row 0
  | Next #n previous step ->
      fun j -> if j < n then chain_state previous j else Seq.index step.after_row 1

inline_for_extraction noextract
let rec chain_index (#e:pos) (c:selection_chain e)
  : Tot (Shared.index e) (decreases e) =
  match c with
  | One _ -> 0
  | Next #n previous step ->
      if selected_index step.chosen = 0 then chain_index previous else n

inline_for_extraction noextract
let chain_value (#e:pos) (c:selection_chain e) : Shared.probability =
  match c with
  | One leaf -> gathered leaf
  | Next _ step -> gathered step

inline_for_extraction noextract
let rec valid_chain (#e:pos) (c:selection_chain e) : Tot prop (decreases e) =
  match c with
  | One leaf -> call_post leaf
  | Next previous step ->
      valid_chain previous /\ call_post step /\
      Seq.index step.before_row 0 == chain_value previous

inline_for_extraction noextract
let rec prefix_extensional (#e1 #e2:pos)
  (p:Shared.row e1) (q:Shared.row e2) (n:pos{n <= e1 /\ n <= e2})
  : Lemma
      (requires forall (j:Shared.index n). p j == q j)
      (ensures Shared.best_prefix p n == Shared.best_prefix q n)
      (decreases n)
  = if n > 1 then prefix_extensional p q (n - 1)

inline_for_extraction noextract
let pair_choice (c:call_observation 2)
  : Lemma
      (requires call_post c)
      (ensures selected_index c.chosen ==
        (if FP.lt #f32 (Seq.index c.before_row 0) (Seq.index c.before_row 1)
         then 1 else 0))
  = ()

inline_for_extraction noextract
let rec chain_correct (#e:pos) (c:selection_chain e)
  : Lemma
      (requires valid_chain c)
      (ensures
        (forall (j:Shared.index e). chain_state c j == chain_input c j) /\
        chain_index c == Shared.top1_index (chain_input c) /\
        chain_value c == chain_input c (chain_index c))
      (decreases e)
  = match c with
    | One leaf -> ()
    | Next #n previous step ->
        chain_correct previous;
        pair_choice step;
        prefix_extensional (chain_input c) (chain_input previous) n;
        let preserves_cell (j:Shared.index e)
          : Lemma (chain_state c j == chain_input c j)
          = if j < n then () else assert (j == n)
        in
        FStar.Classical.forall_intro preserves_cell

inline_for_extraction noextract
let full_input (#e:pos) (c:selection_chain e) : Shared.probabilities 1 e =
  fun _ -> chain_input c

inline_for_extraction noextract
let full_state (#e:pos) (c:selection_chain e) : Shared.probabilities 1 e =
  fun _ -> chain_state c

inline_for_extraction noextract
let full_output (#e:pos) (c:selection_chain e) : Shared.outputs 1 e = {
  selected_expert = (fun _ -> chain_index c);
  selected_probability = (fun _ -> A.to_real #f32 (chain_value c))
}

inline_for_extraction noextract
let chain_refines_shared (#e:pos) (c:selection_chain e)
  : Lemma
      (requires valid_chain c)
      (ensures
        Shared.post (full_input c) (full_state c) (full_output c) /\
        A.v_approximates #f32 (chain_value c)
          ((full_output c).selected_probability 0))
  = chain_correct c;
    A.to_real_ok #f32 (chain_value c)

inline_for_extraction noextract
let frame_for (#width:pos) (row:row_values width)
  : routing_state width width width = {
  counts = Seq.create width 0L;
  indices = Seq.create width 0L;
  probabilities = row;
  survival = Seq.create width 0l
}

(* A constructive consistency/coverage witness for a local transition,
   not an implementation of kplay_run or a premise about its result. *)
inline_for_extraction noextract
let choose_small (width:pos{width = 1 \/ width = 2}) (row:row_values width)
  : c:call_observation width {call_post c /\ c.before_row == row}
  = let best = Shared.top1_index (logical_input row 0) in
    let chosen:selected_values width = Seq.create 1 (if best = 0 then 0l else 1l) in
    Shared.best_prefix_correct (logical_input row 0) width;
    copied_selection_refines row (frame_for row) chosen;
    {
      before_row = row;
      after_row = row;
      chosen = chosen;
      before_frame = frame_for row;
      after_frame = frame_for row
    }

inline_for_extraction noextract
let rec cover_prefix (#e:pos) (p:Shared.row e) (n:pos{n <= e})
  : Tot (c:selection_chain n {
      valid_chain c /\
      (forall (j:Shared.index n). chain_input c j == p j)
    }) (decreases n)
  = if n = 1 then One (choose_small 1 (Seq.create #f32 1 (p 0)))
    else
      let previous = cover_prefix p (n - 1) in
      let pair:row_values 2 =
        Seq.init #f32 2 (fun j -> if j = 0 then chain_value previous else p (n - 1))
      in
      let step = choose_small 2 pair in
      Next previous step

inline_for_extraction noextract
let cover_all_experts (#e:pos) (p:Shared.row e)
  : c:selection_chain e {
      valid_chain c /\
      (forall (j:Shared.index e). chain_input c j == p j) /\
      Shared.post (fun _ -> p) (full_state c) (full_output c)
    }
  = let c = cover_prefix p e in
    chain_refines_shared c;
    prefix_extensional (chain_input c) p e;
    c

inline_for_extraction noextract
let small_call_dimensions (width:pos{width = 1 \/ width = 2})
  : Lemma (
      width <= 2147483648 /\
      Kuiper.SizeT.fits width /\
      width % width == 0 /\
      width == (I64.v 0L + 1) * width)
  = ()

inline_for_extraction noextract
let rec chain_call_count (#e:pos) (c:selection_chain e)
  : Tot (n:pos{n == e}) (decreases e) =
  match c with
  | One _ -> 1
  | Next previous _ -> chain_call_count previous + 1

inline_for_extraction noextract
let batch_chain_input (#s:nat) (#e:pos)
  (calls:Shared.index s -> selection_chain e) : Shared.probabilities s e =
  fun t -> chain_input (calls t)

inline_for_extraction noextract
let batch_chain_state (#s:nat) (#e:pos)
  (calls:Shared.index s -> selection_chain e) : Shared.probabilities s e =
  fun t -> chain_state (calls t)

inline_for_extraction noextract
let batch_chain_output (#s:nat) (#e:pos)
  (calls:Shared.index s -> selection_chain e) : Shared.outputs s e = {
  selected_expert = (fun t -> chain_index (calls t));
  selected_probability = (fun t -> A.to_real #f32 (chain_value (calls t)))
}

inline_for_extraction noextract
let batch_chains_refine_shared (#s:nat) (#e:pos)
  (calls:Shared.index s -> selection_chain e)
  : Lemma
      (requires forall (t:Shared.index s). valid_chain (calls t))
      (ensures
        Shared.post (batch_chain_input calls) (batch_chain_state calls)
          (batch_chain_output calls))
  = let row_post (t:Shared.index s)
      : Lemma (
          (forall (j:Shared.index e).
            batch_chain_state calls t j == batch_chain_input calls t j) /\
          (batch_chain_output calls).selected_expert t ==
            Shared.top1_index (batch_chain_input calls t) /\
          (batch_chain_output calls).selected_probability t ==
            A.to_real #f32
              (batch_chain_input calls t (Shared.top1_index (batch_chain_input calls t))))
      = chain_correct (calls t)
    in
    FStar.Classical.forall_intro row_post

inline_for_extraction noextract
let cover_all_batches (#s:nat) (#e:pos) (p:Shared.probabilities s e)
  : calls:(Shared.index s -> selection_chain e) {
      (forall (t:Shared.index s).
        valid_chain (calls t) /\
        (forall (j:Shared.index e). chain_input (calls t) j == p t j)) /\
      Shared.post p (batch_chain_state calls) (batch_chain_output calls)
    }
  = let calls = (fun (t:Shared.index s) -> cover_all_experts (p t)) in
    let covers_token (t:Shared.index s)
      : Lemma (
          (forall (j:Shared.index e). chain_state (calls t) j == p t j) /\
          chain_index (calls t) == Shared.top1_index (p t) /\
          A.to_real #f32 (chain_value (calls t)) ==
            A.to_real #f32 (p t (Shared.top1_index (p t))))
      = assert (Shared.post (fun _ -> p t)
          (full_state (calls t)) (full_output (calls t)))
    in
    FStar.Classical.forall_intro covers_token;
    calls

inline_for_extraction noextract
unfold let device_array (a:Type0) (n:nat) : Type0 =
  ar:larray a n { KA.is_global_array ar }

fn kplay_run
  (experts:u64 { 0 < U64.v experts /\ U64.v experts <= 2147483648 })
  (slots:u64 {
    U64.v slots < 4294967296 /\ U64.v slots % U64.v experts == 0
  })
  (history:u64 { U64.v history < 4294967296 })
  (token:i64 {
    0 <= I64.v token /\
    U64.v history == (I64.v token + 1) * U64.v experts
  })
  (#row0:erased (row_values (U64.v experts)))
  (#state0:erased
    (routing_state (U64.v experts) (U64.v slots) (U64.v history)))
  (input_row:device_array f32 (U64.v experts))
  (input_counts:device_array i64 (U64.v experts))
  (input_indices:device_array i64 (U64.v slots))
  (input_probabilities:device_array f32 (U64.v slots))
  (input_survival:device_array i32 (U64.v history))
  (output_selected:device_array i32 1)
  (output_row:device_array f32 (U64.v experts))
  (output_counts:device_array i64 (U64.v experts))
  (output_indices:device_array i64 (U64.v slots))
  (output_probabilities:device_array f32 (U64.v slots))
  (output_survival:device_array i32 (U64.v history))
  requires cpu ** on gpu_loc (
    PA.pts_to input_row row0 **
    PA.pts_to input_counts state0.counts **
    PA.pts_to input_indices state0.indices **
    PA.pts_to input_probabilities state0.probabilities **
    PA.pts_to input_survival state0.survival **
    PA.pts_to output_selected 'old_selected **
    PA.pts_to output_row 'old_row **
    PA.pts_to output_counts 'old_counts **
    PA.pts_to output_indices 'old_indices **
    PA.pts_to output_probabilities 'old_probabilities **
    PA.pts_to output_survival 'old_survival)
  ensures exists* (selected:selected_values (U64.v experts)).
    cpu ** on gpu_loc (
      PA.pts_to input_row row0 **
      PA.pts_to input_counts state0.counts **
      PA.pts_to input_indices state0.indices **
      PA.pts_to input_probabilities state0.probabilities **
      PA.pts_to input_survival state0.survival **
      PA.pts_to output_selected selected **
      PA.pts_to output_row row0 **
      PA.pts_to output_counts state0.counts **
      PA.pts_to output_indices state0.indices **
      PA.pts_to output_probabilities state0.probabilities **
      PA.pts_to output_survival state0.survival) **
    pure (semantic_post row0 state0 selected row0 state0)

(* Every node of the coverage representation is an ordinary, well-typed
   physical call with capacity one and token zero. No new primitive is used. *)
inline_for_extraction noextract
fn run_small_call
  (width:u64 { U64.v width == 1 \/ U64.v width == 2 })
  (#row0:erased (row_values (U64.v width)))
  (#state0:erased
    (routing_state (U64.v width) (U64.v width) (U64.v width)))
  (input_row:device_array f32 (U64.v width))
  (input_counts:device_array i64 (U64.v width))
  (input_indices:device_array i64 (U64.v width))
  (input_probabilities:device_array f32 (U64.v width))
  (input_survival:device_array i32 (U64.v width))
  (output_selected:device_array i32 1)
  (output_row:device_array f32 (U64.v width))
  (output_counts:device_array i64 (U64.v width))
  (output_indices:device_array i64 (U64.v width))
  (output_probabilities:device_array f32 (U64.v width))
  (output_survival:device_array i32 (U64.v width))
  requires cpu ** on gpu_loc (
    PA.pts_to input_row row0 **
    PA.pts_to input_counts state0.counts **
    PA.pts_to input_indices state0.indices **
    PA.pts_to input_probabilities state0.probabilities **
    PA.pts_to input_survival state0.survival **
    PA.pts_to output_selected 'old_selected **
    PA.pts_to output_row 'old_row **
    PA.pts_to output_counts 'old_counts **
    PA.pts_to output_indices 'old_indices **
    PA.pts_to output_probabilities 'old_probabilities **
    PA.pts_to output_survival 'old_survival)
  ensures exists* (selected:selected_values (U64.v width)).
    cpu ** on gpu_loc (
      PA.pts_to input_row row0 **
      PA.pts_to input_counts state0.counts **
      PA.pts_to input_indices state0.indices **
      PA.pts_to input_probabilities state0.probabilities **
      PA.pts_to input_survival state0.survival **
      PA.pts_to output_selected selected **
      PA.pts_to output_row row0 **
      PA.pts_to output_counts state0.counts **
      PA.pts_to output_indices state0.indices **
      PA.pts_to output_probabilities state0.probabilities **
      PA.pts_to output_survival state0.survival) **
    pure (semantic_post row0 state0 selected row0 state0)
{
  small_call_dimensions (U64.v width);
  kplay_run width width width 0L #row0 #state0
    input_row input_counts input_indices input_probabilities input_survival
    output_selected output_row output_counts output_indices
    output_probabilities output_survival;
}
