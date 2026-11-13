module KPlay.Functional

#lang-pulse

open Kuiper
module Shared = KPlay.SharedSpec
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base
module I32 = FStar.Int32
module U64 = FStar.UInt64
module SZ = Kuiper.SizeT
module S = FStar.Seq
module PA = Pulse.Lib.Array

inline_for_extraction noextract
let negative_infinity : f32 = F.neg F.infinity

inline_for_extraction noextract
let admissible_dimensions (tokens experts:nat) : prop =
  0 < tokens /\ 0 < experts /\ experts * tokens <= 2147483647

inline_for_extraction noextract
unfold let probability_values (n:nat) : Type0 =
  values:lseq f32 n {
    forall (j:Shared.index n).
      F.kind #f32 (S.index values j) == F.Finite /\
      F.lte #f32 F.zero (S.index values j) /\
      F.lte #f32 (S.index values j) F.one
  }

inline_for_extraction noextract
unfold let selected_values (tokens:nat) (experts:pos) : Type0 =
  values:lseq i32 tokens {
    forall (t:Shared.index tokens).
      0 <= I32.v (S.index values t) /\ I32.v (S.index values t) < experts
  }

inline_for_extraction noextract
let row_input (#e:pos) (row:probability_values e) : Shared.row e =
  fun j -> S.index row j

inline_for_extraction noextract
let row_at (tokens:nat) (experts:pos)
  (rows:probability_values (experts * tokens)) (token:Shared.index tokens)
  : probability_values experts =
  let row = S.slice rows (experts * token) (experts * token + experts) in
  let cell (j:Shared.index experts)
    : Lemma (
        F.kind #f32 (S.index row j) == F.Finite /\
        F.lte #f32 F.zero (S.index row j) /\
        F.lte #f32 (S.index row j) F.one)
    = S.lemma_index_slice rows (experts * token)
        (experts * token + experts) j
  in
  FStar.Classical.forall_intro cell;
  row

inline_for_extraction noextract
let logical_input (#s:nat) (#e:pos)
  (rows:probability_values (e * s)) : Shared.probabilities s e =
  fun t j -> S.index rows (e * t + j)

inline_for_extraction noextract
let selected_index (#s:nat) (#e:pos)
  (selected:selected_values s e) (t:Shared.index s) : Shared.index e =
  I32.v (S.index selected t)

inline_for_extraction noextract
let logical_output (#s:nat) (#e:pos)
  (selected:selected_values s e) (best:probability_values s)
  : Shared.outputs s e = {
  selected_expert = (fun t -> selected_index selected t);
  selected_probability = (fun t -> A.to_real #f32 (S.index best t))
}

inline_for_extraction noextract
let semantic_post (#s:nat) (#e:pos)
  (before after:probability_values (e * s))
  (selected:selected_values s e) (best:probability_values s) : prop =
  after == before /\
  (forall (t:Shared.index s).
    S.index best t == S.index before (e * t + selected_index selected t)) /\
  Shared.post (logical_input before) (logical_input after)
    (logical_output selected best) /\
  (forall (t:Shared.index s).
    A.v_approximates #f32 (S.index best t)
      ((logical_output selected best).selected_probability t))

(* Retained scan invariant: representation identity is stronger than a tie. *)
inline_for_extraction noextract
let candidate (x:f32) : prop =
  not (F.is_nan x) /\ F.lt negative_infinity x

inline_for_extraction noextract
let row_result (#e:pos) (row:lseq f32 e) (selected:i32) (best:f32) : prop =
  (I32.v selected == 0 /\
   F.bit_eq best negative_infinity /\
   (forall (j:Shared.index e). not (candidate (S.index row j))))
  \/
  (exists (chosen:Shared.index e).
    I32.v selected == chosen /\
    candidate (S.index row chosen) /\
    F.bit_eq best (S.index row chosen) /\
    (forall (j:Shared.index e).
      candidate (S.index row j) ==>
      (not (F.lt (S.index row chosen) (S.index row j)) /\
       (F.ieee_eq (S.index row j) (S.index row chosen) ==> chosen <= j))))

inline_for_extraction noextract
let argmax_post (tokens:nat) (experts:pos)
  (rows:probability_values (experts * tokens))
  (selected:lseq i32 tokens) (best:lseq f32 tokens) : prop =
  forall (t:Shared.index tokens).
    row_result (row_at tokens experts rows t) (S.index selected t) (S.index best t)

inline_for_extraction noextract
let stable_maximum_unique (#e:pos) (p:Shared.row e) (chosen:Shared.index e)
  : Lemma
      (requires Shared.stable_maximum p e chosen)
      (ensures chosen == Shared.top1_index p)
  = Shared.best_prefix_correct p e;
    Shared.comparison (p chosen) (p (Shared.top1_index p))

(* Observations contain the returned best buffer, not a recomputed gather. *)
inline_for_extraction noextract
noeq type call_observation (width:pos) = {
  before_row: probability_values width;
  after_row: probability_values width;
  chosen: selected_values 1 width;
  best: probability_values 1
}

inline_for_extraction noextract
let call_post (#width:pos) (c:call_observation width) : prop =
  semantic_post #1 #width c.before_row c.after_row c.chosen c.best

inline_for_extraction noextract
let observed_value (#width:pos) (c:call_observation width) : Shared.probability =
  S.index c.best 0

inline_for_extraction noextract
noeq type selection_chain : pos -> Type0 =
  | One : call_observation 1 -> selection_chain 1
  | Next : #n:pos -> selection_chain n -> call_observation 2 -> selection_chain (n + 1)

inline_for_extraction noextract
let rec chain_input (#e:pos) (c:selection_chain e)
  : Tot (Shared.row e) (decreases e) =
  match c with
  | One leaf -> fun _ -> S.index leaf.before_row 0
  | Next #n previous step ->
      fun j -> if j < n then chain_input previous j else S.index step.before_row 1

inline_for_extraction noextract
let rec chain_state (#e:pos) (c:selection_chain e)
  : Tot (Shared.row e) (decreases e) =
  match c with
  | One leaf -> fun _ -> S.index leaf.after_row 0
  | Next #n previous step ->
      fun j -> if j < n then chain_state previous j else S.index step.after_row 1

inline_for_extraction noextract
let rec chain_index (#e:pos) (c:selection_chain e)
  : Tot (Shared.index e) (decreases e) =
  match c with
  | One _ -> 0
  | Next #n previous step ->
      if selected_index step.chosen 0 = 0 then chain_index previous else n

inline_for_extraction noextract
let chain_value (#e:pos) (c:selection_chain e) : Shared.probability =
  match c with
  | One leaf -> observed_value leaf
  | Next _ step -> observed_value step

inline_for_extraction noextract
let rec valid_chain (#e:pos) (c:selection_chain e) : Tot prop (decreases e) =
  match c with
  | One leaf -> call_post leaf
  | Next previous step ->
      valid_chain previous /\ call_post step /\
      S.index step.before_row 0 == chain_value previous

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
      (ensures selected_index c.chosen 0 ==
        (if F.lt #f32 (S.index c.before_row 0) (S.index c.before_row 1)
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
        let cell (j:Shared.index e)
          : Lemma (chain_state c j == chain_input c j)
          = if j < n then () else assert (j == n)
        in
        FStar.Classical.forall_intro cell

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

(* A request consumes a checked observation of the same entrypoint.
   No coverage constructor supplies an index or a winning-value output. *)
inline_for_extraction noextract
noeq type selection_requests (e:pos) (p:Shared.row e) : nat -> Type0 =
  | Finished :
      c:selection_chain e {
        valid_chain c /\
        (forall (j:Shared.index e). chain_input c j == p j) /\
        Shared.post (fun _ -> p) (full_state c) (full_output c)
      } -> selection_requests e p 0
  | Request :
      #remaining:nat ->
      width:pos{width = 1 \/ width = 2} ->
      row:probability_values width ->
      (c:call_observation width {call_post c /\ c.before_row == row} ->
        Tot (selection_requests e p remaining)) ->
      selection_requests e p (remaining + 1)

inline_for_extraction noextract
let rec request_suffix (#e:pos) (p:Shared.row e) (#n:pos{n <= e})
  (previous:selection_chain n {
    valid_chain previous /\
    (forall (j:Shared.index n). chain_input previous j == p j)
  })
  : Tot (selection_requests e p (e - n)) (decreases (e - n)) =
  if n = e then (
    chain_refines_shared previous;
    prefix_extensional (chain_input previous) p e;
    Finished previous
  ) else
    let pair:probability_values 2 =
      S.init #f32 2 (fun j -> if j = 0 then chain_value previous else p n)
    in
    Request 2 pair (fun observed -> request_suffix p (Next previous observed))

inline_for_extraction noextract
let cover_all_experts (#e:pos) (p:Shared.row e) : selection_requests e p e =
  Request 1 (S.create 1 (p 0))
    (fun observed -> request_suffix p (One observed))

inline_for_extraction noextract
let cover_all_batches (#s:nat) (#e:pos) (p:Shared.probabilities s e)
  : (t:Shared.index s -> selection_requests e (p t) e) =
  fun t -> cover_all_experts (p t)

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
  (p:Shared.probabilities s e) (calls:Shared.index s -> selection_chain e)
  : Lemma
      (requires forall (t:Shared.index s).
        valid_chain (calls t) /\
        (forall (j:Shared.index e). chain_input (calls t) j == p t j))
      (ensures
        Shared.post p (batch_chain_state calls) (batch_chain_output calls) /\
        (forall (t:Shared.index s).
          A.v_approximates #f32 (chain_value (calls t))
            ((batch_chain_output calls).selected_probability t)))
  = let row_post (t:Shared.index s)
      : Lemma (
          (forall (j:Shared.index e). chain_state (calls t) j == p t j) /\
          chain_index (calls t) == Shared.top1_index (p t) /\
          chain_value (calls t) == p t (Shared.top1_index (p t)) /\
          A.v_approximates #f32 (chain_value (calls t))
            (A.to_real #f32 (chain_value (calls t))))
      = chain_correct (calls t);
        prefix_extensional (chain_input (calls t)) (p t) e;
        A.to_real_ok #f32 (chain_value (calls t))
    in
    FStar.Classical.forall_intro row_post

inline_for_extraction noextract
unfold let device_array (a:Type0) (n:nat) : Type0 =
  ar:larray a n { is_global_array ar }

fn kplay_run
  (tokens:u64{0 < U64.v tokens /\ U64.v tokens <= 2147483647})
  (experts:u64{admissible_dimensions (U64.v tokens) (U64.v experts)})
  (#rows:erased (probability_values (U64.v experts * U64.v tokens)))
  (normalized_rows:device_array f32 (U64.v experts * U64.v tokens))
  (selected_experts:device_array i32 (U64.v tokens))
  (best_values:device_array f32 (U64.v tokens))
  requires cpu ** on gpu_loc (
    PA.pts_to normalized_rows rows **
    PA.pts_to selected_experts 'selected_before **
    PA.pts_to best_values 'best_before)
  ensures exists*
    (selected:selected_values (U64.v tokens) (U64.v experts))
    (best:probability_values (U64.v tokens)).
    cpu ** on gpu_loc (
      PA.pts_to normalized_rows rows **
      PA.pts_to selected_experts selected **
      PA.pts_to best_values best) **
    pure (semantic_post rows rows selected best)

inline_for_extraction noextract
let small_call_dimensions (width:pos{width = 1 \/ width = 2})
  : Lemma (admissible_dimensions 1 width /\ SZ.fits width)
  = ()

inline_for_extraction noextract
fn run_small_call
  (width:u64{U64.v width == 1 \/ U64.v width == 2})
  (#row:erased (probability_values (U64.v width)))
  (input:device_array f32 (U64.v width))
  (selected:device_array i32 1)
  (best:device_array f32 1)
  requires cpu ** on gpu_loc (
    PA.pts_to input row **
    PA.pts_to selected 'old_selected **
    PA.pts_to best 'old_best)
  returns observed:erased (call_observation (U64.v width))
  ensures cpu ** on gpu_loc (
    PA.pts_to input observed.after_row **
    PA.pts_to selected observed.chosen **
    PA.pts_to best observed.best) **
    pure (call_post observed /\ observed.before_row == row)
{
  small_call_dimensions (U64.v width);
  kplay_run 1UL width
    #(hide (reveal row <: probability_values (U64.v width * 1)))
    input selected best;
  with chosen. assert on gpu_loc (PA.pts_to selected chosen);
  with values. assert on gpu_loc (PA.pts_to best values);
  let observed:erased (call_observation (U64.v width)) = hide {
    before_row = row;
    after_row = row;
    chosen = chosen;
    best = values
  };
  observed
}

(* The continuation can only be advanced with the actual owned outputs.
   The mathematical remaining-call count decreases by exactly one. *)
inline_for_extraction noextract
fn fulfill_request
  (#e:erased pos) (#p:erased (Shared.row e)) (#remaining:erased nat)
  (width:u64{U64.v width == 1 \/ U64.v width == 2})
  (#row:erased (probability_values (U64.v width)))
  (#resume:erased (
    c:call_observation (U64.v width) {call_post c /\ c.before_row == row} ->
    Tot (selection_requests e p remaining)))
  (input:device_array f32 (U64.v width))
  (selected:device_array i32 1)
  (best:device_array f32 1)
  requires cpu ** on gpu_loc (
    PA.pts_to input row **
    PA.pts_to selected 'old_selected **
    PA.pts_to best 'old_best)
  returns next:erased (selection_requests e p remaining)
  ensures exists* (observed:call_observation (U64.v width)).
    cpu ** on gpu_loc (
      PA.pts_to input observed.after_row **
      PA.pts_to selected observed.chosen **
      PA.pts_to best observed.best) **
    pure (call_post observed /\ observed.before_row == row /\
      reveal next == (reveal resume) observed)
{
  let observed = run_small_call width #row input selected best;
  let next = hide ((reveal resume) (reveal observed));
  next
}
