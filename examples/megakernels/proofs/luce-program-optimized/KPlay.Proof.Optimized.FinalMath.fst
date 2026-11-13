module KPlay.Proof.Optimized.FinalMath

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module C = FStar.Classical
module I32 = FStar.Int32
module Approx = Kuiper.Approximates.Base
module Root = FStar.Math.Sqrt
module P = KPlay.Proof.Shared.Projection
module E = KPlay.Proof.Shared.Embedding
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Layout = KPlay.Proof.Optimized.Layout
module Embedding = KPlay.Proof.Optimized.Embedding
module Norm = KPlay.Proof.Optimized.Norm
module N = KPlay.Proof.Optimized.NormMath
module Sum = KPlay.Proof.Optimized.Sum
module Dot = KPlay.Proof.Optimized.DotMath
module Scores = KPlay.Proof.Optimized.Scores
module ScoreMath = KPlay.Proof.Optimized.ScoreMath

inline_for_extraction noextract
let weight_model (w:Layout.packed) : GTot (lseq real 752393024) =
  S.init_ghost 752393024 (fun i -> Approx.to_real (S.index w i))

let weight_model_ok (w:Layout.packed)
  : Lemma ((w <: lseq bf16 752393024) %~ weight_model w) =
  let at (i:nat{i < 752393024})
    : Lemma (S.index w i %~ S.index (weight_model w) i) =
    Approx.to_real_ok (S.index w i)
  in C.forall_intro at

inline_for_extraction noextract
let hidden_model (h:P.vector real 1024) : GTot (lseq real 1024) =
  S.init_ghost 1024 h

inline_for_extraction noextract
let normalized (w:Layout.packed) (h:P.vector real 1024) : GTot (P.vector real 1024) =
  P.normalized_for_logits P.reference_epsilon h
    (O.model_parameters (Layout.parameter_values w)).Model.final_norm

inline_for_extraction noextract
let squares (x:P.vector real 1024) : P.vector real 1024 = fun i -> x i *. x i

let rms_argument_form (x:P.vector real 1024)
  : Lemma (
      P.rms_argument P.reference_epsilon x ==
        Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 1024 (squares x) /. 1024.0R +. 0.000001R) =
  assert (
    P.rms_argument P.reference_epsilon x ==
      Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 1024 (squares x) /. 1024.0R +. 0.000001R)
  by (FStar.Tactics.V2.norm [delta_only [
        `%P.rms_argument; `%P.real_sum; `%P.reference_epsilon; `%squares]; zeta; iota];
      FStar.Tactics.V2.trefl ())

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let rms_argument_ext (x y:P.vector real 1024)
  : Lemma
      (requires forall (i:P.index 1024). x i == y i)
      (ensures P.rms_argument P.reference_epsilon x == P.rms_argument P.reference_epsilon y) =
  FStar.Pervasives.norm_spec [delta_only [`%squares]; zeta; iota] (squares x);
  FStar.Pervasives.norm_spec [delta_only [`%squares]; zeta; iota] (squares y);
  Sum.bounded_sum_ext #1024 (squares x) (squares y) 1024;
  rms_argument_form x;
  rms_argument_form y
#pop-options

let normalization_cell (w:Layout.packed) (h:P.vector real 1024) (i:P.index 1024)
  : Lemma (
      S.index (Norm.normalized (weight_model w, hidden_model h) 254279680) i ==
        normalized w h i) =
  let x = hidden_model h in
  N.scale_shared x;
  rms_argument_ext (N.source_view x) h

inline_for_extraction noextract
let logits (w:Layout.packed) (h:P.vector real 1024) : GTot (P.vector real 248320) =
  P.project (normalized w h)
    (fun token i -> S.index
      (S.index (O.model_parameters (Layout.parameter_values w)).Model.embedding token) i)

#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"
let embedding_model_cell (w:Layout.packed) (row:P.index 248320) (i:P.index 1024)
  : Lemma (
      S.index (S.index (O.model_parameters (Layout.parameter_values w)).Model.embedding row) i ==
        S.index (weight_model w) (row * 1024 + i)) =
  let actual = (Layout.parameter_values w).O.embedding in
  let modeled_row (t:E.token_id) : E.hidden_vector real =
    S.init #real 1024 (fun j -> Approx.to_real (S.index (S.index actual t) j)) in
  FStar.Pervasives.norm_spec [delta_only [`%O.model_parameters]; zeta; iota]
    (O.model_parameters (Layout.parameter_values w)).Model.embedding;
  S.init_index_ #(E.hidden_vector real) 248320 modeled_row row;
  S.init_index_ #real 1024 (fun j -> Approx.to_real (S.index (S.index actual row) j)) i;
  assert (S.index (S.index (O.model_parameters (Layout.parameter_values w)).Model.embedding row) i ==
    Approx.to_real (S.index (S.index actual row) i));
  Embedding.row_binding w row;
  assert (S.index actual row == Embedding.row w row);
  FStar.Pervasives.norm_spec [delta_only [`%Embedding.row]; zeta; iota] (Embedding.row w row);
  S.init_ghost_index_ #bf16 1024 (fun j -> S.index w (row * 1024 + j)) i;
  assert (S.index (S.index actual row) i == S.index w (row * 1024 + i));
  S.init_ghost_index_ #real 752393024 (fun j -> Approx.to_real (S.index w j)) (row * 1024 + i)
#pop-options

let projection_term (w:Layout.packed) (h:P.vector real 1024)
  (row:P.index 248320) (i:P.index 1024)
  : Lemma (
      Dot.term (weight_model w, Norm.normalized (weight_model w, hidden_model h) 254279680)
        (row * 1024) i ==
      normalized w h i *.
        S.index (S.index (O.model_parameters (Layout.parameter_values w)).Model.embedding row) i) =
  normalization_cell w h i;
  embedding_model_cell w row i

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
let projection_cell (w:Layout.packed) (h:P.vector real 1024) (row:P.index 248320)
  : Lemma (
      ScoreMath.score_model
        (weight_model w, Norm.normalized (weight_model w, hidden_model h) 254279680) row ==
        logits w h row) =
  let norm = Norm.normalized (weight_model w, hidden_model h) 254279680 in
  let term = Dot.term (weight_model w, norm) (row * 1024) in
  let target : P.vector real 1024 = fun i -> normalized w h i *.
    S.index (S.index (O.model_parameters (Layout.parameter_values w)).Model.embedding row) i in
  C.forall_intro (normalization_cell w h);
  C.forall_intro (projection_term w h row);
  Sum.prefix_init 1024 term;
  Sum.prefix_kuiper 1024 term;
  Sum.bounded_sum_ext #1024 term target 1024;
  FStar.Pervasives.norm_spec [delta_only [`%ScoreMath.score_model]; zeta; iota]
    (ScoreMath.score_model (weight_model w, norm) row);
  FStar.Pervasives.norm_spec [delta_only [`%logits; `%P.project; `%P.real_sum]; zeta; iota]
    (logits w h row);
  assert (ScoreMath.score_model (weight_model w, norm) row ==
    rsum (S.init_ghost 1024 term));
  assert (rsum (S.init_ghost 1024 term) == Sum.prefix 1024 term);
  assert (ScoreMath.score_model (weight_model w, norm) row ==
    Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 1024 term);
  assert (ScoreMath.score_model (weight_model w, norm) row ==
    Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 1024 target);
  assert (logits w h row ==
    Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 1024 target)
#pop-options

let propagated (w:Layout.packed) (hidden:lseq bf16 1024)
  (ideal:P.vector real 1024) (selected:I32.t) (observed:Scores.observation)
  : Lemma
      (requires Scores.post w hidden selected observed /\
        (forall (i:P.index 1024). S.index hidden i %~ ideal i))
      (ensures
        (forall (i:P.index 1024). Approx.v_approximates (S.index observed.normalized i) (normalized w ideal i)) /\
        (forall (i:P.index 248320). Approx.v_approximates (S.index observed.scores i) (logits w ideal i))) =
  weight_model_ok w;
  assert (Norm.inputs w hidden (weight_model w, hidden_model ideal));
  assert (observed.normalized %~ Norm.normalized (weight_model w, hidden_model ideal) 254279680);
  C.forall_intro (normalization_cell w ideal);
  assert (Dot.float_inputs w observed.normalized
    (weight_model w, Norm.normalized (weight_model w, hidden_model ideal) 254279680));
  C.forall_intro (projection_cell w ideal)
