module KPlay.Proof.Base.AttentionProjection

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module BF = Kuiper.BFloat16
module C = Kuiper.Chest
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Pipeline = KPlay.Proof.Base.AttentionPipeline
module Inputs = KPlay.Proof.Base.Inputs
module Native = KPlay.Proof.Base.NativeProjection
module Source = KPlay.Proof.Base.SourceProjection
module Schedule = KPlay.Proof.Base.Schedule
module Layout = KPlay.Proof.Base.MatvecLayout
module Matmul = KPlay.Proof.Base.Matmul
module Sum = KPlay.Proof.Base.ReductionMath

type weight_phase = p:nat{p < 3 \/ (19 <= p /\ p < 23)}

inline_for_extraction noextract
let rows (phase:weight_phase) : pos =
  if phase = 0 then 4096 else if phase = 1 || phase = 2 then 512
  else if phase = 20 || phase = 21 then 3584 else 1024

inline_for_extraction noextract
let columns (phase:weight_phase) : pos =
  if phase = 19 then 2048 else if phase = 22 then 3584 else 1024

inline_for_extraction noextract
let weight (w:A.layer_weights) (phase:weight_phase) : A.matrix (rows phase) (columns phase) =
  if phase = 0 then w.A.query_gate_proj else if phase = 1 then w.A.key_proj
  else if phase = 2 then w.A.value_proj else if phase = 19 then w.A.output_proj
  else if phase = 20 then w.A.mlp_gate_proj else if phase = 21 then w.A.mlp_up_proj else w.A.mlp_down_proj

inline_for_extraction noextract
let input (#position:nat) (stages:Pipeline.stages position) (phase:weight_phase) : A.vector (columns phase) =
  if phase < 3 then stages.Pipeline.normalized else if phase = 19 then stages.Pipeline.gated
  else if phase = 22 then stages.Pipeline.product else stages.Pipeline.postnorm

inline_for_extraction noextract
let output (#position:nat) (stages:Pipeline.stages position) (phase:weight_phase) : A.vector (rows phase) =
  if phase = 0 then stages.Pipeline.projected_qg else if phase = 1 then stages.Pipeline.projected_k
  else if phase = 2 then stages.Pipeline.projected_v else if phase = 19 then stages.Pipeline.mixed
  else if phase = 20 then stages.Pipeline.mlp_gate else if phase = 21 then stages.Pipeline.mlp_up else stages.Pipeline.down

inline_for_extraction noextract
let source_dimensions (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:weight_phase)
  : Lemma (
      (Schedule.projection_at call position layer phase).Schedule.rows == rows phase /\
      (Schedule.projection_at call position layer phase).Schedule.columns == columns phase) =
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then ()
  else if phase = 19 then () else if phase = 20 then () else if phase = 21 then () else assert (phase = 22)

inline_for_extraction noextract
let stage_equation (call:Pipeline.logical_call) (stages:Pipeline.stages call.Pipeline.call.A.position)
  (phase:weight_phase) (row:P.index (rows phase))
  : Lemma
      (requires stages == Pipeline.build call)
      (ensures output stages phase row ==
        A.linear (rows phase) (columns phase) (weight call.Pipeline.weight phase) (input stages phase) row) =
  let canonical = Pipeline.build call in
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then ()
  else if phase = 19 then () else if phase = 20 then () else if phase = 21 then () else assert (phase = 22)

inline_for_extraction noextract
let weight_cell (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:weight_phase) (values:C.chest2 BF.t 752393024 1)
  (row:P.index (rows phase)) (col:P.index (columns phase))
  : Lemma (
      weight (O.model_full ((Inputs.parameter_values (Native.packed_values values)).O.full layer)) phase row col ==
      Approx.to_real (Layout.matrix (Source.weight_geometry call position layer phase) values row col)) =
  source_dimensions call position layer phase;
  Inputs.layer_extent layer;
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then ()
  else if phase = 19 then () else if phase = 20 then () else if phase = 21 then () else assert (phase = 22)

inline_for_extraction noextract
let weights_approx (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:A.full_attention_layer) (phase:weight_phase) (values:C.chest2 BF.t 752393024 1)
  : Lemma (O.matrix_approximates (Layout.matrix (Source.weight_geometry call position layer phase) values)
      (weight (O.model_full ((Inputs.parameter_values (Native.packed_values values)).O.full layer)) phase)) =
  source_dimensions call position layer phase;
  let at_cell (row:P.index (rows phase)) (col:P.index (columns phase))
    : Lemma (Approx.v_approximates (Layout.matrix (Source.weight_geometry call position layer phase) values row col)
        (weight (O.model_full ((Inputs.parameter_values (Native.packed_values values)).O.full layer)) phase row col)) =
    weight_cell call position layer phase values row col;
    Approx.to_real_ok (Layout.matrix (Source.weight_geometry call position layer phase) values row col)
  in
  FStar.Classical.forall_intro_2 at_cell

inline_for_extraction noextract
let linear_equation (rows columns:pos) (weight:A.matrix rows columns) (input:A.vector columns) (row:P.index rows)
  : Lemma (Matmul.project #rows #columns weight input row == A.linear rows columns weight input row) =
  let products:P.vector real columns = fun i -> weight row i *. input i in
  FStar.Classical.forall_intro (Matmul.term_inside #rows #columns weight input row);
  Matmul.sum_restrict_full columns (Matmul.terms #rows #columns weight input row) products;
  Sum.sum_window #columns products 0 columns;
  Sum.sum_ext columns (fun i -> products (0 + i)) (fun i -> weight row i *. input i);
  norm_spec [delta_only ["KPlay.Proof.Base.Matmul.project"; "KPlay.Proof.Base.Matmul.total_sum"]]
    (Matmul.project #rows #columns weight input row);
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.linear"]] (A.linear rows columns weight input row)
