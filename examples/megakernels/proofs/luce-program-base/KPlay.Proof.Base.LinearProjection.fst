module KPlay.Proof.Base.LinearProjection

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module SZ = Kuiper.SizeT
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Schedule = KPlay.Proof.Base.Schedule
module Source = KPlay.Proof.Base.SourceProjection
module Native = KPlay.Proof.Base.NativeProjection
module Layout = KPlay.Proof.Base.MatvecLayout
module Matmul = KPlay.Proof.Base.Matmul
module Writes = KPlay.Proof.Base.MatvecWrites
module Inputs = KPlay.Proof.Base.Inputs
module Prior = KPlay.Proof.Base.RecurrentPipeline

inline_for_extraction noextract
let rows (phase:nat{phase < 8}) : pos =
  if phase = 0 then 6144 else if phase = 1 then 2048 else if phase = 2 || phase = 3 then 16
  else if phase = 5 || phase = 6 then 3584 else 1024

inline_for_extraction noextract
let columns (phase:nat{phase < 8}) : pos =
  if phase = 4 then 2048 else if phase = 7 then 3584 else 1024

inline_for_extraction noextract
let weight (w:R.weights) (phase:nat{phase < 8}) : R.matrix (rows phase) (columns phase) =
  if phase = 0 then w.R.in_qkv else if phase = 1 then w.R.in_z else if phase = 2 then w.R.in_beta
  else if phase = 3 then w.R.in_a else if phase = 4 then w.R.out_projection else if phase = 5 then w.R.mlp_gate
  else if phase = 6 then w.R.mlp_up else w.R.mlp_down

inline_for_extraction noextract
let input (stages:Prior.decoder_stages) (phase:nat{phase < 8}) : R.vector (columns phase) =
  if phase < 4 then stages.Prior.normalized else if phase = 4 then stages.Prior.gated_values
  else if phase = 7 then stages.Prior.product else stages.Prior.postnorm

inline_for_extraction noextract
let output (stages:Prior.decoder_stages) (phase:nat{phase < 8}) : R.vector (rows phase) =
  if phase = 0 then stages.Prior.raw else if phase = 1 then stages.Prior.projected_z
  else if phase = 2 then stages.Prior.projected_b else if phase = 3 then stages.Prior.projected_a
  else if phase = 4 then stages.Prior.mixed else if phase = 5 then stages.Prior.mlp_gate
  else if phase = 6 then stages.Prior.mlp_up else stages.Prior.down

inline_for_extraction noextract
let source_dimensions (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.linear_layer) (phase:nat{phase < 8})
  : Lemma (
      (Schedule.projection_at call position layer phase).Schedule.rows == rows phase /\
      (Schedule.projection_at call position layer phase).Schedule.columns == columns phase) =
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then () else if phase = 3 then ()
  else if phase = 4 then () else if phase = 5 then () else if phase = 6 then () else assert (phase = 7)

inline_for_extraction noextract
let stage_equation (call:Prior.logical_call) (stages:Prior.decoder_stages)
  (phase:nat{phase < 8}) (row:P.index (rows phase))
  : Lemma
      (requires stages == Prior.build_stages call)
      (ensures output stages phase row ==
        R.project #(rows phase) #(columns phase) (weight call.Prior.weight phase) (input stages phase) row) =
  let canonical = Prior.build_stages call in
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then () else if phase = 3 then ()
  else if phase = 4 then () else if phase = 5 then () else if phase = 6 then () else assert (phase = 7)

inline_for_extraction noextract
let weight_cell (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.linear_layer) (phase:nat{phase < 8}) (values:C.chest2 BF.t 752393024 1)
  (row:P.index (rows phase)) (col:P.index (columns phase))
  : Lemma (
      weight (O.model_linear ((Inputs.parameter_values (Native.packed_values values)).O.linear layer)) phase row col ==
      Approx.to_real
        (Layout.matrix (Source.weight_geometry call position layer phase) values row col)) =
  source_dimensions call position layer phase;
  Inputs.layer_extent layer;
  if phase = 0 then () else if phase = 1 then () else if phase = 2 then () else if phase = 3 then ()
  else if phase = 4 then () else if phase = 5 then () else if phase = 6 then () else assert (phase = 7)

inline_for_extraction noextract
let weights_approx (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.linear_layer) (phase:nat{phase < 8}) (values:C.chest2 BF.t 752393024 1)
  : Lemma (O.matrix_approximates (Layout.matrix (Source.weight_geometry call position layer phase) values)
      (weight (O.model_linear ((Inputs.parameter_values (Native.packed_values values)).O.linear layer)) phase)) =
  source_dimensions call position layer phase;
  let at_cell (row:P.index (rows phase)) (col:P.index (columns phase))
    : Lemma (Approx.v_approximates
        (Layout.matrix (Source.weight_geometry call position layer phase) values row col)
        (weight (O.model_linear ((Inputs.parameter_values (Native.packed_values values)).O.linear layer)) phase row col)) =
    weight_cell call position layer phase values row col;
    Approx.to_real_ok (Layout.matrix (Source.weight_geometry call position layer phase) values row col)
  in
  FStar.Classical.forall_intro_2 at_cell

inline_for_extraction noextract
let project_equation (phase:nat{phase < 8}) (w:R.weights) (x:R.vector (columns phase))
  (row:P.index (rows phase))
  : Lemma (Matmul.project #(rows phase) #(columns phase) (weight w phase) x row ==
    R.project #(rows phase) #(columns phase) (weight w phase) x row) =
  Matmul.project_expanded #(rows phase) #(columns phase) (weight w phase) x row;
  norm_spec [delta_once ["KPlay.Proof.Shared.Recurrent.project"]]
    (R.project #(rows phase) #(columns phase) (weight w phase) x row)
