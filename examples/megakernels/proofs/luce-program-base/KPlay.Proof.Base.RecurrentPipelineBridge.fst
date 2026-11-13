module KPlay.Proof.Base.RecurrentPipelineBridge

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module H = KPlay.Proof.Base.RecurrentHeadModel
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Dot = KPlay.Proof.Base.RecurrentDotMath

inline_for_extraction noextract
let inputs (p:Prior.head_model) : H.inputs = {
  H.raw_query = p.Prior.qinput;
  H.raw_key = p.Prior.kinput;
  H.raw_value = p.Prior.value;
  H.gate = p.Prior.z;
  H.norm_weight = p.Prior.weight;
  H.beta_input = p.Prior.b;
  H.decay_input = p.Prior.a;
  H.decay_bias = p.Prior.dt_bias;
  H.decay_log = p.Prior.a_log;
  H.state = p.Prior.old;
}

inline_for_extraction noextract
let normalization (p:Prior.head_model)
  : Lemma
      (requires Prior.valid_head p)
      (ensures H.beta (inputs p) == p.Prior.beta /\ H.gain (inputs p) == p.Prior.decay /\
        (forall (i:P.index 128).
          H.query (inputs p) i == p.Prior.query i /\ H.key (inputs p) i == p.Prior.key i) /\
        (forall (i j:P.index 128). H.decayed_column (inputs p) j i == p.Prior.decayed i j)) =
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.valid_head"]] (Prior.valid_head p)

inline_for_extraction noextract
let memory (p:Prior.head_model) (column:P.index 128)
  : Lemma
      (requires Prior.valid_head p)
      (ensures H.memory (inputs p) column ==
        Prior.sum_n 128 (fun i -> p.Prior.decayed i column *. p.Prior.key i)) =
  normalization p;
  Prior.sum_shared_ext 128 (fun i -> p.Prior.decayed i column *. p.Prior.key i)
    (Dot.products (H.decayed_column (inputs p) column) (H.key (inputs p)));
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentHeadModel.memory"]] (H.memory (inputs p) column)

inline_for_extraction noextract
let updated (p:Prior.head_model) (column:P.index 128)
  : Lemma
      (requires Prior.valid_head p)
      (ensures H.delta (inputs p) column == p.Prior.delta column /\
        (forall (i:P.index 128). H.next_column (inputs p) column i == p.Prior.updated i column)) =
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.valid_head"]] (Prior.valid_head p);
  normalization p;
  memory p column

inline_for_extraction noextract
let core (p:Prior.head_model) (column:P.index 128)
  : Lemma
      (requires Prior.valid_head p)
      (ensures H.core (inputs p) column == p.Prior.core column) =
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.valid_head"]] (Prior.valid_head p);
  normalization p;
  updated p column;
  Prior.sum_shared_ext 128 (fun i -> p.Prior.updated i column *. p.Prior.query i)
    (Dot.products (H.next_column (inputs p) column) (H.query (inputs p)));
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentHeadModel.core"]] (H.core (inputs p) column)

inline_for_extraction noextract
let rms_congr (left right:R.vector 128)
  : Lemma
      (requires forall (i:P.index 128). left i == right i)
      (ensures R.rms_scale #128 left == R.rms_scale #128 right) =
  assert (R.sum_squares #128 left == R.sum_squares #128 right) by (
    FStar.Tactics.V2.norm [iota; reduce_projections; unascribe; primops; simplify; delta_only [
      "KPlay.Proof.Shared.Recurrent.sum_squares"]];
    FStar.Tactics.V2.apply_lemma (`(Prior.native_sum_ext 128));
    FStar.Tactics.V2.smt ());
  norm_spec [delta_once ["KPlay.Proof.Shared.Recurrent.rms_scale"]] (R.rms_scale #128 left);
  norm_spec [delta_once ["KPlay.Proof.Shared.Recurrent.rms_scale"]] (R.rms_scale #128 right)

inline_for_extraction noextract
let gated (p:Prior.head_model) (i:P.index 128)
  : Lemma
      (requires Prior.valid_head p)
      (ensures H.gated (inputs p) i == p.Prior.gated i) =
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentPipeline.valid_head"]] (Prior.valid_head p);
  FStar.Classical.forall_intro (core p);
  rms_congr (H.core (inputs p)) p.Prior.core;
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentHeadModel.gated"]] (H.gated (inputs p) i)
