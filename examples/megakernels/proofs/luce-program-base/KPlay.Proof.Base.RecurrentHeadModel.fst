module KPlay.Proof.Base.RecurrentHeadModel

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
open Kuiper
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module Approx = Kuiper.Approximates.Base
module Sum = Kuiper.Sum
module Root = FStar.Math.Sqrt
module Exp = FStar.Math.Exp
module P = KPlay.Proof.Shared.Projection
module S = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module Dot = KPlay.Proof.Base.RecurrentDotMath
module Transfer = KPlay.Proof.Base.Transfer

noeq type inputs = {
  raw_query: R.vector 128;
  raw_key: R.vector 128;
  raw_value: R.vector 128;
  gate: R.vector 128;
  norm_weight: R.vector 128;
  beta_input: real;
  decay_input: real;
  decay_bias: real;
  decay_log: real;
  state: R.matrix 128 128;
}

inline_for_extraction noextract
let raw_value_at (x:inputs) (i:P.index 128) : GTot real = x.raw_value i

inline_for_extraction noextract
let query (x:inputs) : S.vector 128 =
  fun i -> R.l2_normalize x.raw_query i *. Root.rsqrt 128.0R

inline_for_extraction noextract
let key (x:inputs) : S.vector 128 =
  fun i -> R.l2_normalize x.raw_key i

inline_for_extraction noextract
let beta (x:inputs) : GTot real = R.sigmoid x.beta_input

inline_for_extraction noextract
let gain (x:inputs) : GTot real =
  Exp.exp ((0.0R -. Exp.exp x.decay_log) *. R.softplus (x.decay_input +. x.decay_bias))

inline_for_extraction noextract
let decayed_column (x:inputs) (column:P.index 128) : S.vector 128 =
  fun i -> x.state i column *. gain x

inline_for_extraction noextract
let memory (x:inputs) (column:P.index 128) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 128 (Dot.products (decayed_column x column) (key x))

inline_for_extraction noextract
let delta (x:inputs) (column:P.index 128) : GTot real =
  (x.raw_value column -. memory x column) *. beta x

inline_for_extraction noextract
let next_column (x:inputs) (column:P.index 128) : S.vector 128 =
  fun i -> decayed_column x column i +. key x i *. delta x column

inline_for_extraction noextract
let core (x:inputs) : R.vector 128 =
  fun column -> Sum.sum #real #Sum.real_add_semigroup 0 128 (Dot.products (next_column x column) (query x))

inline_for_extraction noextract
let gated (x:inputs) : R.vector 128 =
  fun i -> (x.norm_weight i *. (core x i *. R.rms_scale #128 (core x))) *. R.silu (x.gate i)

inline_for_extraction noextract
let represented (weights:Transfer.packed) (hidden:Transfer.scratch) (state:lseq FP.t 4718592)
  (base:nat{base + 19458208 <= 752393024}) (offset:nat{offset + 16384 <= 4718592})
  (head:P.index 16) (x:inputs) : prop =
  (forall (i:P.index 128).
    Approx.v_approximates (Seq.index hidden (2048 + head * 128 + i)) (x.raw_query i) /\
    Approx.v_approximates (Seq.index hidden (4096 + head * 128 + i)) (x.raw_key i) /\
    Approx.v_approximates (Seq.index hidden (6144 + head * 128 + i)) (x.raw_value i) /\
    Approx.v_approximates (Seq.index hidden (8192 + head * 128 + i)) (x.gate i) /\
    Approx.v_approximates (Seq.index weights (base + 19458080 + i)) (x.norm_weight i)) /\
  (forall (i j:P.index 128).
    Approx.v_approximates (Seq.index state (offset + i * 128 + j)) (x.state i j)) /\
  Approx.v_approximates (Seq.index hidden (10240 + head)) x.beta_input /\
  Approx.v_approximates (Seq.index hidden (10256 + head)) x.decay_input /\
  Approx.v_approximates (Seq.index weights (base + 19458064 + head)) x.decay_bias /\
  Approx.v_approximates (Seq.index weights (base + 19458048 + head)) x.decay_log
