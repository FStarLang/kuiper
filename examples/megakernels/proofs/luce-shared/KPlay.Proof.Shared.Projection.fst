module KPlay.Proof.Shared.Projection

open FStar.Real
module Root = FStar.Math.Sqrt
module Sum = Kuiper.Sum
module BF16 = Kuiper.BFloat16
module Approx = Kuiper.Approximates.Base
module Scalar = Kuiper.Scalars.Base

(* Scalar BF16 values retain the exact observation; computed logits have the
   library's non-injective real approximation semantics, not bitwise equality.
   No allocation, layout, reduction schedule, or decoder implementation occurs
   in this contract. *)

inline_for_extraction noextract
let index (n:nat) = i:nat{i < n}

inline_for_extraction noextract
let vector (a:Type0) (n:nat) = index n -> GTot a

inline_for_extraction noextract
let matrix (a:Type0) (rows columns:nat) =
  index rows -> GTot (vector a columns)

inline_for_extraction noextract
let reference_hidden_size : pos = 1024

inline_for_extraction noextract
let reference_vocab_size : pos = 248320

inline_for_extraction noextract
let reference_epsilon : Root.rpos = 0.000001R

inline_for_extraction noextract
let real_vector (#n:nat) (x:vector BF16.t n) : vector real n =
  fun i -> Approx.to_real (x i)

inline_for_extraction noextract
let real_matrix (#rows #columns:nat) (x:matrix BF16.t rows columns)
  : matrix real rows columns =
  fun row -> real_vector (x row)

inline_for_extraction noextract
let real_sum (#n:nat) (f:vector real n) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 n f

inline_for_extraction noextract
let rec real_sum_nonnegative_from
  (#n:nat) (f:vector real n) (first:nat{first <= n})
  : Lemma
      (requires forall (i:index n). f i >=. 0.0R)
      (ensures
        Sum.sum #real #Sum.real_add_semigroup first n f >=. 0.0R)
      (decreases (n - first)) =
  if first < n then (
    Sum.sum_pop_left #real #Sum.real_add_semigroup first n f;
    real_sum_nonnegative_from f (first + 1)
  )

inline_for_extraction noextract
let rms_argument
  (#hidden_size:pos) (epsilon:Root.rpos) (x:vector real hidden_size)
  : GTot Root.rpos =
  let squares : vector real hidden_size = fun i -> x i *. x i in
  real_sum_nonnegative_from squares 0;
  real_sum squares /. of_int hidden_size +. epsilon

(* Qwen3_5RMSNorm.forward computes in float32 and casts back to x.dtype only
   after the gain. Casts preserve approximation in the supplied library; they
   are not asserted to preserve the floating-point representation. *)
inline_for_extraction noextract
let normalized_for_logits
  (#hidden_size:pos)
  (epsilon:Root.rpos)
  (x weight:vector real hidden_size)
  : GTot (vector real hidden_size) =
  let scale = Root.rsqrt (rms_argument epsilon x) in
  fun i -> (x i *. scale) *. (1.0R +. weight i)

inline_for_extraction noextract
let project
  (#hidden_size #vocab_size:pos)
  (normalized:vector real hidden_size)
  (embedding:matrix real vocab_size hidden_size)
  : vector real vocab_size =
  fun token -> real_sum (fun i -> normalized i *. embedding token i)

inline_for_extraction noextract
let logical_logits
  (#hidden_size #vocab_size:pos)
  (epsilon:Root.rpos)
  (hidden weight:vector BF16.t hidden_size)
  (embedding:matrix BF16.t vocab_size hidden_size)
  : GTot (vector real vocab_size) =
  project
    (normalized_for_logits epsilon (real_vector hidden) (real_vector weight))
    (real_matrix embedding)

inline_for_extraction noextract
let logical_outputs (hidden_size vocab_size:pos) (state:Type0) =
  vector real vocab_size & vector BF16.t hidden_size & state

(* In decode_reference this input row is arguments[0][0, -1, :].
   Every model call has one batch element and one token, including prefill.
   State is the entire caller-supplied decoder state at this boundary:
   position plus all convolution, recurrent, key, and value state. *)
inline_for_extraction noextract
let final_norm_projection
  (#hidden_size #vocab_size:pos) (#state:Type0)
  (epsilon:Root.rpos)
  (hidden weight:vector BF16.t hidden_size)
  (embedding:matrix BF16.t vocab_size hidden_size)
  (before:state)
  : GTot (logical_outputs hidden_size vocab_size state) =
  (logical_logits epsilon hidden weight embedding, hidden, before)

inline_for_extraction noextract
let logical_postcondition
  (#hidden_size #vocab_size:pos) (#state:Type0)
  (epsilon:Root.rpos)
  (hidden weight:vector BF16.t hidden_size)
  (embedding:matrix BF16.t vocab_size hidden_size)
  (before:state)
  (result:logical_outputs hidden_size vocab_size state)
  : prop =
  let (logits, observed_hidden, after) = result in
  (forall (token:index vocab_size).
    logits token == logical_logits epsilon hidden weight embedding token) /\
  (forall (i:index hidden_size). observed_hidden i == hidden i) /\
  after == before

inline_for_extraction noextract
let final_norm_projection_satisfies
  (#hidden_size #vocab_size:pos) (#state:Type0)
  (epsilon:Root.rpos)
  (hidden weight:vector BF16.t hidden_size)
  (embedding:matrix BF16.t vocab_size hidden_size)
  (before:state)
  : Lemma
      (logical_postcondition epsilon hidden weight embedding before
        (final_norm_projection epsilon hidden weight embedding before)) =
  ()

(* This relation does not identify two scores that approximate the same real.
   In particular, it supplies no argmax-stability or IEEE error-bound theorem.
   A role must prove its own mapping to these logical values and exact copies. *)
inline_for_extraction noextract
let postcondition
  (#hidden_size #vocab_size:pos) (#state #score:Type0)
  {| Scalar.scalar score, Approx.real_like score |}
  (epsilon:Root.rpos)
  (hidden weight:vector BF16.t hidden_size)
  (embedding:matrix BF16.t vocab_size hidden_size)
  (before:state)
  (logits:vector score vocab_size)
  (observed_hidden:vector BF16.t hidden_size)
  (after:state)
  : prop =
  (forall (token:index vocab_size).
    Approx.v_approximates (logits token)
      (logical_logits epsilon hidden weight embedding token)) /\
  (forall (i:index hidden_size). observed_hidden i == hidden i) /\
  after == before
