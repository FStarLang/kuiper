module KPlay.Proof.Shared.ProjectionSoftmax

open Kuiper

module C = Kuiper.Chest
module BF16 = Kuiper.BFloat16
module FP = Kuiper.Floating
module Scalar = Kuiper.Scalars
module GEMM = Kuiper.Spec.GEMM
module Softmax = Kuiper.Spec.Softmax

(* Only the supplied base.py projection/softmax slice and input-domain.txt
   determine this contract. In particular, this is the dense local
   probabilities matrix, not the packed output of the complete router.

   The numeric target uses Kuiper's algebraic %~ abstraction: to_real_chest
   interprets the operands, and real arithmetic defines the result. Existing
   real_like/floating_real_like and float_cast rules justify the modeled
   arithmetic and BF16-to-FP32 promotion. No rounding-error or bitwise-equality
   claim is made. Representation, contiguity, and device mappings are separate
   obligations; chest2 carries only logical indices and values. *)

inline_for_extraction noextract
let token_count (smoke:bool) : pos =
  if smoke then 256 else 1024

inline_for_extraction noextract
let hidden_count (smoke:bool) : pos =
  if smoke then 512 else 5120

inline_for_extraction noextract
let expert_count (smoke:bool) : pos =
  if smoke then 8 else 16

inline_for_extraction noextract
let token_matrix (smoke:bool) : Type0 =
  C.chest2 BF16.t (token_count smoke) (hidden_count smoke)

inline_for_extraction noextract
let weight_matrix (smoke:bool) : Type0 =
  C.chest2 BF16.t (hidden_count smoke) (expert_count smoke)

inline_for_extraction noextract
let probability_matrix (smoke:bool) : Type0 =
  C.chest2 real (token_count smoke) (expert_count smoke)

inline_for_extraction noextract
let finite_bounded_bf16 (bound:Int64.t) (value:BF16.t) : prop =
  FP.is_finite value /\
  Scalar.lte (FP.abs value) (FP.of_int bound)

inline_for_extraction noextract
let input_domain
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  : prop
  =
  C.chest_forall (finite_bounded_bf16 32L) tokens /\
  C.chest_forall (finite_bounded_bf16 4L) router_weights

inline_for_extraction noextract
let projection
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  : GTot (probability_matrix smoke)
  =
  GEMM.matmul (C.to_real_chest tokens) (C.to_real_chest router_weights)

inline_for_extraction noextract
let projection_entry
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  (i:natlt (token_count smoke))
  (j:natlt (expert_count smoke))
  : Lemma (
      C.acc2 (projection smoke tokens router_weights) i j ==
      GEMM.gmatmul_single 0.0R ( *. ) ( +. )
        (C.to_real_chest tokens) (C.to_real_chest router_weights) i j)
  =
  GEMM.lemma_matmul_index
    (C.to_real_chest tokens) (C.to_real_chest router_weights) i j

inline_for_extraction noextract
let row_normalizer
  (smoke:bool)
  (scores:probability_matrix smoke)
  (i:natlt (token_count smoke))
  : GTot real
  =
  C.chest1_rsum (C.chest_map exp (C.chest2_row scores i))

inline_for_extraction noextract
let row_normalizer_positive
  (smoke:bool)
  (scores:probability_matrix smoke)
  (i:natlt (token_count smoke))
  : Lemma (row_normalizer smoke scores i >. 0.0R)
  =
  let exps = C.chest1_to_seq (C.chest_map exp (C.chest2_row scores i)) in
  Kuiper.Real.sum_non_zero exps 0.0R

inline_for_extraction noextract
let projection_softmax
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  : GTot (probability_matrix smoke)
  =
  let scores = projection smoke tokens router_weights in
  C.mk2 (fun i j ->
    C.acc1 (Softmax.softmax_real (C.chest2_row scores i)) j)

inline_for_extraction noextract
let probability_entry
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  (i:natlt (token_count smoke))
  (j:natlt (expert_count smoke))
  : Lemma (
      C.acc2 (projection_softmax smoke tokens router_weights) i j ==
      exp (C.acc2 (projection smoke tokens router_weights) i j) /.
        row_normalizer smoke (projection smoke tokens router_weights) i)
  = ()

(* No persistent state exists in this slice. Inputs are preserved exactly;
   output equality is at the real-model level, not a claim about FP32 bits. *)
inline_for_extraction noextract
let postcondition
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  (tokens_after:token_matrix smoke)
  (router_weights_after:weight_matrix smoke)
  (probabilities:probability_matrix smoke)
  : prop
  =
  input_domain smoke tokens router_weights /\
  tokens_after == tokens /\
  router_weights_after == router_weights /\
  probabilities == projection_softmax smoke tokens router_weights

inline_for_extraction noextract
let projection_softmax_satisfies_postcondition
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  : Lemma
      (requires input_domain smoke tokens router_weights)
      (ensures postcondition smoke tokens router_weights
        tokens router_weights (projection_softmax smoke tokens router_weights))
  = ()

inline_for_extraction noextract
let postcondition_unique
  (smoke:bool)
  (tokens:token_matrix smoke)
  (router_weights:weight_matrix smoke)
  (tokens_after:token_matrix smoke)
  (router_weights_after:weight_matrix smoke)
  (p q:probability_matrix smoke)
  : Lemma
      (requires
        postcondition smoke tokens router_weights tokens_after router_weights_after p /\
        postcondition smoke tokens router_weights tokens_after router_weights_after q)
      (ensures p == q)
  = ()
