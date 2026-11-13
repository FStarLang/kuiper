# Specifications, tightness, and trust

The authoritative claims are the checked declarations, not the names of the
upstream projects. A proof relates a supplied Pulse implementation to a
specification under its explicit premises. Source review and finite native
comparisons connected these translations to retained CUDA historically; the
replay script is not a verified CUDA parser, compiler, or reduction algorithm.

## Complete programs

Complete proof sources: Mirage [base](proofs/mirage-program-base/) /
[optimized](proofs/mirage-program-optimized/), LUCE
[base](proofs/luce-program-base/) / [optimized](proofs/luce-program-optimized/),
and FlashMoE [base](proofs/flashmoe-program-base/) /
[optimized](proofs/flashmoe-program-optimized/).
Each directory includes its contract and complete implementation; helper
modules are checked too. Shared-contract and compatibility checks are in
`proofs/<project>-shared/`.

| Program | Covered workload and observations | Important boundary |
| --- | --- | --- |
| Mirage | Qwen3-0.6B; all 28 layers, prompt length 7, one decode step, hidden width 1024, vocabulary 151936. Complete prompt-plus-generated tokens and seven returned hidden rows, with the modeled caller/cache/resource chain. | Repaired/authorized retained programs; finite-score selection and bilateral numerical premises. Not arbitrary prompt/decode lengths. |
| LUCE | Qwen3.5-0.8B; all 24 layers: 18 recurrent and six full-attention layers. Prompt lengths 1..520, decode counts 1/5/30/128, and `P+S-1<=2048`. Complete generated-token and pre-final-normalization BF16 hidden histories, recurrent/convolution/KV state, feedback, and caller ownership/lifetimes. | Repaired optimized scalar loads; stable, non-materializing inputs, including ordinary contiguous offset views. FP32 final-normalization observations are distinct from BF16 hidden/history observations. |
| FlashMoE | Retained router only: `S=256,H=512,E=8,C=32,top_k=1`, finite BF16 tokens bounded by 32 and expert-major weights bounded by 4. Complete demand, packed indices, normalized probabilities, and survival. | No expert MLP or larger-profile claim. The exact original optimized candidate and its original-to-retained binding are unavailable. |

The base and optimized roles refine a common **relative algebraic task**.
This is not bitwise equivalence, an IEEE error bound, proof of numerical
stability, or a guarantee of equal selected tokens/experts. Each execution
follows its own permitted numerical observations and selection history.
Satisfaction of the numerical premises is not newly established by replay.

Mirage's shared check includes its mixed-observation/legacy compatibility
bridge. LUCE's shared model similarly separates normalization and hidden
types. These representation changes do not identify unequal native types or
change the real-valued equations. FlashMoE retains actual `q / (+0 + q)`
normalization and the first capacity-eligible tokens in ascending token order;
it does not replace the computation with a literal one.

## Family contracts and their tightness proofs

Each `<project>-<family>-tightness` directory contains the exact shared source
bound by both family acceptances and a `KPlay.Proof.Tightness.fst` module.
`witness` constructs a permitted result. The subsequent lemmas establish the
appropriate uniqueness or complete allowed-result characterization.

| Project | Family | Determined observations or retained freedom |
| --- | --- | --- |
| Mirage | [embedding_position](proofs/mirage-embedding_position-tightness/) | Embedding values and specified positional/rotary observations |
| Mirage | [decoder](proofs/mirage-decoder-tightness/) | Complete hidden result and every modeled cache slot, including contents, length, and initialization |
| Mirage | [final_norm_projection](proofs/mirage-final_norm_projection-tightness/) | Normalized values, projected scores, and the specified preserved state |
| Mirage | [argmax](proofs/mirage-argmax-tightness/) | Prompt/output relation and allowed maxima of the same actual finite scores; uniqueness after fixing the permitted choice |
| LUCE | [embedding](proofs/luce-embedding-tightness/) | Complete embedding vector |
| LUCE | [recurrent_decoder](proofs/luce-recurrent_decoder-tightness/) | Hidden values and complete convolution/recurrent state |
| LUCE | [full_attention_decoder](proofs/luce-full_attention_decoder-tightness/) | Hidden values and complete specified KV state |
| LUCE | [final_norm_projection](proofs/luce-final_norm_projection-tightness/) | Mathematical normalization/projection targets and preserved state; native values remain subject to the declared relation |
| LUCE | [argmax](proofs/luce-argmax-tightness/) | Complete output/history relation; NaN priority, infinities, and authorized any-maximum ties remain. Fixing choices determines the state history |
| FlashMoE | [projection_softmax](proofs/flashmoe-projection_softmax-tightness/) | Complete mathematical projection/softmax targets under the declared native observation relation |
| FlashMoE | [argmax](proofs/flashmoe-argmax-tightness/) | Selected IDs and gathered probabilities, including the native tie rule |
| FlashMoE | [normalization_capacity](proofs/flashmoe-normalization_capacity-tightness/) | Exact demand, owners/indices and survival; complete mathematical probability targets with the existing approximation freedom |

Inspect `unique`, `full_contract`, `no_extra_results`, `fixed_choice_unique`,
`fixed_choices_unique`, and `all_targets_unique` where present. Not every
family has the same theorem shape. In particular, float-to-real approximation
does **not** make two allowed floating-point bit patterns identical.

These are **family-level** tightness proofs. They are not additional
whole-program tightness certificates, task-correctness proofs, hardware
models, or native termination proofs. A determinate but wrong task could still
be tight; reference fidelity and the approved observation/freedom policies
remain separate obligations. An inconsistent trusted theory would also defeat
the intended interpretation.

## Trusted boundary

* F*, Pulse, Z3, their foundational logic, and the mathematical/numerical model.
  The qualitative approximation relation is not a quantitative error model.
  Existing `assume`/`admit` declarations in the support libraries are visible
  and retained, not silently discharged by strict checking of an example.
* Native scalar operations, BF16/FP32 casts, transcendental and fast/FTZ
  operations, comparison/classification, guarded range/finiteness contracts,
  and BF16 MMA/register-layout contracts.
* CUDA allocation/alignment/residency, arrays and native views, current-stream
  launches and epochs, PyTorch input borrowing and owning results, and the
  specified success/failure/cleanup behavior. These contracts are not proofs
  of CUDA hardware, the C++ standard library, or PyTorch internals.
* Reviewed task interpretation and translation/source correspondence. Fresh
  proof checking establishes the supplied source theorems; hashes establish
  identity, not semantic correspondence.

`support/mirage` and `support/luce` contain only changed/new source dependencies
relative to the pinned public Kuiper package. FlashMoE layers its changes over
the LUCE support snapshot. `support/foundation` contains the five changed
F*/Pulse source files. Unrelated sparse/GEMM patches were excluded using the
actual dependency closures. The historical compiler and the public replay
compiler have the same F* commit but were built with OCaml 5.4.0 and 5.3.0,
respectively; no historical binary-installation fingerprint is claimed.

## Native checks in this release

The complete-program smoke tests use synthetic, non-secret inputs. FlashMoE
checks all four outputs for both a tie/overflow case and a balanced-routing
case. Mirage and LUCE check complete token/hidden output shapes, token range,
zero-weight hidden results, and input preservation; LUCE also uses offset-one
BF16 weights. These are finite diagnostics, not the universal theorem.
Compilation of a family entry point is not execution or numerical validation.

Proof receipts from the original work are represented by their hashes, not
redistributed private logs or review transcripts. Those hashes are provenance
identifiers, not standalone proof of historical acceptance. New replay
results are generated from the source files included here.

The original Mirage optimized-decoder source returned SMT `unknown` on the
real-division rewrite in `rms_radicand_at` during public-toolchain replay.
[replay-adjustments.patch](replay-adjustments.patch) makes that algebraic step
explicit with a proved helper. The original source SHA-256 remains recorded in
`manifest.json`. This changes proof presentation only: the theorem statement,
specification, native operations, and trusted assumptions are unchanged.
