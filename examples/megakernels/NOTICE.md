# Attribution and public-release checklist

**This is a local release candidate, not legal clearance to publish.** Keep it
local until the applicable rightsholders/organization approve disclosure and
the licenses for newly released material are confirmed.

| Material | Source and status |
| --- | --- |
| Mirage upstream | `mirage-project/mirage`, pinned in `sources.json`; Apache-2.0 text retained in `licenses/Mirage-Apache-2.0.txt` |
| LUCE upstream | **`PixelML/luce-megakernel`**, not `Luce-Org/lucebox`; at `bab084443f14e6ab3ba76f8e8e3f721ad2e56144`, the complete tree has no LICENSE, COPYING, or NOTICE file. Redistribution permission for derivative retained CUDA remains unresolved |
| FlashMoE upstream | `osayamenja/FlashMoE`, pinned in `sources.json`; BSD-3-Clause, copyright 2026 Osayamen Jonathan Aimuyo; original notice and conditions retained in `licenses/FlashMoE-BSD-3-Clause.txt` |
| F*/Pulse support | Source snapshots and changes relative to the pinned compiler package; applicable Apache-2.0 texts retained |
| Reference semantics | Qwen3 and Qwen3.5 Transformers model implementations, with model identities pinned in `sources.json`; Transformers and PyTorch license texts retained. No model weights, tokenizers, or input tensors are included |
| Proofs and new replay code | Authored during the proof-artifact work. Confirm the organization's intended outbound license before public release; dependency notices do not automatically license this material |

Existing notices inside source files are preserved. Dependency license texts
are not a blanket grant for every file in this directory. A public repository
without a license is not automatically permission to redistribute derivative
code, and a public draft branch/PR would already disclose it. See GitHub's
[licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository);
the release decision requires the appropriate rightsholders' approval.

## Retained changes and provenance

The CUDA files are **reductions/translations and repaired variants**, not
verbatim claims about the upstream revisions. Their exact bytes and
role-specific historical source/acceptance identities are in `manifest.json`.
The single proof-only replay adjustment is recorded separately with its
original source hash and reversible diff; it is not misidentified as the
original receipt-bound proof bytes.
No runtime source rewriting is performed by the smoke runner; its generated
C++ file only adds a Python binding to the existing `kplay_run` function.

Mirage includes its authorized finite-score/selection and reference/task
repairs. Its current shared specification carries the checked
mixed-observation compatibility bridge. LUCE's complete optimized program
replaces alignment-dependent `uint4`/`float4` reinterpretation reads with typed
scalar reads while preserving indices, casts, FMA/FTZ operations and
accumulation order. Its shared normalization/hidden type separation is a
representation correction. Earlier family artifacts keep their own accepted
source revisions; they must not be silently relabeled as the later complete
program repair.

FlashMoE's retained router is fully identified, but **the original optimized
candidate binding remains unavailable**. The optional upstream checkout is
only a source reference; it does not close that gap or authorize an
original-megakernel equivalence claim.

Before public release, obtain LUCE redistribution permission (or remove that
material), confirm the outbound license and organizational approval for the
new artifacts, and preserve the documented scientific/provenance limits.
No RESOLVE/DTester implementation, private agent transcript, environment
dictionary, historical executable/cache, captured tensor, or model weight is
part of this package.
