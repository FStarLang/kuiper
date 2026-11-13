# Megakernel proof artifacts

Standalone artifacts from the RESOLVE megakernel case studies: **six complete
program refinements, 24 family refinements, three shared-specification checks,
and 12 family-tightness proofs**. No RESOLVE installation, agent, DTester,
model weights, private checkout, or historical checked cache is required.

**Release candidate.** The release owner confirmed the required redistribution
permissions and organizational publication approval on November 13, 2026.
See [NOTICE.md](NOTICE.md) for attribution and licensing boundaries. FlashMoE
still has an explicitly unresolved original-candidate provenance gap.

## Check the proofs

Use Python 3.12+ on Linux x86_64; Ubuntu 24.04 is the tested environment.
The full suite includes a successful check exceeding **20 GiB peak child
memory**; use a suitably large host, or select `tightness`/`shared`.
From this directory:

```sh
python3 replay.py list
python3 replay.py integrity
python3 replay.py verify --kind tightness --jobs 4
python3 replay.py verify --case flashmoe-program-base --jobs 4
python3 replay.py verify --jobs 4
```

The first verification downloads the checksum-pinned public toolchain in
`toolchain.json`, stages the historical source overlays, and checks their
interfaces and implementation bodies. Dependency checking is timed separately;
download and extraction are not included. Subsequent invocations reuse that
preparation, **not example proof results**. Every requested `.fst` and `.fsti`
is checked afresh with
`--force --cache_off --admit_smt_queries false`, F* commit
`da21c3925bf09a0a1d9e7558170be82da5edb37c`, and Z3 4.13.3.

Results and per-module logs go to `.work/proofs/runs/<run>/`. JSON records wall
time, child CPU time, peak child RSS, source hashes, tool versions, and worker
count. This measures **checking the supplied proof**, not finding it. Existing
native/numerical axioms are still trusted; disabling SMT admission does not
turn those axioms into hardware proofs. See [CLAIMS.md](CLAIMS.md).

One proof-only replay adjustment is disclosed in
[replay-adjustments.patch](replay-adjustments.patch): an explicit, proved
real-division identity in Mirage's optimized decoder. Its original source hash
is retained in `manifest.json`; contracts, executable operations, and CUDA are
unchanged. All other proof payloads retain their historical bytes.

`--case` accepts a quoted pattern and is repeatable. `--kind` selects `family`,
`program`, `shared`, or `tightness`; the default is `all`. A changed support
profile requires a new `--work` directory. The old compiler's packaged checked
caches produced verification failures during preparation, so replay deliberately
retains the historical cache-free checking policy.

## Inspect and run the reduced kernels

`kernels/` contains the exact retained CUDA, including repaired revisions.
`manifest.json` binds every proof/support/kernel file by SHA-256, records the
role-specific historical receipt identities, and lists each CUDA entry's ABI.
The `KPlay.*` names are historical F* module names, not a Python dependency.

```sh
# Optional full upstream source checkouts; no builds or model downloads.
python3 fetch_sources.py

# In a separate environment with the CUDA 13.0 toolkit and an NVIDIA GPU:
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install -r requirements-gpu.txt
python run_kernels.py --project all --device 0 --jobs 2
python run_kernels.py --project all --kind family --compile-only --jobs 2
```

The smoke runner builds **all six complete reduced programs** and checks their
complete outputs on synthetic fixtures, including FlashMoE ties/capacity and
LUCE's offset-one input repair. It does not demand identical tokens between
implementations when their specifications permit different choices.
The 24 family entry points can be compiled independently; execution requires
their documented caller state, rather than arbitrary zero-filled arguments.
These are reduced-CUDA checks, **not new extraction of the Pulse witnesses**
or replays of the original megakernels. No weights or tensors are downloaded.

Full upstream pins, kernel/reference entry paths, model identities, and
provenance limitations are in `sources.json`. An upstream checkout is a
reference, **not a substitute for the exact repaired code in `kernels/`**.
The optional `--submodules` flag fetches
pinned submodules too. Inspect [NOTICE.md](NOTICE.md) before redistribution.
The model revisions identify models, not an external Transformers code
revision. Inspecting the upstream entry paths does not reconstruct the original
run or close its source-correspondence obligations.
Native extension builds may be reused on subsequent invocations; select a new
`--work` directory to measure fresh compilation.

## Where this fits in RESOLVE

| Stage | What is released here | Evidence boundary |
| --- | --- | --- |
| Original workload and task | Pinned upstream references and explicit mathematical specifications | Reference/task interpretation was reviewed, not proved by checking a Git hash |
| Reduction and decomposition | Exact complete and family CUDA sources | Original/reduced correspondence is historical finite/reviewed evidence, not a universal equivalence theorem |
| Specification and implementation | Shared contracts, interfaces, complete Pulse implementations and helper proofs | F* checks refinement under the stated numerical and native contracts |
| Tightness | Twelve constructive family-level proofs | Existence and full-observation determination, or the stated choice/approximation policy; not IEEE uniqueness |
| Replay | Small standalone proof and CUDA runners | No discovery, agent orchestration, fuzzing pipeline, or DTester implementation |

This directory is intentionally outside `src/`: cases reuse module names and
must have isolated include paths. It does not alter current Kuiper's compiler,
ordinary build, or generated `dist/` snapshots. Frozen payloads preserve their
original whitespace and are excluded from formatting/whitespace cleanup.
[CLAIMS.md](CLAIMS.md) describes
the exact theorem scopes. See [benchmarks/](benchmarks/) for measured costs and
the CI recommendation; source integrity and runner tests are also available through
`make -j4 integrity test`.
