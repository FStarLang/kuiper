# Checking cost and CI policy

These are measurements of **checking supplied proofs**, not proof discovery.
The source inventory, public toolchain pin, and individual source hashes bind
the results. Measurements were taken on November 13, 2026; hardware and
software details are in [environment.json](environment.json).

## Measurement method

```sh
python3 replay.py verify --jobs 8
```

Each case checks all its listed interfaces, implementation bodies, helper
proofs, and shared source. Cases run sequentially; up to eight independent
module checks within a case run concurrently. Every module uses a fresh
process/scratch directory and `--force --cache_off --admit_smt_queries false`.
Successful results are never substituted for a requested example check.
Parsing and checking imported source are included, so repeated dependencies
are deliberately charged again.

The per-case table below reports elapsed wall time. The machine-readable
`proof-checks.json` also records **every individual module's** wall time,
child user/system CPU time, peak child RSS, source hash, and exit status.
Summed module wall times are not elapsed batch time. Peak child RSS is the
largest individual child-process RSS, **not aggregate concurrent memory**.
This is one run on a shared host, not a distribution or CI-time guarantee.

## First complete sweep

The initial 45-case sweep took **13,460.5 s (3 h 44 min)** after dependency
preparation. All six complete programs, all 12 tightness cases, all three
shared checks, and 23 of the 24 family checks passed. The optimized Mirage
decoder returned SMT `unknown` on a real-division rewrite; the failed attempt
is retained, not counted as successful. The runner correctly exited nonzero.

[proof-checks.json](proof-checks.json) is the original complete record, including
that failure and its original source hash. The narrowly scoped proof adjustment
is described in [CLAIMS.md](../CLAIMS.md) and
[replay-adjustments.patch](../replay-adjustments.patch).
Times below are **first-sweep measurements**, not a fabricated all-pass batch.
The six complete-program cases together took 5,206.6 s (86.8 min);
the tightness cases took 198.8 s (3.3 min), excluding setup.

| Case | Roots | Elapsed (s) | First-sweep status |
| --- | ---: | ---: | --- |
| `flashmoe-argmax-base` | 3 | 54.1 | passed |
| `flashmoe-argmax-optimized` | 3 | 57.3 | passed |
| `flashmoe-argmax-tightness` | 2 | 28.7 | passed |
| `flashmoe-normalization_capacity-base` | 3 | 59.6 | passed |
| `flashmoe-normalization_capacity-optimized` | 3 | 69.8 | passed |
| `flashmoe-normalization_capacity-tightness` | 2 | 30.1 | passed |
| `flashmoe-program-base` | 16 | 250.3 | passed |
| `flashmoe-program-optimized` | 29 | 314.2 | passed |
| `flashmoe-projection_softmax-base` | 3 | 79.4 | passed |
| `flashmoe-projection_softmax-optimized` | 3 | 119.0 | passed |
| `flashmoe-projection_softmax-tightness` | 2 | 47.1 | passed |
| `flashmoe-shared` | 5 | 61.0 | passed |
| `luce-argmax-base` | 3 | 52.1 | passed |
| `luce-argmax-optimized` | 3 | 66.6 | passed |
| `luce-argmax-tightness` | 2 | 10.3 | passed |
| `luce-embedding-base` | 3 | 46.8 | passed |
| `luce-embedding-optimized` | 3 | 44.2 | passed |
| `luce-embedding-tightness` | 2 | 3.1 | passed |
| `luce-final_norm_projection-base` | 3 | 589.2 | passed |
| `luce-final_norm_projection-optimized` | 3 | 234.1 | passed |
| `luce-final_norm_projection-tightness` | 2 | 19.0 | passed |
| `luce-full_attention_decoder-base` | 3 | 336.1 | passed |
| `luce-full_attention_decoder-optimized` | 3 | 467.3 | passed |
| `luce-full_attention_decoder-tightness` | 2 | 2.1 | passed |
| `luce-program-base` | 140 | 1904.4 | passed |
| `luce-program-optimized` | 73 | 673.6 | passed |
| `luce-recurrent_decoder-base` | 3 | 283.1 | passed |
| `luce-recurrent_decoder-optimized` | 3 | 326.6 | passed |
| `luce-recurrent_decoder-tightness` | 2 | 18.1 | passed |
| `luce-shared` | 9 | 59.4 | passed |
| `mirage-argmax-base` | 3 | 45.8 | passed |
| `mirage-argmax-optimized` | 3 | 163.9 | passed |
| `mirage-argmax-tightness` | 2 | 3.6 | passed |
| `mirage-decoder-base` | 3 | 262.0 | passed |
| `mirage-decoder-optimized` | 3 | 3964.1 | **failed; original source** |
| `mirage-decoder-tightness` | 2 | 18.1 | passed |
| `mirage-embedding_position-base` | 3 | 45.9 | passed |
| `mirage-embedding_position-optimized` | 3 | 102.3 | passed |
| `mirage-embedding_position-tightness` | 2 | 1.4 | passed |
| `mirage-final_norm_projection-base` | 3 | 135.7 | passed |
| `mirage-final_norm_projection-optimized` | 3 | 290.6 | passed |
| `mirage-final_norm_projection-tightness` | 2 | 17.2 | passed |
| `mirage-program-base` | 3 | 1228.1 | passed |
| `mirage-program-optimized` | 3 | 836.0 | passed |
| `mirage-shared` | 3 | 39.1 | passed |

## Corrected decoder and current coverage

[decoder-recheck.json](decoder-recheck.json) records a fresh check of the
complete corrected `mirage-decoder-optimized` case: **all three roots passed in
1,805.0 s (30.1 min)** using three workers and the same strict, cache-free
verifier configuration. Its largest child reached **23.1 GiB RSS**.
No alternate solver setting, admission, or weakened contract was retained.

The 44 passing first-sweep cases plus this recheck cover **all 45 current cases
and 377 module roots**. Each successful root's recorded hash was checked against
the current manifest. The first sweep retains the original inventory fingerprint;
the recheck binds the updated inventory containing the disclosed proof change.

The sum of these 45 successful case wall times is **11,301.3 s (3 h 8 min)**,
excluding setup and the failed original attempt. This is a sum across two runs,
**not a separately measured all-pass batch duration**. Proof development and
the small diagnostic experiments are not included in successful checking costs.

## Dependency preparation

The following successful checks are separate from example timings.
Download, unpacking, and source staging are **not included**.
Details are in [support-checks.json](support-checks.json).

| Profile | Checked roots | Workers | Elapsed |
| --- | ---: | ---: | ---: |
| Mirage | 24 | 8 | 102.5 s |
| LUCE | 36 | 8 | 138.7 s |
| FlashMoE | 43 | 8 | 169.0 s |

These were separately measured profile checks, not one end-to-end cold setup.
Successful dependency checks can be reused by subsequent example runs.
The archived compiler's packaged checked caches caused failures during
preparation; this package retains cache-free verification instead.

[isolated-replay.json](isolated-replay.json) records an additional standalone
check with an empty HOME, an otherwise cleared environment, `/usr/bin/python3`,
and `PATH=/usr/bin:/bin`. A new work directory unpacked and authenticated the
public toolchain archive; no private F* installation or prior proof caches were
used. The archive was copied from the original authenticated download, rather
than downloaded again. This was **not a fresh OS installation**.
At two workers, FlashMoE dependency checking took 615.9 s and its argmax
tightness case took 25.9 s.

## CUDA replay

[gpu-programs.json](gpu-programs.json) records successful fresh compilation
and synthetic smoke tests of all six complete reduced programs.
[gpu-families.json](gpu-families.json) records successful independent
compilation of all 24 family entry points; those entries were **not executed**.
Both used two compiler workers, PyTorch 2.14.0+cu130, nvcc 13.0.88, and an H100.

The complete-program builds took 41.7--93.7 s each. Recorded CUDA-event and host
wall times are single synthetic smoke observations, **not performance
benchmarks or speedup claims**. Kernel execution, native compilation, and proof
checking are different measurements. Native builds can be reused on subsequent
runs; choose a fresh `--work` directory for a fresh compilation measurement.
The final runner's reporting changes were also exercised by rerunning all six
program smoke tests; these files retain the first, fresh-build measurements.

## CI recommendation

After the release approvals in [NOTICE.md](../NOTICE.md), the artifacts can
live on main without making full historical proof checking a required PR gate.
The included workflow runs source-integrity checks and runner unit tests on
relevant PRs. Fresh `tightness` and `shared` proof suites are available through
manual dispatch, with two workers and retained diagnostic/timing artifacts.
Run `program`, `family`, or `all` locally/on an appropriately provisioned larger
runner using `replay.py`; they are deliberately not offered on the standard
hosted workflow.

Start with that policy. Even the isolated FlashMoE dependency setup took about
ten minutes at two workers on this host. Successful family checks reached
**20.4 GiB** (LUCE base final-normalization/projection) and **23.1 GiB**
(corrected Mirage optimized decoder) peak child RSS, already exceeding the
[documented standard public Linux runner's 16 GB](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
Measure a hosted-runner baseline before promoting tightness to automatic CI.
Full-suite capacity planning must also allow for concurrent child processes;
aggregate memory was not measured.
No remote workflow was run while the branch remained private/local.
