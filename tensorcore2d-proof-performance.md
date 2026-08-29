# TensorCore2D proof-performance diagnosis

The tensor-core instruction is not the main problem. Verification performance is dominated by two separate pathologies:

1. Badly shaped arithmetic verification conditions in the original descriptor.
2. F*/Pulse repeatedly elaborating enormous separation-logic predicates in the out-of-place setup and teardown.

## Measurements

These are forced module-local verification times with dependencies cached. Normal verification was compared with `--admit_smt_queries true` to separate SMT/VC costs from Pulse elaboration and core checking.

| Module | Normal | SMT admitted | Main hotspot |
|---|---:|---:|---|
| `TensorCore2D.KernelDesc` | 494 s | 16 s | `reconstruct_from_warp_approx`, `tiles_approx_lemma` |
| `To.KernelDesc` | 302 s | 224 s | `split_output_to_lanes`, `block_teardown_to` |
| `To.KernelDesc.Teardown` | 463 s | 343 s | `gather_warp`, `gather_kernel_outputs` |
| `TensorCore2D` | 50 s | 7 s | `kf`, sendability proofs |
| `To.Finish` | 40 s | 5 s | `finish`, `nested_comb_tile_eq` |

Those five modules total about 22.5 serial minutes. Parallel Make hides some wall time but not CPU or memory usage; teardown peaked at approximately 3 GB.

## Implemented and measured in this pass

The first round of targeted changes is verified and repeatable:

| Module | Before | After | Change |
|---|---:|---:|---:|
| `TensorCore2D.KernelDesc` | 494 s | 335 s | -32% |
| `To.KernelDesc` | 302 s | 270 s | -11% |
| `To.Finish` | 40.5 s | 29.6 s | -27% |

Together these three modules fell from about 836 seconds to 635 seconds, saving
about 201 serial seconds (24%). Two independent descriptor runs completed in
334.98 and 334.90 seconds, and `tiles_approx_lemma` completed in 43.3 and 43.9
seconds, so the improvement is not a one-off solver fluctuation.

The main declaration-level changes were:

| Declaration | Before | After |
|---|---:|---:|
| `warp_tile_i` | 16.5 s | 0.29 s |
| `c_subtile_approx_lemma` | 74.6 s | 17.1 s |
| `rhs_is_constant_for_warps_approx` | 44.6 s | 12.0 s |
| `tiles_approx_lemma` | 94.9 s | 43.3 s |
| `reconstruct_from_warp_approx` | 185.1 s | 170.6 s |
| `block_teardown_to` | 123.3 s | 104.0 s |
| `Finish.finish` | 12.8 s | 2.0 s |

The code now selects the known-good initial fuel only for declarations where
that configuration works reliably. It also introduces `tile_flatten_div_mod`,
a small pure bridge that packages flatten/unflatten division and modulo facts
outside the large Pulse reconstruction context.

## The SMT-bound problem

`Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc.fst` already acknowledges that its preconditions and postconditions are poorly structured. It combines:

- Eight tile parameters plus `m`, `n`, `k`, block/thread counts, and many divisibility refinements.
- Raw nonlinear multiplication, division, and modulo arithmetic.
- Nested `ematrix_subtile`, `ematrix_from_tiles`, `wt_target`, `matmul`, and approximation predicates.
- Hundreds of leaf obligations under one large local context.

The concrete retry behavior is particularly wasteful:

- `warp_tile_i`: a 16.09-second failure at ifuel 1, followed by a 0.22-second success at ifuel 2.
- `c_subtile_approx_lemma`: failures of 15.50 and 19.59 seconds, followed by a 0.78-second success at fuel 4.
- The division assertion around line 979 of `KernelDesc.fst`: two retries consume 31.18 and 20.28 seconds; the same goal succeeds in 1.18 seconds at ifuel 2.
- `tiles_approx_lemma` has 219 labeled obligations and takes about 95 seconds.
- `Finish.finish` wastes another 10.5 seconds trying fuel 0 before succeeding almost immediately at fuel 1.

The `--retry 2` option around `reconstruct_from_warp_approx` is expensive, but
the declaration is a mixed workload: some leaves require more unfolding while
other leaves become harder with it. Removing the retries safely requires
isolating the hard `forevery_ext_2` leaf behind a specialized lemma.

### Immediate SMT fixes

- Select known-good initial configurations locally. This is now done for
  `warp_tile_i`, `rhs_is_constant_for_warps_approx`, `tiles_approx_lemma`,
  `Finish.finish`, and the out-of-place block setup/teardown.
- Do not globally raise rlimits. That merely makes doomed first attempts more expensive.
- Keep adaptive retries in `c_subtile_approx_lemma` and
  `reconstruct_from_warp_approx` until their hard leaves are extracted. Starting
  either whole declaration at its eventual retry fuel caused other leaves to
  time out.
- Replace universal division/modulo assertions with small top-level lemmas
  proved in a minimal context. This is now done by `tile_flatten_div_mod`; the
  existing `div_mod_of_mul_add` in teardown follows the same pattern.
- Split `tiles_approx_lemma` into:
  1. coordinate reconstruction,
  2. nested-tile indexing,
  3. matmul decomposition,
  4. approximation transport.
- Experiment with `--smtencoding.nl_arith_repr native` on pure coordinate modules, not globally.
- Use narrower `--using_facts_from` scopes where possible, though most pollution here is local, so this will not be transformative.

An attempted fuel-1 override for teardown's `gather_block` was also reverted:
the leaf succeeds at fuel 1 when retried after fuel 0, but times out when fuel 1
is its initial encoding. Fuel is not monotonic in practice because it changes
the full SMT problem, not just a numeric search allowance.

## The Pulse/elaboration problem

The out-of-place modules have the opposite profile: approximately 74% of runtime remains with SMT admitted. Even `--lax` remained in `gather_warp` for over a minute.

The worst examples are:

- `gather_warp`: 298 seconds normally, 265 seconds with SMT admitted.
- `gather_kernel_outputs`: 58 seconds normally, 54 seconds with SMT admitted.
- `split_output_to_lanes`: 136 seconds normally, 116 seconds with SMT admitted.
- `block_teardown_to`: 123 seconds normally, 100 seconds with SMT admitted.

Their Z3 goals are mostly reported as taking 0.00–0.05 seconds. The time is spent pushing large dependent lambdas containing `forall+`, `exists*`, `**`, refinements, and nested views through generic `forevery_map`, `forevery_factor`, `forevery_commute`, and related combinators.

Naming repeated matrix expressions already helped enormously: the previous `epilogue_warp_output` change reduced one proof from 52 minutes to seconds. More local `let` bindings will not fully solve the residual problem, however. The ownership transformation itself needs a nominal abstraction boundary.

## Recommended partition abstraction

Introduce an abstraction resembling:

```fstar
type tc2d_shape = ...
type block_coord shape = ...
type warp_coord shape = ...
type output_partition shape et gD target : slprop

ghost fn split_output_partition ...
  requires gD |-> eD
  ensures output_partition shape gD (tiles_of eD)

ghost fn gather_output_partition ...
  requires output_partition shape gD target
  ensures exists* eD. gD |-> eD ** pure (eD %~ target)
```

Internally, this abstraction can use the existing `forevery_*` machinery once. Kernel proofs should see an opaque `output_partition`, rather than repeatedly restating the block-to-warp-to-fragment-to-lane hierarchy.

Related improvements:

- Represent block, warp, fragment, and lane coordinates structurally, with pre-proved flatten/unflatten bijections. Stop recovering them everywhere with division and modulo.
- Separate ownership from functional approximation. Avoid threading `exists* em. ownership em ** pure (em %~ target)` through every spatial permutation.
- Add general laws such as `subtile_subtile`, `from_tiles_approximates`, `partition_roundtrip`, and approximation preservation. The recent `subtile_acc2` and `chest_comb_subtile` additions in `Kuiper.EMatrix.Tiling.fsti` are exactly the right direction.
- Put the abstraction behind an `.fsti` or module boundary so F* cannot repeatedly delta-reduce its representation.
- Split the 1,148-line descriptor and 973-line teardown into independently cached modules. This also exposes more parallelism.

## Should this move to Lean?

Moving to Lean is not a practical proof-performance fix.

Lean would likely handle the pure coordinate algebra pleasantly, but the dominant out-of-place cost is separation-logic ownership transport. Migrating would require rebuilding Pulse's GPU memory model, Kuiper's DSL, and the extraction pipeline. The same poorly structured predicates could also be slow in Lean.

The partition abstraction should first be redesigned in F*. If a broader rewrite is desired later, Lean becomes a reasonable architectural choice, but it is not a tactical remedy for the current performance problem.

## Generated specializations

`src/klas/Klas.GEMM.TensorCore2D.fst.sh` emits 144 instances in one module. This takes another approximately 30 seconds and produces a 15 MB checked file. The instances should be sharded by tile family or generated only for requested configurations during development builds.

## Suggested order of work

1. Add a repeatable performance harness that records normal time, admitted-SMT time, declaration timings, retry failures, and peak RSS.
2. Apply local fuel/ifuel changes only where the initial encoding is measured to
   succeed; retain adaptive retries for mixed declarations.
3. Extract division, modulo, flattening, and tile-coordinate facts into small pure lemmas.
4. Replace pointwise nested-tile proofs with reusable laws such as `subtile_subtile` and approximation homomorphisms.
5. Implement an opaque, reusable output-partition split/gather abstraction.
6. Split the large descriptor modules and shard generated Klas instances.
