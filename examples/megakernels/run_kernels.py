#!/usr/bin/env python3
"""Compile and smoke-test the retained reduced CUDA programs on synthetic inputs."""

import argparse
import datetime
import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import time

from replay import ROOT, inventory, sha256, write_json


def compile_kernel(torch, case, work, jobs):
    from torch.utils.cpp_extension import load

    source = ROOT / case["kernel"]
    name = case["id"].replace("-", "_") + "_" + sha256(source)[:12]
    directory = work / name
    directory.mkdir(parents=True, exist_ok=True)
    parameters = case["parameters"]
    if any(parameter not in ("torch::Tensor", "int64_t", "bool") for parameter in parameters):
        raise ValueError(f"Unsupported kernel parameter: {parameters}")
    binding = (
        "#include <torch/extension.h>\n"
        f"std::vector<torch::Tensor> kplay_run({', '.join(parameters)});\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"run\", &kplay_run); }\n"
    )
    path = directory / "binding.cpp"
    if not path.exists() or path.read_text() != binding:
        path.write_text(binding)
    os.environ["MAX_JOBS"] = str(jobs)
    started = time.monotonic()
    module = load(
        name=name, sources=[str(source), str(path)], build_directory=str(directory),
        extra_cflags=["-O2"], extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    return module, time.monotonic() - started


def invoke(torch, module, inputs):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall = time.monotonic()
    start.record()
    outputs = module.run(*inputs)
    finish.record()
    finish.synchronize()
    return outputs, {"cuda_ms": start.elapsed_time(finish),
                     "wall_seconds": time.monotonic() - wall}


def flashmoe_smoke(torch, module, device):
    observations = []
    for fixture in ("ties-and-overflow", "balanced-routing"):
        # Contiguous offset-one BF16 views exercise the stated native input scope.
        tokens = torch.zeros(256 * 512 + 1, dtype=torch.bfloat16, device=device)[1:].view(256, 512)
        weights = torch.zeros(8 * 512 + 1, dtype=torch.bfloat16, device=device)[1:].view(8, 512)
        selected = torch.zeros(256, dtype=torch.int64, device=device)
        if fixture == "balanced-routing":
            selected = torch.arange(256, device=device) % 8
            tokens[torch.arange(256, device=device), selected] = 1
            weights[torch.arange(8, device=device), torch.arange(8, device=device)] = 4
        before = (tokens.clone(), weights.clone())
        outputs, timing = invoke(torch, module, [tokens, weights])
        if len(outputs) != 4:
            raise AssertionError("Expected demand, indices, probabilities, and survival.")
        expected = [
            torch.bincount(selected, minlength=8).to(torch.int32),
            torch.full((8, 32), -1, dtype=torch.int64, device=device),
            torch.zeros((8, 32), dtype=torch.float32, device=device),
            torch.zeros((256, 8), dtype=torch.int32, device=device),
        ]
        for expert in range(8):
            admitted = torch.nonzero(selected == expert).flatten()[:32]
            expected[1][expert, :admitted.numel()] = admitted
            expected[2][expert, :admitted.numel()] = 1
            expected[3][admitted, expert] = 1
        for output, reference in zip(outputs, expected):
            if output.dtype != reference.dtype or not torch.equal(output, reference):
                raise AssertionError(f"Incorrect FlashMoE output for {fixture}.")
        if not torch.equal(tokens, before[0]) or not torch.equal(weights, before[1]):
            raise AssertionError("Input storage changed.")
        observations.append({"fixture": fixture, "status": "passed", **timing})
    return observations


def model_smoke(torch, module, project, device):
    mirage = project == "mirage"
    count = 596049920 if mirage else 752393024
    offset = 0 if mirage else 1
    weights = torch.zeros(count + offset, dtype=torch.bfloat16, device=device)[offset:]
    prompt = torch.arange(7 if mirage else 1, dtype=torch.int64, device=device)
    original_prompt = prompt.clone()
    outputs, timing = invoke(torch, module, [weights, prompt, 1])
    if len(outputs) != 2:
        raise AssertionError("Expected complete token and hidden outputs.")
    tokens, hidden = outputs
    token_shape = (1, 8) if mirage else (1,)
    hidden_shape = (7, 1024) if mirage else (1, 1024)
    if tuple(tokens.shape) != token_shape or tokens.dtype != torch.int64:
        raise AssertionError("Incorrect token shape or dtype.")
    if tuple(hidden.shape) != hidden_shape or hidden.dtype != torch.bfloat16:
        raise AssertionError("Incorrect hidden shape or dtype.")
    vocabulary = 151936 if mirage else 248320
    if not bool(((tokens >= 0) & (tokens < vocabulary)).all()):
        raise AssertionError("Generated token is outside the vocabulary.")
    if mirage and not torch.equal(tokens[0, :7], original_prompt):
        raise AssertionError("Prompt prefix changed.")
    if torch.count_nonzero(hidden).item() != 0:
        raise AssertionError("Zero weights should produce zero returned hidden values.")
    if torch.count_nonzero(weights).item() != 0 or not torch.equal(prompt, original_prompt):
        raise AssertionError("Input storage changed.")
    # The specification permits ties; do not require equal tokens between roles.
    return [{"fixture": "zero-weights", "status": "passed", "tokens": tokens.cpu().tolist(),
             "weight_storage_offset": offset, **timing}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=("all", "flashmoe", "mirage", "luce"), default="flashmoe")
    parser.add_argument("--role", choices=("both", "base", "optimized"), default="both")
    parser.add_argument("--kind", choices=("program", "family", "all"), default="program")
    parser.add_argument("--case", help="Case ID or shell-style pattern")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent compiler processes")
    parser.add_argument("--work", type=Path, default=ROOT / ".work/gpu")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    manifest = inventory()
    cases = [
        case for case in manifest["cases"]
        if case["kind"] in ("program", "family")
        and (args.kind == "all" or case["kind"] == args.kind)
        and (args.project == "all" or case["project"] == args.project)
        and (args.role == "both" or case["id"].endswith("-" + args.role))
        and (not args.case or fnmatch.fnmatchcase(case["id"], args.case))
    ]
    if not cases:
        parser.error("No matching CUDA cases.")
    if not args.compile_only and any(case["kind"] == "family" for case in cases):
        parser.error("Family entry points require caller state; use --compile-only, "
                     "or run the complete-program smoke tests.")
    if "CUDA_HOME" not in os.environ and shutil.which("nvcc"):
        os.environ["CUDA_HOME"] = str(Path(shutil.which("nvcc")).resolve().parent.parent)
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install the documented GPU dependencies.")
    if CUDA_HOME is None:
        parser.error("CUDA toolkit not found; set CUDA_HOME to its installation.")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] < 8:
        parser.error("Native BF16 MMA requires compute capability 8.0 or newer.")
    device = torch.device("cuda", args.device)
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    results = {
        "status": "running", "requested_cases": [case["id"] for case in cases],
        "mode": "compile-only" if args.compile_only else "smoke-test",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "nvcc_version": subprocess.check_output(
            [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "scope": "synthetic reduced-CUDA smoke tests; not proof extraction or original/reduced equivalence",
        "compile_flags": ["-O2", "--expt-relaxed-constexpr"], "cases": [],
    }
    write_json(args.work / "results.json", results)
    for case in cases:
        print(f"Compiling {case['id']}", flush=True)
        module, compile_seconds = compile_kernel(torch, case, args.work, args.jobs)
        observations = [] if args.compile_only else (
            flashmoe_smoke(torch, module, device) if case["project"] == "flashmoe"
            else model_smoke(torch, module, case["project"], device)
        )
        results["cases"].append({
            "id": case["id"], "kernel_sha256": sha256(ROOT / case["kernel"]),
            "compile_seconds": compile_seconds, "observations": observations,
            "status": "compiled" if args.compile_only else "passed",
        })
        write_json(args.work / "results.json", results)
    results["status"] = "passed"
    results["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(args.work / "results.json", results)
    print(f"Results: {args.work / 'results.json'}")


if __name__ == "__main__":
    main()
