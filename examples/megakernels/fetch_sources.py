#!/usr/bin/env python3
"""Fetch optional upstream inspection checkouts at the recorded exact revisions."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from replay import ROOT, read_json


def git(directory, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


def main():
    sources = read_json(ROOT / "sources.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=["all", *sources], default="all")
    parser.add_argument("--directory", type=Path, default=ROOT / ".work/originals")
    parser.add_argument("--submodules", action="store_true")
    args = parser.parse_args()
    for name, source in sources.items():
        if args.project not in ("all", name):
            continue
        destination = args.directory.resolve() / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{name}-") as temporary:
                git(temporary, "init", "--quiet")
                git(temporary, "remote", "add", "origin", source["repository"])
                git(temporary, "fetch", "--quiet", "--depth=1", "--no-tags",
                    "origin", source["commit"])
                git(temporary, "-c", "advice.detachedHead=false", "checkout", "--quiet",
                    "--detach", source["commit"])
                Path(temporary).rename(destination)
        if git(destination, "rev-parse", "HEAD") != source["commit"]:
            raise RuntimeError(f"Different revision at {destination}; use another directory.")
        if git(destination, "status", "--porcelain"):
            raise RuntimeError(f"Modified checkout at {destination}; no changes were overwritten.")
        if args.submodules:
            git(destination, "submodule", "update", "--init", "--recursive", "--depth=1")
        print(f"{name}: {source['commit']} ({destination})")


if __name__ == "__main__":
    main()
