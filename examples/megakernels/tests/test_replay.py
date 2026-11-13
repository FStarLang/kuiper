import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "proofs/case/Example.fst"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("module Example\n")
        self.manifest = {
            "files": {"proofs/case/Example.fst": replay.sha256(self.source)},
            "cases": [{"id": "case", "kind": "family", "profile": "profile", "directory": "proofs/case",
                       "roots": ["proofs/case/Example.fst"]}],
        }
        replay.write_json(self.root / "manifest.json", self.manifest)

    def test_changed_source_is_rejected(self):
        self.source.write_text("module Example\nlet changed = 0\n")
        with patch.object(replay, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                replay.inventory()

    def test_unindexed_dependency_is_rejected(self):
        (self.source.parent / "Unindexed.fst").write_text("module Unindexed\n")
        with patch.object(replay, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                replay.inventory()

    def test_missing_case_does_not_prepare_toolchain(self):
        args = argparse.Namespace(kind="all", case=["missing-*"])
        with patch.object(replay, "setup") as setup:
            with self.assertRaisesRegex(ValueError, "No matching cases"):
                replay.verify(args, self.manifest)
            setup.assert_not_called()

    def test_failed_verifier_is_recorded_and_propagated(self):
        output = self.root / "output"
        output.mkdir()
        config = output / "command.json"
        replay.write_json(config, {
            "output": str(output),
            "command": [sys.executable, "-c", "print('deliberate failure'); exit(7)",
                        "--cache_dir", "unused", "--odir", "unused"],
        })
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = replay.check_module(config, self.source)
        self.assertEqual(status, 7)
        record = replay.read_json(output / "Example.fst.json")
        self.assertEqual(record["returncode"], 7)
        self.assertIn("deliberate failure", (output / "Example.fst.log").read_text())

    def test_failed_worker_cannot_reuse_a_successful_record(self):
        output = self.root / "output"
        output.mkdir()
        replay.write_json(output / "Example.fst.json", {"returncode": 0, "module": "Example.fst"})
        with patch.object(replay.subprocess, "call", return_value=7):
            result = replay.check_set(self.root, "profile", [self.source], output, 1)
        self.assertEqual(result["status"], "failed")

    def test_non_utf8_failure_log_preserves_verifier_status(self):
        output = self.root / "output"
        output.mkdir()
        config = output / "command.json"
        replay.write_json(config, {
            "output": str(output),
            "command": [sys.executable, "-c",
                        "import sys; sys.stdout.buffer.write(b'failure: \\xff'); exit(7)",
                        "--cache_dir", "unused", "--odir", "unused"],
        })
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            self.assertEqual(replay.check_module(config, self.source), 7)
        self.assertIn("failure:", errors.getvalue())
        self.assertEqual((output / "Example.fst.log").read_bytes(), b"failure: \xff")

    def test_duplicate_case_ids_are_rejected(self):
        self.manifest["cases"].append(self.manifest["cases"][0])
        replay.write_json(self.root / "manifest.json", self.manifest)
        with patch.object(replay, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                replay.inventory()

    def test_interrupted_run_is_not_marked_complete(self):
        work = self.root / "work"
        work.mkdir()
        replay.write_json(work / "prepared.json", {})
        args = argparse.Namespace(kind="all", case=None, work=work, jobs=1)
        with patch.object(replay, "setup"), patch.object(replay, "input_fingerprint", return_value="test"):
            with patch.object(replay, "check_set", side_effect=KeyboardInterrupt):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                    replay.verify(args, self.manifest)
        result = replay.read_json(next((work / "runs").glob("*/results.json")))
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["requested_cases"], ["case"])
        self.assertEqual(result["cases"], [])


if __name__ == "__main__":
    unittest.main()
