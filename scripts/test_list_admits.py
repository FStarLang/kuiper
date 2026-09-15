"""Regression tests for the list-admits command's source scanning."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("list-admits.py")


class ListAdmitsTests(unittest.TestCase):
    def assert_listing(self, source, expected_lines, suffix=".fst"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("Test Input" + suffix)
            path.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = source.splitlines()
            self.assertEqual(
                result.stdout.splitlines(),
                [f"{path}:{line}:{lines[line - 1]}" for line in expected_lines],
            )
            self.assertEqual(result.stderr, "")

    def test_verification_disabling_options(self):
        self.assert_listing(
            '#set-options "--admit_smt_queries true"\n'
            '#push-options "--z3rlimit 40 --lax"\n'
            '#reset-options "--admit_smt_queries true --fuel 0"\n',
            [1, 2, 3],
            suffix=".fsti",
        )

    def test_safe_options_and_ordinary_strings(self):
        self.assert_listing(
            '#set-options "--admit_smt_queries false"\n'
            '#push-options "--z3rlimit 40"\n'
            '#set-options "--debug magic"\n'
            'let message = "admit magic assume --lax --admit_smt_queries true"\n'
            'let ordinary_options = "--lax"\n',
            [],
        )

    def test_nested_comments_and_commented_directives(self):
        self.assert_listing(
            '(* #set-options "--lax"\n'
            '   (* admit () #push-options "--admit_smt_queries true" *)\n'
            '   magic () *)\n'
            '// #reset-options "--lax"\n'
            'let proof () = admit () // another admit\n',
            [5],
        )

    def test_comments_and_newlines_before_option_string(self):
        self.assert_listing(
            '#push-options (* comment (* nested *) *)\n'
            '  "  --admit_smt_queries\ttrue --fuel 0"\n'
            '#set-options // comment\n'
            '  "--lax"\n',
            [2, 4],
        )

    def test_escaped_quotes_do_not_expose_strings(self):
        self.assert_listing(
            r'let text = "#set-options \"--lax\" and \\ admit"' + '\n'
            '#set-options "--admit_smt_queries true"\n',
            [2],
        )

    def test_argumentless_directive_does_not_capture_later_string(self):
        self.assert_listing(
            '#push-options\n'
            'let message = "--lax"\n'
            '#reset-options\n'
            'let text = "--admit_smt_queries true"\n',
            [],
        )

    def test_option_boundaries_and_one_report_per_line(self):
        self.assert_listing(
            '#set-options "--lax-extra"\n'
            '#set-options "--admit_smt_queries trueish"\n'
            '#set-options "prefix--lax"\n'
            '#set-options "--lax --admit_smt_queries true"\n'
            'let proof () = admit (); magic ()\n',
            [4, 5],
        )

    def test_existing_trust_identifiers(self):
        self.assert_listing(
            'assume val axiom : unit\n'
            'assume_ (); admit (); tadmit (); magic ()\n'
            'let admitted = 0\n'
            'let magic_number = 1\n'
            'let reassume = 2\n',
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
