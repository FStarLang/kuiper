#!/usr/bin/env python3
"""List trust-related identifiers outside F*/Pulse comments and strings."""

import re
import sys


TRUST_WORD = re.compile(
    r"(?<!\w)(?:assume_|assume|admit|tadmit|magic)(?!\w)|(?<!\w)--lax(?!\w)"
)


def without_comments_and_strings(source: str) -> str:
    """Mask nested OCaml comments, line comments, and string literals."""
    masked = list(source)
    depth = 0
    in_string = False
    escaped = False
    i = 0

    while i < len(source):
        pair = source[i : i + 2]

        if depth:
            if pair == "(*":
                masked[i : i + 2] = "  "
                depth += 1
                i += 2
            elif pair == "*)":
                masked[i : i + 2] = "  "
                depth -= 1
                i += 2
            else:
                if source[i] != "\n":
                    masked[i] = " "
                i += 1
            continue

        if in_string:
            if source[i] != "\n":
                masked[i] = " "
            if escaped:
                escaped = False
            elif source[i] == "\\":
                escaped = True
            elif source[i] == '"':
                in_string = False
            i += 1
            continue

        if pair == "(*":
            masked[i : i + 2] = "  "
            depth = 1
            i += 2
        elif pair == "//":
            while i < len(source) and source[i] != "\n":
                masked[i] = " "
                i += 1
        elif source[i] == '"':
            masked[i] = " "
            in_string = True
            i += 1
        else:
            i += 1

    return "".join(masked)


def list_file(path: str) -> None:
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()

    original_lines = source.splitlines()
    code_lines = without_comments_and_strings(source).splitlines()
    for line_number, (original, code) in enumerate(
        zip(original_lines, code_lines), start=1
    ):
        if TRUST_WORD.search(code):
            print(f"{path}:{line_number}:{original}")


def main() -> None:
    for path in sys.argv[1:]:
        list_file(path)


if __name__ == "__main__":
    main()
