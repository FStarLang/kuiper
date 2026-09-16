#!/usr/bin/env python3
"""List trust-related identifiers and verification-disabling F*/Pulse options."""

import re
import sys


TRUST_WORD = re.compile(
    r"(?<!\w)(?:assume_|assume|admit|tadmit|magic)(?!\w)|(?<!\w)--lax(?!\w)"
)
OPTION_DIRECTIVE = re.compile(r"#(?:set|push|reset)-options\b")
TRUST_OPTION = re.compile(r"(?<!\S)(?:--lax|--admit_smt_queries\s+true)(?=\s|$)")


def scan_source(source: str) -> tuple[str, list[tuple[int, str]]]:
    """Mask comments and strings, retaining option strings and their offsets."""
    masked = list(source)
    options = []
    depth = 0
    in_string = False
    escaped = False
    pending_options = False
    option_start = None
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
                if option_start is not None:
                    options.append((option_start, source[option_start:i]))
                    option_start = None
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
            if pending_options:
                option_start = i + 1
            pending_options = False
            i += 1
        elif source[i] == "#" and (directive := OPTION_DIRECTIVE.match(source, i)):
            pending_options = True
            i = directive.end()
        else:
            if not source[i].isspace():
                pending_options = False
            i += 1

    return "".join(masked), options


def list_file(path: str) -> None:
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()

    original_lines = source.splitlines()
    code, options = scan_source(source)
    option_lines = {
        source.count("\n", 0, offset + match.start()) + 1
        for offset, value in options
        for match in TRUST_OPTION.finditer(value)
    }
    for line_number, (original, code_line) in enumerate(
        zip(original_lines, code.splitlines()), start=1
    ):
        if TRUST_WORD.search(code_line) or line_number in option_lines:
            print(f"{path}:{line_number}:{original}")


def main() -> None:
    for path in sys.argv[1:]:
        list_file(path)


if __name__ == "__main__":
    main()
