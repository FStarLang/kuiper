#!/usr/bin/env python3

"""
Find unused top-level definitions (let, val, type, instance) in Kuiper F*/Pulse source files.

For each definition, checks whether its name appears anywhere else in the codebase
(outside its own definition site and its val/let counterpart). Definitions that are
never referenced externally are reported.

By default, excludes:
  - Files under src/examples/ (standalone demos)
  - Files under src/lib/inst/ (auto-generated kernel instantiations)
  - Auto-generated files (containing "AUTOMATICALLY GENERATED")
  - Private definitions
  - Definitions starting with __
  - val declarations in .fsti files (interface exports)
  - SMTPat lemmas (used implicitly by Z3)
  - Short identifiers (< 3 chars)

Usage:
    scripts/find-unused.py [options] [directory ...]

    If no directories given, defaults to src/ extraction/.

Examples:
    scripts/find-unused.py                  # default scan
    scripts/find-unused.py --all            # include everything
    scripts/find-unused.py --json           # machine-readable output
    scripts/find-unused.py src/lib/kuiper/  # scan only core library
"""

import argparse
import os
import re
import sys
import json
from collections import defaultdict


QUALIFIERS = r'(?:(?:inline_for_extraction|noextract|unfold|irreducible|private|noeq|abstract|rec)\s+)*'
DEF_RE = re.compile(
    r'^(\s*)' +                          # leading whitespace (group 1)
    r'(?:and\s+|' +                      # 'and' for mutual recursion, or:
    QUALIFIERS +                         # optional qualifiers
    r'(let|val|type|instance)\s+)' +     # keyword (group 2)
    r'(\w+)',                            # definition name (group 3)
    re.MULTILINE
)

MODULE_RE = re.compile(r'^\s*module\s+([\w.]+)', re.MULTILINE)
GENERATED_RE = re.compile(r'AUTOMATICALLY GENERATED|THIS FILE IS AUTO', re.IGNORECASE)

SKIP_NAMES = {
    '_', 'rec', 'mut', 'x', 'y', 'z', 'i', 'j', 'k', 'n', 'm', 'f', 'g', 'h',
    'a', 'b', 'c', 'd', 'e', 's', 't', 'v', 'p', 'q', 'r', 'w',
    'aux', 'main', 'op', 'id', 'eq',
    'requires', 'ensures', 'returns', 'exists_', 'forall_',
}


def module_files(filepath):
    """Return the .fst and .fsti paths for a given file."""
    if filepath.endswith('.fsti'):
        return filepath[:-1], filepath
    return filepath, filepath + 'i'


def get_module_name(content):
    m = MODULE_RE.search(content)
    return m.group(1) if m else None


def is_generated(content):
    return bool(GENERATED_RE.search(content[:500]))


def is_excluded_path(filepath, args):
    if not args.include_examples and '/examples/' in filepath:
        return True
    if not args.include_inst and '/inst/' in filepath:
        return True
    return False


def extract_definitions(filepath, content, args):
    defs = []
    module_name = get_module_name(content)
    is_fsti = filepath.endswith('.fsti')
    lines_cache = None

    for m in DEF_RE.finditer(content):
        indent = m.group(1)
        keyword = m.group(2)
        name = m.group(3)

        if len(indent) > 2:
            continue
        if name in SKIP_NAMES:
            continue
        if len(name) < args.min_length:
            continue
        if name.startswith('__') and not args.include_underscored:
            continue

        full_match = m.group(0)
        if 'private' in full_match and not args.include_private:
            continue
        if keyword == 'val' and is_fsti and not args.include_vals:
            continue
        if keyword == 'instance':
            continue

        line_no = content[:m.start()].count('\n') + 1

        # Check for attributes on the preceding line(s) or same line
        # Look back to the previous blank line or definition
        prev_lines_start = content.rfind('\n\n', 0, m.start())
        if prev_lines_start == -1:
            prev_lines_start = 0
        attr_context = content[prev_lines_start:m.end()]
        if 'coercion' in attr_context or '@@"public"' in attr_context:
            continue

        defs.append({
            'name': name,
            'keyword': keyword or 'and',
            'file': filepath,
            'module': module_name,
            'line': line_no,
            'is_private': 'private' in full_match,
        })

    return defs


def find_fstar_files(dirs):
    files = []
    for d in dirs:
        for root, _, filenames in os.walk(d):
            for fn in filenames:
                if fn.endswith('.fst') or fn.endswith('.fsti'):
                    files.append(os.path.join(root, fn))
    return sorted(files)


def build_def_lines(all_defs):
    """Build a map of (file, line) -> True for all definition sites."""
    sites = set()
    for d in all_defs:
        sites.add((d['file'], d['line']))
    return sites


def count_references(name, all_contents, own_module_files, def_sites):
    """
    Count references to `name` across all files, excluding:
    - definition sites (any file+line in def_sites for this name)
    - the module's own .fst/.fsti pair (val/let counterparts)

    Returns (external_refs, internal_refs, external_files).
    """
    pattern = re.compile(r'\b' + re.escape(name) + r'\b')

    ext_count = 0
    int_count = 0
    ext_files = set()

    for filepath, content in all_contents.items():
        is_own = filepath in own_module_files
        for m in pattern.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            if (filepath, line_no) in def_sites:
                continue
            if is_own:
                int_count += 1
            else:
                ext_count += 1
                ext_files.add(filepath)

    return ext_count, int_count, ext_files


def is_smtpat_def(content, line_no):
    lines = content.split('\n')
    start = max(0, line_no - 1)
    end = min(len(lines), line_no + 25)
    chunk = '\n'.join(lines[start:end])
    return 'SMTPat' in chunk or 'smt_pat' in chunk


def main():
    parser = argparse.ArgumentParser(
        description='Find unused definitions in Kuiper F*/Pulse code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('dirs', nargs='*', default=['src/', 'extraction/'],
                        help='Directories to scan (default: src/ extraction/)')

    filt = parser.add_argument_group('filtering')
    filt.add_argument('--all', action='store_true',
                      help='Include everything (examples, inst, generated, private, etc.)')
    filt.add_argument('--include-examples', action='store_true',
                      help='Include src/examples/ files')
    filt.add_argument('--include-inst', action='store_true',
                      help='Include src/lib/inst/ files (auto-generated instantiations)')
    filt.add_argument('--include-generated', action='store_true',
                      help='Include auto-generated files')
    filt.add_argument('--include-private', action='store_true',
                      help='Include private definitions')
    filt.add_argument('--include-underscored', action='store_true',
                      help='Include __ prefixed definitions')
    filt.add_argument('--include-vals', action='store_true',
                      help='Include val declarations in .fsti files')
    filt.add_argument('--include-smtpat', action='store_true',
                      help='Include SMTPat lemmas (auto-triggered by Z3)')
    filt.add_argument('--min-length', type=int, default=3,
                      help='Minimum identifier length (default: 3)')

    out = parser.add_argument_group('output')
    out.add_argument('--json', action='store_true',
                     help='Output as JSON')
    out.add_argument('--verbose', action='store_true',
                     help='Show reference counts for all definitions')

    args = parser.parse_args()

    if args.all:
        args.include_examples = True
        args.include_inst = True
        args.include_generated = True
        args.include_private = True
        args.include_underscored = True
        args.include_vals = True
        args.include_smtpat = True

    files = find_fstar_files(args.dirs)
    if not files:
        print(f"No .fst/.fsti files found in {args.dirs}", file=sys.stderr)
        sys.exit(1)

    # Read all files
    all_contents = {}
    for f in files:
        try:
            all_contents[f] = open(f).read()
        except Exception as e:
            print(f"Warning: cannot read {f}: {e}", file=sys.stderr)

    # Extract definitions, skipping excluded files for definition extraction
    # (but keeping all files for reference counting)
    all_defs = []
    skipped_files = 0
    for f, content in all_contents.items():
        if is_excluded_path(f, args):
            skipped_files += 1
            continue
        if not args.include_generated and is_generated(content):
            skipped_files += 1
            continue
        all_defs.extend(extract_definitions(f, content, args))

    def_sites = build_def_lines(all_defs)

    print(f"Scanned {len(files)} files ({skipped_files} excluded), "
          f"found {len(all_defs)} definitions to check", file=sys.stderr)

    # Group definitions by (module, name) to handle val/let pairs
    by_key = defaultdict(list)
    for defn in all_defs:
        by_key[(defn['module'], defn['name'])].append(defn)

    unused = []
    for (module, name), defs_group in by_key.items():
        own_files = set()
        for d in defs_group:
            fst, fsti = module_files(d['file'])
            own_files.add(fst)
            own_files.add(fsti)

        ext_refs, int_refs, ext_files = count_references(
            name, all_contents, own_files, def_sites
        )

        for d in defs_group:
            d['ext_refs'] = ext_refs
            d['int_refs'] = int_refs
            d['ext_files'] = sorted(ext_files)

        if ext_refs == 0 and int_refs == 0:
            # No references at all
            rep = defs_group[0]

            if not args.include_smtpat and is_smtpat_def(
                all_contents[rep['file']], rep['line']
            ):
                continue

            unused.append(rep)

    # Sort by file and line
    unused.sort(key=lambda d: (d['file'], d['line']))

    if args.json:
        result = []
        for defn in unused:
            result.append({
                'name': defn['name'],
                'keyword': defn['keyword'],
                'file': defn['file'],
                'line': defn['line'],
                'module': defn['module'],
            })
        print(json.dumps(result, indent=2))
    else:
        if not unused:
            print("No unused definitions found.")
        else:
            print(f"\nFound {len(unused)} potentially unused definitions:\n")
            current_file = None
            for defn in unused:
                if defn['file'] != current_file:
                    current_file = defn['file']
                    print(f"  {current_file}:")
                print(f"    {defn['line']:4d}: {defn['keyword']} {defn['name']}")
            print()
            print(f"Total: {len(unused)} unused definitions across "
                  f"{len(set(d['file'] for d in unused))} files")

    if args.verbose:
        print("\n--- All definitions with reference counts ---", file=sys.stderr)
        for defn in sorted(all_defs, key=lambda d: d.get('ext_refs', 0)):
            print(f"  {defn.get('ext_refs', '?'):>4} ext, {defn.get('int_refs', '?'):>4} int: "
                  f"{defn['keyword']} {defn['name']} "
                  f"({defn['file']}:{defn['line']})", file=sys.stderr)


if __name__ == '__main__':
    main()
