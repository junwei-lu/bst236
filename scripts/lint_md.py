#!/usr/bin/env python3
"""Lint Markdown files for Python-Markdown compatibility.

Rules:
  1. A blank line is required before a list item when preceded by a non-list line.
  2. Nested list items must use 4-space indentation (multiples of 4).
"""

import os
import re
import sys


def is_list_item(line):
    """Return True if line (after stripping leading spaces) starts with a list marker."""
    s = line.lstrip()
    return bool(re.match(r"[-*+] ", s) or re.match(r"\d+\. ", s))


def leading_spaces(line):
    """Count leading spaces on a line."""
    return len(line) - len(line.lstrip(" "))


def _skip_prev_line(prev):
    """Return True if prev is a block element that doesn't need a blank line before a list."""
    s = prev.strip()
    if not s:
        return True
    if s.startswith("#"):
        return True
    if s.startswith("!!!") or s.startswith("???"):
        return True
    if s.startswith(">"):
        return True
    if re.match(r"^-{3,}$", s) or re.match(r"^\*{3,}$", s) or re.match(r"^_{3,}$", s):
        return True
    return False


_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)?$")


def lint_file(filepath):
    """Return list of (line_number, message) for issues found in *filepath*."""
    issues = []

    with open(filepath, encoding="utf-8") as f:
        lines = [l.rstrip("\n\r") for l in f]

    # Skip YAML front matter
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                start = i + 1
                break

    in_fence = False
    fence_char = None
    fence_count = 0

    for i in range(start, len(lines)):
        line = lines[i]

        # Track fenced code blocks
        m = _FENCE_RE.match(line)
        if m:
            char = m.group(2)[0]
            count = len(m.group(2))
            info = (m.group(3) or "").strip()
            if not in_fence:
                in_fence = True
                fence_char = char
                fence_count = count
                continue
            else:
                if char == fence_char and count >= fence_count and not info:
                    in_fence = False
                    continue

        if in_fence:
            continue

        # Rule 1: blank line required before a list item
        if is_list_item(line) and i > 0:
            prev = lines[i - 1]
            if not _skip_prev_line(prev) and not is_list_item(prev):
                issues.append(
                    (i + 1, f"Missing blank line before list item: {line.strip()}")
                )

        # Rule 2: nested list indentation must be a multiple of 4
        if is_list_item(line):
            spaces = leading_spaces(line)
            if spaces > 0 and spaces % 4 != 0:
                issues.append(
                    (
                        i + 1,
                        f"List indented by {spaces} spaces (must be multiple of 4): {line.strip()}",
                    )
                )

    return issues


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(2)

    search_dir = sys.argv[1]
    total = 0

    for root, _dirs, files in os.walk(search_dir):
        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue
            path = os.path.join(root, fname)
            for lineno, msg in lint_file(path):
                print(f"{path}:{lineno}: {msg}")
                total += 1

    if total:
        print(f"\n{total} issue(s) found.")
        sys.exit(1)
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
