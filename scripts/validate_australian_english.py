#!/usr/bin/env python3
"""
Validates that markdown documentation files adhere to Australian English spelling conventions.
Skips code blocks (fenced triple-backtick blocks) and inline code snippets (`...`).
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Mapping of American spellings (regex pattern) to Australian English recommendation
US_TO_AU_MAP = {
    # -ize / -ization -> -ise / -isation
    r"\boptimiz(e|es|ed|ing|ation|ations)\b": "optimis\\1",
    r"\bquantiz(e|es|ed|ing|ation|ations)\b": "quantis\\1",
    r"\binitializ(e|es|ed|ing|ation|ations)\b": "initialis\\1",
    r"\bserializ(e|es|ed|ing|ation|ations)\b": "serialis\\1",
    r"\bspecializ(e|es|ed|ing|ation|ations)\b": "specialis\\1",
    r"\bcentraliz(e|es|ed|ing|ation|ations)\b": "centralis\\1",
    r"\bmodulariz(e|es|ed|ing|ation|ations)\b": "modularis\\1",
    r"\bstandardiz(e|es|ed|ing|ation|ations)\b": "standardis\\1",
    r"\bcategoriz(e|es|ed|ing|ation|ations)\b": "categoris\\1",
    r"\bsynchroniz(e|es|ed|ing|ation|ations)\b": "synchronis\\1",
    r"\bvectoriz(e|es|ed|ing|ation|ations)\b": "vectoris\\1",
    r"\bprioritiz(e|es|ed|ing|ation|ations)\b": "prioritis\\1",
    r"\bcustomiz(e|es|ed|ing|ation|ations)\b": "customis\\1",
    r"\butiliz(e|es|ed|ing|ation|ations)\b": "utilis\\1",
    r"\btokeniz(e|es|ed|ing|er|ers|ation|ations)\b": "tokenis\\1",
    r"\brecogniz(e|es|ed|ing|ation|ations)\b": "recognis\\1",
    r"\bharmoniz(e|es|ed|ing|ation|ations)\b": "harmonis\\1",
    r"\bmaximiz(e|es|ed|ing|ation|ations)\b": "maximis\\1",
    r"\bminimiz(e|es|ed|ing|ation|ations)\b": "minimis\\1",
    
    # -or -> -our
    r"\bbehavior(s)?\b": "behaviour\\1",
    r"\bcolor(s)?\b": "colour\\1",
    r"\bflavor(s)?\b": "flavour\\1",
    
    # -er -> -re
    r"\bcenter\b": "centre",
    r"\bcenters\b": "centres",
    r"\bcentered\b": "centred",
    r"\bcentering\b": "centring",
    
    # artifact -> artefact
    r"\bartifact(s)?\b": "artefact\\1",
    
    # license (noun) -> licence
    r"\b(Apache|MIT|BSD|GPL|software|the|a|an|under)\s+license\b": "\\1 licence",
    r"\bLicense\b": "Licence",
    
    # -og -> -ogue
    r"\banalog(s)?\b": "analogue\\1",
    r"\bcatalog(s)?\b": "catalogue\\1",
    r"\bdialog(s)?\b": "dialogue\\1",
}

COMPILED_RULES = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in US_TO_AU_MAP.items()]

def check_text(content: str) -> List[Dict]:
    """
    Checks text for American English spellings while ignoring code blocks and inline code.
    Returns a list of violations with line number, matched word, and suggestion.
    """
    violations = []
    lines = content.splitlines()
    in_code_block = False

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Strip inline code snippets (`...`)
        line_no_inline = re.sub(r"`[^`]*`", "", line)
        # Strip markdown URLs in links: [text](http...)
        line_no_links = re.sub(r"\]\([^)]+\)", "]", line_no_inline)

        for pattern, replacement in COMPILED_RULES:
            for match in pattern.finditer(line_no_links):
                matched_str = match.group(0)
                # Compute recommendation
                suggestion = pattern.sub(replacement, matched_str)
                violations.append({
                    "line": line_idx,
                    "matched": matched_str,
                    "suggestion": suggestion,
                    "context": line.strip()
                })

    return violations

def scan_file(file_path: Path) -> List[Dict]:
    """Scans a file and returns violations."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []
    violations = check_text(content)
    for v in violations:
        v["file"] = str(file_path)
    return violations

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_australian_english.py <path-to-file-or-dir> ...", file=sys.stderr)
        sys.exit(2)

    files_to_scan = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file():
            if p.suffix.lower() == ".md":
                files_to_scan.append(p)
        elif p.is_dir():
            files_to_scan.extend(p.rglob("*.md"))

    total_violations = []
    for file_path in sorted(set(files_to_scan)):
        # Ignore files inside hidden dirs or specs/plans that record exact prompts or external filenames
        if "/.superpowers/" in str(file_path) or "/.git/" in str(file_path) or "/.ort/" in str(file_path):
            continue
        file_violations = scan_file(file_path)
        if file_violations:
            total_violations.extend(file_violations)

    if total_violations:
        print(f"FAILED: Found {len(total_violations)} American English spelling violation(s):\n")
        for v in total_violations:
            print(f"  {v['file']}:{v['line']}: '{v['matched']}' -> suggested '{v['suggestion']}'")
            print(f"    Context: {v['context']}\n")
        sys.exit(1)
    else:
        print(f"PASSED: All {len(files_to_scan)} markdown file(s) adhere to Australian English spelling.")
        sys.exit(0)

if __name__ == "__main__":
    main()
