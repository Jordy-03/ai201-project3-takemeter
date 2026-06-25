#!/usr/bin/env python3
"""
build_csv.py — turn data/reviews_raw.txt into a clean data/data.csv.

Reads the paste-friendly block format (see data/reviews_raw.txt), validates
labels, joins multi-line review text into one field, and writes a proper CSV
with columns: text, label, notes, game. Handles all the quoting for you.

Usage:
    python scripts/build_csv.py
    python scripts/build_csv.py --in data/reviews_raw.txt --out data/data.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

VALID = {"positive", "negative"}


def parse_blocks(raw: str):
    blocks, current, in_text = [], None, False
    for line in raw.splitlines():
        if line.strip() == "@@@":
            if current is not None:
                blocks.append(current)
            current = {"game": "", "label": "", "notes": "", "text": []}
            in_text = False
            continue
        if current is None:
            continue                      # preamble before first @@@
        if not in_text and line.lstrip().startswith("#"):
            continue                      # comment line
        stripped = line.strip()
        if not in_text and stripped.lower().startswith("game:"):
            current["game"] = stripped[5:].strip()
        elif not in_text and stripped.lower().startswith("label:"):
            current["label"] = stripped[6:].strip().lower()
        elif not in_text and stripped.lower().startswith("notes:"):
            current["notes"] = stripped[6:].strip()
        elif not in_text and stripped.lower().startswith("text:"):
            in_text = True
            rest = line.split(":", 1)[1].strip()
            if rest:
                current["text"].append(rest)
        elif in_text:
            current["text"].append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def clean_text(lines):
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/reviews_raw.txt")
    ap.add_argument("--out", default="data/data.csv")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        sys.exit(f"Input not found: {args.inp}")

    with open(args.inp, encoding="utf-8") as f:
        blocks = parse_blocks(f.read())

    rows, skipped, errors = [], 0, []
    for i, b in enumerate(blocks, 1):
        text = clean_text(b["text"])
        if not text and not b["label"]:
            skipped += 1                  # empty template stub
            continue
        if b["label"] not in VALID:
            errors.append(f"  block {i} (game={b['game']!r}): bad label {b['label']!r}, "
                          f"must be one of {sorted(VALID)}")
            continue
        if len(text) < 15:
            errors.append(f"  block {i} (game={b['game']!r}): text too short / empty")
            continue
        rows.append({"text": text, "label": b["label"], "game": b["game"]})

    if errors:
        print("PROBLEMS (fix these in reviews_raw.txt):")
        print("\n".join(errors))
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "game"])
        w.writeheader()
        w.writerows(rows)

    dist = Counter(r["label"] for r in rows)
    print(f"Wrote {len(rows)} rows -> {args.out}  (skipped {skipped} empty stubs)")
    print("Label distribution:", dict(dist))
    if rows:
        top = max(dist.values()) / len(rows)
        print(f"Largest class: {top:.0%}  ", "OK" if top <= 0.70 else "OVER 70%, collect more of the others")
    print(f"Total toward the 200 minimum: {len(rows)}")


if __name__ == "__main__":
    main()
