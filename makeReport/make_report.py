#!/usr/bin/env python3
"""Generate human-readable .txt reports from moral_generation_plus.jsonl.

Reads the parsed generation records, asks the user which model and which input
condition to report on, then writes one plain-text report per unique book for
that (model, condition) pair into the reports/ directory.

Every element of the story schema is included EXCEPT the story morals. The
SUMMARY section has no source field in the schema and is left as an "ADD"
placeholder for manual completion, mirroring the example report.
"""

import argparse
import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.normpath(
    os.path.join(HERE, "..", "outputs", "moral_generation_plus.jsonl")
)
DEFAULT_OUTDIR = os.path.join(HERE, "reports")
WIDTH = 90


# ---------------------------------------------------------------------------
# Text-wrapping helpers (match the layout of the example report)
# ---------------------------------------------------------------------------

def wrap_indented(text, indent=2):
    """Wrap `text` to WIDTH with every line indented by `indent` spaces."""
    pad = " " * indent
    text = (text or "").strip()
    if not text:
        return ""
    return textwrap.fill(text, width=WIDTH, initial_indent=pad, subsequent_indent=pad)


def wrap_labeled(label, text, indent=4):
    """Wrap "<indent>Label: text", continuation lines indented by `indent`."""
    pad = " " * indent
    text = (text or "").strip()
    return textwrap.fill(
        f"{label}: {text}", width=WIDTH,
        initial_indent=pad, subsequent_indent=pad,
    )


def header(title, char):
    return f"{title}\n{char * len(title)}"


# ---------------------------------------------------------------------------
# Report body
# ---------------------------------------------------------------------------

def build_report(book, res, include_morals=False):
    """Return the full report text for one book given its result dict."""
    out = []
    out.append(header(book, "="))
    out.append("")

    # PROTAGONIST
    prot = res.get("protagonist") or {}
    out.append(header("PROTAGONIST", "-"))
    out.append(f"Name: {prot.get('name', '') or ''}")
    out.append("Description:")
    out.append(wrap_indented(prot.get("description")))
    out.append("Primary goal:")
    out.append(wrap_indented(prot.get("primary_goal")))
    out.append("")

    # SUMMARY
    out.append(header("SUMMARY", "-"))
    summary = res.get("summary")
    if isinstance(summary, list):
        summary_text = " ".join(s for s in summary if s)
    else:
        summary_text = (summary or "").strip()
    out.append(wrap_indented(summary_text) if summary_text else "  ADD")
    out.append("")

    # MAJOR THEMES
    out.append(header("MAJOR THEMES", "-"))
    for theme in res.get("major_themes") or []:
        out.append(f"- {theme}")
    out.append("")

    # MAJOR LOCATIONS
    out.append(header("MAJOR LOCATIONS", "-"))
    for loc in res.get("major_locations") or []:
        out.append(f"- {loc.get('location', '') or ''}")
        if loc.get("location_type"):
            out.append(f"    Type: {loc['location_type']}")
        if loc.get("social_or_symbolic_significance"):
            out.append(wrap_labeled("Significance", loc["social_or_symbolic_significance"]))
    out.append("")

    # CENTRAL CONFLICT
    conf = res.get("central_conflict") or {}
    out.append(header("CENTRAL CONFLICT", "-"))
    out.append("Summary:")
    out.append(wrap_indented(conf.get("summary")))
    out.append(f"Structural type: {conf.get('structural_type', '') or ''}")
    out.append("Thematic opposing force:")
    out.append(wrap_indented(conf.get("thematic_opposing_force")))
    out.append("")

    # PRINCIPAL EVENTS
    out.append(header("PRINCIPAL EVENTS", "-"))
    for ev in res.get("principal_events") or []:
        event_text = (ev.get("event", "") or "").strip()
        out.append(textwrap.fill(
            f"- {event_text}", width=WIDTH,
            initial_indent="", subsequent_indent="  ",
        ))
        if ev.get("relationship_to_values_or_goals"):
            out.append(wrap_labeled("Relationship to values/goals",
                                    ev["relationship_to_values_or_goals"]))
    out.append("")

    # VALUES PURSUED
    out.append(header("VALUES PURSUED", "-"))
    for v in res.get("values_pursued") or []:
        out.append(f"- {v.get('value', '') or ''}: {v.get('explanation', '') or ''}")
    out.append("")

    # VALUES THREATENED
    out.append(header("VALUES THREATENED", "-"))
    for v in res.get("values_threatened") or []:
        out.append(f"- {v.get('value', '') or ''}: {v.get('explanation', '') or ''}")
    out.append("")

    # CONSEQUENCE AND RESOLUTION
    cons = res.get("consequence_resolution") or {}
    out.append(header("CONSEQUENCE AND RESOLUTION", "-"))
    out.append("Ending summary:")
    out.append(wrap_indented(cons.get("ending_summary")))
    out.append("Moral significance:")
    out.append(wrap_indented(cons.get("moral_significance")))
    out.append("")

    # STORY MORALS (optional)
    if include_morals:
        out.append(header("STORY MORALS", "-"))
        for moral in res.get("story_morals") or []:
            out.append(textwrap.fill(
                f"- {moral}", width=WIDTH,
                initial_indent="", subsequent_indent="  ",
            ))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Record loading and interactive selection
# ---------------------------------------------------------------------------

def load_records(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: input file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def choose(label, options):
    """Prompt the user to pick one of `options` (a sorted list); return it."""
    print(f"\nChoose {label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"Enter number (1-{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid selection, try again.")


def sanitize(name):
    return re.sub(r"[^\w\-. ]+", "_", name).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"Path to moral_generation_plus.jsonl (default: {DEFAULT_INPUT})")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help=f"Directory for reports (default: {DEFAULT_OUTDIR})")
    args = ap.parse_args()

    records = load_records(args.input)
    # Keep only usable records (parsed result dict).
    usable = [r for r in records if isinstance(r.get("result"), dict)]
    if not usable:
        sys.exit("ERROR: no parsed records with a result object found in input.")

    models = sorted({r["model"] for r in usable})
    conditions = sorted({r["input_condition"] for r in usable})

    model = choose("model", models)
    condition = choose("input condition", conditions)
    include_morals = choose("whether to include story morals", ["No", "Yes"]) == "Yes"

    # For each unique book, find the matching (model, condition) record.
    # If duplicates exist, keep the last one (latest write wins).
    selected = {}
    for r in usable:
        if r["model"] == model and r["input_condition"] == condition:
            selected[r["filename"]] = r

    if not selected:
        sys.exit(f"No records for model='{model}', condition='{condition}'.")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nGenerating reports for model='{model}', condition='{condition}':")
    for book in sorted(selected):
        report = build_report(book, selected[book]["result"], include_morals)
        out_path = os.path.join(args.outdir, f"{sanitize(book)}_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  wrote {out_path}")

    print(f"\nDone. {len(selected)} report(s) in {args.outdir}")


if __name__ == "__main__":
    main()
