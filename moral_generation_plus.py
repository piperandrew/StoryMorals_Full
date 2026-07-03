#!/usr/bin/env python3
"""Moral generation + narrative dimensions workflow (was notebooks 2–5).

This single workflow runs all four downstream steps, in order:

  generate  (was 2_moral_generation_plus) — for each book x input condition x
            moral-generation model, run prompt.txt to produce a structured JSON
            result (three story morals + narrative dimensions).
            -> moral_generation_plus.jsonl  (+ .csv index)

  parse     (was 3_parse_outputs) — flatten the JSONL into tidy tables.
            -> tidy/morals_long.csv, tidy/categories_wide.csv

  values    (was 4/5_value_extraction) — label each moral against the value
            taxonomy, with each value-extraction model (calls run in parallel).
            -> moral_values.csv

  validate  (was 5_validation_table) — flatten one story's outputs field-by-
            field for manual QA.
            -> validation_table.csv

Models come from `models.txt`:
  [moral_generation]  -> the `generate` step
  [value_extraction]  -> the `values` step
(The two lists may differ.) The extraction prompt/schema is `prompt.txt`.

Usage:
    python moral_generation_plus.py                 # run all steps in order
    python moral_generation_plus.py generate        # just one step
    python moral_generation_plus.py values validate # a subset, in order
    python moral_generation_plus.py --conditions full_text memory

API keys are read from the environment (OPENAI_API_KEY, GEMINI_API_KEY /
GOOGLE_API_KEY) or a .env file; if unset and interactive, you are prompted.
Every step checkpoints and resumes.
"""

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

import common

STEPS = ["generate", "parse", "values", "validate"]


# ===========================================================================
# Step 1 (generate) — morals + narrative dimensions
# ===========================================================================

_LEAD = "Given the following story,"


def build_prompt_parts(prompt_txt: Path):
    """Load prompt.txt and split off the fixed opening + trailing placeholder."""
    text = (prompt_txt.read_text(encoding="utf-8")
            .replace(" ", "\n").replace(" ", "\n"))
    body = re.split(r"Story:\s*\[INSERT STORY HERE\]", text)[0].rstrip()
    if not body.startswith(_LEAD):
        raise ValueError("prompt.txt opening changed; update _LEAD in this script.")
    return body[len(_LEAD):].lstrip()  # instructions after the opening phrase


def build_prompt(instructions, condition, text=None, title=None, author=None):
    """Adjust the opening framing + trailing input for each representation."""
    if condition == "full_text":
        opening, tail = "Given the following story,", f"\n\nStory:\n{text}"
    elif condition == "chunk_summary":
        opening = "Given the following plot summary of a story,"
        tail = f"\n\nPlot Summary:\n{text}"
    elif condition == "memory":
        opening, tail = f'For the novel "{title}" by {author},', ""
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return f"{opening} {instructions}{tail}"


def _key(filename, condition, model):
    return (str(filename), str(condition), str(model))


def _moral_at(morals, i):
    return morals[i] if isinstance(morals, list) and len(morals) > i else None


def _dedupe_latest(records):
    """Keep the last record per (filename, condition, model).

    The JSONL is append-only, so a retried generation adds a new line after the
    old (failed) one; keeping the last occurrence makes the success win.
    """
    latest = {}
    for d in records:
        latest[_key(d["filename"], d["input_condition"], d["model"])] = d
    return list(latest.values())


def _rebuild_index(raw_jsonl: Path, output_csv: Path):
    index_cols = ["filename", "input_condition", "model", "parse_ok",
                  "story_moral_1", "story_moral_2", "story_moral_3", "timestamp"]
    rows = []
    if raw_jsonl.exists():
        recs = []
        with open(raw_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        for d in _dedupe_latest(recs):
            res = d.get("result")
            morals = res.get("story_morals") if isinstance(res, dict) else None
            rows.append({
                "filename": d["filename"], "input_condition": d["input_condition"],
                "model": d["model"], "parse_ok": d["parse_ok"],
                "story_moral_1": _moral_at(morals, 0),
                "story_moral_2": _moral_at(morals, 1),
                "story_moral_3": _moral_at(morals, 2), "timestamp": d["timestamp"],
            })
    df = pd.DataFrame(rows, columns=index_cols)
    df.to_csv(output_csv, index=False)
    return df


def run_generate(args):
    print("\n=== STEP: generate ===")
    models = common.models_for("moral_generation", args.models_txt)
    print(f"Moral-generation models: {models}")

    instructions = build_prompt_parts(args.prompt_txt)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_jsonl = args.output_csv.with_suffix(".jsonl")

    # Title/Author keyed by canonical id — used by the memory condition.
    meta_df = pd.read_csv(args.meta_csv)
    meta = {common.normalize_id(row["filename"]):
            {"title": row.get("Title"), "author": row.get("Author")}
            for _, row in meta_df.iterrows()}

    # Chunk summaries keyed by canonical id — used by the chunk_summary condition.
    summaries = {}
    if args.summaries_csv.exists():
        sdf = pd.read_csv(args.summaries_csv)
        summaries = {common.normalize_id(r["filename"]): r["chunk_summary"]
                     for _, r in sdf.iterrows()}
    else:
        print(f"WARNING: {args.summaries_csv} not found — chunk_summary condition skipped.")
    print(f"{len(meta)} book(s) in metadata; {len(summaries)} chunk summary(ies).")

    # Resume: a (book, condition, model) counts as done only if it PARSED.
    # Before spending an API call to retry an unparsed record, try to SALVAGE
    # its saved raw reply with the current (tolerant) parser — a reply that
    # failed under an older parser often parses now, recovering it for free.
    # Anything still unparsed is retried; the fresh record supersedes the old
    # one (see _dedupe_latest, which keeps the last per key).
    done = set()
    if raw_jsonl.exists():
        recs = []
        with open(raw_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        recovered = retry = 0
        for d in _dedupe_latest(recs):
            key = _key(d["filename"], d["input_condition"], d["model"])
            if d.get("parse_ok"):
                done.add(key)
                continue
            salvaged = common.extract_json(d.get("raw_text") or "")
            if isinstance(salvaged, dict):
                rec = {
                    "filename": d["filename"], "input_condition": d["input_condition"],
                    "model": d["model"], "parse_ok": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "result": salvaged, "raw_text": None,
                }
                with open(raw_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.add(key)
                recovered += 1
            else:
                retry += 1
        msg = f"Resuming — {len(done)} successful generation(s) in {raw_jsonl}"
        if recovered:
            msg += f"; recovered {recovered} by re-parsing saved replies"
        if retry:
            msg += f"; {retry} still unparsed, will retry"
        print(msg)

    n_new = 0
    for book_id, info in meta.items():
        for condition in args.conditions:
            text = None
            if condition == "full_text":
                p = args.input_dir / f"{book_id}.txt"
                if not p.exists():
                    print(f"SKIP  {book_id} / full_text (no .txt found)")
                    continue
                text = common.read_text_file(p)
            elif condition == "chunk_summary":
                text = summaries.get(book_id)
                if not text:
                    print(f"SKIP  {book_id} / chunk_summary (not in summaries.csv)")
                    continue
            # memory: no text needed (uses title/author from meta.csv)

            for model in models:
                if _key(book_id, condition, model) in done:
                    print(f"SKIP  {book_id} / {condition} / {model} (done)")
                    continue
                print(f"GEN   {book_id} / {condition} / {model}")
                try:
                    prompt = build_prompt(instructions, condition, text=text,
                                          title=info["title"], author=info["author"])
                    raw = common.call_model(
                        model, prompt, max_tokens=args.max_output_tokens,
                        temperature=args.temperature, use_temperature=args.use_temperature,
                        json_mode=True, timeout=args.request_timeout,
                        max_retries=args.max_retries)
                    parsed = common.extract_json(raw)
                    record = {
                        "filename": book_id, "input_condition": condition, "model": model,
                        "parse_ok": parsed is not None,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "result": parsed,
                        "raw_text": None if parsed is not None else raw,
                    }
                    with open(raw_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done.add(_key(book_id, condition, model))
                    n_new += 1
                    time.sleep(args.delay_seconds)
                except Exception as e:  # noqa: BLE001
                    print(f"      ERROR: {e}")

    df = _rebuild_index(raw_jsonl, args.output_csv)
    print(f"Done generate: {len(df)} generation(s) ({n_new} new).")
    print(f"  raw   -> {raw_jsonl}")
    print(f"  index -> {args.output_csv}")


# ===========================================================================
# Step 2 (parse) — tidy tables
# ===========================================================================

def _gen_id(r):
    return f"{r['filename']}__{r['input_condition']}__{r['model']}"


def _as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def _as_json(x):
    return None if x is None else json.dumps(x, ensure_ascii=False)


def _pick(items, field=None):
    out = []
    for it in _as_list(items):
        out.append(it.get(field) if (isinstance(it, dict) and field is not None) else it)
    return out


def _one_or_json(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else json.dumps(vals, ensure_ascii=False)


def _load_records(raw_jsonl: Path, raw_csv: Path):
    if raw_jsonl.exists():
        recs = []
        with open(raw_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        recs = _dedupe_latest(recs)
        print(f"Loaded {len(recs)} record(s) from {raw_jsonl}")
        return recs
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        if "output_json" not in df.columns:
            raise ValueError(f"{raw_csv} has no `output_json` column and {raw_jsonl} "
                             "is missing — nothing to parse.")
        recs = []
        for _, row in df.iterrows():
            try:
                result = json.loads(row["output_json"])
            except Exception:
                result = None
            recs.append({"filename": row["filename"],
                         "input_condition": row["input_condition"],
                         "model": row["model"], "result": result})
        print(f"Loaded {len(recs)} record(s) from {raw_csv} (output_json column)")
        return recs
    raise FileNotFoundError(f"Neither {raw_jsonl} nor {raw_csv} found.")


def run_parse(args):
    print("\n=== STEP: parse ===")
    raw_jsonl = args.output_csv.with_suffix(".jsonl")
    records = _load_records(raw_jsonl, args.output_csv)
    args.tidy_dir.mkdir(parents=True, exist_ok=True)

    morals, wide = [], []
    for r in records:
        res = r.get("result")
        base = {"generation_id": _gen_id(r), "filename": r["filename"],
                "input_condition": r["input_condition"], "model": r["model"]}
        if not isinstance(res, dict):
            continue

        for i, m in enumerate(_as_list(res.get("story_morals")), 1):
            morals.append({**base, "moral_index": i, "moral_text": m})

        prot = res.get("protagonist") or {}
        conf = res.get("central_conflict") or {}
        cons = res.get("consequence_resolution") or {}
        wide.append({**base,
            "protagonist":            prot.get("name"),
            "major_themes":           _one_or_json(_pick(res.get("major_themes"))),
            "major_locations":        _one_or_json(_pick(res.get("major_locations"), "location")),
            "central_conflict":       conf.get("thematic_opposing_force"),
            "principal_actions":      _one_or_json(_pick(res.get("principal_actions"), "action")),
            "values_pursued":         _one_or_json(_pick(res.get("values_pursued"), "value")),
            "values_threatened":      _one_or_json(_pick(res.get("values_threatened"), "value")),
            "consequence_resolution": cons.get("moral_significance"),
            "story_morals":           _as_json(res.get("story_morals")),
        })

    pd.DataFrame(morals, columns=["generation_id", "filename", "input_condition",
                                  "model", "moral_index", "moral_text"]
                 ).to_csv(args.tidy_dir / "morals_long.csv", index=False)

    wide_cols = ["generation_id", "filename", "input_condition", "model", "protagonist",
                 "major_themes", "major_locations", "central_conflict", "principal_actions",
                 "values_pursued", "values_threatened", "consequence_resolution", "story_morals"]
    pd.DataFrame(wide, columns=wide_cols).to_csv(args.tidy_dir / "categories_wide.csv", index=False)

    print(f"morals_long     {len(morals):4d} rows -> {args.tidy_dir / 'morals_long.csv'}")
    print(f"categories_wide {len(wide):4d} rows -> {args.tidy_dir / 'categories_wide.csv'}")


# ===========================================================================
# Step 3 (values) — label morals against the taxonomy (parallel)
# ===========================================================================

def _build_label_prompt(moral_text, labels):
    label_block = "\n- ".join(labels)
    return (
        "You are annotating the moral of a story using a fixed taxonomy of values.\n\n"
        "Select ALL labels from the taxonomy that apply. Multiple labels are allowed. "
        "Use ONLY labels from this list (verbatim):\n\n"
        f"- {label_block}\n\n"
        f'Moral to annotate:\n"""\n{moral_text}\n"""\n\n'
        'Return ONLY a JSON array of the chosen label strings, e.g. ["Label A", "Label B"]. '
        "No prose, no explanation, no markdown."
    )


def _row_seed(run_seed, generation_id, moral_index):
    h = sha256(f"{run_seed}|{generation_id}|{moral_index}".encode())
    return int.from_bytes(h.digest()[:4], "big")


def _shuffled_labels(labels, seed):
    lab = list(labels)
    random.Random(seed).shuffle(lab)
    return lab


def run_values(args):
    print("\n=== STEP: values ===")
    models = common.models_for("value_extraction", args.models_txt)
    print(f"Value-extraction models: {models}")

    morals_long = args.tidy_dir / "morals_long.csv"
    if not morals_long.exists():
        raise FileNotFoundError(f"{morals_long} not found — run the parse step first.")
    args.values_csv.parent.mkdir(parents=True, exist_ok=True)

    label_list = pd.read_csv(args.taxonomy_csv).iloc[:, 0].dropna().astype(str).tolist()
    print(f"{len(label_list)} taxonomy labels loaded.")

    df = pd.read_csv(morals_long)

    # Resume: reuse any (moral, model) already labeled non-empty.
    existing = {}
    if args.values_csv.exists():
        prev = pd.read_csv(args.values_csv)
        if "Values" in prev.columns:
            for r in prev.itertuples():
                key = (str(r.generation_id), int(r.moral_index))
                try:
                    existing[key] = (json.loads(r.Values)
                                     if isinstance(r.Values, str) and r.Values.strip() else {})
                except Exception:
                    existing[key] = {}

    keys = [(str(r.generation_id), int(r.moral_index)) for r in df.itertuples()]
    results = {k: dict(existing.get(k, {})) for k in keys}

    tasks = []
    for r in df.itertuples():
        key = (str(r.generation_id), int(r.moral_index))
        labels = (_shuffled_labels(label_list, _row_seed(args.run_seed, *key))
                  if args.shuffle_labels else label_list)
        for model in models:
            val = results[key].get(model)
            if isinstance(val, list) and len(val) > 0:
                continue
            tasks.append((key, model, r.moral_text, labels))

    print(f"{len(tasks)} call(s) across {args.max_workers} workers "
          f"({len(df)} morals x {len(models)} models, minus done).")

    def _label_one(task):
        key, model, moral_text, labels = task
        try:
            raw = common.call_model(model, _build_label_prompt(moral_text, labels),
                                    max_tokens=args.label_max_tokens,
                                    temperature=args.temperature,
                                    use_temperature=args.use_temperature,
                                    timeout=args.request_timeout,
                                    max_retries=args.max_retries, quiet=True)
            return key, model, (common.extract_json_array(raw) or []), None
        except Exception as e:  # noqa: BLE001
            return key, model, None, str(e)

    def _write():
        out = df.copy()
        out["Values"] = [json.dumps(results[k], ensure_ascii=False) for k in keys]
        out.to_csv(args.values_csv, index=False)

    done_n = 0
    if tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(_label_one, t) for t in tasks]
            for fut in as_completed(futures):
                key, model, vals, err = fut.result()
                results[key][model] = vals
                done_n += 1
                print(f"[{done_n}/{len(tasks)}] {key} / {model} -> "
                      + ("OK" if err is None else f"ERROR: {err}"))
                if done_n % args.save_every == 0:
                    _write()

    _write()
    print(f"Done values: {len(df)} moral(s) labeled -> {args.values_csv}")


# ===========================================================================
# Step 4 (validate) — flatten one story for manual review
# ===========================================================================

def _walk(val, prefix=""):
    out = []
    if isinstance(val, dict):
        for k, v in val.items():
            sub = k if not prefix else f"{prefix}.{k}"
            out.extend(_walk(v, sub))
    elif isinstance(val, list):
        for i, item in enumerate(val, 1):
            sub = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.extend(_walk(item, sub))
    else:
        out.append((prefix, val))
    return out


def run_validate(args):
    print("\n=== STEP: validate ===")
    args.validation_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_jsonl = args.output_csv.with_suffix(".jsonl")
    records = []
    with open(raw_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {raw_jsonl}")
    records = _dedupe_latest(records)

    story = args.story if args.story is not None else records[0]["filename"]
    selected = [r for r in records
                if r["filename"] == story
                and (args.validate_conditions is None
                     or r["input_condition"] in args.validate_conditions)
                and (args.validate_models is None or r["model"] in args.validate_models)]
    if not selected:
        raise ValueError(f"No records match story={story!r} with the given filters.")

    print(f"Story: {story!r} — {len(selected)} generation(s)")

    rows = []
    for rec in selected:
        base = {"filename": rec["filename"], "input_condition": rec["input_condition"],
                "model": rec["model"]}
        result = rec.get("result")
        if not isinstance(result, dict):
            rows.append({**base, "main_category": None, "sub_category": None,
                         "answer": "<no parsed result (parse_ok=False)>"})
            continue
        for main, val in result.items():
            leaves = _walk(val)
            if not leaves:
                rows.append({**base, "main_category": main, "sub_category": "", "answer": None})
            for sub, ans in leaves:
                rows.append({**base, "main_category": main, "sub_category": sub, "answer": ans})

    vt = pd.DataFrame(rows, columns=["filename", "input_condition", "model",
                                     "main_category", "sub_category", "answer"])
    vt.to_csv(args.validation_csv, index=False)
    print(f"Done validate: {len(vt)} row(s) -> {args.validation_csv}")


# ===========================================================================
# CLI
# ===========================================================================

_DISPATCH = {"generate": run_generate, "parse": run_parse,
             "values": run_values, "validate": run_validate}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="*", choices=STEPS,
                   help=f"Steps to run, in order (default: all — {' '.join(STEPS)}).")

    # Paths
    # Inputs live at the repo root; all generated files go under outputs/.
    p.add_argument("--input-dir", type=Path, default=Path("InputTexts"))
    p.add_argument("--summaries-csv", type=Path, default=Path("outputs/summaries.csv"))
    p.add_argument("--meta-csv", type=Path, default=Path("meta.csv"))
    p.add_argument("--prompt-txt", type=Path, default=Path("prompt.txt"))
    p.add_argument("--models-txt", type=Path, default=Path("models.txt"))
    p.add_argument("--output-csv", type=Path,
                   default=Path("outputs/moral_generation_plus.csv"),
                   help="Index CSV; the .jsonl beside it is the source of truth.")
    p.add_argument("--tidy-dir", type=Path, default=Path("outputs/tidy"))
    p.add_argument("--taxonomy-csv", type=Path, default=Path("Values_Taxonomy.csv"))
    p.add_argument("--values-csv", type=Path, default=Path("outputs/moral_values.csv"))
    p.add_argument("--validation-csv", type=Path,
                   default=Path("outputs/validation_table.csv"))

    # generate
    p.add_argument("--conditions", nargs="+",
                   default=["full_text", "chunk_summary", "memory"],
                   choices=["full_text", "chunk_summary", "memory"],
                   help="Input representations to run in the generate step.")
    p.add_argument("--max-output-tokens", type=int, default=2000)

    # values
    p.add_argument("--label-max-tokens", type=int, default=512)
    p.add_argument("--shuffle-labels", action="store_true", default=True)
    p.add_argument("--no-shuffle-labels", dest="shuffle_labels", action="store_false")
    p.add_argument("--run-seed", type=int, default=20240601)
    p.add_argument("--max-workers", type=int, default=8)

    # validate
    p.add_argument("--story", default=None,
                   help="Filename to validate (default: first story in the JSONL).")
    p.add_argument("--validate-conditions", nargs="+", default=None)
    p.add_argument("--validate-models", nargs="+", default=None)

    # Shared generation / robustness
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--no-temperature", dest="use_temperature", action="store_false")
    p.add_argument("--delay-seconds", type=float, default=0.1)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--request-timeout", type=int, default=120)
    p.add_argument("--save-every", type=int, default=10)

    args = p.parse_args(argv)
    if not args.steps:
        args.steps = list(STEPS)
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Running steps: {args.steps}")
    for step in args.steps:
        _DISPATCH[step](args)
    print("\nAll requested steps complete.")


if __name__ == "__main__":
    main()
