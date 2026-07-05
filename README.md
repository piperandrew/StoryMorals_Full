# Book → Morals → Values workflow

Python scripts that read the **moral content of books at scale**. Point them
at a folder of plain-text books and they produce, for each book: short plot
summaries, three "story morals" plus a structured set of narrative dimensions,
and a labeling of each moral against a fixed value taxonomy — all optionally
across several models and several ways of "seeing" the book — plus an optional
readable per-book report.

## The scripts

| Script | What it does |
|--------|--------------|
| **`summarization.py`** | Reduces each book to a chunk-by-chunk plot summary and a one-paragraph short summary. |
| **`moral_generation_plus.py`** | Everything downstream, in one script: generate morals + narrative dimensions, parse to tidy tables, label morals against the value taxonomy, and build a validation table. |
| **`makeReport/make_report.py`** | Renders a readable per-book `.txt` report from the generated results; you pick one model and one input condition, and optionally include the story morals. |

`moral_generation_plus.py` runs four steps in order — `generate`, `parse`,
`values`, `validate` (formerly notebooks 2–5). Run them all, or name a subset.

## What it produces (high level)

For every book:

- **Summaries** — a chunk-by-chunk plot summary and a one-paragraph summary.
- **Morals + narrative dimensions** — three story morals, plus a brief plot
  summary, the protagonist, themes, settings, central conflict, principal events,
  values pursued, values threatened, and the consequence/resolution.
- **Value labels** — the taxonomy values each moral expresses.
- **Readable reports** — an optional per-book `.txt` report rendered from the
  generated dimensions (see `makeReport/make_report.py`).

These are generated under combinations of:

- **Models** — e.g. GPT and Gemini, to compare how different models read a book.
  The models for moral generation and for value extraction are configured
  **separately** (they may differ) — see [`models.txt`](#modelstxt).
- **Input representations ("conditions")** — `full_text` (the whole book),
  `chunk_summary` (the summary), and `memory` (no text supplied — the model uses
  only the title and author).

**Note:** a sample book *Heart of Darkness* is included with its outputs for
demonstration.

---

## Requirements

- **Python 3.10+** with `openai`, `tiktoken`, `pandas`, `requests`:
  ```bash
  pip install -r requirements.txt
  ```

### API keys (never hard-coded)

Keys are read at runtime from **environment variables** or an optional **`.env`**
file, and are never written to disk. If a needed key is not set and you are
running interactively, the script prompts for it (hidden input).

| Provider | Environment variable | When needed |
|----------|----------------------|-------------|
| OpenAI | `OPENAI_API_KEY` | Any `gpt*` / `o1*` / `o3*` model. |
| Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Any `gemini-*` model. |

Set them in your shell:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

…or copy `.env.example` to `.env` and fill it in (`.env` is git-ignored):

```bash
cp .env.example .env      # then edit .env
```

---

## What you provide (inputs)

| File / folder | What it is |
|---------------|------------|
| `InputTexts/` | One `.txt` file per book. The file name (without `.txt`) is the book's id. |
| `meta.csv` | Columns `filename`, `Title`, `Author`. Supplies title/author for the `memory` condition. |
| `prompt.txt` | The instruction + JSON schema used to extract morals and narrative dimensions. |
| `models.txt` | The models per stage — see below. |
| `Values_Taxonomy.csv` | The value taxonomy; **column 1** is the list of value labels. |

### `models.txt`

Lists the models used by `moral_generation_plus.py`, in two sections, because the
models for moral generation and for value extraction may differ. One model per
line; the provider is inferred from the name prefix. Comment out a line with `#`
to drop that model.

```
[moral_generation]
gpt-5.4
gemini-3.1-pro-preview

[value_extraction]
gpt-5.4
gemini-3.1-pro-preview
```

(The summarization script uses a single model, set with `--model`; default
`gpt-5.4-mini`.)

---

## How to run

**1 — Summarize** the books:

```bash
python summarization.py
# writes outputs/summaries.csv
```

**2 — Generate morals, parse, label values, and validate** (all four steps):

```bash
python moral_generation_plus.py
# writes outputs/moral_generation_plus.jsonl (+ .csv), outputs/tidy/*.csv,
#        outputs/moral_values.csv, outputs/validation_table.csv
```

Run a single step, or a subset in order:

```bash
python moral_generation_plus.py generate
python moral_generation_plus.py values validate
```

**3 — (Optional) Render readable reports** from the generated results:

```bash
python makeReport/make_report.py
# prompts for a model + input condition (and whether to include story morals),
# then writes one report per book to makeReport/reports/<book>_report.txt
```

All generated files land in **`outputs/`** (created automatically, and
git-ignored), so a run over new material is self-contained — clear `outputs/` to
start fresh. Inputs stay at the repo root. Override any path with its flag (e.g.
`--output-csv`, `--tidy-dir`).

Both scripts **checkpoint and resume**: rerunning skips work already completed
and only fills in what's missing, so interrupted runs are safe to restart. Every
setting has a command-line flag with a sensible default — run either script with
`--help` to see them all.

---

## The steps in detail

### `summarization.py`
Splits each book into fixed-size token chunks (via `tiktoken`), summarizes each
chunk as its ten most significant plot events and concatenates them into a
**chunk summary**, then condenses that into a one-paragraph **short summary**.

- **Key flags:** `--input-dir`, `--output-csv`, `--model` (default
  `gpt-5.4-mini`), `--language` (default English), `--chunk-size-tokens`.
- **Reads:** `InputTexts/*.txt` · **Writes:** `outputs/summaries.csv`
  (`filename`, `chunk_summary`, `short_summary`).

### `moral_generation_plus.py`
One script, four steps (run all by default, or name a subset):

**`generate`** — For each book × input condition × moral-generation model, runs
the `prompt.txt` extractor to produce a structured result: three story morals
plus the narrative dimensions. The prompt is adapted automatically per condition
(`full_text`, `chunk_summary`, `memory`).
- **Models:** the `[moral_generation]` section of `models.txt`.
- **Flags:** `--conditions`, `--max-output-tokens`.
- **Reads:** `InputTexts/`, `outputs/summaries.csv`, `meta.csv`, `prompt.txt`.
- **Writes:** `outputs/moral_generation_plus.jsonl` (full structured result — the
  source of truth) and `outputs/moral_generation_plus.csv` (a slim index).

**`parse`** — Flattens the JSONL into tidy tables:
- `outputs/tidy/morals_long.csv` — one row per moral.
- `outputs/tidy/categories_wide.csv` — one row per generation, one column per category.
- **Reads:** `outputs/moral_generation_plus.jsonl` · **Writes:** the two `outputs/tidy/` CSVs.

**`values`** — Labels each moral from `morals_long` against the taxonomy (column 1
of `Values_Taxonomy.csv`), with each value-extraction model. API calls run in
parallel across a thread pool.
- **Models:** the `[value_extraction]` section of `models.txt`.
- **Flags:** `--shuffle-labels` / `--no-shuffle-labels`, `--max-workers`.
- **Reads:** `outputs/tidy/morals_long.csv`, `Values_Taxonomy.csv`.
- **Writes:** `outputs/moral_values.csv` — the `morals_long` columns plus a **`Values`**
  column: a JSON object mapping each model to its chosen labels, e.g.
  `{"gpt-5.4": ["Harm", "Power"], "gemini-3.1-pro-preview": ["Authority"]}`.

**`validate`** — Flattens one story's outputs into a long table for manual QA —
one row per answer, with its model and condition.
- **Flags:** `--story` (default: first story in the file),
  `--validate-conditions`, `--validate-models`.
- **Reads:** `outputs/moral_generation_plus.jsonl` · **Writes:**
  `outputs/validation_table.csv` (`filename`, `input_condition`, `model`,
  `main_category`, `sub_category`, `answer`).

### `makeReport/make_report.py`
Renders the structured results into a plain-text report per book — protagonist,
summary, themes, locations, central conflict, principal events, values, and the
consequence/resolution, laid out for reading. Interactive: it lists the models
and input conditions found in the data and asks you to pick one of each, then
whether to include the story morals. One report is written per unique book for
that model + condition.
- **Reads:** `outputs/moral_generation_plus.jsonl` (override with `--input`).
- **Writes:** `makeReport/reports/<book>_report.txt` (override dir with `--outdir`).

---

## Output files

All under `outputs/`:

| File | From | Contents |
|------|------|----------|
| `outputs/summaries.csv` | `summarization.py` | `filename`, `chunk_summary`, `short_summary` |
| `outputs/moral_generation_plus.jsonl` | `generate` | Full structured result per generation (source of truth) |
| `outputs/moral_generation_plus.csv` | `generate` | Slim index: morals per generation |
| `outputs/tidy/morals_long.csv` | `parse` | One row per moral |
| `outputs/tidy/categories_wide.csv` | `parse` | One row per generation; one column per category |
| `outputs/moral_values.csv` | `values` | `morals_long` + `Values` (labels per model) |
| `outputs/validation_table.csv` | `validate` | One story flattened for review |
| `makeReport/reports/<book>_report.txt` | `make_report.py` | Readable per-book report for a chosen model + condition |

---

## Configuration at a glance

- **Models** live in `models.txt` — one per line, in `[moral_generation]` and
  `[value_extraction]` sections (they may differ). Summarization uses `--model`.
- **API keys** come from the environment or a `.env` file — never hard-coded.
- **Language** (`--language` on `summarization.py`) controls the summary language.
- **Taxonomy** is `Values_Taxonomy.csv` (column 1 = labels); replace it to use a
  different value set.
- **Prompt** for the morals + dimensions is `prompt.txt`; edit it to change what
  is extracted (the script adapts to it automatically).
- **Shared code** (API dispatch, key loading, `models.txt` parsing, JSON
  helpers) lives in `common.py`.
