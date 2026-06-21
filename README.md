# Book → Morals → Values workflow

A set of Jupyter notebooks that read the **moral content of books at scale**.
Point it at a folder of plain-text books and it produces, for each book: short
plot summaries, three "story morals" plus a structured set of narrative
dimensions, and a labeling of each moral against a fixed value taxonomy — all
optionally across several models and several ways of "seeing" the book.

## What it produces (high level)

For every book the workflow generates:

- **Summaries** — a chunk-by-chunk plot summary and a one-paragraph summary.
- **Morals + narrative dimensions** — three story morals, plus the protagonist,
  themes, settings, central conflict, principal actions, values pursued, values
  threatened, and the consequence/resolution.
- **Value labels** — the taxonomy values each moral expresses.

These are generated under combinations of:

- **Models** — e.g. GPT and Gemini, to compare how different models read a book.
- **Input representations ("conditions")** — `full_text` (the whole book),
  `chunk_summary` (the summary), and `memory` (no text supplied — the model uses
  only the title and author).

**Note: a sample book *Heart of Darkness* is included with its respective outputs for demonstration purposes.**

---

## Requirements

- **Python 3** with: `openai`, `tiktoken`, `pandas`, `requests` (and `pyarrow`
  if you want Parquet). Install with:
  ```bash
  pip install openai tiktoken pandas requests
  ```
- **API keys**, entered at runtime when each notebook prompts (never stored):
  - **OpenAI** (`OPENAI_API_KEY`) — required.
  - **Google Gemini** (`GEMINI_API_KEY` / `GOOGLE_API_KEY`) — required when a
    `gemini-*` model is in the model list (the default).

---

## What you provide (inputs)

| File / folder | What it is |
|---------------|------------|
| `InputTexts/` | One `.txt` file per book. The file name (without `.txt`) is the book's id. |
| `meta.csv` | Columns `filename`, `Title`, `Author`. Supplies title/author for the `memory` condition. |
| `prompt.txt` | The instruction + JSON schema used to extract morals and narrative dimensions. |
| `Values_Taxonomy.csv` | The value taxonomy; **column 1** is the list of value labels. |

---

## The pipeline

Run the notebooks in number order. Each reads the previous step's output, so
the files flow straight through.

| # | Notebook | What it addresses | Reads | Writes |
|---|----------|-------------------|-------|--------|
| 1 | `1_chunk_summarization.ipynb` | Makes each book usable at scale by reducing it to compact summaries. | `InputTexts/*.txt` | `summaries.csv` |
| 2 | `2_moral_generation_plus.ipynb` | The core extraction: morals + narrative dimensions, across models and input conditions. | `InputTexts/`, `summaries.csv`, `meta.csv`, `prompt.txt` | `moral_generation_plus.jsonl` (+ `.csv` index) |
| 3 | `3_parse_outputs.ipynb` | Turns the nested results into flat, analysis-ready tables. | `moral_generation_plus.jsonl` | `tidy/morals_long.csv`, `tidy/categories_wide.csv` |
| 4 | `4_value_extraction.ipynb` | Maps each moral onto the fixed value taxonomy. | `tidy/morals_long.csv`, `Values_Taxonomy.csv` | `moral_values.csv` |
| 5 | `5_validation_table.ipynb` | QA: lay out one story's outputs field-by-field for manual review. | `moral_generation_plus.jsonl` | `validation_table.csv` |

---

## How to run

1. Put your books in `InputTexts/` and fill in `meta.csv`.
2. Open each notebook **in order (1 → 5)** in Jupyter.
3. Edit the **Parameters** cell near the top if you want to change anything
   (paths, models, language, etc.).
4. Run all cells. The notebook prompts for the needed API key(s) and writes its
   output file(s).
5. Move on to the next notebook.

Every stage **checkpoints and resumes**: rerunning a notebook skips work already
completed and only fills in what's missing, so interrupted runs are safe to
restart.

---

## The steps in detail

### 1 — `1_chunk_summarization.ipynb`
Splits each book into fixed-size token chunks, summarizes each chunk as its ten
most significant plot events, and concatenates them into a **chunk summary**;
then condenses that into a one-paragraph **short summary**.

- **Key parameters:** `INPUT_DIR`, `MODEL` (default `gpt-5.4-mini`), `LANGUAGE`
  (default English), `CHUNK_SIZE_TOKENS`.
- **Output:** `summaries.csv` — `filename`, `chunk_summary`, `short_summary`.

### 2 — `2_moral_generation_plus.ipynb`
For each book, each selected **input condition**, and each **model**, runs the
`prompt.txt` extractor to produce a structured result: three story morals plus
the narrative dimensions. The prompt is adapted automatically per condition
(`full_text`, `chunk_summary`, `memory`).

- **Key parameters:** `MODELS` and `CONDITIONS` (one item per line — comment a
  line out to drop that model/condition), `LANGUAGE`.
- **Output:** `moral_generation_plus.jsonl` — one line per generation holding the
  full structured result (the source of truth) — and `moral_generation_plus.csv`,
  a slim index with the three morals.

### 3 — `3_parse_outputs.ipynb`
Reads the JSONL and builds flat tables:

- `tidy/morals_long.csv` — **one row per moral**.
- `tidy/categories_wide.csv` — **one row per generation**, one column per
  category (e.g. `protagonist` = the name; multi-valued fields are a single
  scalar, or a JSON list when there is more than one value).

Additional per-dimension tables (themes, values, actions, locations) are
included as commented-out blocks you can enable.

- **Output:** `tidy/morals_long.csv`, `tidy/categories_wide.csv`.

### 4 — `4_value_extraction.ipynb`
Labels each moral from `morals_long` against the taxonomy (column 1 of
`Values_Taxonomy.csv`), using each model in `LABELING_MODELS`.

- **Key parameters:** `LABELING_MODELS` (default GPT + Gemini, one per line),
  `SHUFFLE_LABELS`.
- **Output:** `moral_values.csv` — the `morals_long` columns plus a **`Values`**
  column: a JSON object mapping each model to its chosen labels, e.g.
  `{"gpt-5.4": ["Harm", "Power"], "gemini-3.1-pro-preview": ["Authority"]}`.
  Stays one row per moral.

### 5 — `5_validation_table.ipynb`
Flattens one story's outputs into a long table for manual checking — one row per
answer, with its model and condition.

- **Key parameters:** `STORY` (default: the first story in the file),
  `CONDITIONS` and `MODELS` (default: all).
- **Output:** `validation_table.csv` — `filename`, `input_condition`, `model`,
  `main_category`, `sub_category`, `answer`.

---

## Output files

| File | From | Contents |
|------|------|----------|
| `summaries.csv` | 1 | `filename`, `chunk_summary`, `short_summary` |
| `moral_generation_plus.jsonl` | 2 | Full structured result per generation (source of truth) |
| `moral_generation_plus.csv` | 2 | Slim index: morals per generation |
| `tidy/morals_long.csv` | 3 | One row per moral |
| `tidy/categories_wide.csv` | 3 | One row per generation; one column per category |
| `moral_values.csv` | 4 | `morals_long` + `Values` (labels per model) |
| `validation_table.csv` | 5 | One story flattened for review |

---

## Configuration at a glance

- **Models / conditions** are lists at the top of stages 2 and 4 (one per line) —
  comment out a line to drop that model or condition.
- **Language** (`LANGUAGE`) controls the output language of summaries (stage 1)
  and morals (stage 2); default is English.
- **Taxonomy** is `Values_Taxonomy.csv` (column 1 = labels); replace it to use a
  different value set.
- **Prompt** for the morals + dimensions is `prompt.txt`; edit it to change what
  is extracted (the notebooks adapt to it automatically).
