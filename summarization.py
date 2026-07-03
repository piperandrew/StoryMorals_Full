#!/usr/bin/env python3
"""Summarization workflow (was 1_chunk_summarization.ipynb).

For every `.txt` book in an input directory, produce two summaries with a single
model:

  1. chunk_summary — the text is split into fixed-size token chunks (via
     tiktoken); each chunk is summarized as its ten most significant plot
     events, and the chunk summaries are concatenated.
  2. short_summary — the chunk summary condensed into one paragraph.

Output: a CSV with one row per book — `filename, chunk_summary, short_summary`.
The run checkpoints after each book and skips books already present on a rerun.

Usage:
    python summarization.py                       # defaults below
    python summarization.py --input-dir InputTexts --output-csv summaries.csv
    python summarization.py --model gpt-5.4-mini --language English

API keys are read from the environment (OPENAI_API_KEY) or a .env file; if
neither is set and the terminal is interactive, you are prompted. Nothing is
hard-coded or written to disk.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import tiktoken

import common

# Default summarization model. `models.txt` covers the moral_generation_plus
# workflow; summarization uses a single model, chosen here or via --model.
DEFAULT_MODEL = "gpt-5.4-mini"

CHUNK_SYSTEM_PROMPT = "You are a helpful assistant that summarizes literature."


def build_chunk_prompt(chunk_text: str, language: str) -> str:
    return (
        f"Here is a portion of a novel. Please summarize it IN {language.upper()} "
        "by listing the TEN most significant events and plot developments:\n\n"
        f"{chunk_text}"
    )


def build_short_prompt(language: str) -> str:
    return (
        f"Please provide a 1 paragraph summary IN {language.upper()} "
        "of the following plot events of a novel:"
    )


def chunk_text_by_tokens(text, encoding, chunk_size_tokens):
    """Split text into fixed-size chunks using real LLM tokenization."""
    token_ids = encoding.encode(text)
    chunks = []
    for start in range(0, len(token_ids), chunk_size_tokens):
        chunks.append(encoding.decode(token_ids[start:start + chunk_size_tokens]))
    return chunks


def make_chunk_summary(full_text, args, encoding):
    """Stage A — summarize the full text chunk by chunk and concatenate."""
    chunks = chunk_text_by_tokens(full_text, encoding, args.chunk_size_tokens)
    print(f"  {len(chunks)} chunk(s)")
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"    summarizing chunk {i}/{len(chunks)}")
        summaries.append(common.call_model(
            args.model, build_chunk_prompt(chunk, args.language),
            max_tokens=args.chunk_max_tokens, temperature=args.temperature,
            use_temperature=args.use_temperature, system=CHUNK_SYSTEM_PROMPT,
            timeout=args.request_timeout, max_retries=args.max_retries))
        time.sleep(args.delay_seconds)
    return "\n\n".join(summaries)


def make_short_summary(chunk_summary, args):
    """Stage B — condense the chunk summary into a single paragraph."""
    out = common.call_model(
        args.model, chunk_summary, max_tokens=args.short_max_tokens,
        temperature=args.temperature, use_temperature=args.use_temperature,
        system=build_short_prompt(args.language),
        timeout=args.request_timeout, max_retries=args.max_retries)
    time.sleep(args.delay_seconds)
    return out


def run(args):
    encoding = tiktoken.get_encoding(args.token_encoding)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {args.input_dir}")
    print(f"Found {len(files)} book(s) in {args.input_dir}")

    # Resume: load any existing output and skip books already summarized.
    if args.output_csv.exists():
        results = pd.read_csv(args.output_csv)
        print(f"Resuming — {len(results)} book(s) already in {args.output_csv}")
    else:
        results = pd.DataFrame(columns=["filename", "chunk_summary", "short_summary"])

    done = set(results["filename"].astype(str))
    n_ok = 0
    for path in files:
        book_id = path.stem
        if book_id in done:
            print(f"SKIP  {book_id} (already summarized)")
            continue

        print(f"BOOK  {book_id}")
        try:
            full_text = common.read_text_file(path)
            chunk_summary = make_chunk_summary(full_text, args, encoding)
            short_summary = make_short_summary(chunk_summary, args)
            results.loc[len(results)] = [book_id, chunk_summary, short_summary]
            done.add(book_id)
            n_ok += 1
            if n_ok % args.save_every == 0:
                results.to_csv(args.output_csv, index=False)
                print("  checkpoint saved")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR on {book_id}: {e}")

    results.to_csv(args.output_csv, index=False)
    print(f"\nDone. {len(results)} book(s) in {args.output_csv}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("InputTexts"),
                   help="Folder of .txt books (default: InputTexts)")
    p.add_argument("--output-csv", type=Path, default=Path("outputs/summaries.csv"),
                   help="Where the summary table is written (default: outputs/summaries.csv)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model for both summaries (default: {DEFAULT_MODEL})")
    p.add_argument("--language", default="English",
                   help="Output language for the summaries (default: English)")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--no-temperature", dest="use_temperature", action="store_false",
                   help="Omit the temperature param (for reasoning models that reject it)")
    p.add_argument("--chunk-max-tokens", type=int, default=2000)
    p.add_argument("--short-max-tokens", type=int, default=2000)
    p.add_argument("--token-encoding", default="o200k_base",
                   help="tiktoken encoding (default: o200k_base)")
    p.add_argument("--chunk-size-tokens", type=int, default=14000,
                   help="Tokens per chunk (default: 14000)")
    p.add_argument("--delay-seconds", type=float, default=0.1)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--request-timeout", type=int, default=120)
    p.add_argument("--save-every", type=int, default=1,
                   help="Checkpoint the CSV after every N books (default: 1)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
