# Low-rank compression of LLM weights and the KV cache (GPT-2)

Authors: Anikait Rao, Ayush Manocha, Sucheta Sarangi.

This project tests how far low-rank methods (SVD, randomized SVD, ALS) can compress
(1) the weight matrices and (2) the key/value (KV) cache of GPT-2 small, measured by
WikiText-2 perplexity on a CPU-only laptop. It also checks the main KV-cache findings on a
small RoPE model (SmolLM2-135M).

**Main results**
- GPT-2 weights are not close to low rank. Activation-aware SVD keeps perplexity usable (40.1 at 90% of the parameters, dense 33.64); plain SVD does not (1689).
- One basis per layer, shared by all heads, with keys at rank 256 and values at rank 512 (half the stored codes), gives +4.0% perplexity. With the basis refreshed during generation and the newest 32 tokens kept exact, the cost falls to about +0.2%. Counting the per-prompt basis, memory is 0.92x of the full cache at 1024 tokens.
- Most of the gain comes from keeping the newest tokens exact (recency), not from attention weighting. Plain 4-bit quantization reaches similar quality at lower memory.
- On a CPU a shared basis is slower than the full cache (1.3-1.5x per token at 4k-64k tokens).

All numbers come from the JSON files in `experiments/`. Limitations are listed in the report.
The low-rank KV-cache recipe is close to published work (iS-KV, AttSVD, OjaKV, Palu); the
related-work discussion is in the report.

## Layout

| Folder / file | Contents |
|---|---|
| `papers/report.tex`, `papers/report.pdf` | The report (65 pages): theory, weights, KV cache, bounds, basis updates, follow-ups, discussion, appendices. Start here. |
| `experiments/` | All scripts (`kv_*.py`, `model_quality.py`, `llm_compression.py`, ...), their result JSONs (`results_*.json`), and the run logs |
| `experiments/rerun_2026-11-19/` | Idle-CPU timing rerun |
| `experiments/rerun_2026-11-22/` | Follow-up experiments C1-C23 (JSON results, summaries, runner scripts) |
| `figures/` | All figures (pdf + png), drawn with the shared style in `experiments/dataviz_style.py` |
| `LICENSE` | MIT |

Working notes (literature checks, drafts, review lists) are kept on the author's machine and are not part of this repository.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Data: WikiText-2 raw is downloaded from the Hugging Face Hub (`Salesforce/wikitext`). Models: `gpt2` and `HuggingFaceTB/SmolLM2-135M`.

## Reproduce

Run scripts from the project root. Every plotting script supports `--replot`, which redraws its figures from the saved JSON without recomputing:

```
python experiments/kv_update.py --replot --figdir figures
```

Full runs take from minutes to hours on a CPU. The report's reproduction appendix lists the scripts and their run times.
Pass `--out` and `--figdir` when re-running, so the reported JSONs and figures are not overwritten.
Timings need an idle CPU and vary between sessions, so compare ratios, not absolute milliseconds.

## Build the report

From inside `papers/`, with a LaTeX distribution installed (run three passes):

```
cd papers
pdflatex report.tex
pdflatex report.tex
pdflatex report.tex
```

## License

MIT. See `LICENSE`.
