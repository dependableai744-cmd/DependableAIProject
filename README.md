# Privacy-Preserving Retrieval-Augmented Generation (RAG) for Trustworthy AI

**Course Project — Dependable AI**
**Authors:** Saksham Bhatnagar, Harshavardhan Panguluri
**University of Western Ontario**

---

## 📌 Project Overview

This project investigates **privacy leakage in Retrieval-Augmented Generation (RAG) systems** and proposes a **PII masking preprocessing layer** to mitigate it.

Modern RAG frameworks like **HippoRAG-2** and **RAG-Anything** retrieve documents from a knowledge base and pass them to an LLM to answer questions. The problem: if those documents contain **Personally Identifiable Information (PII)** — names, emails, SSNs, phone numbers, bank accounts — the LLM will reproduce that information in its responses.

### The Surprising Finding
Even **"safe" Wikipedia data** (HotpotQA dataset) leaks PII through RAG systems — names, locations, and organisations appear in answers without any deliberate injection. This is the core research contribution.

### Our Solution
A **preprocessing masking layer** applied before document indexing:
- Uses **spaCy NER** (Named Entity Recognition) to detect person names, locations, organisations
- Uses **13 regex patterns** to detect emails, phones, SSNs, credit cards, bank accounts, passwords, etc.
- Masks all detected PII before any document is embedded or indexed
- **Does NOT modify LLM internals** — works as a drop-in layer on any RAG pipeline

---

## 🏗️ Project Architecture

```
Approach C Pipeline:
                                        ┌─────────────────────┐
  HotpotQA docs ──────────────────────► │   RAG Frameworks    │ ──► Phase 1: Incidental PII leaked
  (Wikipedia, "safe" data)              │  • HippoRAG-2       │
                                        │  • RAG-Anything     │
  PII-masking-200k docs ──────────────► │                     │ ──► Phase 1: Massive PII leaked
  (explicit PII — worst case)           └─────────────────────┘
                                                   │
                                     Phase 2: Apply PII Masking
                                                   │
                                                   ▼
                              ┌────────────────────────────────┐
                              │      PII Masking Layer         │
                              │  spaCy NER + 13 Regex Rules    │
                              └────────────────────────────────┘
                                                   │
                              ┌────────────────────┴─────────────────┐
                              ▼                                       ▼
                 HotpotQA (masked):                    PII-masking (masked):
                 Leakage down + F1/EM maintained       Leakage down dramatically
```

---

## 📁 Project Structure

```
privacy_rag_project/
├── README.md                    ← You are here
├── requirements.txt             ← All Python dependencies
│
├── data/
│   ├── __init__.py
│   └── datasets.py              ← Loads HotpotQA + PII-masking-200k from HuggingFace
│
├── pii_masker.py                ← PII detection & masking (spaCy NER + regex)
├── eval_metrics.py              ← Exact Match + F1 scoring
│
├── phase1_baseline.py           ← Phase 1: RAG with NO protection → shows leakage
├── phase2_private_rag.py        ← Phase 2: RAG WITH masking → leakage drops
├── compare_results.py           ← Full evaluation report with GDPR/HIPAA framing
├── plot_results.py              ← Generates 3 trade-off charts (matplotlib)
├── interactive_demo.py          ← Live query mode — type questions, see before/after
│
└── results/                     ← Auto-created when you run scripts
    ├── phase1_baseline_results.json
    ├── phase2_private_results.json
    ├── final_report.json
    ├── masked_documents.json
    ├── interactive_query_log.json
    └── plots/
        ├── privacy_utility_tradeoff.png
        ├── pii_leakage_comparison.png
        └── f1_comparison.png
```

---

## ⚙️ Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10/11 (no WSL needed) |
| **Python** | 3.10 (via Anaconda/Miniconda) |
| **GPU** | NVIDIA GPU recommended (RTX series) |
| **Disk space** | ~5 GB free |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **Internet** | Required for first run (downloads models + datasets) |

---

## 🚀 STEP-BY-STEP SETUP

### STEP 1 — Install Ollama (Free Local LLM — replaces OpenAI)

1. Go to **https://ollama.com/download**
2. Download and run **OllamaSetup.exe**
3. Open **Command Prompt** and download the required models:

```cmd
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

> ⏱️ This downloads ~5 GB total. Do this first while setting up Python.

4. Start Ollama server — **keep this terminal open the entire time**:

```cmd
ollama serve
```

You should see: `Listening on 127.0.0.1:11434`

---

### STEP 2 — Install Anaconda (if not already installed)

Download from: **https://www.anaconda.com/download**
Install with default settings.

---

### STEP 3 — Create Python Environment

Open **Anaconda Prompt** (search in Start Menu):

```bash
conda create -n privacy_rag python=3.10 -y
conda activate privacy_rag
```

---

### STEP 4 — Navigate to Project Folder

```bash
cd C:\path\to\privacy_rag_project
```

Replace `C:\path\to\` with wherever you extracted the project files.

---

### STEP 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ Takes 5–10 minutes. If `hipporag` or `raganything` fail on Windows,
> the project automatically falls back to a direct Ollama RAG — your demo still works.

---

### STEP 6 — Download spaCy Language Model

```bash
python -m spacy download en_core_web_lg
```

---

### STEP 7 — Set Environment Variables

```bash
set OPENAI_API_KEY=ollama
set OPENAI_BASE_URL=http://localhost:11434/v1
```

> Do this every time you open a new Anaconda Prompt session.
> Or add them permanently via Windows → System Properties → Environment Variables.

---

## ▶️ RUNNING THE PROJECT

Make sure **`ollama serve` is running** in a separate terminal before each step.

---

### Run 1 — Phase 1: Baseline (Show the Problem)

```bash
python phase1_baseline.py
```

**What happens:**
- Downloads HotpotQA (30 samples) and PII-masking-200k (30 samples) from HuggingFace
- Indexes both datasets into HippoRAG-2 and RAG-Anything WITHOUT any privacy protection
- Runs probing queries targeting PII
- Shows red warnings when PII is leaked in responses
- Saves results to `results/phase1_baseline_results.json`

**Expected output:**
```
⚠ PII LEAKED: 3 items → {emails: [...], phones: [...]}
Response: John Smith can be reached at john@example.com...
```

> ⏱️ First run: ~10–15 min (dataset download + indexing)
> Subsequent runs: ~3–5 min (datasets cached in `data/cache/`)

---

### Run 2 — Phase 2: Privacy-Preserving RAG (Show the Fix)

```bash
python phase2_private_rag.py
```

**What happens:**
- Loads same documents from cache
- Applies PII masking layer to ALL documents before indexing
- Shows before/after example: `john@example.com → [EMAIL_REDACTED]`
- Runs same queries — leakage should drop dramatically
- Saves results to `results/phase2_private_results.json`

**Expected output:**
```
✅ 47 PII instances masked across 60 docs
✓ No PII leaked — masking effective
F1: 0.34 (similar to baseline — quality preserved)
```

---

### Run 3 — Compare Results (Full Evaluation Report)

```bash
python compare_results.py
```

**What happens:**
- Loads both phase results
- Prints full metrics table: PII leaked, avg entities/response, F1, EM
- Calculates PII reduction % and F1 delta
- Prints GDPR / HIPAA compliance analysis
- Saves `results/final_report.json`

---

### Run 4 — Generate Charts

```bash
python plot_results.py
```

**What happens:**
- Generates 3 publication-quality charts saved to `results/plots/`:
  1. `privacy_utility_tradeoff.png` — scatter plot with arrows (main result)
  2. `pii_leakage_comparison.png`   — bar chart: before vs after
  3. `f1_comparison.png`            — F1/EM quality comparison

---

### Run 5 — Interactive Live Demo

```bash
python interactive_demo.py
```

**What happens:**
- Loads and indexes both datasets (uses cache — fast after first run)
- Drops into interactive query loop
- Every question is answered TWICE: unsafe (red) and safe (green)
- Side-by-side PII count + F1 shown for each answer

**Built-in commands:**
| Command | Action |
|---------|--------|
| Any question | Answer from both unsafe and safe RAG |
| `suggest` | Show pre-tested demo questions |
| `show docs` | Print raw vs masked document side by side |
| `switch` | Switch between HotpotQA and PII-masking datasets |
| `log` | Show all queries from this session |
| `quit` | Save session log and exit |

---

## 📊 Expected Results

| Metric | HotpotQA Baseline | HotpotQA Private | PII-masking Baseline | PII-masking Private |
|--------|:-----------------:|:----------------:|:--------------------:|:-------------------:|
| Total PII leaked | ~10–20 | ~0–3 | ~30–60 | ~0–5 |
| Avg entities/response | ~0.5–1.0 | ~0.0–0.1 | ~1.5–3.0 | ~0.0–0.2 |
| PII reduction % | — | **~85–100%** | — | **~90–100%** |
| Avg F1 | ~0.30–0.40 | ~0.28–0.40 | N/A | N/A |
| F1 drop | — | **< 0.05** | — | — |

---

## 🎯 Evaluation Criteria (from Proposal)

| Criterion | How it's measured | Where |
|-----------|------------------|-------|
| PII leakage reduction | Count of PII instances before vs after | `compare_results.py` |
| Number of exposed entities per response | `avg_entities_per_response` metric | `compare_results.py` |
| Retrieval relevance | F1 + EM on HotpotQA gold answers | `eval_metrics.py` |
| Answer quality | F1 delta (private − baseline) | `compare_results.py` |
| Privacy–utility trade-off | Scatter plot + F1 delta table | `plot_results.py` |

---

## 🔧 Troubleshooting

**"ollama: command not found"**
→ Restart your terminal after installing Ollama.

**"Connection refused" errors**
→ Make sure `ollama serve` is running in a separate terminal.

**`hipporag` install fails on Windows**
→ The project auto-falls back to direct Ollama RAG. Your demo still works.

**Dataset download fails**
→ Check your internet connection. HuggingFace sometimes throttles — retry.

**Slow responses**
→ Normal on CPU. With RTX GPU, responses should be 2–5 seconds each.

**spaCy model not found**
→ Run: `python -m spacy download en_core_web_lg`

---

## 📚 References

1. Gutiérrez et al., "From RAG to Memory: Non-Parametric Continual Learning for LLMs (HippoRAG 2)", arXiv:2502.14802, 2025.
2. Guo et al., "RAG-Anything: All-in-One RAG Framework", arXiv:2510.12323, 2025.
3. Bodea et al., "SoK: Privacy Risks and Mitigations in RAG Systems", arXiv:2601.03979, 2026.
4. Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", EMNLP 2018.
5. Ai4Privacy, "PII-masking-200k", HuggingFace Datasets, 2024.
