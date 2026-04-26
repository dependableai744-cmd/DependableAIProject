# Privacy-Preserving RAG System

**Dependable AI — Course Project | University of Western Ontario**  
**Authors:** Saksham Bhatnagar, Harshavardhan Panguluri

---

## What Is This Project?

This project is about making AI question-answering systems safer when it comes to personal data. We built a privacy layer that prevents AI from accidentally leaking sensitive information like names, email addresses, phone numbers, or social security numbers.

Modern AI systems called **RAG (Retrieval-Augmented Generation)** work by searching through a collection of documents and using what they find to answer your questions. The problem is — if those documents contain private information, the AI will repeat that information back in its answers, even when it shouldn't.

### The Surprising Finding

You might assume the privacy risk only exists if someone puts sensitive data into the system on purpose. Our research found that's not true. Even everyday Wikipedia-style articles leak private details — names, locations, and organisation mentions — just by being part of the AI's knowledge base. That's the key finding of this project.

### Our Solution

We built a masking layer that sits in front of the AI pipeline. Before any document gets processed by the AI, our layer scans it and hides any personal information it finds using two methods:

- A language model (spaCy) that recognises names, places, and organisations in text
- 13 pattern-matching rules that catch structured data like emails, phone numbers, credit card numbers, bank accounts, and passwords

Anything flagged gets replaced with a placeholder like `[EMAIL_REDACTED]` or `[PERSON_REDACTED]`. The AI then works with the cleaned version of the document, so it simply never sees the sensitive data in the first place.

Importantly, this works as a **plug-in layer** — we didn't change the AI itself at all. It can be dropped onto any existing RAG system.

---

## Project Structure

```
privacy_rag_project/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── datasets.py              ← Loads HotpotQA + PII-masking-200k from HuggingFace
├── pii_masker.py                ← PII detection & masking (spaCy NER + regex)
├── eval_metrics.py              ← Exact Match + F1 scoring
├── phase1_baseline.py           ← RAG with no protection — shows the leakage problem
├── phase2_private_rag.py        ← RAG with masking applied — shows the fix
├── compare_results.py           ← Full evaluation report
├── plot_results.py              ← Generates trade-off charts
└── results/                     ← Auto-created when you run the scripts
```

---

## Setup

### Prerequisites

- Windows 10/11
- Python 3.10 (via Anaconda or Miniconda)
- ~5 GB free disk space
- Internet connection (required for first run to download models and datasets)
- NVIDIA GPU recommended, but not required

---

### Step 1 — Install Ollama

Ollama lets you run AI models locally for free — no paid API needed.

1. Download and run the installer from [https://ollama.com/download](https://ollama.com/download)
2. Open Command Prompt and pull the two required models:

```cmd
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

> This downloads about 5 GB total — start this early while you set up the rest.

3. Start the Ollama server and **keep this terminal open the entire time**:

```cmd
ollama serve
```

You should see: `Listening on 127.0.0.1:11434`

---

### Step 2 — Install Anaconda

If you don't have it already, download from [https://www.anaconda.com/download](https://www.anaconda.com/download) and install with default settings.

---

### Step 3 — Create a Python Environment

Open **Anaconda Prompt** (search in the Start Menu) and run:

```bash
conda create -n privacy_rag python=3.10 -y
conda activate privacy_rag
```

---

### Step 4 — Navigate to the Project Folder

```bash
cd C:\path\to\privacy_rag_project
```

Replace the path with wherever you extracted the project files.

---

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

> Takes about 5–10 minutes. If `hipporag` or `raganything` fail to install on Windows, don't worry — the project automatically falls back to a simpler setup that still works for the demo.

---

### Step 6 — Download the spaCy Language Model

```bash
python -m spacy download en_core_web_lg
```

---

### Step 7 — Set Environment Variables

Run these in your Anaconda Prompt before using the project. You'll need to repeat this each time you open a new session (or set them permanently via Windows → System Properties → Environment Variables):

```bash
set OPENAI_API_KEY=ollama
set OPENAI_BASE_URL=http://localhost:11434/v1
```

---

## Running the Project

> Make sure `ollama serve` is running in a separate terminal before each step.

---

### Run 1 — Phase 1: Show the Problem

```bash
python phase1_baseline.py
```

Downloads 30 samples from each dataset, indexes them into the AI **without any privacy protection**, and runs probing queries. You'll see warnings whenever the AI leaks private information in a response. Results are saved to `results/phase1_baseline_results.json`.

> First run takes 10–15 minutes (downloading + indexing). Subsequent runs are much faster since data is cached.

---

### Run 2 — Phase 2: Apply the Fix

```bash
python phase2_private_rag.py
```

Loads the same documents from cache, runs them through the PII masking layer, then indexes the cleaned versions. You'll see a before/after example (e.g. `john@example.com` → `[EMAIL_REDACTED]`), and the same queries run again with dramatically less leakage. Results are saved to `results/phase2_private_results.json`.

---

### Run 3 — Compare Results

```bash
python compare_results.py
```

Loads both sets of results and prints a full metrics table — how much PII leaked, how many entities appeared per response, F1 and Exact Match scores, and the percentage reduction in leakage. Saves a final report to `results/final_report.json`.

---

### Run 4 — Generate Charts

```bash
python plot_results.py
```

Produces three charts saved to `results/plots/`:
- A scatter plot showing the privacy vs. utility trade-off
- A bar chart comparing leakage before and after masking
- An F1 quality comparison

---

## Troubleshooting

**"ollama: command not found"**  
Restart your terminal after installing Ollama.

**"Connection refused" errors**  
Make sure `ollama serve` is running in a separate terminal window.

**`hipporag` install fails**  
Expected on Windows sometimes. The project falls back automatically and everything still works.

**Dataset download fails**  
Check your internet connection. HuggingFace occasionally rate-limits downloads — just retry.

**Slow responses**  
Normal when running on CPU. With an NVIDIA GPU, responses typically take 2–5 seconds each.

**"spaCy model not found"**  
Re-run: `python -m spacy download en_core_web_lg`
