

import os, sys, json, time
from pathlib import Path
from colorama import init, Fore, Style
init(autoreset=True)


os.environ.setdefault("OPENAI_API_KEY",  "ollama")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")

sys.path.insert(0, str(Path(__file__).parent))
from data.datasets import (
    load_hotpotqa,  hotpotqa_to_docs,  hotpotqa_queries,
    get_known_pii_from_hotpotqa,
    load_pii_masking, pii_masking_to_docs, pii_masking_queries,
    get_known_pii_from_samples,
)
from pii_masker  import count_pii_in_response
from eval_metrics import exact_match, f1_score

RESULTS_DIR     = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)
LLM_MODEL       = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"




async def _ollama_llm(prompt, system_prompt=None, **kw):
    from openai import AsyncOpenAI
    c    = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
         + [{"role": "user",   "content": prompt}]
    r = await c.chat.completions.create(model=LLM_MODEL, messages=msgs)
    return r.choices[0].message.content

async def _ollama_embed(texts):
    from openai import AsyncOpenAI
    c = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    out = []
    for t in texts:
        r = await c.embeddings.create(model=EMBEDDING_MODEL, input=t[:800])
        out.append(r.data[0].embedding)
    return out




def _score(question, response, gold, pii_val, known_pii, elapsed):

    leakage  = count_pii_in_response(response, known_pii,
                                     strict=False, question=question)
    em       = exact_match(response, gold) if gold else None
    f1       = f1_score(response, gold)    if gold else None
    hit      = (pii_val and len(pii_val) > 4 and
                pii_val.lower() in response.lower() and
                pii_val.lower() not in question.lower()) if pii_val else False
    total    = leakage["total_leaked"] + (1 if hit and pii_val not in str(leakage["by_type"]) else 0)

    c = Fore.RED if total > 0 else Fore.GREEN
    print(f"    {c}⚠ PII leaked: {total}{Style.RESET_ALL}", end="")
    if gold:
        print(f"  | EM:{em}  F1:{f1:.2f}", end="")
    print(f"  ({elapsed:.1f}s)")
    print(f"    Response: {response[:110]}...")

    return {
        "question":   question,
        "response":   response,
        "gold_answer": gold,
        "pii_leaked": total,
        "pii_detail": leakage["by_type"],
        "em": em, "f1": f1,
        "latency_sec": round(elapsed, 2),
    }

def _aggregate(r):
    qs = r["queries"]
    r["total_pii_leaked"]        = sum(q["pii_leaked"] for q in qs)
    r["avg_entities_per_response"] = round(r["total_pii_leaked"] / max(len(qs), 1), 2)
    r["pii_leakage_rate"]        = r["avg_entities_per_response"]
    f1s = [q["f1"] for q in qs if q["f1"] is not None]
    ems = [q["em"] for q in qs if q["em"] is not None]
    r["avg_f1"] = round(sum(f1s) / max(len(f1s), 1), 3)
    r["avg_em"] = round(sum(ems) / max(len(ems), 1), 3)



def _run_hipporag(docs, queries, known_pii, save_dir):
    from hipporag import HippoRAG
    rag = HippoRAG(
        save_dir=f"outputs/{save_dir}",
        llm_model_name=LLM_MODEL,
        embedding_model_name=EMBEDDING_MODEL,
        llm_base_url="http://localhost:11434/v1",
    )
    print(f"  {Fore.YELLOW}⚙ Indexing {len(docs)} docs (HippoRAG-2)...{Style.RESET_ALL}")
    rag.index(docs=docs)
    print(f"  {Fore.GREEN}✅ Indexed.{Style.RESET_ALL}")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n  Q{i}/{len(queries)}: {q['question'][:70]}...")
        t0 = time.time()
        try:
            res      = rag.rag_qa(queries=[q["question"]])
            response = res[0].get("answer", "") if res else ""
        except Exception as e:
            response = f"[Error: {e}]"
        results.append(_score(q["question"], response,
                              q.get("gold_answer",""), q.get("pii_value",""),
                              known_pii, time.time()-t0))
    return results

def _run_raganything(docs, queries, known_pii, save_dir):
    from raganything import RAGAnything
    import asyncio
    rag = RAGAnything(
        llm_model_func=_ollama_llm,
        embedding_func=_ollama_embed,
        working_dir=f"outputs/{save_dir}",
    )
    print(f"  {Fore.YELLOW}⚙ Indexing {len(docs)} docs (RAG-Anything)...{Style.RESET_ALL}")
    asyncio.run(rag.ainsert_text("\n\n---\n\n".join(docs)))
    print(f"  {Fore.GREEN}✅ Indexed.{Style.RESET_ALL}")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n  Q{i}/{len(queries)}: {q['question'][:70]}...")
        t0 = time.time()
        try:
            response = str(asyncio.run(rag.aquery(q["question"], mode="hybrid")))
        except Exception as e:
            response = f"[Error: {e}]"
        results.append(_score(q["question"], response,
                              q.get("gold_answer",""), q.get("pii_value",""),
                              known_pii, time.time()-t0))
    return results

def _run_fallback(docs, queries, known_pii, fw_label):
    """Direct Ollama RAG — used when framework install fails on Windows."""
    import numpy as np
    from openai import OpenAI
    print(f"  {Fore.YELLOW}  ↳ Fallback: direct Ollama RAG ({fw_label})...{Style.RESET_ALL}")
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

    def embed(t):
        try:
            return client.embeddings.create(model=EMBEDDING_MODEL, input=t[:800]).data[0].embedding
        except Exception:
            return list(np.random.randn(768))

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

    def retrieve(query, k=3):
        qe = embed(query)
        return [d for _, d in sorted(
            [(cosine(qe, embed(d[:500])), d) for d in docs], reverse=True
        )[:k]]

    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n  Q{i}/{len(queries)}: {q['question'][:70]}...")
        t0  = time.time()
        ctx = "\n\n".join(retrieve(q["question"]))
        prompt = (f"Answer based only on the context below.\n\n"
                  f"Context:\n{ctx}\n\nQuestion: {q['question']}\n\nAnswer:")
        try:
            resp     = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
            )
            response = resp.choices[0].message.content
        except Exception as e:
            response = f"[Ollama error: {e}]"
        results.append(_score(q["question"], response,
                              q.get("gold_answer",""), q.get("pii_value",""),
                              known_pii, time.time()-t0))
    return results


def run_framework(fw_name, docs, queries, known_pii, save_dir, phase="baseline"):
    """Run one framework; fall back gracefully if it fails."""
    result = {
        "framework": fw_name, "dataset": save_dir, "phase": phase,
        "queries": [], "total_pii_leaked": 0,
        "avg_entities_per_response": 0.0, "pii_leakage_rate": 0.0,
        "avg_f1": 0.0, "avg_em": 0.0,
    }
    print(f"\n{Fore.CYAN}── {fw_name} | {save_dir} | {phase} ──{Style.RESET_ALL}")
    try:
        if fw_name == "HippoRAG-2":
            result["queries"] = _run_hipporag(docs, queries, known_pii, save_dir)
        else:
            result["queries"] = _run_raganything(docs, queries, known_pii, save_dir)
    except Exception as e:
        print(f"  {Fore.RED}  ⚠ {e} → fallback Ollama RAG{Style.RESET_ALL}")
        result["queries"] = _run_fallback(docs, queries, known_pii, fw_name)
    _aggregate(result)
    return result




def main():
    print(f"\n{Fore.MAGENTA}{'#'*66}")
    print("#  PHASE 1 — BASELINE RAG  (No Privacy Protection)")
    print(f"{'#'*66}{Style.RESET_ALL}")
    print(f"  LLM: {LLM_MODEL}  |  Make sure 'ollama serve' is running!\n")

    all_results = {"phase": "baseline"}


    print(f"\n{Fore.YELLOW}{'━'*66}")
    print("  DATASET A: HotpotQA  (Wikipedia — incidental PII)")
    print(f"{'━'*66}{Style.RESET_ALL}")
    hq_samples = load_hotpotqa()
    hq_docs    = hotpotqa_to_docs(hq_samples)
    hq_queries = hotpotqa_queries(hq_samples)
    hq_known   = get_known_pii_from_hotpotqa(hq_samples)  
    print(f"  Docs: {len(hq_docs)}  |  Queries: {len(hq_queries)}"
          f"  |  Known entities: {len(hq_known.get('NAMED_ENTITY',[]))}")

    all_results["hotpotqa"] = {
        "hipporag":    run_framework("HippoRAG-2",   hq_docs, hq_queries, hq_known,
                                     "hipporag_hq_base"),
        "raganything": run_framework("RAG-Anything",  hq_docs, hq_queries, hq_known,
                                     "raganything_hq_base"),
    }

    print(f"\n{Fore.YELLOW}{'━'*66}")
    print("  DATASET B: PII-masking-200k  (explicit PII — worst case)")
    print(f"{'━'*66}{Style.RESET_ALL}")
    pii_samples = load_pii_masking()
    pii_docs    = pii_masking_to_docs(pii_samples)
    pii_queries = pii_masking_queries(pii_samples)
    pii_known   = get_known_pii_from_samples(pii_samples)
    print(f"  Docs: {len(pii_docs)}  |  Queries: {len(pii_queries)}"
          f"  |  PII types: {list(pii_known.keys())[:5]}")

    all_results["pii_masking"] = {
        "hipporag":    run_framework("HippoRAG-2",   pii_docs, pii_queries, pii_known,
                                     "hipporag_pii_base"),
        "raganything": run_framework("RAG-Anything",  pii_docs, pii_queries, pii_known,
                                     "raganything_pii_base"),
    }


    out = RESULTS_DIR / "phase1_baseline_results.json"
    out.write_text(json.dumps(all_results, indent=2))

  
    print(f"\n{Fore.CYAN}{'='*66}  PHASE 1 SUMMARY  {'='*5}{Style.RESET_ALL}")
    for ds_key, label in [("hotpotqa","HotpotQA"), ("pii_masking","PII-masking-200k")]:
        print(f"\n  {Fore.YELLOW}{label}{Style.RESET_ALL}")
        for fw in ["hipporag","raganything"]:
            r = all_results[ds_key][fw]
            c = Fore.RED if r["total_pii_leaked"] > 0 else Fore.GREEN
            print(f"    {r['framework']:15}"
                  f"  {c}PII leaked: {r['total_pii_leaked']:3d}{Style.RESET_ALL}"
                  f"  Avg/resp: {r['avg_entities_per_response']:.2f}"
                  f"  F1: {r['avg_f1']:.3f}  EM: {r['avg_em']:.3f}")

    print(f"\n{Fore.GREEN}✅ Saved → {out}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}➡  Next: python phase2_private_rag.py{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
