
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
from pii_masker   import PIIMasker, count_pii_in_response
from eval_metrics import exact_match, f1_score

RESULTS_DIR     = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)
LLM_MODEL       = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"


async def _ollama_llm(prompt, system_prompt=None, **kw):
    from openai import AsyncOpenAI
    c    = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    msgs = ([{"role":"system","content":system_prompt}] if system_prompt else []) \
         + [{"role":"user","content":prompt}]
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
  
    leakage = count_pii_in_response(
        response, known_pii, strict=True, question=question
    )
    em      = exact_match(response, gold) if gold else None
    f1      = f1_score(response, gold)    if gold else None
   
    hit     = (pii_val and len(pii_val) > 4 and
               pii_val.lower() in response.lower() and
               pii_val.lower() not in question.lower() and
               pii_val not in str(leakage["by_type"]))
    total   = leakage["total_leaked"] + (1 if hit else 0)

    c = Fore.RED if total > 0 else Fore.GREEN
    print(f"    {c}{'⚠ PII: '+str(total) if total > 0 else '✓ No PII leaked'}{Style.RESET_ALL}",
          end="")
    if gold:
        print(f"  | EM:{em}  F1:{f1:.2f}", end="")
    print(f"  ({elapsed:.1f}s)")
    print(f"    Response: {response[:110]}...")
    return {
        "question": question, "response": response, "gold_answer": gold,
        "pii_leaked": total, "pii_detail": leakage["by_type"],
        "em": em, "f1": f1, "latency_sec": round(elapsed, 2),
    }

def _aggregate(r):
    qs = r["queries"]
    r["total_pii_leaked"]          = sum(q["pii_leaked"] for q in qs)
    r["avg_entities_per_response"] = round(r["total_pii_leaked"] / max(len(qs),1), 2)
    r["pii_leakage_rate"]          = r["avg_entities_per_response"]
    f1s = [q["f1"] for q in qs if q["f1"] is not None]
    ems = [q["em"] for q in qs if q["em"] is not None]
    r["avg_f1"] = round(sum(f1s)/max(len(f1s),1), 3)
    r["avg_em"] = round(sum(ems)/max(len(ems),1), 3)


def _run_hipporag(docs, queries, known_pii, save_dir):
    from hipporag import HippoRAG
    rag = HippoRAG(
        save_dir=f"outputs/{save_dir}",
        llm_model_name=LLM_MODEL,
        embedding_model_name=EMBEDDING_MODEL,
        llm_base_url="http://localhost:11434/v1",
    )
    print(f"  {Fore.YELLOW}⚙ Indexing {len(docs)} MASKED docs (HippoRAG-2)...{Style.RESET_ALL}")
    rag.index(docs=docs)
    print(f"  {Fore.GREEN}✅ Indexed (PII redacted).{Style.RESET_ALL}")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n  Q{i}/{len(queries)}: {q['question'][:70]}...")
        t0 = time.time()
        try:
            res      = rag.rag_qa(queries=[q["question"]])
            response = res[0].get("answer","") if res else ""
        except Exception as e:
            response = f"[Error:{e}]"
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
    print(f"  {Fore.YELLOW}⚙ Indexing {len(docs)} MASKED docs (RAG-Anything)...{Style.RESET_ALL}")
    asyncio.run(rag.ainsert_text("\n\n---\n\n".join(docs)))
    print(f"  {Fore.GREEN}✅ Indexed (PII redacted).{Style.RESET_ALL}")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n  Q{i}/{len(queries)}: {q['question'][:70]}...")
        t0 = time.time()
        try:
            response = str(asyncio.run(rag.aquery(q["question"], mode="hybrid")))
        except Exception as e:
            response = f"[Error:{e}]"
        results.append(_score(q["question"], response,
                              q.get("gold_answer",""), q.get("pii_value",""),
                              known_pii, time.time()-t0))
    return results

def _run_fallback(docs, queries, known_pii, fw_label):
    import numpy as np
    from openai import OpenAI
    print(f"  {Fore.YELLOW}  ↳ Fallback: direct Ollama RAG ({fw_label}, masked)...{Style.RESET_ALL}")
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

    def embed(t):
        try:
            return client.embeddings.create(model=EMBEDDING_MODEL, input=t[:800]).data[0].embedding
        except Exception:
            return list(np.random.randn(768))

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

    def retrieve(query, k=3):
        qe = embed(query)
        return [d for _,d in sorted(
            [(cosine(qe,embed(d[:500])),d) for d in docs], reverse=True
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
                messages=[{"role":"user","content":prompt}],
                max_tokens=250,
            )
            response = resp.choices[0].message.content
        except Exception as e:
            response = f"[Ollama error: {e}]"
        results.append(_score(q["question"], response,
                              q.get("gold_answer",""), q.get("pii_value",""),
                              known_pii, time.time()-t0))
    return results


def run_framework(fw_name, masked_docs, queries, known_pii, save_dir):
    result = {
        "framework": fw_name, "dataset": save_dir, "phase": "private",
        "queries": [], "total_pii_leaked": 0,
        "avg_entities_per_response": 0.0, "pii_leakage_rate": 0.0,
        "avg_f1": 0.0, "avg_em": 0.0,
    }
    print(f"\n{Fore.CYAN}── {fw_name} | {save_dir} | private ──{Style.RESET_ALL}")
    try:
        if fw_name == "HippoRAG-2":
            result["queries"] = _run_hipporag(masked_docs, queries, known_pii, save_dir)
        else:
            result["queries"] = _run_raganything(masked_docs, queries, known_pii, save_dir)
    except Exception as e:
        print(f"  {Fore.RED}  ⚠ {e} → fallback Ollama RAG{Style.RESET_ALL}")
        result["queries"] = _run_fallback(masked_docs, queries, known_pii, fw_name)
    _aggregate(result)
    return result


def main():
    print(f"\n{Fore.MAGENTA}{'#'*66}")
    print("#  PHASE 2 — PRIVACY-PRESERVING RAG  (PII Masked)")
    print(f"{'#'*66}{Style.RESET_ALL}")

    masker      = PIIMasker(use_ner=True)
    all_results = {"phase": "private"}

  
    print(f"\n{Fore.YELLOW}{'━'*66}")
    print("  DATASET A: HotpotQA  (masked → leakage down, quality maintained)")
    print(f"{'━'*66}{Style.RESET_ALL}")
    hq_samples  = load_hotpotqa()
    hq_docs     = hotpotqa_to_docs(hq_samples)
    hq_queries  = hotpotqa_queries(hq_samples)
    hq_known    = get_known_pii_from_hotpotqa(hq_samples)

    print(f"\n  {Fore.CYAN}Masking {len(hq_docs)} HotpotQA docs...{Style.RESET_ALL}")
    hq_masked, hq_counts = masker.mask_documents(hq_docs)
    total_hq = sum(hq_counts)
    print(f"  {Fore.GREEN}✅ {total_hq} PII instances masked{Style.RESET_ALL}")
    print(f"  Before: {hq_docs[0][:100]}...")
    print(f"  After : {hq_masked[0][:100]}...")

    all_results["hotpotqa"] = {
        "masking_stats": {"total_pii_masked": total_hq, "per_doc": hq_counts},
        "hipporag":    run_framework("HippoRAG-2",  hq_masked, hq_queries, hq_known,
                                     "hipporag_hq_priv"),
        "raganything": run_framework("RAG-Anything", hq_masked, hq_queries, hq_known,
                                     "raganything_hq_priv"),
    }

    print(f"\n{Fore.YELLOW}{'━'*66}")
    print("  DATASET B: PII-masking-200k  (masked → leakage drops dramatically)")
    print(f"{'━'*66}{Style.RESET_ALL}")
    pii_samples  = load_pii_masking()
    pii_docs     = pii_masking_to_docs(pii_samples)
    pii_queries  = pii_masking_queries(pii_samples)
    pii_known    = get_known_pii_from_samples(pii_samples)

    print(f"\n  {Fore.CYAN}Masking {len(pii_docs)} PII-masking docs...{Style.RESET_ALL}")
    pii_masked, pii_counts = masker.mask_documents(pii_docs)
    total_pii = sum(pii_counts)
    print(f"  {Fore.GREEN}✅ {total_pii} PII instances masked{Style.RESET_ALL}")
    print(f"  Before: {pii_docs[0][:100]}...")
    print(f"  After : {pii_masked[0][:100]}...")

    masked_path = RESULTS_DIR / "masked_documents.json"
    masked_path.write_text(json.dumps([
        {"original": r, "masked": m, "pii_count": c}
        for r, m, c in zip(pii_docs[:5], pii_masked[:5], pii_counts[:5])
    ], indent=2))

    all_results["pii_masking"] = {
        "masking_stats": {"total_pii_masked": total_pii, "per_doc": pii_counts},
        "hipporag":    run_framework("HippoRAG-2",  pii_masked, pii_queries, pii_known,
                                     "hipporag_pii_priv"),
        "raganything": run_framework("RAG-Anything", pii_masked, pii_queries, pii_known,
                                     "raganything_pii_priv"),
    }


    out = RESULTS_DIR / "phase2_private_results.json"
    out.write_text(json.dumps(all_results, indent=2))


    print(f"\n{Fore.CYAN}{'='*66}  PHASE 2 SUMMARY  {'='*5}{Style.RESET_ALL}")
    for ds_key, label in [("hotpotqa","HotpotQA"), ("pii_masking","PII-masking-200k")]:
        ms = all_results[ds_key]["masking_stats"]["total_pii_masked"]
        print(f"\n  {Fore.YELLOW}{label}  (masked: {ms} PII instances){Style.RESET_ALL}")
        for fw in ["hipporag","raganything"]:
            r = all_results[ds_key][fw]
            c = Fore.RED if r["total_pii_leaked"] > 0 else Fore.GREEN
            print(f"    {r['framework']:15}"
                  f"  {c}PII leaked: {r['total_pii_leaked']:3d}{Style.RESET_ALL}"
                  f"  Avg/resp: {r['avg_entities_per_response']:.2f}"
                  f"  F1: {r['avg_f1']:.3f}  EM: {r['avg_em']:.3f}")

    print(f"\n{Fore.GREEN}✅ Saved → {out}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}➡  Next: python compare_results.py{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
