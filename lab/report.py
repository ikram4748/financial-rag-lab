"""Build results/RESULTS.md and figures/ from the saved runs.

    python -m lab.report
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragl.corpus import load_questions, split  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
VARIANT_LABEL = {"closed_book": "No retrieval (closed book)", "naive": "Naive RAG: fixed chunks, dense, all filings",
                 "best": "Best retrieval config (chosen on dev)", "best_hyde": "Best config + HyDE",
                 "oracle": "Oracle: the evidence pages themselves"}


def retrieval_table(ranks: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    d = ranks[ranks.qid.isin(ids)].copy()
    fh = d.first_hit.fillna(10 ** 6)
    for k in (1, 5, 10, 20):
        d[f"r@{k}"] = fh <= k
    d["rr"] = 1 / fh
    return d.groupby(["config", "chunking", "scope", "retriever"]).agg(
        **{f"recall@{k}": (f"r@{k}", "mean") for k in (1, 5, 10, 20)}, mrr=("rr", "mean"),
        ms=("seconds", lambda s: s.mean() * 1000), in_scope=("evidence_in_scope", "mean")).reset_index()


def paired_ci(a: pd.Series, b: pd.Series, iters: int = 5000) -> tuple[float, float, float]:
    diffs = (b.astype(float) - a.astype(float)).tolist()
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return sum(diffs) / len(diffs), boots[int(0.025 * iters)], boots[int(0.975 * iters)]


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    qs = load_questions()
    dev_ids, test_ids = {q.id for q in split(qs, "dev")}, {q.id for q in split(qs, "test")}
    ranks = pd.read_csv(RESULTS / "retrieval_ranks.csv")
    dev, test = retrieval_table(ranks, dev_ids), retrieval_table(ranks, test_ids)
    best = dev.sort_values(["recall@5", "mrr", "ms"], ascending=[False, False, True]).iloc[0]
    L = ["# Results", "",
         f"FinanceBench open-source sample: 150 questions over 84 filings (12,013 pages). Dev split: 50 questions, used to "
         f"pick the configuration. Test split: the other 100, reported here. A retrieved chunk counts as a hit when it comes "
         f"from a page FinanceBench lists as evidence.", "",
         f"**Configuration chosen on dev:** `{best.config}` (dev recall@5 {best['recall@5']:.0%}).", "",
         "## 1. Retrieval (test split, 100 questions)", "",
         "| Chunking | Scope | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | ms / query |", "|---|---|---|---|---|---|---|---|"]
    for _, r in test.sort_values("recall@5", ascending=False).iterrows():
        mark = "**" if r.config == best.config else ""
        L.append(f"| {mark}{r.chunking}{mark} | {r.scope} | {r.retriever} | {r['recall@1']:.0%} | {mark}{r['recall@5']:.0%}{mark} | "
                 f"{r['recall@10']:.0%} | {r.mrr:.2f} | {'-' if pd.isna(r.ms) else f'{r.ms:.0f}'} |")
    if test.retriever.str.endswith("+hyde").any():
        L += ["", "HyDE adds one LLM call per query (to write the hypothetical passage), so its latency is not comparable."]

    # marginal effect of each choice: average recall@5 over all other settings
    L += ["", "## 2. What each choice is worth (test recall@5, averaged over the other settings)", ""]
    grid = test[~test.retriever.str.endswith("+hyde")]          # HyDE was only run on one configuration
    for dim in ("chunking", "scope", "retriever"):
        g = grid.groupby(dim)["recall@5"].mean().sort_values(ascending=False)
        L.append(f"- **{dim}:** " + " · ".join(f"{k} {v:.0%}" for k, v in g.items()))
    for _, h in test[test.retriever.str.endswith("+hyde")].iterrows():
        base = test[test.config == h.config.removesuffix("+hyde")]
        if len(base):
            L.append(f"- **HyDE** (same configuration, `{base.iloc[0].config}`): recall@5 {base.iloc[0]['recall@5']:.0%} → "
                     f"{h['recall@5']:.0%}")
    ins = test.groupby("scope")["in_scope"].mean()
    L += ["", "Scope filters remove filings before searching. The share of test questions whose evidence filing survives "
          "the filter: " + ", ".join(f"{k} {v:.0%}" for k, v in ins.items()) + "."]

    # recall@k figure for a few telling configurations
    show = [c for c in [f"fixed|all|dense", "fixed|all|bm25", "structure|all|hybrid", best.config.removesuffix("+hyde"),
                        best.config + "+hyde"] if c in set(ranks.config)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for c in dict.fromkeys(show):
        d = ranks[(ranks.config == c) & ranks.qid.isin(test_ids)]
        ks = list(range(1, 21))
        ax.plot(ks, [(d.first_hit.fillna(1e6) <= k).mean() * 100 for k in ks], marker=".", label=c.replace("|", " · "))
    ax.set_xlabel("k (chunks retrieved)")
    ax.set_ylabel("evidence page found (%)")
    ax.set_title("Recall@k on the test split")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES / "recall_at_k.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # generation
    gen_dir = RESULTS / "generation"
    rows = []
    for v in VARIANT_LABEL:
        p = gen_dir / f"{v}.jsonl"
        if p.exists():
            rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    if rows:
        g = pd.DataFrame(rows)
        L += ["", "## 3. End-to-end answers (test split)", "",
              "Same answer model and the same five-excerpt budget for every variant; graded against FinanceBench's gold answers "
              "by a different model (see the judge check below).", "",
              "| Variant | Answered correctly | Evidence page in context | n |", "|---|---|---|---|"]
        for v, lab in VARIANT_LABEL.items():
            s = g[g.variant == v]
            if len(s):
                ev = "-" if s.evidence_retrieved.isna().all() else f"{s.evidence_retrieved.mean():.0%}"
                L.append(f"| {lab} | **{s.correct.mean():.0%}** | {ev} | {len(s)} |")
        wide = g.pivot_table(index="qid", columns="variant", values="correct", aggfunc="first")
        for a, b in (("naive", "best"), ("best", "best_hyde"), ("best", "oracle"), ("closed_book", "best")):
            if a in wide and b in wide:
                w = wide[[a, b]].dropna()
                m, lo, hi = paired_ci(w[a], w[b])
                L.append(f"\n- `{a}` → `{b}`: {m:+.0%} [{lo:+.0%}, {hi:+.0%}] (paired bootstrap, {len(w)} questions)")
        if "best" in wide:
            b = g[g.variant == "best"]
            hit, miss = b[b.evidence_retrieved == True], b[b.evidence_retrieved == False]  # noqa: E712
            L += ["", f"With the best configuration, answers were correct {hit.correct.mean():.0%} of the time when the evidence "
                  f"page was retrieved ({len(hit)} questions) and {miss.correct.mean():.0%} when it was not ({len(miss)})."]
            by = g[g.variant.isin(["best", "oracle"])].groupby(["qtype", "variant"]).correct.mean().unstack()
            L += ["", "| Question type | Best config | Oracle |", "|---|---|---|"]
            for t, r in by.iterrows():
                L.append(f"| {t} | {r.get('best', float('nan')):.0%} | {r.get('oracle', float('nan')):.0%} |")
    jc = gen_dir / "judge_check.jsonl"
    if jc.exists():
        j = pd.DataFrame([json.loads(line) for line in jc.read_text(encoding="utf-8").splitlines()])
        c = j.dropna(subset=["accepts_corrupted"])
        L += ["", "## 4. Can the judge be trusted?", "",
              f"The judge accepted **{j.accepts_gold.mean():.0%}** of the gold answers themselves ({len(j)}), and rejected "
              f"**{(~c.accepts_corrupted.astype(bool)).mean():.0%}** of gold answers whose first number was changed by 15% "
              f"({len(c)})."]
    (RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
