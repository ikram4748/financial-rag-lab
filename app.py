"""Streamlit demo: ask questions about 84 company filings, see the answer next to the pages it came from.

    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from ragl import chunking
from ragl.config import ROOT
from ragl.corpus import doc_meta, load_pages, load_questions
from ragl.generation import Generator
from ragl.llm import LLMUnavailable
from ragl.retrieval import SCOPES, Index, scope

st.set_page_config(page_title="Financial RAG Lab", page_icon="🏦", layout="wide")
RERANKERS = {"none": None, "MiniLM-L12 cross-encoder": "Xenova/ms-marco-MiniLM-L-12-v2", "bge-reranker-base": "BAAI/bge-reranker-base"}


@st.cache_resource(show_spinner="Loading filings and building the index (first time only)...")
def index(name: str) -> Index:
    return Index(name, chunking.STRATEGIES[name](load_pages(), doc_meta()))


@st.cache_resource
def generator() -> Generator:
    return Generator()


def main() -> None:
    st.title("🏦 Financial RAG Lab")
    st.caption("Questions over 84 public company filings (10-K, 10-Q, 8-K, earnings) from the FinanceBench benchmark. "
               "Change the retrieval settings in the sidebar and compare what gets retrieved.")
    if not (ROOT / "data" / "processed" / "pages.jsonl").exists() and not (ROOT / "data" / "raw" / "pdfs").exists():
        st.error("Run `python scripts/setup_data.py` first.")
        return
    tab_ask, tab_results = st.tabs(["Ask", "Benchmark results"])
    with tab_results:
        p = ROOT / "results" / "RESULTS.md"
        st.markdown(p.read_text(encoding="utf-8") if p.exists() else "Run the experiments to produce results.")
        for f in ("recall_at_k.png",):
            if (ROOT / "figures" / f).exists():
                st.image(str(ROOT / "figures" / f))

    st.sidebar.header("Retrieval")
    # defaults = the configuration chosen on the dev split (structure chunks, company+year scope, dense, no reranker)
    strat = st.sidebar.selectbox("Chunking", list(chunking.STRATEGIES), index=list(chunking.STRATEGIES).index("structure"))
    method = st.sidebar.selectbox("Retriever", ["dense", "hybrid", "bm25"])
    rr = st.sidebar.selectbox("Reranker", list(RERANKERS), index=0)
    sc = st.sidebar.selectbox("Scope", SCOPES, index=list(SCOPES).index("company+year"),
                              help="Only search the filings of the company (and year) named in the question.")
    k = st.sidebar.slider("Excerpts given to the model", 3, 10, 5)

    with tab_ask:
        examples = [q.question for q in load_questions()[:150:15]]
        pick = st.selectbox("Example questions from FinanceBench", ["(write your own)"] + examples)
        question = st.text_input("Question", value="" if pick.startswith("(") else pick)
        if not st.button("Answer", type="primary", disabled=not question.strip()):
            return
        meta = doc_meta()
        allowed = scope(question, meta, sc)
        ix = index(strat)
        with st.spinner("Retrieving..."):
            ids = ix.search(question, method, allowed, k, RERANKERS[rr])
        chunks = [ix.chunks[i] for i in ids]
        st.caption("Searched " + ("all 84 filings" if allowed is None else ", ".join(sorted(allowed))))
        try:
            with st.spinner("Answering..."):
                answer = generator().answer(question, chunks, meta)
        except LLMUnavailable as e:
            st.error(str(e))
            answer = None
        if answer:
            st.subheader("Answer")
            st.write(answer)
        gold = next((q for q in load_questions() if q.question == question), None)
        if gold:
            st.info(f"FinanceBench reference answer: **{gold.answer}**  \nEvidence: "
                    + ", ".join(f"{d} page {p + 1}" for d, p in gold.evidence))
        st.subheader("Retrieved excerpts")
        for n, c in enumerate(chunks, 1):
            m = meta.get(c.doc, {})
            hit = gold and any((c.doc, p) in set(gold.evidence) for p in c.pages)
            with st.expander(f"{'✅ ' if hit else ''}[{n}] {m.get('company', c.doc)} {m.get('doc_type', '').upper()} "
                             f"{m.get('period', '')} · page {', '.join(str(p + 1) for p in c.pages)}"):
                st.text(c.display or c.text)


main()
