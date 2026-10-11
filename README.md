# Financial RAG Lab

A benchmark lab for retrieval-augmented generation over real company filings. Instead of building one RAG pipeline
and hoping it works, it **measures which design choices matter**: chunking, metadata filtering, keyword vs dense vs
hybrid retrieval, cross-encoder reranking and HyDE. Each is scored on whether the *exact evidence page* is retrieved,
and then end to end on answer accuracy.

Data: [FinanceBench](https://github.com/patronus-ai/financebench) (open-source sample): **150 analyst questions over
84 filings** (10-K, 10-Q, 8-K and earnings releases of 32 US companies, 12,013 pages), each with a verified answer and
the page(s) that contain the evidence.

> **Results: in progress.** The code, tests and demo are complete. The full experiment grid (embedding ~140k chunks
> and cross-encoder reranking) is being run on a GPU machine; the tables below will be filled from
> `results/RESULTS.md` when it finishes.

## The experiment

| Dimension | Options compared |
|---|---|
| **Chunking** | `fixed` 800-character windows (the usual default) · `page` never crosses a page boundary · `structure` paragraph/table blocks with a "[company \| filing \| year]" header (cheap contextual retrieval) |
| **Scope** | all 84 filings · the company named in the question · company and year |
| **Retriever** | BM25 · dense (bge-small, ONNX) · hybrid (reciprocal rank fusion) · hybrid + MiniLM-L12 cross-encoder · hybrid + bge-reranker-base |
| **Query** | the question · HyDE (embed an LLM-written hypothetical answer) |
| **Generation** | closed book · naive RAG · best configuration · best + HyDE · oracle (the evidence pages themselves) |

Method:
- **Retrieval is scored without an LLM.** A retrieved chunk is a hit when it comes from a FinanceBench evidence page,
  so recall@k and MRR can be computed for all 45 retrieval configurations at no API cost.
- **Dev/test split.** The configuration is chosen on 50 dev questions and reported on the other 100, so picking the
  best of 45 does not inflate the result.
- **Answers are graded by a different model** from the one that wrote them, and the judge is checked: it must accept
  the gold answers and reject gold answers with one number changed by 15%.
- **GraphRAG is deliberately not included.** It needs an LLM pass over every chunk to build its graph, tens of
  thousands of calls for this corpus.

## Results

*To be filled from `results/RESULTS.md`.*

## Run it

**Windows:** double-click `run.bat`. **macOS:** double-click `run.command`. The first run installs packages, downloads
FinanceBench (~170 MB) and opens the demo, where you can switch chunking, retriever, reranker and scope and see which
pages come back.

```bash
pip install -r requirements.txt
python scripts/setup_data.py                 # questions + 84 filings (CC BY-NC 4.0; downloaded, not redistributed)
python -m lab.run_retrieval                  # retrieval grid, no API key needed (GPU: see requirements.txt)
python -m lab.run_generation hyde            # HyDE passages (API)
python -m lab.run_generation answer          # end-to-end answers + grading (API)
python -m lab.run_generation judge-check     # validate the judge (API)
python -m lab.report                         # results/RESULTS.md + figures/
streamlit run app.py
python -m pytest                             # 11 offline tests
```

## Project layout

```
ragl/
  corpus.py       questions, evidence pages (0-based, verified against the PDFs), page parsing (PyMuPDF), dev/test split
  chunking.py     fixed / page / structure chunkers, each chunk keeps its page provenance
  retrieval.py    BM25, dense, hybrid (RRF), cross-encoder reranking, company/year scope from the question
  generation.py   answer prompt, closed-book prompt, HyDE, judge
  llm.py          Gemini client: two keys, strict model preference, rate limiting, local cache
lab/
  run_retrieval.py    the retrieval grid (first evidence-page rank per question and configuration)
  run_generation.py   HyDE, end-to-end answers, judge check
  report.py           tables and figures
app.py              Streamlit demo
tests/              offline tests
```

## Stack

Python · PyMuPDF · rank-bm25 · fastembed (bge-small-en-v1.5, MiniLM / bge cross-encoders, ONNX) · Gemini
(`google-genai`) · pandas · matplotlib · Streamlit · pytest

## License

Code: MIT. FinanceBench (Islam et al., 2023, Patronus AI) is CC BY-NC 4.0 and is downloaded by the setup script, not
included in this repository.
