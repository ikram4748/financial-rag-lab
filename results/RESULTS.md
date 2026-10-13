# Results

FinanceBench open-source sample: 150 questions over 84 filings (12,013 pages). Dev split: 50 questions, used to pick the configuration. Test split: the other 100, reported here. A retrieved chunk counts as a hit when it comes from a page FinanceBench lists as evidence.

**Configuration chosen on dev:** `structure|company+year|dense+hyde` (dev recall@5 80%).

## 1. Retrieval (test split, 100 questions)

| Chunking | Scope | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | ms / query |
|---|---|---|---|---|---|---|---|
| **structure** | company+year | dense+hyde | 30% | **60%** | 79% | 0.44 | - |
| page | company+year | dense | 26% | 53% | 63% | 0.38 | 7 |
| fixed | company+year | dense | 19% | 51% | 66% | 0.33 | 9 |
| page | company | dense | 20% | 46% | 54% | 0.31 | 9 |
| page | company+year | hybrid+rerank-bge | 22% | 44% | 60% | 0.33 | 756 |
| structure | company+year | dense | 18% | 44% | 56% | 0.30 | 8 |
| page | all | dense | 18% | 42% | 50% | 0.29 | 5 |
| structure | company+year | hybrid+rerank-bge | 15% | 42% | 56% | 0.27 | 785 |
| structure | all | hybrid+rerank-bge | 16% | 42% | 52% | 0.27 | 787 |
| page | company | hybrid+rerank-bge | 17% | 40% | 55% | 0.28 | 772 |
| structure | company+year | hybrid+rerank-minilm | 21% | 39% | 52% | 0.30 | 542 |
| structure | company | hybrid+rerank-bge | 15% | 39% | 51% | 0.26 | 786 |
| fixed | company | dense | 16% | 38% | 54% | 0.26 | 15 |
| page | company+year | hybrid+rerank-minilm | 20% | 37% | 51% | 0.29 | 523 |
| page | all | hybrid+rerank-bge | 17% | 37% | 56% | 0.28 | 763 |
| structure | company | dense | 13% | 34% | 46% | 0.23 | 9 |
| fixed | company+year | hybrid+rerank-minilm | 16% | 34% | 49% | 0.25 | 544 |
| fixed | all | dense | 15% | 34% | 45% | 0.23 | 10 |
| structure | all | dense | 13% | 33% | 45% | 0.23 | 6 |
| page | all | hybrid+rerank-minilm | 15% | 32% | 50% | 0.24 | 538 |
| structure | company | hybrid+rerank-minilm | 18% | 32% | 45% | 0.26 | 543 |
| structure | all | hybrid+rerank-minilm | 17% | 31% | 49% | 0.26 | 545 |
| fixed | company | hybrid+rerank-bge | 14% | 31% | 43% | 0.22 | 759 |
| fixed | company+year | hybrid+rerank-bge | 13% | 31% | 46% | 0.23 | 752 |
| page | company | hybrid+rerank-minilm | 14% | 31% | 47% | 0.23 | 534 |
| fixed | company | hybrid+rerank-minilm | 12% | 30% | 44% | 0.21 | 577 |
| fixed | all | hybrid+rerank-minilm | 13% | 28% | 37% | 0.20 | 855 |
| fixed | all | hybrid+rerank-bge | 12% | 27% | 34% | 0.19 | 2547 |
| fixed | company+year | hybrid | 12% | 26% | 38% | 0.19 | 449 |
| structure | all | hybrid | 12% | 21% | 33% | 0.17 | 328 |
| fixed | company | hybrid | 9% | 20% | 30% | 0.16 | 430 |
| structure | company+year | hybrid | 12% | 20% | 28% | 0.17 | 328 |
| structure | company | hybrid | 11% | 20% | 24% | 0.16 | 330 |
| page | company+year | hybrid | 11% | 19% | 26% | 0.16 | 309 |
| structure | company+year | bm25 | 11% | 18% | 22% | 0.14 | 319 |
| fixed | all | hybrid | 10% | 18% | 26% | 0.14 | 412 |
| page | company+year | bm25 | 6% | 17% | 20% | 0.11 | 300 |
| page | all | hybrid | 10% | 16% | 30% | 0.15 | 325 |
| structure | all | bm25 | 9% | 15% | 17% | 0.11 | 320 |
| fixed | company | bm25 | 6% | 15% | 17% | 0.10 | 421 |
| fixed | company+year | bm25 | 8% | 15% | 18% | 0.12 | 392 |
| page | company | hybrid | 9% | 15% | 20% | 0.13 | 312 |
| structure | company | bm25 | 10% | 15% | 18% | 0.13 | 319 |
| fixed | all | bm25 | 5% | 14% | 15% | 0.09 | 404 |
| page | company | bm25 | 6% | 12% | 16% | 0.09 | 310 |
| page | all | bm25 | 6% | 11% | 15% | 0.09 | 293 |

HyDE adds one LLM call per query (to write the hypothetical passage), so its latency is not comparable.

## 2. What each choice is worth (test recall@5, averaged over the other settings)

- **chunking:** page 30% · structure 30% · fixed 27%
- **scope:** company+year 33% · company 28% · all 27%
- **retriever:** dense 42% · hybrid+rerank-bge 37% · hybrid+rerank-minilm 33% · hybrid 19% · bm25 15%
- **HyDE** (same configuration, `structure|company+year|dense`): recall@5 44% → 60%

Scope filters remove filings before searching. The share of test questions whose evidence filing survives the filter: all 100%, company 100%, company+year 99%.

## 3. End-to-end answers (test split)

Same answer model and the same five-excerpt budget for every variant; graded against FinanceBench's gold answers by a different model (see the judge check below).

| Variant | Answered correctly | Evidence page in context | n |
|---|---|---|---|
| No retrieval (closed book) | **41%** | - | 100 |
| Naive RAG: fixed chunks, dense, all filings | **24%** | 34% | 100 |
| Best retrieval config (chosen on dev) | **47%** | 44% | 100 |
| Best config + HyDE | **61%** | 60% | 100 |
| Oracle: the evidence pages themselves | **84%** | 100% | 100 |

- `naive` → `best`: +23% [+13%, +34%] (paired bootstrap, 100 questions)

- `best` → `best_hyde`: +14% [+5%, +23%] (paired bootstrap, 100 questions)

- `best` → `oracle`: +37% [+26%, +48%] (paired bootstrap, 100 questions)

- `closed_book` → `best`: +6% [-6%, +17%] (paired bootstrap, 100 questions)

With the best configuration, answers were correct 68% of the time when the evidence page was retrieved (44 questions) and 30% when it was not (56).

| Question type | Best config | Oracle |
|---|---|---|
| domain-relevant | 53% | 91% |
| metrics-generated | 44% | 76% |
| novel-generated | 44% | 84% |

## 4. Can the judge be trusted?

The judge accepted **100%** of the gold answers themselves (100), and rejected **75%** of gold answers whose first number was changed by 15% (85).
