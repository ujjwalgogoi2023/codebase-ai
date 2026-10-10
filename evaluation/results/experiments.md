# Retrieval experiments

55 questions, dataset `dataset.json`.

| Configuration | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| naive chunking | 23.6% | 72.7% | 85.5% | 0.475 |
| code-aware chunking | 23.6% | 54.5% | 72.7% | 0.435 |
| code-aware + reranking | 38.2% | 70.9% | 78.2% | 0.552 |
| code-aware, top_k=3 | 23.6% | 54.5% | 54.5% | 0.373 |
| code-aware, top_k=5 | 23.6% | 54.5% | 72.7% | 0.416 |
| code-aware, top_k=10 | 23.6% | 54.5% | 72.7% | 0.435 |
| chunk_lines=40 | 27.3% | 58.2% | 67.3% | 0.467 |
| chunk_lines=120 | 23.6% | 54.5% | 72.7% | 0.435 |
| chunk_lines=250 | 18.2% | 45.5% | 61.8% | 0.371 |
