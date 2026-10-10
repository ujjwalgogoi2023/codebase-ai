# Retrieval experiments

45 questions, dataset `dataset_flask.json`.

| Configuration | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| naive chunking | 53.3% | 68.9% | 84.4% | 0.650 |
| code-aware chunking | 60.0% | 84.4% | 91.1% | 0.729 |
| code-aware + reranking | 77.8% | 91.1% | 93.3% | 0.845 |
| code-aware, top_k=3 | 60.0% | 84.4% | 84.4% | 0.707 |
| code-aware, top_k=5 | 60.0% | 84.4% | 91.1% | 0.723 |
| code-aware, top_k=10 | 60.0% | 84.4% | 91.1% | 0.729 |
| chunk_lines=40 | 60.0% | 84.4% | 91.1% | 0.733 |
| chunk_lines=120 | 60.0% | 84.4% | 91.1% | 0.729 |
| chunk_lines=250 | 57.8% | 84.4% | 86.7% | 0.711 |
